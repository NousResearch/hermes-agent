"""Context continuity helpers for quality-preserving session handoff.

This module is intentionally separate from ``context_compressor``.  Compression
keeps an oversized session alive; continuity handoff creates an explicit packet
that a fresh session can use to resume work without inheriting all historical
noise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ContextContinuityStatus:
    """Inputs used to recommend the next context-continuity action."""

    context_tokens: int = 0
    context_length: int = 0
    remaining_todos: int = 0
    compression_count: int = 0
    high_risk_task: bool = False
    tool_call_count: int = 0
    failed_attempts: int = 0


@dataclass(frozen=True)
class ContextContinuityRecommendation:
    """Recommendation returned by ``recommend_continuity_action``."""

    level: str
    recommended_action: str
    usage_percent: int
    reason: str


@dataclass(frozen=True)
class AutomaticCompressionGate:
    """Decision for proactive automatic compression.

    This gate intentionally applies only to proactive compression. Explicit
    user requests such as ``/compress`` and emergency provider-error recovery
    can still use the compressor as a survival fallback.
    """

    defer: bool
    recommended_action: str
    usage_percent: int
    reason: str


def _usage_percent(context_tokens: int, context_length: int) -> int:
    if context_length <= 0:
        return 0
    return max(0, min(100, round((max(0, context_tokens) / context_length) * 100)))


def recommend_continuity_action(
    status: ContextContinuityStatus,
) -> ContextContinuityRecommendation:
    """Recommend continue/checkpoint/handoff/stop based on context risk.

    Policy: handoff is preferred before lossy compression for quality risk;
    compression remains a safety fallback elsewhere.
    """

    pct = _usage_percent(status.context_tokens, status.context_length)
    risk = pct

    if status.high_risk_task:
        risk += 10
    risk += min(10, status.remaining_todos * 2)
    risk += min(8, status.compression_count * 4)
    risk += min(6, status.failed_attempts * 2)
    if status.tool_call_count >= 20:
        risk += 5

    if pct >= 90:
        return ContextContinuityRecommendation(
            level="hard_stop",
            recommended_action="handoff_required",
            usage_percent=pct,
            reason="대화가 한계에 가까워졌습니다. 더 진행하기 전에 새 세션용 이어가기 안내를 만드세요.",
        )
    if pct >= 85 or risk >= 85:
        return ContextContinuityRecommendation(
            level="strong_handoff",
            recommended_action="handoff",
            usage_percent=pct,
            reason="다음 단계는 현재 세션을 줄이기보다 새 세션으로 넘기는 편이 안전합니다.",
        )
    if pct >= 75 or risk >= 75:
        return ContextContinuityRecommendation(
            level="handoff_recommended",
            recommended_action="handoff",
            usage_percent=pct,
            reason="대화가 길어져 품질이 떨어질 수 있습니다. 이어가기 안내를 준비하세요.",
        )
    if pct >= 65 or risk >= 65:
        return ContextContinuityRecommendation(
            level="checkpoint",
            recommended_action="checkpoint",
            usage_percent=pct,
            reason="작업이 길어지고 있습니다. 다음 큰 단계 전에 체크포인트를 남기는 편이 좋습니다.",
        )
    return ContextContinuityRecommendation(
        level="continue",
        recommended_action="continue",
        usage_percent=pct,
        reason="아직 현재 세션에서 계속 진행해도 됩니다.",
    )


def should_defer_automatic_compression(
    status: ContextContinuityStatus,
) -> AutomaticCompressionGate:
    """Defer proactive automatic compression behind continuity policy.

    Automatic compression is lossy and can hide context loss from the user. For
    proactive threshold checks, prefer visible checkpoint/handoff guidance and
    let explicit ``/compress`` or emergency context-overflow recovery remain the
    only compression paths.
    """

    recommendation = recommend_continuity_action(status)
    if recommendation.recommended_action in {"handoff", "handoff_required"}:
        reason = (
            f"{recommendation.reason} "
            "권장: /m으로 새 세션을 만들거나 /h로 이동 준비 인계문을 만드세요. "
            "같은 세션 유지가 꼭 필요할 때만 /c를 사용하세요."
        )
    elif recommendation.recommended_action == "checkpoint":
        reason = (
            f"{recommendation.reason} "
            "다음 큰 단계 전에 /h 또는 /m으로 이어가기 지점을 준비하세요."
        )
    else:
        reason = (
            "현재 세션은 계속 사용할 수 있습니다. 단, 긴 작업에서는 /m 세션 이동이 "
            "손실 압축보다 품질 보존에 유리합니다."
        )
    return AutomaticCompressionGate(
        defer=True,
        recommended_action=recommendation.recommended_action,
        usage_percent=recommendation.usage_percent,
        reason=reason,
    )


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(p for p in parts if p).strip()
    return str(content).strip()


def _first_text(messages: Iterable[Mapping[str, Any]], role: str) -> str:
    for msg in messages:
        if msg.get("role") == role:
            text = _message_text(msg)
            if text:
                return text
    return ""


def _last_text(messages: Iterable[Mapping[str, Any]], role: str) -> str:
    last = ""
    for msg in messages:
        if msg.get("role") == role:
            text = _message_text(msg)
            if text:
                last = text
    return last


def _truncate(text: str, limit: int = 700) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."



def _outline_messages(messages: list[Mapping[str, Any]], *, start_index: int = 0, limit: int = 24) -> str:
    visible = [m for m in messages if m.get("role") in {"user", "assistant"} and _message_text(m)]
    if not visible:
        return "표시 가능한 user/assistant 메시지가 없습니다."
    rows: list[str] = []
    for idx, msg in enumerate(visible[-limit:], start=max(start_index + 1, len(visible) - limit + 1)):
        role = "User" if msg.get("role") == "user" else "Hermes"
        rows.append(f"{idx:03d}. {role}: {_truncate(_message_text(msg), 320)}")
    if len(visible) > limit:
        rows.insert(0, f"[앞부분 {len(visible) - limit}개 메시지는 길이 보호로 생략]")
    return "\n".join(rows)


def _matching_evidence(messages: list[Mapping[str, Any]], needles: tuple[str, ...], *, limit: int = 3) -> str:
    rows: list[str] = []
    for msg in reversed(messages):
        text = _message_text(msg)
        if not text:
            continue
        hay = text.lower()
        if any(n.lower() in hay for n in needles):
            role = "User" if msg.get("role") == "user" else "Hermes" if msg.get("role") == "assistant" else str(msg.get("role") or "message")
            rows.append(f"- {role}: {_truncate(text, 220)}")
            if len(rows) >= limit:
                break
    return "\n".join(reversed(rows))

def build_handoff_packet(
    messages: list[Mapping[str, Any]],
    *,
    session_id: str | None = None,
    context_tokens: int | None = None,
    context_length: int | None = None,
    current_step: str | None = None,
    title: str | None = None,
    created_at: str | None = None,
) -> str:
    """Build an efficient operational handoff packet for resuming in a new session.

    This is intentionally not a raw transcript transfer. It preserves the
    operational state needed to continue safely: goal/current state, verification
    anchors, next step, cautions, and a bounded user/assistant outline.
    """

    messages = list(messages or [])
    goal = _first_text(messages, "user") or "Not captured; ask the user to restate the goal."
    latest_user = _last_text(messages, "user") or "No latest user message captured."
    latest_assistant = _last_text(messages, "assistant") or "No assistant progress captured."

    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    message_count = len(messages)
    created_at = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    pct = _usage_percent(context_tokens or 0, context_length or 0)
    usage = "unknown"
    if context_tokens is not None and context_length:
        usage = f"{pct}% 사용 중 ({context_tokens:,}/{context_length:,} 토큰)"

    next_step = current_step or "Read this packet, inspect the current files/state, then continue from the latest user request."
    title_line = f"- 제목: {title}" if title else "- 제목: unknown"
    verification = _matching_evidence(
        messages,
        ("pass", "검증", "확인", "console", "served", "active", "running", "오류 없음", "js errors"),
    )

    packet_body = "\n".join(
        [
            "## 운영 상태 체크리스트",
            f"- 현재 목표: {_truncate(goal, 360)}",
            f"- 완료/변경: {_truncate(latest_assistant, 520)}",
            f"- 검증 결과: {verification or '확인 필요 — 새 세션에서 관련 파일/상태/테스트를 재검증할 것.'}",
            f"- 남은 작업: {_truncate(next_step, 420)}",
            "- 주의/금지: 전체 원문 이동이 아니라 운영 요약 + bounded outline입니다. 오래된 요청을 재실행하지 마세요.",
            f"- 참조: source_session_id={session_id or 'unknown'}; message_count={message_count}; user={user_count}; assistant={assistant_count}; tool={tool_count}",
            "",
            "## 현재 상태",
            title_line,
            f"- 대화 용량: {usage}",
            f"- 마지막 사용자 요청: {_truncate(latest_user, 360)}",
            "",
            "## 계승된 운영 요약",
            f"- {_truncate(latest_assistant, 900)}",
            "",
            "## 이번 세션 visible 메시지 outline",
            _outline_messages(messages),
            "",
            "## 이어가기 지시",
            f"- {_truncate(next_step, 420)}",
            "- 먼저 실제 파일/상태를 확인하고, 완료 전 가장 작은 관련 테스트를 다시 실행하세요.",
            "- 이 인계문은 작업 재개 기준점이며 증명이나 전체 원문 보존이 아닙니다.",
        ]
    )
    packet_hash = hashlib.sha256(packet_body.encode("utf-8")).hexdigest()

    return "\n".join(
        [
            "[이동 준비: 새 세션 이어가기 안내]",
            f"원본 세션: {session_id or 'unknown'}",
            "범위: 운영 요약 + bounded visible outline",
            f"source_session_id: {session_id or 'unknown'}",
            "packet_scope: operational_summary_plus_bounded_visible_outline",
            "packet_limit_note: not_full_raw_transcript; verify live files/state before finalizing",
            f"message_count: {message_count}",
            f"created_at: {created_at}",
            f"packet_hash: sha256:{packet_hash}",
            "목적: 새 세션에서 같은 작업을 효율적으로 이어가기 위한 준비 인계문입니다.",
            "",
            packet_body,
        ]
    )
