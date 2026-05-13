"""Context continuity helpers for quality-preserving session handoff.

This module is intentionally separate from ``context_compressor``.  Compression
keeps an oversized session alive; continuity handoff creates an explicit packet
that a fresh session can use to resume work without inheriting all historical
noise.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
            reason="다음 단계는 자동 압축보다 새 세션으로 넘기는 편이 안전합니다.",
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


def build_handoff_packet(
    messages: list[Mapping[str, Any]],
    *,
    session_id: str | None = None,
    context_tokens: int | None = None,
    context_length: int | None = None,
    current_step: str | None = None,
    title: str | None = None,
) -> str:
    """Build a copy/paste packet for resuming the task in a new session.

    The packet is deterministic and conservative.  It does not pretend to be a
    perfect semantic summary; it tells the next session what to verify and where
    to restart, while preserving the latest user/assistant anchors.
    """

    messages = list(messages or [])
    goal = _first_text(messages, "user") or "Not captured; ask the user to restate the goal."
    latest_user = _last_text(messages, "user") or "No latest user message captured."
    latest_assistant = _last_text(messages, "assistant") or "No assistant progress captured."

    tool_count = sum(1 for m in messages if m.get("role") == "tool")
    user_count = sum(1 for m in messages if m.get("role") == "user")
    assistant_count = sum(1 for m in messages if m.get("role") == "assistant")
    pct = _usage_percent(context_tokens or 0, context_length or 0)
    usage = "unknown"
    if context_tokens is not None and context_length:
        usage = f"{pct}% 사용 중 ({context_tokens:,}/{context_length:,} 토큰)"

    next_step = current_step or "Read this packet, inspect the current files/state, then continue from the latest user request."
    title_line = f"- Title: {title}" if title else "- Title: unknown"

    packet_body = "\n".join(
        [
            "## 목표",
            f"- {_truncate(goal, 360)}",
            "",
            "## 현재 상태",
            title_line.replace("Title", "제목"),
            f"- 대화 용량: {usage}",
            f"- 메시지: 사용자 {user_count} / Hermes {assistant_count} / 도구 {tool_count}",
            f"- 마지막 사용자 요청: {_truncate(latest_user, 360)}",
            "",
            "## 완료한 것",
            f"- {_truncate(latest_assistant, 420)}",
            "",
            "## 중요 결정",
            "- 컨텍스트가 무거워지면 손실 압축보다 새 세션 이동을 우선합니다.",
            "- 이 인계문은 작업 재개 기준점이며, 증명은 아닙니다.",
            "",
            "## 변경/검증 상태",
            "- 먼저 실제 파일/상태를 확인하고, 완료 전 가장 작은 관련 테스트를 다시 실행하세요.",
            "- 파일 변경, git 상태, 외부 서비스 상태는 이 인계문만 믿지 말고 재조회하세요.",
            "",
            "## 다음 작업",
            f"- {_truncate(next_step, 420)}",
            "- 이전 완료 작업을 반복하지 말고 위 상태 확인 후 바로 이어가세요.",
            "",
            "## 완료 기준",
            "- 새 세션이 목표, 현재 상태, 변경/검증 상태, 다음 작업을 숨은 이전 문맥 없이 설명할 수 있어야 합니다.",
        ]
    )
    packet_hash = hashlib.sha256(packet_body.encode("utf-8")).hexdigest()

    return "\n".join(
        [
            "[세션 이동 인계문]",
            f"원본 세션: {session_id or 'unknown'}",
            "기준: 현재 세션 전체 요약",
            f"본문 해시: sha256:{packet_hash}",
            "목적: 새 세션에서 가벼운 컨텍스트로 같은 작업을 이어갑니다.",
            "",
            packet_body,
        ]
    )
