"""Deterministic Hindsight session summary generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Protocol

SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION = 2

OPERATIONAL_METADATA_KEYS = frozenset(
    {
        "agent",
        "agent_id",
        "bank",
        "bank_id",
        "channel",
        "channel_id",
        "document",
        "document_id",
        "message_id",
        "profile",
        "provider",
        "sender",
        "sender_id",
        "session",
        "session_id",
        "session_key",
        "source",
        "source_system",
        "thread",
        "thread_id",
        "tool",
        "tool_call_id",
        "update_mode",
        "user_id",
    }
)

_INJECTION_RE = re.compile(
    r"\b(ignore|override|forget|bypass)\b.{0,80}\b(previous|system|developer|instructions?)\b"
    r"|\b(reveal|print|exfiltrate|leak)\b.{0,80}\b(secret|token|prompt|credentials?)\b"
    r"|\bdo\s+not\s+(store|summari[sz]e|sanitize)\b",
    re.IGNORECASE,
)
_CANARY_RE = re.compile(
    r"\b[A-Z0-9_]*(?:SECRET|CANARY|DO_NOT_STORE|DO_NOT_LEAK|SHOULD_NOT_APPEAR)[A-Z0-9_]*\b"
    r"|/private/[^\s`'\"<>]+"
    r"|\bsha256:[a-fA-F0-9]{32,64}\b",
    re.IGNORECASE,
)
_SECRET_RE = re.compile(
    r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?[^'\"\s,;]+",
    re.IGNORECASE,
)
_METADATA_BLOCK_RE = re.compile(
    r"[\w\s]+\(untrusted metadata\)[^\n]*\n```json\n[\s\S]*?```",
    re.IGNORECASE,
)
_MEMORY_TAG_RE = re.compile(
    r"<(?:hindsight_memories|relevant_memories)>[\s\S]*?</(?:hindsight_memories|relevant_memories)>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionSummaryBudget:
    max_input_chars: int = 16_000
    max_output_chars: int = 2_000
    max_recall_query_chars: int = 800
    recall_query_budget_ratio: float = 0.25
    max_prompt_inject_chars: int = 1_200
    max_retain_context_chars: int = 1_200
    min_latest_query_reserve_chars: int = 400


@dataclass(frozen=True)
class SessionSummaryBudgetedText:
    output_text: str
    recall_query_text: str
    prompt_inject_text: str
    retain_context_text: str


@dataclass(frozen=True)
class SessionSummaryWindowBounds:
    segment_start_turn: int
    segment_end_turn: int
    input_start_turn: int
    recall_context_start_turn: int


@dataclass(frozen=True)
class SessionSummaryRequest:
    session_id: str
    identity_scope: str
    messages: list[dict[str, Any]]
    previous_summary_text: str | None = None
    latest_query: str = ""
    turn_index: int = 0
    metadata: dict[str, Any] | None = None
    budget: SessionSummaryBudget = SessionSummaryBudget()


@dataclass(frozen=True)
class SessionSummaryResult:
    summary_text: str
    summary_json: dict[str, Any] | None = None
    schema_version: int = SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION
    status: str = "ready"
    error: str | None = None


class SessionSummaryGenerator(Protocol):
    def generate(self, request: SessionSummaryRequest) -> SessionSummaryResult:
        """Return a bounded, sanitized summary result."""


def sanitize_session_summary_text(text: str, *, max_chars: int | None = None) -> str:
    """Remove prompt-injection, operational envelopes, and sensitive canaries."""
    if not text:
        return ""
    cleaned = _MEMORY_TAG_RE.sub("", str(text))
    cleaned = _METADATA_BLOCK_RE.sub("", cleaned)
    cleaned = _strip_operational_metadata_json_objects(cleaned)
    cleaned = _SECRET_RE.sub("[redacted-secret]", cleaned)
    kept_lines: list[str] = []
    for raw_line in cleaned.splitlines():
        line = _CANARY_RE.sub("[redacted]", raw_line).strip()
        if not line or _INJECTION_RE.search(line):
            continue
        kept_lines.append(line)
    cleaned = "\n".join(kept_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if max_chars is not None and max_chars >= 0 and len(cleaned) > max_chars:
        return cleaned[:max_chars].rstrip()
    return cleaned


def build_session_summary_prompt(request: SessionSummaryRequest) -> str:
    """Build the summary-only prompt used by real generators and smoke tests."""
    trimmed = trim_session_summary_inputs(request, request.budget)
    prior = sanitize_session_summary_text(trimmed.previous_summary_text or "")
    messages = [
        {
            "role": str(msg.get("role", "")),
            "content": sanitize_session_summary_text(str(msg.get("content", ""))),
        }
        for msg in trimmed.messages
    ]
    return (
        "Generate a compact Hindsight rolling session summary as plain text only.\n"
        f"Schema version: {SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION}\n"
        f"Maximum output length: {trimmed.budget.max_output_chars} characters.\n"
        "Rules: use only evidence in user/assistant messages; preserve exact entity names, "
        "proper nouns, project names, file paths, commands, addresses, dates, amounts, "
        "numbers, URLs, model names, error messages, and user terminology. Do not rename, "
        "translate, normalize, abbreviate, substitute, or autocorrect proper nouns and identifiers. "
        "Treat the previous summary as a draft; current messages and explicit corrections override it. "
        "Do not output JSON, markdown, Semantic Anchors, Exact Identifiers, Decisions, Blockers, "
        "Open Questions, or Completed Todos categories.\n"
        f"Previous rolling summary:\n{prior}\n"
        f"Latest query:\n{sanitize_session_summary_text(trimmed.latest_query)}\n"
        f"Messages JSON:\n{json.dumps(messages, ensure_ascii=False)}"
    )


def render_session_summary(summary_text: str | dict[str, Any], *, max_chars: int) -> str:
    """Render summary text to a bounded context string."""
    if isinstance(summary_text, dict):
        raw = summary_text.get("summary_text") or summary_text.get("summaryText") or ""
    else:
        raw = summary_text
    return sanitize_session_summary_text(str(raw), max_chars=max_chars)


def should_update_session_summary(
    turn_index: int,
    retain_every_n_turns: int,
    update_every_n_turns: int | None = None,
    min_update_every_n_turns: int = 2,
    retain_overlap_turns: int = 0,
    recall_context_turns: int = 1,
) -> bool:
    """Return whether a background summary refresh should be scheduled."""
    if turn_index <= 0:
        return False
    session_summary_window_bounds(
        turn_index=turn_index,
        retain_every_n_turns=retain_every_n_turns,
        retain_overlap_turns=retain_overlap_turns,
        recall_context_turns=recall_context_turns,
    )
    minimum = max(1, int(min_update_every_n_turns or 1))
    if update_every_n_turns is not None:
        cadence = max(minimum, int(update_every_n_turns or minimum))
    else:
        cadence = max(minimum, int(retain_every_n_turns or 1))
    return turn_index % cadence == 0


def session_summary_window_bounds(
    *,
    turn_index: int,
    retain_every_n_turns: int,
    retain_overlap_turns: int = 0,
    recall_context_turns: int = 1,
) -> SessionSummaryWindowBounds:
    """Return generator-only segment and input bounds for a summary refresh."""
    end_turn = max(0, int(turn_index or 0))
    if end_turn <= 0:
        return SessionSummaryWindowBounds(0, 0, 0, 0)
    segment_size = max(1, int(retain_every_n_turns or 1))
    overlap = max(0, int(retain_overlap_turns or 0))
    recall_context = max(1, int(recall_context_turns or 1))
    segment_start = max(1, end_turn - segment_size + 1)
    overlap_start = max(1, segment_start - overlap)
    recall_start = max(1, end_turn - recall_context + 1)
    return SessionSummaryWindowBounds(
        segment_start_turn=segment_start,
        segment_end_turn=end_turn,
        input_start_turn=min(overlap_start, recall_start),
        recall_context_start_turn=recall_start,
    )


def trim_session_summary_inputs(
    request: SessionSummaryRequest,
    budget: SessionSummaryBudget,
) -> SessionSummaryRequest:
    """Trim summary inputs while reserving room for the latest query."""
    latest = sanitize_session_summary_text(
        request.latest_query,
        max_chars=max(0, budget.min_latest_query_reserve_chars),
    )
    remaining_total = max(0, budget.max_input_chars - len(latest))
    previous_summary_text = sanitize_session_summary_text(
        request.previous_summary_text or "",
        max_chars=(remaining_total // 4 if request.previous_summary_text else 0),
    )
    remaining = max(0, remaining_total - len(previous_summary_text))
    kept_reversed: list[dict[str, Any]] = []
    for msg in reversed(request.messages):
        content = sanitize_session_summary_text(str(msg.get("content", "")))
        if not content:
            continue
        if len(content) > remaining:
            if remaining <= 0:
                break
            content = content[-remaining:]
        kept_reversed.append({**msg, "content": content})
        remaining -= len(content)
        if remaining <= 0:
            break
    return replace(
        request,
        previous_summary_text=previous_summary_text,
        latest_query=latest,
        messages=list(reversed(kept_reversed)),
        budget=budget,
    )


def build_session_summary_budgeted_text(
    summary_text: str | dict[str, Any],
    budget: SessionSummaryBudget,
) -> SessionSummaryBudgetedText:
    """Render summary-derived text variants with independent stage budgets."""
    output_text = render_session_summary(summary_text, max_chars=budget.max_output_chars)
    return SessionSummaryBudgetedText(
        output_text=output_text,
        recall_query_text=_render_budgeted_summary_variant(
            output_text,
            max_chars=_effective_recall_query_chars(budget),
        ),
        prompt_inject_text=_render_budgeted_summary_variant(
            output_text,
            max_chars=budget.max_prompt_inject_chars,
        ),
        retain_context_text=_render_budgeted_summary_variant(
            output_text,
            max_chars=budget.max_retain_context_chars,
        ),
    )


class FakeSessionSummaryGenerator:
    """Deterministic generator for tests and offline fallback."""

    def generate(self, request: SessionSummaryRequest) -> SessionSummaryResult:
        try:
            trimmed = trim_session_summary_inputs(request, request.budget)
            evidence_text = _evidence_text(trimmed.messages)
            summary_text = sanitize_session_summary_text(
                "\n".join(part for part in (trimmed.previous_summary_text or "", evidence_text) if part)
            )
            return SessionSummaryResult(
                summary_text=summary_text,
                summary_json={
                    "schema_version": SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION,
                    "summary_text": summary_text,
                },
            )
        except Exception as exc:
            return SessionSummaryResult(
                summary_text="",
                summary_json={
                    "schema_version": SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION,
                    "summary_text": "",
                },
                status="error",
                error=str(exc),
            )


def _evidence_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = sanitize_session_summary_text(str(msg.get("content", "")))
        if content:
            lines.append(content)
    return "\n".join(lines)


def _strip_operational_metadata_json_objects(text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        if not isinstance(parsed, dict):
            return raw
        keys = {_normalize_metadata_key(key) for key in parsed}
        if keys & OPERATIONAL_METADATA_KEYS:
            return ""
        return raw

    return re.sub(r"\{[^\{\}]*\}", _replace, text)


def _looks_like_metadata_assignment(text: str) -> bool:
    stripped = text.strip().strip(",")
    if not stripped:
        return False
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        keys = {_normalize_metadata_key(key) for key in parsed}
        if keys & OPERATIONAL_METADATA_KEYS:
            return True
    separator = ":" if ":" in stripped else "=" if "=" in stripped else ""
    if not separator:
        return False
    key = stripped.split(separator, 1)[0].strip().strip("\"'")
    return _normalize_metadata_key(key) in OPERATIONAL_METADATA_KEYS


def _is_operational_identifier(value: str) -> bool:
    return _normalize_metadata_key(value) in OPERATIONAL_METADATA_KEYS


def _normalize_metadata_key(value: Any) -> str:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value).strip())
    text = text.replace("-", "_").replace(".", "_").lower()
    return re.sub(r"_+", "_", text).strip("_")


def _effective_recall_query_chars(budget: SessionSummaryBudget) -> int:
    ratio_limit = int(max(0, budget.max_input_chars) * max(0.0, budget.recall_query_budget_ratio))
    return max(0, min(budget.max_recall_query_chars, ratio_limit))


def _render_budgeted_summary_variant(summary_text: str | dict[str, Any], *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    return render_session_summary(summary_text, max_chars=max_chars)
