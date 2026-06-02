"""Deterministic Hindsight session summary generation helpers."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import re
from typing import Any, Protocol

SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION = 1

OPERATIONAL_METADATA_KEYS = frozenset(
    {
        "agent",
        "agent_id",
        "bank",
        "bank_id",
        "channel",
        "channel_id",
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
_IDENTIFIER_RE = re.compile(r"\b[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)+\b")
_PROJECT_CUE_RE = re.compile(
    r"\b(?:project|repo|repository|package|module|app|service|workspace)\s+([A-Za-z][\w.-]{2,})",
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
    drop_completed_todos_after_turns: int = 20


@dataclass(frozen=True)
class SessionSummaryRequest:
    session_id: str
    identity_scope: str
    messages: list[dict[str, Any]]
    previous_summary: dict[str, Any] | None = None
    latest_query: str = ""
    turn_index: int = 0
    metadata: dict[str, Any] | None = None
    budget: SessionSummaryBudget = SessionSummaryBudget()


@dataclass(frozen=True)
class SessionSummaryResult:
    summary_json: dict[str, Any]
    summary_text: str
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
    prior = json.dumps(trimmed.previous_summary or {}, ensure_ascii=False, sort_keys=True)
    messages = [
        {
            "role": str(msg.get("role", "")),
            "content": sanitize_session_summary_text(str(msg.get("content", ""))),
        }
        for msg in trimmed.messages
    ]
    return (
        "Generate a compact Hindsight session summary as JSON only.\n"
        f"Schema version: {SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION}\n"
        "Rules: use only evidence in user/assistant messages; do not promote "
        "bank, source, session, sender, profile, provider, or tool metadata into "
        "semantic entities; carry forward previous anchors only when grounded.\n"
        f"Previous summary JSON:\n{prior}\n"
        f"Latest query:\n{sanitize_session_summary_text(trimmed.latest_query)}\n"
        f"Messages JSON:\n{json.dumps(messages, ensure_ascii=False)}"
    )


def render_session_summary(summary_json: dict[str, Any], *, max_chars: int) -> str:
    """Render summary JSON to a bounded plain-text context string."""
    sections = []
    for key, label in (
        ("active_projects", "Active projects"),
        ("semantic_anchors", "Semantic anchors"),
        ("exact_identifiers", "Exact identifiers"),
        ("decisions", "Decisions"),
        ("blockers", "Blockers"),
        ("open_questions", "Open questions"),
    ):
        values = _as_string_list(summary_json.get(key))
        if values:
            sections.append(f"{label}: " + "; ".join(values))
    text = sanitize_session_summary_text("\n".join(sections), max_chars=max_chars)
    return text


def should_update_session_summary(
    turn_index: int,
    retain_every_n_turns: int,
    update_every_n_turns: int | None = None,
    min_update_every_n_turns: int = 2,
) -> bool:
    """Return whether a background summary refresh should be scheduled."""
    if turn_index <= 0:
        return False
    minimum = max(1, int(min_update_every_n_turns or 1))
    if update_every_n_turns is not None:
        cadence = max(minimum, int(update_every_n_turns or minimum))
    else:
        cadence = max(minimum, int(retain_every_n_turns or 1))
    return turn_index % cadence == 0


def trim_session_summary_inputs(
    request: SessionSummaryRequest,
    budget: SessionSummaryBudget,
) -> SessionSummaryRequest:
    """Trim summary inputs while reserving room for the latest query."""
    latest = sanitize_session_summary_text(
        request.latest_query,
        max_chars=max(0, budget.min_latest_query_reserve_chars),
    )
    remaining = max(0, budget.max_input_chars - len(latest))
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
    return replace(request, latest_query=latest, messages=list(reversed(kept_reversed)), budget=budget)


class FakeSessionSummaryGenerator:
    """Deterministic generator for tests and offline fallback."""

    def generate(self, request: SessionSummaryRequest) -> SessionSummaryResult:
        try:
            trimmed = trim_session_summary_inputs(request, request.budget)
            evidence_text = _evidence_text(trimmed.messages)
            summary_json = {
                "schema_version": SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION,
                "active_projects": _active_projects(evidence_text, trimmed.previous_summary),
                "semantic_anchors": _semantic_anchors(evidence_text),
                "exact_identifiers": _exact_identifiers(evidence_text),
                "decisions": _matching_lines(evidence_text, ("decided", "decision", "use ", "chosen")),
                "blockers": _matching_lines(evidence_text, ("blocked", "failing", "failure", "error", "risk")),
                "open_questions": _matching_lines(evidence_text, ("?", "open question", "unknown")),
                "completed_todos": _matching_lines(evidence_text, ("done", "completed", "fixed")),
            }
            summary_text = render_session_summary(
                summary_json,
                max_chars=trimmed.budget.max_output_chars,
            )
            return SessionSummaryResult(summary_json=summary_json, summary_text=summary_text)
        except Exception as exc:
            return SessionSummaryResult(
                summary_json={
                    "schema_version": SESSION_SUMMARY_GENERATOR_SCHEMA_VERSION,
                    "active_projects": [],
                    "semantic_anchors": [],
                    "exact_identifiers": [],
                    "decisions": [],
                    "blockers": [],
                    "open_questions": [],
                    "completed_todos": [],
                },
                summary_text="",
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


def _active_projects(evidence_text: str, previous_summary: dict[str, Any] | None) -> list[str]:
    candidates: list[str] = []
    lower_evidence = evidence_text.lower()
    for value in _as_string_list((previous_summary or {}).get("active_projects")):
        if value.lower() in lower_evidence:
            candidates.append(value)
    for match in _PROJECT_CUE_RE.finditer(evidence_text):
        candidates.append(match.group(1))
    for line in evidence_text.splitlines():
        if _looks_like_metadata_assignment(line):
            continue
        for ident in _IDENTIFIER_RE.findall(line):
            if "-" in ident and not _is_operational_identifier(ident):
                candidates.append(ident)
    return _dedupe(candidates, limit=8)


def _semantic_anchors(evidence_text: str) -> list[str]:
    anchors = []
    for line in evidence_text.splitlines():
        text = line.strip(" -")
        if len(text) < 8 or _looks_like_metadata_assignment(text):
            continue
        anchors.append(text[:180])
    return _dedupe(anchors, limit=8)


def _exact_identifiers(evidence_text: str) -> list[str]:
    return _dedupe(
        [
            ident
            for line in evidence_text.splitlines()
            if not _looks_like_metadata_assignment(line)
            for ident in _IDENTIFIER_RE.findall(line)
            if not _is_operational_identifier(ident)
        ],
        limit=16,
    )


def _matching_lines(evidence_text: str, needles: tuple[str, ...]) -> list[str]:
    out = []
    for line in evidence_text.splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in needles) and not _looks_like_metadata_assignment(line):
            out.append(line.strip()[:180])
    return _dedupe(out, limit=8)


def _looks_like_metadata_assignment(text: str) -> bool:
    stripped = text.strip().strip(",")
    if not stripped:
        return False
    key = stripped.split(":", 1)[0].strip().strip("\"'").lower().replace("-", "_")
    return key in OPERATIONAL_METADATA_KEYS


def _is_operational_identifier(value: str) -> bool:
    normalized = value.lower().replace("-", "_").replace(".", "_")
    return normalized in OPERATIONAL_METADATA_KEYS


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dedupe(values: list[str], *, limit: int) -> list[str]:
    out = []
    seen = set()
    for value in values:
        text = sanitize_session_summary_text(str(value)).strip(" .,;")
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out
