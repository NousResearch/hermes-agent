"""Request-time semantic memory overlay for newly written durable facts.

This module is deliberately read-only. It renders a small ephemeral system-prompt
suffix from live typed memory records so current-session durable writes can be
seen by the model without changing ``agent/system_prompt.py``,
``agent._cached_system_prompt``, or the persisted session system-prompt row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

MAX_OVERLAY_RECORDS = 8
MAX_RECORDS_PER_TARGET = 4
MAX_OVERLAY_LINE_CHARS = 160
_TRUNCATION_MARKER = "…"
_ALLOWED_KINDS = {"semantic_fact", "user_profile_fact"}
_BLOCKED_ACTIONS = {
    "procedural_skill_candidate",
    "episodic_only",
    "working_memory_only",
    "discard",
}
_BLOCKED_WARNING_HINTS = ("procedural", "episodic", "task-local", "session_search")


@dataclass(frozen=True)
class OverlayRecord:
    target: str
    text: str
    salience: float
    confidence: float
    updated_at: int


def _compact_line(value: Any, limit: int = MAX_OVERLAY_LINE_CHARS) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER


def _clamp(value: Any, default: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def _int_value(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _security_scan_passes(text: str) -> bool:
    try:
        from tools.threat_patterns import scan_for_threats

        return not scan_for_threats(text, scope="strict")
    except Exception:
        return False


def _warnings_are_safe(record: Mapping[str, Any]) -> bool:
    warnings = record.get("consolidation_warnings") or ()
    if not isinstance(warnings, (list, tuple)):
        warnings = (warnings,)
    warning_text = " ".join(str(item).casefold() for item in warnings)
    return not any(hint in warning_text for hint in _BLOCKED_WARNING_HINTS)


def _record_is_eligible(
    record: Mapping[str, Any],
    *,
    target: str,
    base_system_prompt: str,
) -> bool:
    text = _compact_line(record.get("text"), MAX_OVERLAY_LINE_CHARS)
    if not text:
        return False
    if text in base_system_prompt or str(record.get("text") or "").strip() in base_system_prompt:
        return False

    kind = str(record.get("kind") or "")
    if kind not in _ALLOWED_KINDS:
        return False
    if target == "user" and kind != "user_profile_fact":
        return False
    if target == "memory" and kind != "semantic_fact":
        return False

    if str(record.get("consolidation_action") or "") in _BLOCKED_ACTIONS:
        return False
    if not _warnings_are_safe(record):
        return False

    salience = _clamp(record.get("salience"), 0.5)
    confidence = _clamp(record.get("confidence"), 0.7)
    if confidence < 0.6:
        return False
    if target != "user" and salience < 0.5:
        return False

    return _security_scan_passes(text)


def _records_for_target(store: Any, target: str) -> Sequence[Mapping[str, Any]]:
    records_by_target = getattr(store, "semantic_records", {}) or {}
    records = records_by_target.get(target, ()) if isinstance(records_by_target, dict) else ()
    if not isinstance(records, (list, tuple)):
        return ()
    return [record for record in records if isinstance(record, Mapping)]


def select_overlay_records(
    records_by_target: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    base_system_prompt: str = "",
    limit: int = MAX_OVERLAY_RECORDS,
    per_target_limit: int = MAX_RECORDS_PER_TARGET,
) -> list[OverlayRecord]:
    """Select a compact deterministic set of live semantic records to overlay."""
    selected: list[OverlayRecord] = []
    for target in ("user", "memory"):
        records = records_by_target.get(target, ()) or ()
        candidates: list[OverlayRecord] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            if not _record_is_eligible(record, target=target, base_system_prompt=base_system_prompt):
                continue
            candidates.append(
                OverlayRecord(
                    target=target,
                    text=_compact_line(record.get("text"), MAX_OVERLAY_LINE_CHARS),
                    salience=_clamp(record.get("salience"), 0.5),
                    confidence=_clamp(record.get("confidence"), 0.7),
                    updated_at=_int_value(record.get("updated_at") or record.get("created_at")),
                )
            )
        candidates.sort(key=lambda r: (-r.salience, -r.confidence, -r.updated_at, r.text))
        selected.extend(candidates[:per_target_limit])

    selected.sort(key=lambda r: (0 if r.target == "user" else 1, -r.salience, -r.confidence, -r.updated_at, r.text))
    return selected[:limit]


def render_semantic_overlay(records: Sequence[OverlayRecord]) -> str:
    if not records:
        return ""

    lines = [
        "RECENT SEMANTIC MEMORY (ephemeral, request-time only)",
        "These are durable memory writes visible in live state but not necessarily present in the frozen session system prompt. Use them as current durable facts, but do not save them again verbatim.",
    ]

    user_records = [record for record in records if record.target == "user"]
    memory_records = [record for record in records if record.target == "memory"]
    if user_records:
        lines.append("User profile:")
        lines.extend(f"- {record.text}" for record in user_records)
    if memory_records:
        lines.append("Memory:")
        lines.extend(f"- {record.text}" for record in memory_records)
    return "\n".join(lines)


def build_semantic_memory_ephemeral_overlay(
    agent: Any,
    *,
    base_system_prompt: str | None = None,
) -> str:
    """Build the request-time semantic overlay for an agent, if any.

    Fail closed: any unexpected store/record/scanner issue returns an empty
    overlay instead of risking prompt pollution.
    """
    try:
        store = getattr(agent, "_memory_store", None)
        if store is None:
            return ""
        base = base_system_prompt
        if base is None:
            base = getattr(agent, "_cached_system_prompt", "") or ""
        records_by_target = {
            "user": _records_for_target(store, "user"),
            "memory": _records_for_target(store, "memory"),
        }
        records = select_overlay_records(records_by_target, base_system_prompt=str(base or ""))
        return render_semantic_overlay(records)
    except Exception:
        return ""


__all__ = [
    "MAX_OVERLAY_LINE_CHARS",
    "MAX_OVERLAY_RECORDS",
    "MAX_RECORDS_PER_TARGET",
    "OverlayRecord",
    "build_semantic_memory_ephemeral_overlay",
    "render_semantic_overlay",
    "select_overlay_records",
]
