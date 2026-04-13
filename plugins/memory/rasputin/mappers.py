"""Payload mappers for Rasputin commit bodies.

These helpers preserve the architectural rule from the planning docs:
Rasputin is a derived retrieval plane, not the canonical memory store.
Mappings therefore keep provenance metadata so Rasputin commits remain
rebuildable from Hermes-native sources.

TODO:
- wire build_jsonl_fact_commit() into the separate backfill/import scripts
- refine text rendering once real canonical JSONL samples are exercised
"""

from __future__ import annotations

from typing import Any, Dict, Mapping


_TYPE_IMPORTANCE = {
    "fact": 80,
    "decision": 90,
    "pattern": 70,
    "gotcha": 85,
    "incident": 90,
    "preference": 85,
}


def build_turn_commit(
    user_content: str,
    assistant_content: str,
    *,
    session_id: str = "",
    platform: str = "cli",
    agent_context: str = "primary",
    user_id: str = "",
    namespace: str = "hermes",
) -> Dict[str, Any]:
    """Map one completed Hermes turn into a Rasputin window commit."""
    return {
        "text": _strip_join(
            f"USER: {(user_content or '').strip()}",
            f"ASSISTANT: {(assistant_content or '').strip()}",
        ),
        "source": "hermes-turn",
        "importance": 55,
        "metadata": {
            "namespace": namespace,
            "kind": "conversation_window",
            "session_id": session_id,
            "platform": platform,
            "agent_context": agent_context,
            "user_id": user_id or None,
        },
    }


def build_memory_write_commit(
    action: str,
    target: str,
    content: str,
    *,
    namespace: str = "hermes",
    session_id: str = "",
) -> Dict[str, Any]:
    """Map a built-in Hermes memory write into a Rasputin mirror commit."""
    clean_action = (action or "").strip() or "add"
    clean_target = (target or "").strip() or "memory"
    clean_content = (content or "").strip()
    return {
        "text": f"MEMORY WRITE [target={clean_target} action={clean_action}]: {clean_content}",
        "source": "hermes-memory-write",
        "importance": 75,
        "metadata": {
            "namespace": namespace,
            "kind": "memory_write",
            "session_id": session_id or None,
            "target": clean_target,
            "action": clean_action,
        },
    }


def build_jsonl_fact_commit(
    record: Mapping[str, Any],
    *,
    namespace: str = "ryan-book",
    source_path: str = "",
) -> Dict[str, Any]:
    """Scaffold mapper for canonical JSONL typed memory backfill.

    This is intentionally not wired into the live provider yet. It exists so the
    backfill/import phase can reuse a single mapping definition later.
    """
    memory_type = str(record.get("type") or "fact").strip() or "fact"
    pinned = bool(record.get("pinned"))
    trust = _coerce_float(record.get("trust"))
    importance = _importance_for_record(memory_type, pinned=pinned, trust=trust)
    metadata = {
        "namespace": namespace,
        "kind": "typed_memory",
        "canonical_id": record.get("id"),
        "memory_type": memory_type,
        "fleet": record.get("fleet"),
        "tags": list(record.get("tags") or []),
        "trust": trust,
        "pinned": pinned,
        "source_path": source_path or None,
        "source_created_at": record.get("created_at"),
    }
    return {
        "text": _render_typed_memory_text(record, memory_type),
        "source": f"jsonl-{memory_type}",
        "importance": importance,
        "metadata": metadata,
    }


def _render_typed_memory_text(record: Mapping[str, Any], memory_type: str) -> str:
    content = record.get("content")
    pieces = [f"[{memory_type}]"]
    if isinstance(content, Mapping):
        for key, value in content.items():
            rendered = _render_value(value)
            if rendered:
                pieces.append(f"{key}: {rendered}")
    else:
        rendered = _render_value(content)
        if rendered:
            pieces.append(rendered)
    return "\n".join(pieces)


def _importance_for_record(memory_type: str, *, pinned: bool, trust: float | None) -> int:
    importance = _TYPE_IMPORTANCE.get(memory_type, 80)
    if pinned:
        importance = min(100, importance + 5)
    if trust is not None and trust < 0.6:
        importance = max(40, importance - 10)
    return importance


def _render_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [part for part in (_render_value(item) for item in value) if part]
        return ", ".join(parts)
    if isinstance(value, Mapping):
        parts = []
        for key, item in value.items():
            rendered = _render_value(item)
            if rendered:
                parts.append(f"{key}={rendered}")
        return ", ".join(parts)
    return str(value).strip()


def _strip_join(*parts: str) -> str:
    return "\n".join(part for part in parts if part and part.strip())


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
