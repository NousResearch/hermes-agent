"""Build Hermes-specific ACP ``_meta`` session payloads.

ACP models expose an optional ``field_meta`` field that serializes to ``_meta``.
This module keeps Hermes' session lineage/compaction extension centralized so
new/load/resume/list/update responses do not each invent subtly different
metadata shapes.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_COMPRESSION_SESSION_KINDS = {"continuation", "compression_continuation"}


def _clean_dict(values: Mapping[str, Any]) -> dict[str, Any]:
    """Drop ``None`` values while preserving falsey data like ``0``/``False``."""
    return {key: value for key, value in values.items() if value is not None}


def _row_to_dict(row: Any) -> dict[str, Any] | None:
    if row is None:
        return None
    if isinstance(row, dict):
        return dict(row)
    try:
        return dict(row)
    except Exception:
        return None


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def current_hermes_session_id(state: Any) -> str:
    """Return the current Hermes DB session id for an ACP session state.

    ``SessionState.session_id`` is the stable ACP/editor session id. Automatic
    Hermes compression can rotate ``state.agent.session_id`` to a continuation
    row in ``SessionDB``. Ignore MagicMock/autovivified attributes and fall back
    to the ACP id when the agent does not expose a real string id.
    """
    for candidate in (
        getattr(getattr(state, "agent", None), "session_id", None),
        getattr(state, "hermes_session_id", None),
        getattr(state, "session_id", None),
    ):
        value = _string_or_none(candidate)
        if value:
            return value
    return ""


def _get_session_row(
    db: Any,
    session_id: str,
    fallback_row: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if fallback_row and fallback_row.get("id") == session_id:
        return dict(fallback_row)
    if db is None or not session_id:
        return dict(fallback_row) if fallback_row else None
    try:
        return _row_to_dict(db.get_session(session_id))
    except Exception:
        return dict(fallback_row) if fallback_row else None


def _lineage_rows(
    db: Any,
    session_id: str,
    current_row: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return root→current rows by following parent_session_id defensively."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = session_id
    fallback = current_row
    for _ in range(100):
        if not cursor or cursor in seen:
            break
        seen.add(cursor)
        row = _get_session_row(db, cursor, fallback)
        fallback = None
        if not row:
            break
        rows.append(row)
        parent_id = _string_or_none(row.get("parent_session_id"))
        if not parent_id:
            break
        cursor = parent_id
    rows.reverse()
    return rows


def _compression_depth(rows: Sequence[Mapping[str, Any]]) -> int:
    depth = 0
    for row in rows[1:]:
        if row.get("creator_kind") == "compression" or row.get("session_kind") in _COMPRESSION_SESSION_KINDS:
            depth += 1
    if depth == 0 and rows:
        current = rows[-1]
        if current.get("creator_kind") == "compression" or current.get("session_kind") in _COMPRESSION_SESSION_KINDS:
            depth = 1
    return depth


def build_hermes_session_meta(
    db: Any,
    *,
    acp_session_id: str,
    hermes_session_id: str | None = None,
    row: Mapping[str, Any] | None = None,
    in_place_compaction_count: int = 0,
    last_compaction_mode: str | None = None,
) -> dict[str, Any]:
    """Build the Hermes ACP ``field_meta`` payload for a session.

    Args:
        db: ``SessionDB``-like object used for lineage lookups. May be ``None``;
            the function then emits a conservative id-only payload.
        acp_session_id: Stable session id visible to the ACP client/editor.
        hermes_session_id: Current canonical Hermes DB session id. This may
            differ from ``acp_session_id`` after automatic context compression.
        row: Optional already-fetched session row for list/session update paths.
        in_place_compaction_count: Number of manual ACP ``/compact`` operations
            performed without rotating the Hermes DB session id.
        last_compaction_mode: Explicit last compaction mode, e.g. ``"in_place"``.
    """
    acp_id = _string_or_none(acp_session_id) or ""
    hermes_id = _string_or_none(hermes_session_id) or acp_id
    current_row = _row_to_dict(row)
    if current_row is None:
        current_row = _get_session_row(db, hermes_id)

    rows = _lineage_rows(db, hermes_id, current_row)
    if not rows and current_row:
        rows = [current_row]

    row_for_current = rows[-1] if rows else current_row or {}
    lineage_ids = [str(r["id"]) for r in rows if r.get("id")]
    if not lineage_ids and hermes_id:
        lineage_ids = [hermes_id]

    root_id = (
        _string_or_none(row_for_current.get("root_session_id"))
        or (lineage_ids[0] if lineage_ids else None)
        or acp_id
    )
    is_user_facing_raw = row_for_current.get("is_user_facing")
    is_user_facing = True if is_user_facing_raw is None else bool(is_user_facing_raw)
    depth = _compression_depth(rows)

    provenance = _clean_dict(
        {
            "schemaVersion": 1,
            "acpSessionId": acp_id,
            "hermesSessionId": hermes_id,
            "parentHermesSessionId": _string_or_none(row_for_current.get("parent_session_id")),
            "rootHermesSessionId": root_id,
            "sessionKind": row_for_current.get("session_kind") or "main",
            "creatorKind": row_for_current.get("creator_kind"),
            "creatorToolName": row_for_current.get("creator_tool_name"),
            "creatorToolCallId": row_for_current.get("creator_tool_call_id"),
            "creatorTaskIndex": row_for_current.get("creator_task_index"),
            "creatorCommand": row_for_current.get("creator_command"),
            "isUserFacing": is_user_facing,
            "compressionDepth": depth,
            "lineageHermesSessionIds": lineage_ids,
        }
    )

    hermes_meta: dict[str, Any] = {"sessionProvenance": provenance}

    inferred_split = depth > 0 and hermes_id != acp_id
    if last_compaction_mode == "in_place" or in_place_compaction_count:
        hermes_meta["compaction"] = _clean_dict(
            {
                "lastMode": "in_place",
                "compressionDepth": depth,
                "inPlaceCount": int(in_place_compaction_count or 0),
                "currentHermesSessionId": hermes_id,
            }
        )
    elif inferred_split or last_compaction_mode == "split":
        hermes_meta["compaction"] = {
            "lastMode": "split",
            "compressionDepth": depth,
            "currentHermesSessionId": hermes_id,
        }

    return {"hermes": hermes_meta}


def build_hermes_session_meta_for_state(db: Any, state: Any) -> dict[str, Any]:
    """Convenience wrapper for ``SessionState``-like objects."""
    return build_hermes_session_meta(
        db,
        acp_session_id=getattr(state, "session_id", ""),
        hermes_session_id=current_hermes_session_id(state),
        in_place_compaction_count=getattr(state, "in_place_compaction_count", 0) or 0,
        last_compaction_mode=getattr(state, "last_compaction_mode", None),
    )
