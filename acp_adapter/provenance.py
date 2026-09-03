"""Derive ACP session-provenance metadata from the existing compression chain.

This is an additive Hermes extension surfaced under ACP ``_meta.hermes`` so
existing ACP clients ignore it. It carries no new persisted state: everything
is derived on demand from the ``sessions`` table (``parent_session_id`` /
``end_reason``), which already models compression-continuation chains.

The ACP/editor ``session_id`` stays the stable public handle. When context
compression rotates the internal Hermes head, ``build_session_provenance`` lets
a client see the previous/current internal ids and the lineage root without
parsing status text, guessing from token drops, or reading ``state.db``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from hermes_state_common import is_continuation_end_reason

# Bound defensive walks; compression chains this deep are pathological.
_MAX_WALK = 100


def build_session_provenance(
    db: Any,
    acp_session_id: str,
    current_hermes_session_id: str,
    *,
    previous_hermes_session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Build ``_meta.hermes.sessionProvenance`` for an ACP session.

    Args:
        db: A ``SessionDB`` (must expose ``get_session``).
        acp_session_id: The stable ACP/editor-facing session handle.
        current_hermes_session_id: The live internal Hermes DB session id
            (``state.agent.session_id``).
        previous_hermes_session_id: The internal id from before the most recent
            turn, when known. Supplied by ``prompt()`` to flag a rotation.

    Returns:
        A dict suitable for ``{"hermes": {"sessionProvenance": <dict>}}`` under
        ACP ``_meta``, or ``None`` if the session can't be read.
    """
    try:
        row = db.get_session(current_hermes_session_id)
    except Exception:
        return None
    if not row:
        return None

    parent_id = row.get("parent_session_id")
    end_reason = row.get("end_reason")

    # Walk parents to the lineage root and count actual compression depth.
    # Turn-boundary rollover is a continuation, but not a compaction boundary.
    # Delegate/branch children also share parent_session_id without being either.
    root_id = current_hermes_session_id
    compression_depth = 0
    cursor_parent = parent_id
    seen = {current_hermes_session_id}
    for _ in range(_MAX_WALK):
        if not cursor_parent or cursor_parent in seen:
            break
        seen.add(cursor_parent)
        try:
            prow = db.get_session(cursor_parent)
        except Exception:
            prow = None
        if not prow:
            break
        root_id = cursor_parent
        if prow.get("end_reason") == "compression":
            compression_depth += 1
        cursor_parent = prow.get("parent_session_id")

    # A session continues when its parent was ended by any continuation reason.
    # Keep that lineage classification separate from compression provenance.
    is_continuation = False
    immediate_end_reason = None
    if parent_id:
        try:
            immediate_parent = db.get_session(parent_id)
        except Exception:
            immediate_parent = None
        if immediate_parent and is_continuation_end_reason(immediate_parent.get("end_reason")):
            is_continuation = True
            immediate_end_reason = immediate_parent.get("end_reason")

    rotated = bool(
        previous_hermes_session_id
        and previous_hermes_session_id != current_hermes_session_id
    )

    provenance: Dict[str, Any] = {
        "acpSessionId": acp_session_id,
        "currentHermesSessionId": current_hermes_session_id,
        "rootHermesSessionId": root_id,
        "parentHermesSessionId": parent_id,
        "sessionKind": "continuation" if is_continuation else "root",
        "compressionDepth": compression_depth,
    }
    if previous_hermes_session_id:
        provenance["previousHermesSessionId"] = previous_hermes_session_id
    if rotated:
        # Tell clients which continuation mechanism moved the internal head.
        # Preserve the established compression payload exactly while exposing
        # rollover as its own truthful non-compression creator kind.
        if immediate_end_reason == "compression":
            provenance["reason"] = "compression"
            provenance["creatorKind"] = "compression"
        elif immediate_end_reason in {
            "turn_boundary_rollover", "turn_boundary_rollover_recovered",
        }:
            provenance["reason"] = immediate_end_reason
            provenance["creatorKind"] = "rollover"

    return provenance


def session_provenance_meta(
    db: Any,
    acp_session_id: str,
    current_hermes_session_id: str,
    *,
    previous_hermes_session_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Return a ready ``_meta`` payload: ``{"hermes": {"sessionProvenance": ...}}``."""
    prov = build_session_provenance(
        db,
        acp_session_id,
        current_hermes_session_id,
        previous_hermes_session_id=previous_hermes_session_id,
    )
    if prov is None:
        return None
    return {"hermes": {"sessionProvenance": prov}}
