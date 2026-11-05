"""Parent-bound lineage intent shared by the reference repositories."""

from typing import Any, Mapping

from persistence.conversation.sessions import SessionLineageKind


def direct_lineage_kind(row: Mapping[str, Any]) -> SessionLineageKind:
    """Inherited intent is not a fork of this parent; absent origins retain legacy meaning."""
    origin = row.get("lineage_origin_id")
    if origin is not None and origin != row.get("parent_session_id"):
        return SessionLineageKind.NONE
    return row.get("lineage_kind", SessionLineageKind.NONE)
