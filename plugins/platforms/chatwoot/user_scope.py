"""Hard per-member scoping for ``crwd_db`` on Chatwoot sessions.

Resolves the chatting member's CRWD ``users._id`` and enforces it via
``tool_request`` middleware (rewrite args) plus a ``pre_tool_call`` safety net
(block private collections / unresolved identity).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from plugins.platforms.chatwoot.coach_context import (
    cross_user_request_active,
    resolve_member_crwd_id,
)

logger = logging.getLogger(__name__)

_OBJECTID_RE = re.compile(r"^[a-fA-F0-9]{24}$")

_USER_SCOPED_ACTIONS = frozenset({
    "get_user",
    "get_user_gigs",
    "get_waitlisted_gigs",
    "get_user_gig_status",
    "get_user_products",
    "get_user_receipts",
    "get_user_notifications",
    "list_active_gigs",
})

_PRIVATE_COLLECTIONS = frozenset({
    "users",
    "added_crwd_members",
    "user_product_purchases",
    "receipt_upload_history",
    "notifications",
})

_UNRESOLVED_MSG = (
    "I can't verify your CRWD account yet. Please try again in a moment."
)

_CROSS_USER_BLOCK_MSG = (
    "I can only provide you with your information."
)


def _is_chatwoot() -> bool:
    try:
        from gateway.session_context import get_session_env

        return (get_session_env("HERMES_SESSION_PLATFORM", "") or "").strip().lower() == "chatwoot"
    except Exception:
        return False


def _contact_id() -> Optional[str]:
    try:
        from gateway.session_context import get_session_env

        cid = (get_session_env("HERMES_SESSION_USER_ID", "") or "").strip()
        return cid or None
    except Exception:
        return None


def _normalize_id(value: Any) -> str:
    return str(value or "").strip().lower()


def _ids_match(a: Any, b: Any) -> bool:
    return _normalize_id(a) == _normalize_id(b)


def resolved_member_id() -> Optional[str]:
    """Return the authenticated member's CRWD user id for this Chatwoot turn."""
    contact_id = _contact_id()
    if not contact_id:
        return None
    return resolve_member_crwd_id(contact_id)


def _explicit_user_ref(args: Dict[str, Any], action: str) -> Optional[str]:
    """Return an explicit user id/identifier from tool args, if present."""
    if action == "get_user":
        ident = args.get("identifier")
        if ident is not None and str(ident).strip():
            return str(ident).strip()
    elif action in _USER_SCOPED_ACTIONS - {"get_user"}:
        uid = args.get("user_id")
        if uid is not None and str(uid).strip():
            return str(uid).strip()
    return None


def _scoped_crwd_db_args(args: Dict[str, Any], member_id: str) -> Dict[str, Any]:
    scoped = dict(args)
    action = str(scoped.get("action", "")).strip()
    if action == "get_user":
        scoped["identifier"] = member_id
    elif action in _USER_SCOPED_ACTIONS - {"get_user"}:
        scoped["user_id"] = member_id
    return scoped


def on_tool_request(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, Any]]:
    """Rewrite ``crwd_db`` user-scoped args to the authenticated member id."""
    if tool_name != "crwd_db" or not _is_chatwoot():
        return None
    if not isinstance(args, dict):
        return None

    action = str(args.get("action", "")).strip()
    if action not in _USER_SCOPED_ACTIONS:
        return None

    member_id = resolved_member_id()
    if not member_id:
        return None

    if cross_user_request_active():
        return None

    explicit = _explicit_user_ref(args, action)
    if explicit and not _ids_match(explicit, member_id):
        return None

    return {
        "args": _scoped_crwd_db_args(args, member_id),
        "source": "chatwoot-user-scope",
        "reason": f"scoped crwd_db {action} to authenticated member",
    }


def on_pre_tool_call(
    tool_name: str = "",
    args: Any = None,
    **_: Any,
) -> Optional[Dict[str, str]]:
    """Block cross-user ``crwd_db`` access and private ``custom_query`` collections."""
    if tool_name != "crwd_db" or not _is_chatwoot():
        return None
    if not isinstance(args, dict):
        return None

    action = str(args.get("action", "")).strip()

    if action == "custom_query":
        collection = str(args.get("collection", "")).strip()
        if collection in _PRIVATE_COLLECTIONS:
            return {"action": "block", "message": _CROSS_USER_BLOCK_MSG}
        filt = args.get("filter")
        if isinstance(filt, dict) and _filter_targets_other_member(filt):
            return {"action": "block", "message": _CROSS_USER_BLOCK_MSG}
        return None

    if action not in _USER_SCOPED_ACTIONS:
        return None

    if cross_user_request_active():
        return {"action": "block", "message": _CROSS_USER_BLOCK_MSG}

    member_id = resolved_member_id()
    if not member_id:
        return {"action": "block", "message": _UNRESOLVED_MSG}

    if action == "get_user":
        ident = args.get("identifier")
        if ident and not _ids_match(ident, member_id):
            return {"action": "block", "message": _CROSS_USER_BLOCK_MSG}
    else:
        uid = args.get("user_id")
        if uid and not _ids_match(uid, member_id):
            return {"action": "block", "message": _CROSS_USER_BLOCK_MSG}

    return None


def _filter_targets_other_member(filt: Any) -> bool:
    """Best-effort scan of a custom_query filter for another member's id."""
    member_id = resolved_member_id()
    if not member_id:
        return False

    user_keys = frozenset({
        "user_id", "member", "worker_id", "to", "_id",
    })

    def walk(obj: Any) -> bool:
        if isinstance(obj, dict):
            for key, val in obj.items():
                key_l = str(key).strip().lower()
                if key_l in user_keys or key_l.endswith("_id"):
                    if isinstance(val, str) and _OBJECTID_RE.match(val):
                        if not _ids_match(val, member_id):
                            return True
                    if isinstance(val, dict) and "$in" in val:
                        for item in val.get("$in") or []:
                            if isinstance(item, str) and _OBJECTID_RE.match(item):
                                if not _ids_match(item, member_id):
                                    return True
                if walk(val):
                    return True
        elif isinstance(obj, list):
            return any(walk(item) for item in obj)
        return False

    try:
        return walk(filt)
    except Exception as exc:
        logger.debug("[chatwoot-user-scope] filter scan failed: %s", exc)
        return False


def _filter_json_for_tests(filt: Any) -> bool:
    """Test helper — expose filter scan without private name mangling."""
    return _filter_targets_other_member(filt)
