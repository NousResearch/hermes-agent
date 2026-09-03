"""Approval Delegation — state management for cross-platform approval routing.

When ``approvals.delegation.enabled`` is ``true`` in config.yaml, non-admin
users' dangerous-command approval requests are routed to an admin's chat
instead of the user's own chat.  The admin approves/denies from their own
DM, and the result is relayed back to the original user's session.

This module provides:
- Delegation config loading (``get_delegation_config``)
- Admin lookup and user classification (``is_admin_user``, ``get_admins``)
- Delegation state management (``register_delegation``, ``resolve_delegation``,
  ``clear_delegation``) for mapping admin chat → original session

No monkey-patching — integration is via the standard approval flow in
``tools/approval.py`` and ``gateway/run.py``.

.. note::

    Delegation state is held **in-process** (module-level dict).  In a
    multi-worker gateway deployment each worker maintains its own map, so
    a delegation registered in worker A is not visible to worker B.
    This is acceptable for the current single-worker gateway; horizontal
    scaling would require a shared store (e.g. Redis).
"""

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────

_delegation_config: Optional[Dict] = None
_delegation_config_lock = threading.Lock()


def get_delegation_config() -> Dict:
    """Load ``approvals.delegation`` from config.yaml (lazy, thread-safe).

    Returns a dict with ``enabled`` (bool) and ``admins`` (list of dicts
    with ``platform``, ``user_id``, and optional ``chat_id``).
    """
    global _delegation_config
    if _delegation_config is not None:
        return _delegation_config

    with _delegation_config_lock:
        if _delegation_config is not None:
            return _delegation_config

        cfg: Dict = {"enabled": False, "admins": []}
        try:
            from hermes_cli.config import load_config
            config = load_config()
            approvals = config.get("approvals", {}) or {}
            raw = approvals.get("delegation", {})
            if isinstance(raw, dict):
                cfg["enabled"] = bool(raw.get("enabled", False))
                raw_admins = raw.get("admins", [])
                if isinstance(raw_admins, list):
                    cfg["admins"] = [
                        {
                            "platform": str(a.get("platform", "")).strip().lower(),
                            "user_id": str(a.get("user_id", "")).strip(),
                            "chat_id": str(
                                a.get("chat_id", a.get("user_id", ""))
                            ).strip(),
                        }
                        for a in raw_admins
                        if isinstance(a, dict)
                        and a.get("platform")
                        and a.get("user_id")
                    ]
        except Exception as e:
            logger.warning("[approval-delegation] Failed to load config: %s", e)

        _delegation_config = cfg
        if cfg["enabled"]:
            logger.info(
                "[approval-delegation] Enabled with %d admin(s): %s",
                len(cfg["admins"]),
                [f"{a['platform']}:{a['user_id']}" for a in cfg["admins"]],
            )
        return cfg


def reload_delegation_config() -> None:
    """Force reload of delegation config (e.g. after config edit)."""
    global _delegation_config
    with _delegation_config_lock:
        _delegation_config = None
    get_delegation_config()


def is_delegation_enabled() -> bool:
    """Quick check if delegation is enabled."""
    return get_delegation_config().get("enabled", False)


def get_admins() -> List[Dict]:
    """Return the list of admin dicts (platform, user_id, chat_id)."""
    return get_delegation_config().get("admins", [])


def is_admin_user(platform: str, user_id: str) -> bool:
    """Check if the given user is an admin (never delegated)."""
    plat = str(platform).strip().lower()
    uid = str(user_id).strip()
    return any(
        a["platform"] == plat and a["user_id"] == uid
        for a in get_admins()
    )


# ── Delegation state ────────────────────────────────────────────────────
#
# When an approval is delegated, we store a mapping:
#   admin_chat_key → {session_key → delegation entry}
# so that when the admin sends /approve or /deny, we can resolve the
# original session's pending approval.  Multiple concurrent delegations
# to the same admin are supported (each session_key gets its own entry).
#
# The admin_chat_key format is "platform:chat_id".

_delegation_map: Dict[str, Dict[str, Dict[str, Any]]] = {}
_delegation_lock = threading.Lock()
_DELEGATION_TTL = 600  # 10 minutes


def _admin_chat_key(platform: str, chat_id: str) -> str:
    """Build the lookup key for the delegation map."""
    return f"{str(platform).strip().lower()}:{str(chat_id).strip()}"


def register_delegation(
    *,
    admin_platform: str,
    admin_chat_id: str,
    session_key: str,
    user_platform: str,
    user_chat_id: str,
    user_chat_meta: Any = None,
    command: str = "",
    description: str = "",
) -> None:
    """Register a pending delegation: admin_chat → original session.

    Called when an approval is redirected to an admin.  Supports multiple
    concurrent delegations to the same admin (keyed by session_key).
    """
    key = _admin_chat_key(admin_platform, admin_chat_id)
    entry = {
        "session_key": session_key,
        "user_platform": user_platform,
        "user_chat_id": user_chat_id,
        "user_chat_meta": user_chat_meta,
        "command": command,
        "description": description,
        "created_at": time.monotonic(),
    }
    with _delegation_lock:
        _delegation_map.setdefault(key, {})[session_key] = entry
    logger.info(
        "[approval-delegation] Registered: admin=%s → session=%s cmd=%s",
        key, session_key[:16], command[:60],
    )


def resolve_delegation(platform: str, chat_id: str) -> Optional[Dict[str, Any]]:
    """Look up a pending delegation for the given admin chat.

    Returns the most recent delegation entry dict, or None if no pending
    delegation.  Automatically prunes stale entries.
    """
    key = _admin_chat_key(platform, chat_id)
    now = time.monotonic()
    with _delegation_lock:
        sessions = _delegation_map.get(key)
        if not sessions:
            return None
        # Prune stale entries and find the most recent
        stale = [sk for sk, e in sessions.items() if now - e["created_at"] > _DELEGATION_TTL]
        for sk in stale:
            del sessions[sk]
        if not sessions:
            _delegation_map.pop(key, None)
            return None
        # Return the most recent entry (safe: empty-sessions guard above)
        return max(sessions.values(), key=lambda e: e["created_at"])


def resolve_delegation_for_session(platform: str, chat_id: str, session_key: str) -> Optional[Dict[str, Any]]:
    """Look up a specific delegation entry by admin chat + session_key."""
    key = _admin_chat_key(platform, chat_id)
    with _delegation_lock:
        sessions = _delegation_map.get(key)
        if not sessions:
            return None
        entry = sessions.get(session_key)
        if entry is None:
            return None
        if time.monotonic() - entry["created_at"] > _DELEGATION_TTL:
            del sessions[session_key]
            if not sessions:
                _delegation_map.pop(key, None)
            return None
        return entry


def clear_delegation(platform: str, chat_id: str, session_key: Optional[str] = None) -> None:
    """Remove a delegation entry after it's been resolved.

    If session_key is given, only that entry is removed.  Otherwise all
    entries for the admin chat are removed.
    """
    key = _admin_chat_key(platform, chat_id)
    with _delegation_lock:
        if session_key:
            sessions = _delegation_map.get(key)
            if sessions:
                sessions.pop(session_key, None)
                if not sessions:
                    _delegation_map.pop(key, None)
        else:
            _delegation_map.pop(key, None)


def clear_all_delegations() -> None:
    """Clear all delegation state (e.g. on gateway shutdown)."""
    with _delegation_lock:
        _delegation_map.clear()
