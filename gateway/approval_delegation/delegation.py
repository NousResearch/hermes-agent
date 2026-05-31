"""
Approval Delegation — Route dangerous-command approvals to designated admins.

When approvals.delegate_to is configured in config.yaml, non-admin users'
dangerous-command approval requests are sent to an admin's chat instead
of the user's own chat. The admin approves/denies from their own DM,
and the result is relayed back to the original user's session.

Supports cross-platform delegation: user on Feishu, admin on WeChat, etc.
"""

import logging
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Delegation config (lazy-loaded from config.yaml)
_delegation_config: list = []
_delegation_config_loaded: bool = False
_delegation_config_lock = threading.Lock()

# Delegation map: admin_chat_key -> deque of delegation entries
# Key format: "<platform>:<chat_id>"
_delegation_map: Dict[str, deque] = {}
_delegation_lock = threading.Lock()

# TTL for stale delegations (seconds)
_DELEGATION_TTL = 600  # 10 minutes


def _admin_chat_key(platform: str, chat_id: str) -> str:
    """Build the lookup key for the delegation map."""
    return f"{str(platform).strip().lower()}:{str(chat_id).strip()}"


def _ensure_delegation_config_loaded() -> list:
    """Lazy-load approvals.delegate_to from config.yaml. Returns the list."""
    global _delegation_config, _delegation_config_loaded
    
    if _delegation_config_loaded:
        return _delegation_config
    
    with _delegation_config_lock:
        # Double-check after acquiring lock
        if _delegation_config_loaded:
            return _delegation_config
        
        try:
            from hermes_cli.config import load_config
            config = load_config()
            approvals = config.get("approvals", {}) or {}
            raw = approvals.get("delegate_to", None)
            
            if isinstance(raw, list):
                _delegation_config = [
                    {
                        "platform": str(item.get("platform", "")).strip().lower(),
                        "user_id": str(item.get("user_id", "")).strip(),
                        "chat_id": str(item.get("chat_id", item.get("user_id", ""))).strip(),
                    }
                    for item in raw
                    if isinstance(item, dict) and item.get("platform") and item.get("user_id")
                ]
                logger.info(
                    "[approval-delegation] Loaded %d admin(s): %s",
                    len(_delegation_config),
                    [f"{d['platform']}:{d['user_id']}" for d in _delegation_config],
                )
        except Exception as e:
            logger.warning("[approval-delegation] Failed to load config: %s", e)
        
        _delegation_config_loaded = True
    
    return _delegation_config


def reload_config() -> None:
    """Force reload delegation config (call after config changes)."""
    global _delegation_config_loaded
    _delegation_config_loaded = False
    _ensure_delegation_config_loaded()


def get_delegation_admins() -> List[Dict[str, str]]:
    """Return the current delegation admin config list."""
    return list(_ensure_delegation_config_loaded())


def is_admin_user(platform: str, user_id: str) -> bool:
    """Check whether user_id is a designated approval admin on platform."""
    if not platform or not user_id:
        return False
    
    platform_lower = str(platform).strip().lower()
    user_id_str = str(user_id).strip()
    
    for entry in _ensure_delegation_config_loaded():
        if entry["platform"] == platform_lower and entry["user_id"] == user_id_str:
            return True
    
    return False


def register_delegation(
    admin_platform: str,
    admin_chat_id: str,
    target_session_key: str,
    notify_cb: Optional[Any] = None,
    *,
    user_platform: str = "",
    user_chat_id: str = "",
    user_chat_meta: Optional[Dict] = None,
) -> None:
    """Register a delegated approval so the admin can resolve it via /approve.
    
    Multiple delegations to the same admin are queued (FIFO) so the oldest
    pending request is resolved first when the admin types /approve.
    
    Args:
        admin_platform: Platform where admin will approve (feishu/wecom/etc.)
        admin_chat_id: Admin's chat_id on that platform
        target_session_key: The session_key waiting for approval
        notify_cb: Optional callback (unused, for compatibility)
        user_platform: Original user's platform (for cross-platform notification)
        user_chat_id: Original user's chat_id (for notification after approval)
        user_chat_meta: Metadata dict (thread_id, etc.) for routing
    """
    key = _admin_chat_key(admin_platform, admin_chat_id)
    
    with _delegation_lock:
        queue = _delegation_map.get(key)
        if queue is None:
            queue = deque()
            _delegation_map[key] = queue
        
        entry = {
            "session_key": target_session_key,
            "notify_cb": notify_cb,
            "user_platform": user_platform,
            "user_chat_id": user_chat_id,
            "user_chat_meta": user_chat_meta,
            "timestamp": time.monotonic(),
        }
        queue.append(entry)
        
        logger.info(
            "[approval-delegation] Registered: admin=%s:%s, user=%s:%s, session=%s",
            admin_platform, admin_chat_id,
            user_platform or "?", user_chat_id or "?",
            target_session_key[:16],
        )


def resolve_delegation(admin_platform: str, admin_chat_id: str) -> Optional[Dict[str, Any]]:
    """Look up the OLDEST pending delegation for an admin (FIFO).
    
    Returns the delegation dict or None if nothing is pending.
    Automatically cleans up stale entries.
    """
    key = _admin_chat_key(admin_platform, admin_chat_id)
    
    with _delegation_lock:
        queue = _delegation_map.get(key)
        if not queue:
            return None
        
        # Clean stale entries from front
        now = time.monotonic()
        while queue and (now - queue[0].get("timestamp", 0)) > _DELEGATION_TTL:
            stale = queue.popleft()
            logger.debug(
                "[approval-delegation] Expired stale delegation for session %s",
                stale.get("session_key", "?")[:16],
            )
        
        return queue[0] if queue else None


def clear_delegation(admin_platform: str, admin_chat_id: str) -> None:
    """Remove the OLDEST pending delegation entry (FIFO) for an admin.
    
    Should be called after the admin approves or denies.
    """
    key = _admin_chat_key(admin_platform, admin_chat_id)
    
    with _delegation_lock:
        queue = _delegation_map.get(key)
        if queue:
            removed = queue.popleft()
            logger.debug(
                "[approval-delegation] Cleared delegation for session %s",
                removed.get("session_key", "?")[:16],
            )
            if not queue:
                _delegation_map.pop(key, None)


def clear_delegation_for_session(session_key: str) -> int:
    """Remove ALL delegation entries whose target session has expired.
    
    Walks every admin's queue and removes entries matching session_key.
    Called from the gateway approval timeout path to prevent stale
    delegations from piling up.
    
    Returns the number of entries removed.
    """
    removed = 0
    
    with _delegation_lock:
        for key in list(_delegation_map.keys()):
            queue = _delegation_map.get(key)
            if not queue:
                continue
            
            # Filter out entries for the expired session
            new_queue = deque(
                entry for entry in queue
                if entry.get("session_key") != session_key
            )
            removed += len(queue) - len(new_queue)
            
            if new_queue:
                _delegation_map[key] = new_queue
            else:
                _delegation_map.pop(key, None)
    
    if removed:
        logger.info(
            "[approval-delegation] Cleared %d stale delegation(s) for session %s",
            removed, session_key[:16],
        )
    
    return removed
