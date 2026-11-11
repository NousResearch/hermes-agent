"""Adapter-local identity aliases learned from authenticated Feishu events."""

import logging
import threading
from typing import Any, Optional


logger = logging.getLogger(__name__)


class SenderIdentityCache:
    """Keep bounded ID evidence; authorization is always checked live by the gateway."""

    def __init__(self, max_entries: int = 1024):
        self._max_entries = max_entries
        self._identities: dict[str, tuple[str, str]] = {}
        self._lock = threading.Lock()

    def remember(self, open_id: Optional[str], user_id: Optional[str], union_id: Optional[str]) -> None:
        if not open_id or not (user_id or union_id):
            return
        with self._lock:
            previous = self._identities.pop(open_id, ("", ""))
            self._identities[open_id] = (user_id or previous[0], union_id or previous[1])
            if len(self._identities) > self._max_entries:
                self._identities.pop(next(iter(self._identities)))

    def resolve(self, open_id: str, user_id: str, union_id: str) -> tuple[str, str]:
        with self._lock:
            known = self._identities.get(open_id, ("", ""))
        return user_id or known[0], union_id or known[1]


def is_interactive_operator_authorized(
    adapter: Any, open_id: str, *, user_id: str = "", union_id: str = "", chat_id: str = "",
) -> bool:
    """Check each SDK-associated ID live so learning a tenant ID cannot hide an open-ID grant."""
    normalized = str(open_id or "").strip()
    if not normalized:
        return False
    operator_ids = tuple(dict.fromkeys(value for value in (
        str(user_id or "").strip(), normalized, str(union_id or "").strip(),
    ) if value))
    allowed_ids = set(adapter._admins) | set(adapter._allowed_group_users)
    if allowed_ids:
        return "*" in allowed_ids or bool(set(operator_ids) & allowed_ids)
    canonical_scope = None
    if adapter.gateway_runner is not None:
        canonical_scope = _operator_routing_identity(adapter, operator_ids[0], chat_id)
        if canonical_scope is None:
            return False
    # Keep the tenant ID first, matching SessionSource.user_id. An existing pairing may
    # name an alias, but its lookup must stay in the canonical user's profile.
    for index, principal in enumerate(operator_ids):
        if canonical_scope is not None and index and not _alias_uses_authorization_scope(adapter, principal, chat_id, canonical_scope):
            continue
        verdict = adapter._is_sender_authorized(principal, "group", str(chat_id or "") or None)
        if verdict is True:
            return True
        if verdict is None:
            logger.debug("[Feishu] No gateway authorization verdict for card operator %s; denying", principal)
    return False


def _alias_uses_authorization_scope(adapter: Any, user_id: str, chat_id: str, canonical_scope: Any) -> bool:
    # A dedicated bot's scalar authorization callback is bound to its owner profile,
    # even when message routing selects another runtime. Keep alias grants in that scope.
    owner = adapter._owner_transport_profile()
    if owner is not None and owner != canonical_scope.runtime_profile:
        return False
    return _operator_routing_identity(adapter, user_id, chat_id) == canonical_scope


def _operator_routing_identity(adapter: Any, user_id: str, chat_id: str) -> Any:
    source = adapter.build_source(chat_id=str(chat_id or ""), chat_type="group", user_id=user_id)
    return adapter._canonicalize(source)


async def resolve_sender_profile(adapter: Any, sender_id: Any, *, is_bot: bool = False) -> dict[str, Optional[str]]:
    """Remember the SDK's ID association before an open_id-only card callback arrives."""
    open_id = getattr(sender_id, "open_id", None) or None
    user_id = getattr(sender_id, "user_id", None) or None
    union_id = getattr(sender_id, "union_id", None) or None
    adapter._sender_identities.remember(open_id, user_id, union_id)
    primary_id = user_id or open_id
    # bots/basic_batch only accepts open_id; humans prefer the tenant-scoped ID.
    name_lookup_id = open_id if is_bot else (primary_id or union_id)
    display_name = await adapter._resolve_sender_name_from_api(name_lookup_id, is_bot=is_bot)
    return {"user_id": primary_id, "user_name": display_name, "user_id_alt": union_id}
