"""Adapter-local identity aliases learned from authenticated Feishu events."""

import threading
from typing import Any, Optional


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
