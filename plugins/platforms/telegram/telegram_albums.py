"""Short-lived memory of unaddressed Telegram album items.

Telegram delivers an album (``media_group_id``) as one message per item and puts the caption on a
single item. With ``require_mention`` the gate judges each item alone, so an @mention in one caption,
or a later "@bot" reply to one item, used to reach the agent with only that item. The Bot API cannot
fetch an old message by id, so the adapter keeps the skipped items here (``Message`` objects only;
nothing is downloaded or added to any transcript) and pulls them back in once the album is addressed.
"""

import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

# Long enough for "send the files, then reply @bot a few minutes later"; short enough to stay small.
ALBUM_ITEM_TTL_SECONDS = 1800.0
ALBUM_ITEM_MAX_ENTRIES = 300
# Window in which later items of an addressed album bypass the mention gate.
TRIGGERED_ALBUM_TTL_SECONDS = 60.0


class RecentAlbumItems:
    def __init__(self, ttl: float = ALBUM_ITEM_TTL_SECONDS, max_entries: int = ALBUM_ITEM_MAX_ENTRIES,
                 triggered_ttl: float = TRIGGERED_ALBUM_TTL_SECONDS, clock=time.monotonic):
        self._ttl = ttl
        self._max = max_entries
        self._triggered_ttl = triggered_ttl
        self._clock = clock
        # (chat_id, message_id) -> (stored_at, media_group_id, message, update_id)
        self._items: "OrderedDict[Tuple[str, int], Tuple[float, str, Any, Optional[int]]]" = OrderedDict()
        self._triggered: Dict[Tuple[str, str], float] = {}

    def _prune(self) -> None:
        now = self._clock()
        while self._items:
            key, (stored_at, *_rest) = next(iter(self._items.items()))
            if now - stored_at <= self._ttl and len(self._items) <= self._max:
                break
            self._items.popitem(last=False)
        for key in [k for k, expires in self._triggered.items() if expires < now]:
            del self._triggered[key]

    def remember(self, chat_id: str, message: Any, update_id: Optional[int] = None) -> None:
        media_group_id = getattr(message, "media_group_id", None)
        message_id = getattr(message, "message_id", None)
        if not media_group_id or message_id is None:
            return
        self._items[(str(chat_id), int(message_id))] = (self._clock(), str(media_group_id), message, update_id)
        self._prune()

    def media_group_of(self, chat_id: str, message: Any) -> Optional[str]:
        """``media_group_id`` from the message itself, else from a remembered copy of it."""
        media_group_id = getattr(message, "media_group_id", None)
        if media_group_id:
            return str(media_group_id)
        message_id = getattr(message, "message_id", None)
        if message_id is None:
            return None
        entry = self._items.get((str(chat_id), int(message_id)))
        return entry[1] if entry else None

    def siblings(self, chat_id: str, media_group_id: str, exclude_message_id: Any = None, *,
                 pop: bool = False) -> List[Tuple[Any, Optional[int]]]:
        """Remembered items of the album, oldest first, as ``(message, update_id)``."""
        self._prune()
        chat_id = str(chat_id)
        exclude = int(exclude_message_id) if exclude_message_id is not None else None
        keys = sorted(
            key for key, (_ts, group, _msg, _uid) in self._items.items()
            if key[0] == chat_id and group == str(media_group_id) and key[1] != exclude)
        found = [(self._items[key][2], self._items[key][3]) for key in keys]
        if pop:
            for key in keys:
                del self._items[key]
        return found

    def mark_triggered(self, chat_id: str, media_group_id: str) -> bool:
        """Mark the album addressed; True only the first time (within the window)."""
        self._prune()
        key = (str(chat_id), str(media_group_id))
        first = key not in self._triggered
        self._triggered[key] = self._clock() + self._triggered_ttl
        return first

    def is_triggered(self, chat_id: str, media_group_id: Optional[str]) -> bool:
        if not media_group_id:
            return False
        expires = self._triggered.get((str(chat_id), str(media_group_id)))
        return expires is not None and expires >= self._clock()
