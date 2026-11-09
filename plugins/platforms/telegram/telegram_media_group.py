"""Telegram album (``media_group_id``) debounce: one album, one turn.

Telegram sends each album item as its own update, and the adapter downloads each item before
merging it. Updates of one chat run one after another (``PerChatUpdateProcessor``), so a sibling
whose download is still running has not been merged when the first item's debounce fires.
Without tracking, a slow download split the album across turns (#74965, #150720). Each album
item now counts as *in flight* from the moment its handler starts. The flush waits for in-flight
siblings, bounded by ``MEDIA_GROUP_MAX_WAIT_SECONDS`` so one stuck download cannot hold the album.
"""

import asyncio
import contextlib
import functools
from typing import Any, Iterator

from gateway.platforms.event import MessageEvent

# Private event attribute: an item's own notes (too large, …), which an album merge never dedupes away
# (#151871). Not ``event.metadata``: native admission refuses any metadata key outside its allowlist.
ITEM_NOTES_ATTR = "_telegram_item_notes"


def add_item_note(event: Any, note: str) -> None:
    """Append ``note`` to the event text (the caption survives) and record it as this item's own note."""
    event.text = f"{event.text}\n\n{note}" if event.text else note
    event.__dict__.setdefault(ITEM_NOTES_ATTR, []).append(note)


def item_notes(event: Any) -> list[str]:
    return list(getattr(event, ITEM_NOTES_ATTR, None) or [])

# Upper bound on how long a due flush waits for in-flight siblings before sending what it has.
MEDIA_GROUP_MAX_WAIT_SECONDS = 8.0
_IN_FLIGHT_POLL_SECONDS = 0.05
_MAX_WAIT_SECONDS_CEILING = 10.0


def _in_flight(adapter: Any) -> dict[str, int]:
    return adapter.__dict__.setdefault("_media_group_in_flight", {})


@contextlib.contextmanager
def tracking_album_item(adapter: Any, msg: Any) -> Iterator[None]:
    """Count ``msg`` as an in-flight album item while its handler runs (download, cache, merge)."""
    group = getattr(msg, "media_group_id", None) if msg is not None else None
    if group is None:
        yield
        return
    key, counts = str(group), _in_flight(adapter)
    counts[key] = counts.get(key, 0) + 1
    try:
        yield
    finally:
        left = counts.get(key, 1) - 1
        if left > 0:
            counts[key] = left
        else:
            counts.pop(key, None)


def tracking_album_items(handler):
    """Decorate the media handler: its album item is in flight from arrival until the handler returns."""
    @functools.wraps(handler)
    async def wrapper(adapter, update, context):
        with tracking_album_item(adapter, adapter._effective_update_message(update)):
            return await handler(adapter, update, context)
    return wrapper


def media_group_wait_seconds(adapter: Any) -> float:
    """Debounce after the last merged item: ``platforms.telegram.extra.media_group_wait_seconds``."""
    return adapter._coerce_float_extra(
        "media_group_wait_seconds", adapter.MEDIA_GROUP_WAIT_SECONDS, max_value=_MAX_WAIT_SECONDS_CEILING)


async def queue_media_group_event(adapter: Any, media_group_id: str, event: MessageEvent) -> None:
    """Merge an album item into its pending event and (re)arm the album's flush."""
    if adapter._should_drop_delayed_delivery():
        adapter._hold_inbound_event(event, where="media-group-enqueue")
        return
    adapter._merge_into_pending(adapter._media_group_events, media_group_id, event)
    adapter._accept_update()
    prior_task = adapter._media_group_tasks.get(media_group_id)
    if prior_task:
        prior_task.cancel()
    adapter._media_group_tasks[media_group_id] = asyncio.create_task(adapter._flush_media_group_event(media_group_id))


async def flush_media_group_event(adapter: Any, media_group_id: str) -> None:
    """Debounce, then wait (bounded) for in-flight siblings, then dispatch the merged album.

    Waiting happens before ``_flush_buffered`` pops the event, so a sibling that lands meanwhile
    cancels this task (via ``queue_media_group_event``) and nothing is lost."""
    await asyncio.sleep(media_group_wait_seconds(adapter))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + MEDIA_GROUP_MAX_WAIT_SECONDS
    while _in_flight(adapter).get(media_group_id, 0) > 0 and loop.time() < deadline:
        await asyncio.sleep(_IN_FLIGHT_POLL_SECONDS)
    await adapter._flush_buffered(
        adapter._media_group_events, adapter._media_group_tasks, media_group_id, 0, "media-group")
