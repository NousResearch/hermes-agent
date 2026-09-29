"""Bounded snapshots of reactions to Matrix events."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.client_events import Method, decrypt_raw_event, raw_event


logger = logging.getLogger(__name__)
_REACTION_BATCH_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class MatrixReaction:
    event_id: str
    sender: str
    emoji: str
    target_event_id: str
    emoji_truncated: bool = False

    def to_dict(self, *, sender_authorized: bool) -> dict[str, str | bool]:
        result: dict[str, str | bool] = {
            "event_id": self.event_id,
            "sender": self.sender,
            "emoji": self.emoji,
            "target_event_id": self.target_event_id,
            "sender_authorized": sender_authorized,
        }
        if self.emoji_truncated:
            result["emoji_truncated"] = True
        return result


@dataclass(frozen=True)
class UndecryptableReaction:
    event_id: str
    error: str


@dataclass(frozen=True)
class ReactionSnapshot:
    reactions: tuple[MatrixReaction, ...] = ()
    truncated: bool = False
    undecryptable: tuple[UndecryptableReaction, ...] = ()
    error: str | None = None


async def fetch_event_reactions(
    client: Any, room_id: str, target_event_id: str, *, limit: int = 50,
) -> ReactionSnapshot:
    path = (
        f"/_matrix/client/v1/rooms/{quote(room_id, safe='')}"
        f"/relations/{quote(target_event_id, safe='')}/m.annotation"
    )
    try:
        response = await asyncio.wait_for(
            client.api.request(Method.GET, path, query_params={"dir": "b", "limit": str(limit)}),
            timeout=10.0,
        )
    except Exception as exc:
        logger.debug("Matrix: could not read reactions for %s in %s: %s", target_event_id, room_id, exc)
        return ReactionSnapshot(error=f"reactions unavailable: {type(exc).__name__}")

    chunk = response.get("chunk") if isinstance(response, dict) else None
    if not isinstance(chunk, list):
        return ReactionSnapshot(error="reactions unavailable: invalid response")

    reactions: list[MatrixReaction] = []
    undecryptable: list[UndecryptableReaction] = []
    seen: set[tuple[str, str]] = set()
    for raw in chunk[:limit]:
        if not isinstance(raw, dict):
            continue
        unsigned = raw.get("unsigned")
        if isinstance(unsigned, dict) and unsigned.get("redacted_because"):
            continue
        if raw.get("room_id", room_id) != room_id:
            continue
        event_id = raw.get("event_id")
        sender = raw.get("sender")
        if not isinstance(event_id, str) or not isinstance(sender, str):
            continue
        content = raw.get("content")
        outer_relation = content.get("m.relates_to") if isinstance(content, dict) else None
        if not isinstance(outer_relation, dict):
            continue
        if outer_relation.get("rel_type") != "m.annotation" or outer_relation.get("event_id") != target_event_id:
            continue
        visible = raw
        if raw.get("type") == "m.room.encrypted":
            decrypted, decryption_error = await decrypt_raw_event(client, raw)
            if decryption_error is not None:
                undecryptable.append(UndecryptableReaction(event_id, decryption_error))
                continue
            visible = raw_event(decrypted)
        if visible.get("type") != "m.reaction":
            continue
        key = outer_relation.get("key")
        if not isinstance(key, str) or not key:
            continue
        identity = (sender, key)
        if identity in seen:
            continue
        seen.add(identity)
        reactions.append(MatrixReaction(
            event_id, sender, key[:40], target_event_id, emoji_truncated=len(key) > 40,
        ))

    return ReactionSnapshot(
        tuple(reactions), truncated=bool(response.get("next_batch")), undecryptable=tuple(undecryptable),
    )


async def fetch_reactions_for_events(
    client: Any, room_id: str, event_ids: list[str], *, limit: int = 8,
) -> list[ReactionSnapshot]:
    if not event_ids:
        return []

    semaphore = asyncio.Semaphore(4)

    async def fetch(event_id: str) -> ReactionSnapshot:
        async with semaphore:
            return await fetch_event_reactions(client, room_id, event_id, limit=limit)

    tasks = [asyncio.create_task(fetch(event_id)) for event_id in reversed(event_ids)]
    try:
        await asyncio.wait(tasks, timeout=_REACTION_BATCH_TIMEOUT_SECONDS)
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    snapshots = [
        task.result() if not task.cancelled() else ReactionSnapshot(error="reactions unavailable: timeout")
        for task in tasks
    ]
    return list(reversed(snapshots))
