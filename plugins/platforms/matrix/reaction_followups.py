"""Durable, single-use watches for reactions to delivered Matrix replies."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import quote

from plugins.platforms.matrix.reaction_context import Method
from plugins.platforms.matrix.followup_context import (
    EXCERPT_PART_LIMIT, REPLY_EXCERPT_CHARS, ReplyExcerpt, ReplyExcerptPart, body_digest, source_characters,
)


WATCH_SECONDS = 600
REGISTRATION_REPLAY_SECONDS = 10
REGISTRATION_REPLAY_LIMIT = 32
# Source fields that the follow-up needs to rebuild the original session key. Room and user
# names are resolved again when the reaction arrives.
_SOURCE_BINDING_KEYS = ("chat_type", "scope_id", "parent_chat_id")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingFollowupReaction:
    target_event_id: str
    emoji: str
    sender: str
    event_id: str
    expires_at: float


class PendingFollowupReactions:
    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.clock = clock
        self.events: dict[str, PendingFollowupReaction] = {}
        self.registered = False

    def remember(self, target_event_id: str, emoji: str, sender: str, event_id: str) -> None:
        self.events = {key: event for key, event in self.events.items()
                       if event.expires_at > self.clock()}
        if self.registered or not event_id or not target_event_id or not emoji:
            return
        if event_id in self.events or len(self.events) >= REGISTRATION_REPLAY_LIMIT:
            return
        self.events[event_id] = PendingFollowupReaction(
            target_event_id, emoji, sender, event_id, self.clock() + REGISTRATION_REPLAY_SECONDS,
        )

    def eligible(self, event_id: str) -> bool:
        event = self.events.get(event_id)
        return event is not None and event.expires_at > self.clock()

    def discard(self, event_id: str) -> None:
        self.events.pop(event_id, None)

    def clear(self) -> None:
        self.events.clear()


class FinalDeliveryEvents:
    def __init__(self) -> None:
        self.events: OrderedDict[tuple[str, str], str | None] = OrderedDict()
        self.parts: OrderedDict[tuple[str, str], tuple[str, int]] = OrderedDict()

    def remember(
        self, room_id: str, event_id: str, content: dict[str, Any], *, finalize: bool,
    ) -> None:
        relation = content.get("m.relates_to", {})
        target = relation.get("event_id") if relation.get("rel_type") == "m.replace" else event_id
        key = (room_id, target)
        self.events.pop(key, None)
        self.events[key] = event_id if finalize else None
        body = (content.get("m.new_content", {}) if target != event_id else content).get("body", "")
        self.parts[key] = (body_digest(body), source_characters(body))
        if len(self.events) > 512:
            removed, _ = self.events.popitem(last=False)
            self.parts.pop(removed, None)

    def excerpt(self, room_id: str, visible_ids: tuple[str, ...], text: str) -> ReplyExcerpt | None:
        targets = set(visible_ids)
        ordered = [(key, value) for key, value in self.parts.items()
                   if key[0] == room_id and key[1] in targets]
        if len(ordered) != len(targets):
            return None
        parts = []
        covered = 0
        required = min(len(text), REPLY_EXCERPT_CHARS)
        for key, (digest, count) in ordered[:EXCERPT_PART_LIMIT]:
            parts.append(ReplyExcerptPart(key[1], digest))
            covered += count
            if covered >= required:
                break
        complete = covered >= required or len(parts) == len(ordered)
        return ReplyExcerpt(tuple(parts), complete)

    def target_digests(self, room_id: str, visible_ids: tuple[str, ...]) -> dict[str, str]:
        return {event_id: self.parts[(room_id, event_id)][0] for event_id in visible_ids
                if (room_id, event_id) in self.parts}

    def latest(self, room_id: str, visible_ids: tuple[str, ...]) -> str | None:
        targets = set(visible_ids)
        return next((self.events[key] for key in reversed(self.events)
                     if key[0] == room_id and key[1] in targets), None)


async def reaction_follows_delivery(
    client: Any, room_id: str, delivery_event_id: str, reaction_event_id: str,
) -> bool:
    if client is None or not delivery_event_id or not reaction_event_id:
        return False
    room_path = f"/_matrix/client/v3/rooms/{quote(room_id, safe='')}"
    try:
        async with asyncio.timeout(10.0):
            context = await client.api.request(
                Method.GET, f"{room_path}/context/{quote(delivery_event_id, safe='')}",
                query_params={"limit": "0"},
            )
            token = context.get("end") if isinstance(context, dict) else None
            seen = set()
            for _ in range(10):
                if not isinstance(token, str) or not token or token in seen:
                    return False
                seen.add(token)
                page = await client.api.request(
                    Method.GET, f"{room_path}/messages",
                    query_params={"from": token, "dir": "f", "limit": "100"},
                )
                chunk = page.get("chunk") if isinstance(page, dict) else None
                if not isinstance(chunk, list):
                    return False
                if any(isinstance(event, dict) and event.get("event_id") == reaction_event_id
                       for event in chunk):
                    return True
                token = page.get("end")
    except Exception as exc:
        logger.debug("Matrix: could not establish reaction delivery order: %s", exc)
    return False


@dataclass(frozen=True)
class ReactionWatchClaim:
    turn_id: str
    token: str
    watch: dict[str, Any]


class ReactionWatchStore:
    """Watches for reactions to delivered replies, one row per visible reply event.

    A row keeps the bindings that a claim checks, the emoji filter, the expiry and an excerpt
    of the reply. The excerpt is kept because a streamed or encrypted reply cannot be read back
    after a restart: the original event contains the draft preview, and the final text is in
    an encrypted edit.
    """

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        from hermes_state import _secure_state_db_files

        self.path = path
        self.clock = clock
        self._active_claims: set[str] = set()
        path.parent.mkdir(parents=True, exist_ok=True)
        _secure_state_db_files(path, create_main=True)
        with closing(self._connect()) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS watches (
                    event_id TEXT PRIMARY KEY,
                    turn_id TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    requester TEXT NOT NULL,
                    source_json TEXT NOT NULL,
                    emoji_json TEXT NOT NULL,
                    text_content TEXT NOT NULL DEFAULT '',
                    delivery_event_id TEXT NOT NULL DEFAULT '',
                    expires_at REAL NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(watches)")}
            if "session_id" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")
            if "text_content" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN text_content TEXT NOT NULL DEFAULT ''")
            if "delivery_event_id" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN delivery_event_id TEXT NOT NULL DEFAULT ''")
            if "reply_excerpt_json" not in columns:
                db.execute("ALTER TABLE watches ADD COLUMN reply_excerpt_json TEXT NOT NULL DEFAULT ''")
            for column in ("claim_event_id", "claim_token"):
                if column not in columns:
                    db.execute(f"ALTER TABLE watches ADD COLUMN {column} TEXT NOT NULL DEFAULT ''")
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))

    def purge_expired(self) -> float | None:
        """Delete expired watches and return the next expiry time, if any watch remains."""
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            (next_expiry,) = db.execute("SELECT MIN(expires_at) FROM watches").fetchone()
        return next_expiry

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5)

    def arm(
        self,
        turn_id: str,
        event_ids: Iterable[str],
        *,
        profile: str,
        room_id: str,
        thread_id: str,
        session_key: str,
        session_id: str,
        requester: str,
        source: dict[str, Any],
        emoji_filter: tuple[str, ...],
        delivery_event_id: str,
        text_content: str = "",
        reply_excerpt: ReplyExcerpt | None = None,
        target_digests: dict[str, str] | None = None,
    ) -> None:
        rows = [
            (
                event_id,
                turn_id,
                profile,
                room_id,
                thread_id,
                session_key,
                session_id,
                requester,
                json.dumps({key: source[key] for key in _SOURCE_BINDING_KEYS if source.get(key)}),
                json.dumps(emoji_filter),
                text_content[:REPLY_EXCERPT_CHARS],
                delivery_event_id,
                json.dumps(replace(reply_excerpt, target_digest=(target_digests or {}).get(event_id, "")).to_json())
                if reply_excerpt is not None else "",
                self.clock() + WATCH_SECONDS,
            )
            for event_id in event_ids
            if event_id
        ]
        if not rows:
            return
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            db.executemany(
                """INSERT OR REPLACE INTO watches
                   (event_id, turn_id, profile, room_id, thread_id, session_key,
                    session_id, requester, source_json, emoji_json, text_content, delivery_event_id, reply_excerpt_json, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

    def claim(
        self,
        profile: str,
        room_id: str,
        target_event_id: str,
        sender: str,
        emoji: str,
        *,
        reaction_event_id: str,
        verified_delivery_event_id: str = "",
    ) -> ReactionWatchClaim | None:
        with closing(self._connect()) as db, db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            row = db.execute(
                """
                SELECT turn_id, thread_id, session_key, session_id, requester,
                       source_json, emoji_json, expires_at, text_content, delivery_event_id, reply_excerpt_json,
                       claim_event_id, claim_token
                FROM watches WHERE event_id = ? AND profile = ? AND room_id = ?
            """,
                (target_event_id, profile, room_id),
            ).fetchone()
            if row is None or row[4] != sender or not row[3]:
                return None
            if not row[9] or row[9] != verified_delivery_event_id:
                return None
            allowed = tuple(json.loads(row[6]))
            if allowed and emoji not in allowed:
                return None
            if not reaction_event_id or (row[11] and row[11] != reaction_event_id):
                return None
            if row[12] in self._active_claims:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE watches SET claim_event_id = ?, claim_token = ? WHERE turn_id = ?",
                       (reaction_event_id, token, row[0]))
        self._active_claims.add(token)
        return ReactionWatchClaim(row[0], token, {
            "profile": profile,
            "room_id": room_id,
            "thread_id": row[1],
            "session_key": row[2],
            "session_id": row[3],
            "requester": sender,
            "source": json.loads(row[5]),
            "emoji": emoji,
            "target_event_id": target_event_id,
            "text_content": row[8],
            "reply_excerpt": ReplyExcerpt.from_json(json.loads(row[10])) if row[10] else None,
        })

    def finish_claim(self, claim: ReactionWatchClaim, *, consumed: bool) -> None:
        self._active_claims.discard(claim.token)
        with closing(self._connect()) as db, db:
            if consumed:
                db.execute("DELETE FROM watches WHERE turn_id = ? AND claim_token = ?",
                           (claim.turn_id, claim.token))
                return
            db.execute("UPDATE watches SET claim_event_id = '', claim_token = '' "
                       "WHERE turn_id = ? AND claim_token = ?", (claim.turn_id, claim.token))

    def candidate(self, room_id: str, target_event_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as db, db:
            db.execute("DELETE FROM watches WHERE expires_at <= ?", (self.clock(),))
            row = db.execute(
                """
                SELECT profile, thread_id, session_key, session_id, requester,
                       source_json, expires_at, delivery_event_id
                FROM watches WHERE event_id = ? AND room_id = ?
            """,
                (target_event_id, room_id),
            ).fetchone()
        if row is None or row[6] <= self.clock() or not row[7]:
            return None
        return {
            "profile": row[0],
            "thread_id": row[1],
            "session_key": row[2],
            "session_id": row[3],
            "requester": row[4],
            "source": json.loads(row[5]),
            "delivery_event_id": row[7],
        }
