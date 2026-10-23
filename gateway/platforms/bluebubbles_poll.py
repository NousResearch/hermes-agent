"""Bounded REST recovery for missed BlueBubbles webhooks."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

from utils import atomic_json_write

if TYPE_CHECKING:
    from gateway.platforms.bluebubbles import BlueBubblesAdapter

logger = logging.getLogger(__name__)
DEFAULT_POLL_INTERVAL = 15
LOOKBACK_MS = 60_000
CATCHUP_MS = 600_000
PAGE_LIMIT = 100
MAX_PAGES = 10
MAX_ATTACHMENT_ATTEMPTS = 3
_to_thread = asyncio.to_thread
_sleep = asyncio.sleep


def _now_ms() -> int:
    return int(time.time() * 1000)


def _timestamp(value) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("invalid BlueBubbles timestamp")
    return int(value)


def _created_ms(record: dict) -> int | None:
    try:
        return _timestamp(record.get("dateCreated"))
    except (ValueError, OverflowError):
        logger.warning("[bluebubbles] skipping poll record with invalid dateCreated")
        return None


@dataclass
class BlueBubblesPollState:
    cursor_ms: int
    floor_ms: int
    last_polled_ms: int
    # Persist the same edit-aware keys as the adapter, including equal-timestamp/late arrivals.
    seen: dict[str, int] = field(default_factory=dict)
    scan_after_ms: int = 0
    scan_before_ms: int = 0
    offset: int = 0
    retry_key: str = ""
    retry_attempts: int = 0

    @classmethod
    def load(cls, path: Path) -> BlueBubblesPollState | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            state = cls(**data)
            for name in ("cursor_ms", "floor_ms", "last_polled_ms", "scan_after_ms", "scan_before_ms", "offset", "retry_attempts"):
                setattr(state, name, _timestamp(getattr(state, name)))
            if not isinstance(state.retry_key, str):
                raise ValueError("invalid retry key")
            if not isinstance(state.seen, dict) or any(not isinstance(key, str) for key in state.seen):
                raise ValueError("invalid seen keys")
            state.seen = {key: _timestamp(date) for key, date in state.seen.items()}
            return state
        except FileNotFoundError:
            return None
        except (OSError, ValueError, TypeError, OverflowError):
            logger.warning("[bluebubbles] invalid poll state; initializing without historical replay")
            return None

    def save(self, path: Path) -> None:
        atomic_json_write(path, asdict(self))


class BlueBubblesPoller:
    def __init__(self, adapter: BlueBubblesAdapter, *, interval: float, state_path: Path):
        self.adapter = adapter
        self.interval = interval
        self.state_path = state_path
        self._state_lock = asyncio.Lock()
        self._started_ms = _now_ms()
        self.state = BlueBubblesPollState.load(state_path)
        if self.state and self.state.last_polled_ms < _now_ms() - CATCHUP_MS:
            logger.warning("[bluebubbles] poll state too old, skipping catch-up")
            self.state = None
        if self.state:
            for key in self.state.seen:
                adapter._dedup.is_duplicate(key)

    async def run(self) -> None:
        delay = self.interval
        failures = 0
        while self.adapter._running:
            await _sleep(delay)
            if not self.adapter._running:
                return
            try:
                await self._tick()
            except Exception as exc:  # health: allow BLE001 -- retry transport/ingest failures without logging password-bearing exceptions
                failures = min(failures + 1, 6)
                delay = min(self.interval * 2 ** failures, 60)
                # HTTP exception strings can contain the password-bearing request URL.
                logger.warning("[bluebubbles] REST poll failed (%s); retrying", type(exc).__name__)
            else:
                failures = 0
                delay = self.interval

    async def _query(self, **body) -> list[dict]:
        response = await self.adapter._api_post("/api/v1/message/query", body)
        records = response.get("data")
        if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
            raise ValueError("invalid BlueBubbles query response")
        return records

    async def _write_state(self, state: BlueBubblesPollState) -> None:
        # Finish the immutable snapshot's write before disconnect can start a new poller.
        writer = asyncio.create_task(_to_thread(state.save, self.state_path))
        try:
            await asyncio.shield(writer)
        except asyncio.CancelledError:
            await writer
            self.state = state
            raise
        self.state = state

    async def _persist(self, state: BlueBubblesPollState, *, prune: bool = False) -> None:
        async with self._state_lock:
            # The poll snapshot may predate a webhook handoff during query/ingest.
            if self.state is not None:
                state.seen = {**self.state.seen, **state.seen}
            if prune:
                cutoff = state.cursor_ms - LOOKBACK_MS
                state.seen = {key: date for key, date in state.seen.items() if date >= cutoff}
            await self._write_state(state)

    async def record_webhook(self, record: dict) -> None:
        key = self.adapter._message_key(record)
        if not key:
            return
        # Once intake succeeds, cancellation must wait even for a queued state write.
        writer = asyncio.create_task(self._record_webhook(key, record))
        try:
            await asyncio.shield(writer)
        except asyncio.CancelledError:
            await writer
            raise

    async def _record_webhook(self, key: str, record: dict) -> None:
        async with self._state_lock:
            state = self.state
            if state is None:
                # A webhook can arrive before the first seed query. Exclude pre-start
                # history, but leave all subsequent missed messages eligible for recovery.
                state = BlueBubblesPollState(self._started_ms, self._started_ms, self._started_ms)
            if key in state.seen:
                return
            try:
                date = _timestamp(record.get("dateCreated"))
            except (ValueError, OverflowError):
                date = _now_ms()
            await self._write_state(replace(state, seen={**state.seen, key: date}))

    async def _seed(self) -> None:
        records = await self._query(
            **{"limit": 1, "offset": 0, "sort": "DESC", "before": self._started_ms,
               "with": ["chat", "handle"]})
        date = _created_ms(records[0]) if records else None
        cursor = self._started_ms if date is None else date
        seen = {}
        if records and (key := self.adapter._message_key(records[0])):
            seen[key] = cursor
        # The floor excludes the ENTIRE initial overlap, not only the latest GUID.
        async with self._state_lock:
            if self.state is None:  # A webhook may have initialized state during the query.
                await self._write_state(BlueBubblesPollState(cursor, cursor, _now_ms(), seen))

    def _retry_attachment(self, state: BlueBubblesPollState, key: str) -> bool:
        state.retry_attempts = state.retry_attempts + 1 if state.retry_key == key else 1
        state.retry_key = key
        if state.retry_attempts < MAX_ATTACHMENT_ATTEMPTS:
            logger.warning("[bluebubbles] attachment unavailable; retrying poll page")
            return True
        # A deleted/unavailable asset must not permanently block every later inbound message.
        logger.warning("[bluebubbles] skipping unavailable attachment after %s poll attempts", state.retry_attempts)
        return False

    async def _consume(self, state: BlueBubblesPollState, records: list[dict]) -> int:
        from gateway.platforms.bluebubbles import _Ingest

        for index, record in enumerate(records):
            date = _created_ms(record)
            if date is None:
                continue
            if date <= state.floor_ms:
                continue
            key = self.adapter._message_key(record)
            if key and key in state.seen:
                state.cursor_ms = max(state.cursor_ms, date)
                continue
            result = await self.adapter._ingest_message(record, record, origin="poll")
            if result is _Ingest.RETRYABLE and self._retry_attachment(state, key):
                return index
            state.retry_key, state.retry_attempts = "", 0
            state.cursor_ms = max(state.cursor_ms, date)
            if key and result is not _Ingest.MISSING_FIELDS:
                state.seen[key] = date
        return len(records)

    async def _tick(self) -> None:
        if self.state is None:
            await self._seed()
            return
        state = replace(self.state, seen=dict(self.state.seen))
        if not state.offset:
            state.scan_after_ms = max(0, state.cursor_ms - LOOKBACK_MS)
            state.scan_before_ms = _now_ms()
        for _ in range(MAX_PAGES):
            # BlueBubbles' after/before bounds are inclusive Unix milliseconds. Freeze both
            # across capped ticks so a dense overlap (including timestamp ties) makes progress.
            records = await self._query(**{
                "limit": PAGE_LIMIT, "offset": state.offset, "sort": "ASC",
                "after": state.scan_after_ms, "before": state.scan_before_ms,
                "with": ["chat", "chat.participants", "handle", "attachment"],
            })
            consumed = await self._consume(state, records)
            retry = consumed < len(records)
            complete = not retry and len(records) < PAGE_LIMIT
            state.offset = 0 if complete else state.offset + consumed
            state.last_polled_ms = _now_ms()
            # Checkpoint each handed-off page: a failure on a later page must not lose progress.
            await self._persist(state, prune=complete)
            if complete or retry:
                return
            state = replace(state, seen=dict(state.seen))
