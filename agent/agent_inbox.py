"""One durable per-profile wake queue and turn scheduler — DRAFT, not wired into any path.

Today six producers wake an agent through six queues with six claim rules. This module is
the shape a single inbox would take. It generalises two existing modules rather than adding
a seventh design. From ``tools/bot_live_delivery.py`` it takes the record state machine and
operation semantics: idempotent admission that raises on a payload conflict, claiming the
oldest item by sequence, an immutable terminal receipt where an identical duplicate is a
no-op, a cancel that races the claim under one lock, and a shared waiter. From
``cron/delivery_queue.py`` it takes the storage idiom: a SQLite row claimed by a conditional
UPDATE, an owner stamp per claim, and dead-owner recovery that fences to ``unknown``. It adds
three things neither has: lanes, coalescing, and a recovery policy per source.

Migration map (current behaviour of each path; see the "Agent runtime primitives" developer-guide page):

| Path | Today: queue / storage / claim | Inbox mapping | Stays path-specific |
|---|---|---|---|
| message_agent DM | ``<profile>/bot_live_delivery/<id>.json`` + ``.sequence``; ``claim_pending_delivery`` under ``_FileLock``, owner-pinned ``_matches`` | source=dm, lane=agent, item_id=delivery_id, target_session=pinned session, ``complete``=``complete_delivery``, ``cancel_if_queued``=``cancel_queued_delivery``, ``await_outcome``=``await_delivery`` | lease/live_session pinning and compression-tip match; sender intent/DM files; CLI subprocess lane |
| Cross-machine relay | ``bot_relay/{outbox,claimed,replies}``; ``os.replace`` outbox→claimed; one re-offer; first reply wins | source=relay, lane=agent, item_id=envelope id, ``expires_at``=envelope TTL, recover → queued once (``attempts`` cap 2) | roster refuse-on-offline; Desktop drain RPC; reply file transport across machines |
| Cron → Bot Chat | ``cron/deliveries.db`` (``claim_next`` conditional UPDATE, owner fence) + ``cron/bot_chat_pending/*.json`` (``_drain`` claims before the turn) | source=cron, lane=background, item_id=execution_id; recover → ``unknown``, never replayed (``recover_abandoned``) | platform send for non-Bot-Chat targets; ``transferred`` handoff; tombstone retention |
| Process completions | in-memory ``ProcessRegistry.completion_queue``; pop = claim; requeue when not owned | source=process, lane=background, coalesce_key=route key (``_COMPLETION_BATCH_KEY_FIELDS``); recover → queued | ``_completion_consumed`` (output already read); watch/heartbeat rate limits; formatting |
| Async delegation | ``async_delegations`` in state.db: ``delivery_state`` + ``delivery_claim`` lease; orphan sweep re-offers | source=delegation, lane=background, item_id=delegation_id, coalesce_key=``_ASYNC_GROUP_KEY_FIELDS`` route; recover → queued once | task lifecycle (running/stalling); owner-proving restore (#64484); failure surfacing |
| Kanban notifier | ``kanban_notify_subs.last_event_id`` cursor CAS; rewind on in-process failure | source=kanban, lane=background, item_id=(sub, event range), coalesce_key=sub; recover → queued | event log and cursor stay in the board DB; ping (non-wake) delivery; sub GC |

Gaps this closes:
- Process completion events are not durable (path 4); here they are rows.
- A ``claimed`` DM mailbox or ``bot_chat_pending`` record is never recovered; here ``recover``
  applies a policy and ``Scheduler.stale_claims`` surfaces the record.
- Kanban loses the claimed range if it crashes before send. Here the item stays ``claimed``
  until ``recover`` requeues it.
- The CLI marks a delegation delivered before the turn runs. Here ``complete`` follows the turn.
- The ``bot_relay`` turn flock is a no-op on Windows and only covers delivery turns. Here
  ``claim_next_turn`` uses a SQLite ``BEGIN IMMEDIATE`` on every platform. Session turn
  exclusivity still belongs to ``session_turn_leases``; this module does not replace that.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent.deadline import MAX_SAFE_TIMEOUT_S, clamp_timeout, poll_until
from gateway.hosted_rooms_common import compact_json
from hermes_cli.sqlite_util import open_db, transaction


class Source(StrEnum):
    USER = "user"
    DM = "dm"
    RELAY = "relay"
    CRON = "cron"
    PROCESS = "process"
    DELEGATION = "delegation"
    KANBAN = "kanban"


class Lane(StrEnum):
    INTERACTIVE = "interactive"
    AGENT = "agent"
    BACKGROUND = "background"


class Status(StrEnum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    SETTLED = "settled"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"
    EXPIRED = "expired"


TERMINAL = frozenset({Status.SETTLED, Status.FAILED, Status.CANCELLED, Status.UNKNOWN, Status.EXPIRED})
_LANE_RANK = {Lane.INTERACTIVE: 0, Lane.AGENT: 1, Lane.BACKGROUND: 2}
DEFAULT_LANE = {
    Source.USER: Lane.INTERACTIVE,
    Source.DM: Lane.AGENT, Source.RELAY: Lane.AGENT,
    Source.CRON: Lane.BACKGROUND, Source.PROCESS: Lane.BACKGROUND,
    Source.DELEGATION: Lane.BACKGROUND, Source.KANBAN: Lane.BACKGROUND,
}


@dataclass(frozen=True)
class RecoverPolicy:
    """What ``recover`` does with an item whose claim owner died.

    ``replay=False`` fences the item to ``unknown``: the turn may have run, so another run
    could duplicate it (cron ``recover_abandoned``, the DM mailbox "Do not resend").
    ``replay=True`` requeues it while ``attempts < max_attempts``, then marks it ``failed``.
    ``max_attempts=None`` means no cap.
    """

    replay: bool
    max_attempts: int | None = None


RECOVER_POLICY = {
    Source.USER: RecoverPolicy(replay=False),
    Source.DM: RecoverPolicy(replay=False),
    Source.CRON: RecoverPolicy(replay=False),
    Source.RELAY: RecoverPolicy(replay=True, max_attempts=2),  # bot_relay: re-offered once
    Source.DELEGATION: RecoverPolicy(replay=True, max_attempts=2),
    Source.PROCESS: RecoverPolicy(replay=True),
    Source.KANBAN: RecoverPolicy(replay=True),  # kanban rewinds its cursor and retries
}
AWAIT_POLL_SECONDS = 0.25


class InboxError(Exception):
    """Base class for agent inbox failures."""


class InboxConflictError(InboxError):
    """An item id or terminal receipt is already bound to a different payload/outcome."""


class InboxClaimError(InboxError):
    """A completion or release does not hold the item's current claim."""


class InboxNotFoundError(InboxError):
    """No item has the given id."""


@dataclass(frozen=True)
class InboxItem:
    item_id: str
    target_session: str | None
    source: Source
    lane: Lane
    coalesce_key: str | None
    sequence: int
    author: dict[str, Any] | None
    payload: dict[str, Any]
    payload_digest: str
    status: Status
    claim_id: str | None
    claim_owner: str | None
    claim_fingerprint: str | None
    claimed_at: float | None
    attempts: int
    expires_at: float | None
    outcome: dict[str, Any] | None
    created_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> InboxItem:
        return cls(
            item_id=row["item_id"], target_session=row["target_session"], source=Source(row["source"]),
            lane=Lane(row["lane"]), coalesce_key=row["coalesce_key"], sequence=row["sequence"],
            author=json.loads(row["author_json"]) if row["author_json"] else None,
            payload=json.loads(row["payload_json"]), payload_digest=row["payload_digest"],
            status=Status(row["status"]), claim_id=row["claim_id"], claim_owner=row["claim_owner"],
            claim_fingerprint=row["claim_fingerprint"], claimed_at=row["claimed_at"],
            attempts=row["attempts"], expires_at=row["expires_at"],
            outcome=json.loads(row["outcome_json"]) if row["outcome_json"] else None,
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class TurnBatch:
    """Items one turn consumes together. They share a claim, a lane, and a target session."""

    claim_id: str
    owner: str
    lane: Lane
    target_session: str | None
    items: tuple[InboxItem, ...]

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(item.item_id for item in self.items)


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS inbox_items (
    item_id TEXT PRIMARY KEY,
    target_session TEXT,
    source TEXT NOT NULL CHECK(source IN ({", ".join(repr(s.value) for s in Source)})),
    lane TEXT NOT NULL CHECK(lane IN ({", ".join(repr(lane.value) for lane in Lane)})),
    coalesce_key TEXT,
    sequence INTEGER NOT NULL UNIQUE,
    author_json TEXT,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ({", ".join(repr(s.value) for s in Status)})),
    claim_id TEXT,
    claim_owner TEXT,
    claim_fingerprint TEXT,
    claimed_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    expires_at REAL,
    outcome_json TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS inbox_items_queued ON inbox_items(status, lane, sequence);
CREATE INDEX IF NOT EXISTS inbox_items_claim ON inbox_items(claim_id);
CREATE TABLE IF NOT EXISTS inbox_sequence (id INTEGER PRIMARY KEY CHECK(id = 1), next INTEGER NOT NULL);
INSERT OR IGNORE INTO inbox_sequence (id, next) VALUES (1, 1);
"""
_LANE_ORDER_SQL = "CASE lane " + " ".join(f"WHEN '{lane.value}' THEN {rank}" for lane, rank in _LANE_RANK.items()) + " END"
_LIVE_SQL = "(expires_at IS NULL OR expires_at > ?)"


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)


def _canonical(value: Any) -> str:
    return compact_json(value, ensure_ascii=False)


def _digest(*parts: Any) -> str:
    return hashlib.sha256(_canonical(list(parts)).encode()).hexdigest()


class AgentInbox:
    """A profile's inbox at an explicit SQLite path. Nothing is resolved from HERMES_HOME."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)

    @contextmanager
    def _txn(self) -> Iterator[sqlite3.Connection]:
        conn = open_db(self.db_path, db_label="agent inbox", synchronous_full=True, initialize=_initialize_schema)
        with transaction(conn, immediate=True) as txn:
            yield txn

    @staticmethod
    def _get(conn: sqlite3.Connection, item_id: str) -> InboxItem | None:
        row = conn.execute("SELECT * FROM inbox_items WHERE item_id = ?", (item_id,)).fetchone()
        return None if row is None else InboxItem.from_row(row)

    @classmethod
    def _require(cls, conn: sqlite3.Connection, item_id: str) -> InboxItem:
        item = cls._get(conn, item_id)
        if item is None:
            raise InboxNotFoundError(item_id)
        return item

    def get(self, item_id: str) -> InboxItem | None:
        with self._txn() as conn:
            return self._get(conn, item_id)

    def enqueue(self, item_id: str, source: Source, payload: dict[str, Any], *, now: float,
                lane: Lane | None = None, target_session: str | None = None,
                coalesce_key: str | None = None, author: dict[str, Any] | None = None,
                expires_at: float | None = None) -> InboxItem:
        """Admit an item, idempotent by ``item_id``. The existing record is returned in any state.

        Raises ``InboxConflictError`` when the id already holds a different digest. The digest
        covers the routing fields as well as the payload: a retry must be byte-identical.
        """
        source, lane = Source(source), Lane(lane or DEFAULT_LANE[Source(source)])
        digest = _digest(source.value, lane.value, target_session, coalesce_key, author, payload)
        with self._txn() as conn:
            existing = self._get(conn, item_id)
            if existing is not None:
                if existing.payload_digest != digest:
                    raise InboxConflictError(f"inbox item {item_id!r} already holds a different payload")
                return existing
            sequence = conn.execute("SELECT next FROM inbox_sequence WHERE id = 1").fetchone()[0]
            conn.execute("UPDATE inbox_sequence SET next = next + 1 WHERE id = 1")
            conn.execute(
                "INSERT INTO inbox_items (item_id, target_session, source, lane, coalesce_key, sequence,"
                " author_json, payload_json, payload_digest, status, expires_at, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)",
                (item_id, target_session, source.value, lane.value, coalesce_key, sequence,
                 _canonical(author) if author is not None else None, _canonical(payload), digest,
                 expires_at, now),
            )
            return self._require(conn, item_id)

    def claim_next_turn(self, owner: str, now: float, *, fingerprint: str = "") -> TurnBatch | None:
        """Claim the next turn's worth of items, or None when nothing unexpired is queued.

        The head item is the highest-priority lane first, then the oldest sequence within it.
        If the head has a coalesce_key, every queued item in the same lane and target session
        with that key joins the batch, so a burst becomes one turn. The caller must already
        hold its turn admission (session lease or in-process guard). This is the same contract
        as ``claim_pending_delivery``.
        """
        with self._txn() as conn:
            head = conn.execute(
                f"SELECT * FROM inbox_items WHERE status = 'queued' AND {_LIVE_SQL}"
                f" ORDER BY {_LANE_ORDER_SQL}, sequence LIMIT 1", (now,)).fetchone()
            if head is None:
                return None
            if head["coalesce_key"] is None:
                ids = [head["item_id"]]
            else:
                ids = [row[0] for row in conn.execute(
                    f"SELECT item_id FROM inbox_items WHERE status = 'queued' AND {_LIVE_SQL}"
                    " AND lane = ? AND coalesce_key = ? AND target_session IS ? ORDER BY sequence",
                    (now, head["lane"], head["coalesce_key"], head["target_session"]))]
            claim_id = uuid.uuid4().hex
            conn.executemany(
                "UPDATE inbox_items SET status = 'claimed', claim_id = ?, claim_owner = ?,"
                " claim_fingerprint = ?, claimed_at = ?, attempts = attempts + 1"
                " WHERE item_id = ? AND status = 'queued'",
                [(claim_id, owner, fingerprint, now, item_id) for item_id in ids])
            items = tuple(InboxItem.from_row(row) for row in conn.execute(
                "SELECT * FROM inbox_items WHERE claim_id = ? ORDER BY sequence", (claim_id,)))
        return TurnBatch(claim_id=claim_id, owner=owner, lane=Lane(head["lane"]),
                         target_session=head["target_session"], items=items)

    def complete(self, batch: TurnBatch, outcome: dict[str, Any], *, status: Status = Status.SETTLED,
                 now: float) -> tuple[InboxItem, ...]:
        """Write one immutable terminal receipt to every item in ``batch``.

        Repeating a completion with the same status and outcome is a no-op (``completed_at``
        is ignored in the comparison and the first receipt's timestamp is kept). A different
        receipt raises ``InboxConflictError``. An item whose claim this batch no longer holds
        raises ``InboxClaimError``: it was released, recovered, or never claimed.
        """
        status = Status(status)
        if status not in (Status.SETTLED, Status.FAILED):
            raise ValueError("complete() writes settled or failed; use cancel/recover/expire for the rest")
        body = {key: value for key, value in outcome.items() if key != "completed_at"}
        outcome_json = _canonical({**body, "completed_at": now})
        with self._txn() as conn:
            for item in (self._require(conn, item_id) for item_id in batch.item_ids):
                if item.claim_id != batch.claim_id:
                    raise InboxClaimError(f"inbox item {item.item_id!r} is not held by claim {batch.claim_id}")
                if item.status in TERMINAL:
                    stored = {k: v for k, v in (item.outcome or {}).items() if k != "completed_at"}
                    if item.status != status or _canonical(stored) != _canonical(body):
                        raise InboxConflictError(f"inbox item {item.item_id!r} already has a different receipt")
            conn.execute(
                f"UPDATE inbox_items SET status = ?, outcome_json = ? WHERE claim_id = ? AND status = 'claimed'"
                f" AND item_id IN ({','.join('?' * len(batch.item_ids))})",
                (status.value, outcome_json, batch.claim_id, *batch.item_ids))
            return tuple(self._require(conn, item_id) for item_id in batch.item_ids)

    def release(self, batch: TurnBatch) -> int:
        """Put a batch back as queued because the target was busy (defer). The attempt is refunded.

        The turn never started, so this is not a retry. Items keep their sequence and stay at the
        head of their lane.
        """
        with self._txn() as conn:
            return conn.execute(
                "UPDATE inbox_items SET status = 'queued', claim_id = NULL, claim_owner = NULL,"
                " claim_fingerprint = NULL, claimed_at = NULL, attempts = MAX(attempts - 1, 0)"
                " WHERE claim_id = ? AND status = 'claimed'", (batch.claim_id,)).rowcount

    def cancel_if_queued(self, item_id: str, *, reason: str, now: float) -> InboxItem | None:
        """Cancel ``item_id`` only while it is still queued, and return its current record.

        The check and the write share one IMMEDIATE transaction with ``claim_next_turn``.
        A claim that won returns unchanged, so exactly one side wins.
        """
        with self._txn() as conn:
            conn.execute("UPDATE inbox_items SET status = 'cancelled', outcome_json = ?"
                         " WHERE item_id = ? AND status = 'queued'",
                         (_canonical({"reason": reason, "completed_at": now}), item_id))
            return self._get(conn, item_id)

    def recover(self, dead_owner: Callable[[str, str], bool], now: float) -> list[InboxItem]:
        """Apply ``RECOVER_POLICY`` to claimed items for which ``dead_owner(owner, fingerprint)`` is True.

        The predicate is the caller's: process-liveness belongs to the host layer (for example
        ``cron.executions._owner_is_live``). It must return False when it cannot prove death.
        """
        changed: list[InboxItem] = []
        with self._txn() as conn:
            for row in conn.execute("SELECT * FROM inbox_items WHERE status = 'claimed'").fetchall():
                item = InboxItem.from_row(row)
                if not dead_owner(item.claim_owner or "", item.claim_fingerprint or ""):
                    continue
                policy = RECOVER_POLICY[item.source]
                capped = policy.max_attempts is not None and item.attempts >= policy.max_attempts
                if policy.replay and not capped:
                    conn.execute(
                        "UPDATE inbox_items SET status = 'queued', claim_id = NULL, claim_owner = NULL,"
                        " claim_fingerprint = NULL, claimed_at = NULL WHERE item_id = ? AND claim_id = ?",
                        (item.item_id, item.claim_id))
                else:
                    status, reason = ((Status.FAILED, "attempts_exhausted") if policy.replay
                                      else (Status.UNKNOWN, "owner_died_not_replayed"))
                    conn.execute("UPDATE inbox_items SET status = ?, outcome_json = ? WHERE item_id = ? AND claim_id = ?",
                                 (status.value, _canonical({"reason": reason, "completed_at": now}),
                                  item.item_id, item.claim_id))
                changed.append(self._require(conn, item.item_id))
        return changed

    def expire(self, now: float) -> int:
        """Give queued items past ``expires_at`` a terminal ``expired`` receipt so their waiters resolve."""
        with self._txn() as conn:
            return conn.execute(
                "UPDATE inbox_items SET status = 'expired', outcome_json = ?"
                " WHERE status = 'queued' AND expires_at IS NOT NULL AND expires_at <= ?",
                (_canonical({"reason": "queued_expired", "completed_at": now}), now)).rowcount

    def claims_older_than(self, cutoff: float) -> list[InboxItem]:
        with self._txn() as conn:
            return [InboxItem.from_row(row) for row in conn.execute(
                "SELECT * FROM inbox_items WHERE status = 'claimed' AND claimed_at <= ? ORDER BY claimed_at",
                (cutoff,))]

    def await_outcome(self, item_id: str, timeout: float | None, *,
                      interval: float = AWAIT_POLL_SECONDS) -> InboxItem | None:
        """Wait for a terminal receipt and return it, or None on timeout (``None`` = unbounded).

        This would replace the polling waiters in ``bot_live_delivery.await_delivery``,
        ``delivery_queue.enqueue_and_wait`` and ``bot_mode_dm._wait_reply_main``.
        """
        def _terminal() -> InboxItem | None:
            with self._txn() as conn:
                item = self._require(conn, item_id)
            return item if item.status in TERMINAL else None

        budget = MAX_SAFE_TIMEOUT_S if timeout is None else clamp_timeout(timeout) or 0.0
        return poll_until(_terminal, budget, interval)


class Scheduler:
    """Turn-ordering policy over one inbox.

    ``should_preempt`` encodes today's rule. A human typing into the live session interrupts a background-
    originated turn (TUI ``busy_input_mode`` and gateway ``_busy_input_mode`` both default to
    ``interrupt``). A teammate DM or relay message never interrupts: it waits for idle.
    Background completions and wakes never interrupt either.
    """

    def __init__(self, inbox: AgentInbox) -> None:
        self.inbox = inbox

    @staticmethod
    def should_preempt(running_lane: Lane, incoming_lane: Lane) -> bool:
        return Lane(incoming_lane) is Lane.INTERACTIVE and Lane(running_lane) is Lane.BACKGROUND

    def stale_claims(self, now: float, max_age: float) -> list[InboxItem]:
        """List claims older than ``max_age`` seconds so the TUI and gateway can show them.

        Today a stuck ``claimed`` DM mailbox or ``bot_chat_pending`` record is never recovered
        or surfaced. This only reports
        them. Recovery stays an explicit ``AgentInbox.recover`` decision.
        """
        return self.inbox.claims_older_than(now - max_age)
