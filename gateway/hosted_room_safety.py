"""Room-id reservations, quarantine and shared retention guards for hosted Group Chat stores.

Installed on every room store, the legacy ``shared-state.db`` and each canonical ``state.db``:

* ``hosted_room_quarantine`` keeps a room read-only, and out of pruning, once its history records a
  takeover no exclusive authority proved (an ``authority.claimed`` promoted from a replica, or an
  unmarked ``authority.lost`` or ``authority.transition``), or once one room id names both a local
  room and a stored copy.
* ``hosted_room_verified_transitions`` marks the authority changes exclusive-authority recovery
  verified. A mark admits exactly one authority-change event, in its own transaction (see
  ``mark_verified_transition``); a mark set aside with its event's divergent branch is archived in
  ``hosted_room_branch_transitions``, never deleted.
* ``hosted_room_id_reservations`` binds every room id to the kind that first used it, even after its
  payload is pruned, so a copied room is never recreated as a local one.
* ``hosted_room_event_budget`` counts room and replica events against one byte budget.

A quarantined room never becomes writable again. The one change it accepts is an operator-confirmed
Disband on this gateway (``hosted_room_quarantine_disbands``): a tombstone that appends nothing to its
history and leaves its authority untouched. Its rows and events can't be deleted, so no pruning, not even
an older process's, removes that evidence.

The guards are SQLite triggers, so they also hold for writers that predate them, such as an older
gateway process still sharing the store.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from typing import Any, Mapping

from gateway.hosted_rooms_common import compact_json, identifier, table_columns, table_exists

_QUARANTINE_SCHEMA_COLUMNS = frozenset({"room_id", "reason", "detected_at"})

_ROOM_RESERVATION_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "owner_kind",
    "reserved_at",
})

_REPLICA_SCHEMA_COLUMNS = frozenset({
    "room_id",
    "name",
    "members_json",
    "authority_gateway_id",
    "authority_epoch",
    "last_seq",
    "latest_seq",
    "event_bytes",
    "created_at",
    "updated_at",
    "disbanded_at",
    "quarantined_at",
    "quarantine_reason",
})

_REPLICA_EVENT_SCHEMA_COLUMNS = frozenset({
    "room_id", "seq", "event_id", "kind", "actor_json",
    "authority_epoch", "payload_json", "created_at",
})

_EVENT_BUDGET_SCHEMA_COLUMNS = frozenset({"singleton", "event_bytes"})

_QUARANTINE_DISBAND_SCHEMA_COLUMNS = frozenset({"room_id", "confirmed_at"})

_VERIFIED_TRANSITION_SCHEMA_COLUMNS = frozenset({
    "room_id", "from_epoch", "to_epoch", "successor_gateway_id", "proof_kind", "proof_digest", "created_at",
})
_TRANSITION_USE_SCHEMA_COLUMNS = frozenset({"room_id", "to_epoch", "seq", "event_id"})
_BRANCH_TRANSITION_SCHEMA_COLUMNS = frozenset({
    "room_id", "branch_id", "from_epoch", "to_epoch", "successor_gateway_id", "proof_kind", "proof_digest",
    "marked_at", "seq", "event_id", "archived_at",
})

PROOF_KINDS = frozenset({"attested", "certified", "evidence", "handover"})
# A handover proof is ``{statement, signature}``: the old authority's statement, signed with its room
# identity key, that it hands this room to ``successor`` after its event ``last_seq``.
HANDOVER_STATEMENT_FIELDS = frozenset({"room_id", "from_epoch", "to_epoch", "successor", "last_seq", "last_hash"})
# An evidence proof is ``{statement, signature}`` too, signed by the successor: it had no contact with
# the old authority since ``silent_since``, for ``silent_for_s`` seconds, and continues after ``last_seq``.
EVIDENCE_STATEMENT_FIELDS = HANDOVER_STATEMENT_FIELDS | {"silent_since", "silent_for_s"}

_ROOM_SAFETY_TRIGGERS = frozenset({
    "trg_hosted_rooms_reject_reserved_insert",
    "trg_hosted_rooms_reserve_insert",
    "trg_hosted_replicas_reject_reserved_insert",
    "trg_hosted_replicas_reserve_insert",
    "trg_hosted_events_reject_quarantined_insert",
    "trg_hosted_events_quarantine_unsafe_lineage",
    "trg_hosted_replica_events_verified_lineage",
    "trg_hosted_events_shared_budget",
    "trg_hosted_replica_events_shared_budget",
    "trg_hosted_events_budget_account_insert",
    "trg_hosted_events_budget_account_delete",
    "trg_hosted_replica_events_budget_account_insert",
    "trg_hosted_replica_events_budget_account_delete",
    "trg_hosted_rooms_quarantined_tombstone",
    "trg_hosted_rooms_keep_quarantined",
    "trg_hosted_events_keep_quarantined",
    "trg_branch_transitions_keep",
    "trg_branch_transitions_unchanged",
})
# Triggers whose definition changed with verified transitions; a store holding an older body is migrated.
_REVISED_TRIGGERS = {
    "trg_hosted_events_quarantine_unsafe_lineage": "hosted_room_verified_transitions",
    "trg_hosted_events_shared_budget": "authority.transition",
}


def transition_proof_digest(proof: Mapping[str, Any]) -> str:
    """The digest a verified-transition mark binds: sha256 of the proof's canonical JSON."""
    return hashlib.sha256(compact_json(proof).encode("utf-8")).hexdigest()


def _transition_match_sql(events: str) -> str:
    """Whether a mark verifies ``NEW``, an authority-change event being written to ``events``.

    The mark names this room, the event's epoch and successor, proof kind and digest, and the epoch of
    the event just before it. A mark admits one event identity: unused, or already used by this very
    event (history moved between this store's room and copy tables keeps its verification).
    """
    return f"""EXISTS (
        SELECT 1 FROM hosted_room_verified_transitions AS mark
         WHERE mark.room_id=NEW.room_id AND mark.to_epoch=NEW.authority_epoch
           AND mark.proof_kind IS json_extract(NEW.payload_json, '$.proof_kind')
           AND mark.proof_digest IS json_extract(NEW.payload_json, '$.proof_digest')
           AND mark.successor_gateway_id IS json_extract(NEW.payload_json, CASE NEW.kind
               WHEN 'authority.transition' THEN '$.successor_gateway_id' ELSE '$.authority_gateway_id' END)
           AND (NEW.kind!='authority.transition' OR (
               json_extract(NEW.payload_json, '$.from_epoch') IS mark.from_epoch
               AND json_extract(NEW.payload_json, '$.to_epoch') IS mark.to_epoch))
           AND (NEW.kind!='authority.lost' OR json_extract(NEW.payload_json, '$.authority_epoch') IS mark.to_epoch)
           AND mark.from_epoch IS (SELECT prior.authority_epoch FROM {events} AS prior
                                    WHERE prior.room_id=NEW.room_id AND prior.seq=NEW.seq-1)
           AND NOT EXISTS (
               SELECT 1 FROM hosted_room_verified_transition_uses AS used
                WHERE used.room_id=mark.room_id AND used.to_epoch=mark.to_epoch
                  AND (used.seq IS NOT NEW.seq OR used.event_id IS NOT NEW.event_id)))"""


_USE_TRANSITION_SQL = """INSERT OR IGNORE INTO hosted_room_verified_transition_uses (room_id, to_epoch, seq, event_id)
               SELECT NEW.room_id, NEW.authority_epoch, NEW.seq, NEW.event_id WHERE {match};"""


def mark_verified_transition(
    conn: sqlite3.Connection, *, room_id: str, from_epoch: int, to_epoch: int, successor_gateway_id: str,
    proof_kind: str, proof_digest: str,
) -> None:
    """Mark one verified authority change, inside the transaction that makes it.

    The caller has verified the proof whose digest this records:

    - ``attested``: the room owner, or the owner of a consented successor the owner designated,
      explicitly continued the group on this machine; or, after a split, the rule's choice signed by
      the host it keeps (``decided_by: "rule"``);
    - ``certified``: a certificate of signed promises from a majority of the room's voters, each
      given only once its lease to the old authority had expired;
    - ``handover``: the old authority's own signed statement handing the room to this successor
      directly after its last event (``HANDOVER_STATEMENT_FIELDS``);
    - ``evidence``: with exactly two voters, the successor's own signed statement that it had no
      contact with the old authority for the careful window (``EVIDENCE_STATEMENT_FIELDS``).

    The mark lets the lineage triggers accept exactly one authority-change event (``authority.transition``,
    or a verified ``authority.lost``): in this room, at ``to_epoch`` directly after an event at
    ``from_epoch``, naming this successor, proof kind and digest. Written in any other way, the change
    is quarantined like any unproven takeover. ``to_epoch`` is any later epoch: an attempt that fenced
    an epoch and did not finish retries at a higher one, so an epoch may never have had an authority.

    The event must be written in this same transaction. A mark no event used fails the commit (a
    deferred foreign key to the use row), so a mark can't outlive its transaction, be reused for a
    later event, or count for another room or epoch. There is one mark per room and epoch.
    """
    from gateway.hosted_rooms import VerifiedTransitionError

    def text(value: Any, label: str) -> str:
        return identifier(value, label=label, error=VerifiedTransitionError, max_chars=128)

    room_id, successor_gateway_id = text(room_id, "room_id"), text(successor_gateway_id, "successor_gateway_id")
    for value in (from_epoch, to_epoch):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value < 2**63:
            raise VerifiedTransitionError("transition epochs must be positive integers")
    if to_epoch <= from_epoch:
        raise VerifiedTransitionError("a verified transition moves authority to a later epoch")
    if proof_kind not in PROOF_KINDS:
        raise VerifiedTransitionError("proof_kind must be 'attested', 'certified', 'evidence' or 'handover'")
    if not isinstance(proof_digest, str) or re.fullmatch(r"[0-9a-f]{64}", proof_digest) is None:
        raise VerifiedTransitionError("proof_digest must be a lowercase sha256 hex digest")
    if not conn.in_transaction:
        raise VerifiedTransitionError("a transition is marked inside the transaction that makes it")
    if conn.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
        # The commit-time check that binds a mark to its transaction is a foreign key.
        raise VerifiedTransitionError("verified transitions need foreign key enforcement")
    try:
        conn.execute(
            """INSERT INTO hosted_room_verified_transitions
               (room_id, from_epoch, to_epoch, successor_gateway_id, proof_kind, proof_digest, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (room_id, from_epoch, to_epoch, successor_gateway_id, proof_kind, proof_digest, time.time()))
    except sqlite3.IntegrityError as exc:
        raise VerifiedTransitionError("this room already has a verified transition to that epoch") from exc


def move_transition_mark_to_branch(
    conn: sqlite3.Connection, *, room_id: str, to_epoch: int, branch_id: str,
) -> dict[str, Any]:
    """Archive the mark of a transition whose event is leaving the log for a quarantined branch.

    When two successors claimed the same epoch and the owner kept the other one, this gateway's own
    ``authority.transition`` and its tail move into a quarantined divergent branch. The caller moves
    that event out of the live log first, in this same transaction; then this archives the event's
    mark and use in ``hosted_room_branch_transitions`` (kept and readable, never deleted) and frees
    ``(room_id, to_epoch)`` for the transition that was kept. Refused while the marked event is still
    in the room's log or copy.
    """
    from gateway.hosted_rooms import VerifiedTransitionError

    room_id = identifier(room_id, label="room_id", error=VerifiedTransitionError, max_chars=128)
    branch_id = identifier(branch_id, label="branch_id", error=VerifiedTransitionError, max_chars=128)
    if isinstance(to_epoch, bool) or not isinstance(to_epoch, int) or not 1 <= to_epoch < 2**63:
        raise VerifiedTransitionError("to_epoch must be a positive integer")
    if not conn.in_transaction:
        raise VerifiedTransitionError("a mark moves inside the transaction that moves its event")
    mark = conn.execute(
        """SELECT mark.*, used.seq, used.event_id FROM hosted_room_verified_transitions AS mark
             JOIN hosted_room_verified_transition_uses AS used
               ON used.room_id=mark.room_id AND used.to_epoch=mark.to_epoch
            WHERE mark.room_id=? AND mark.to_epoch=?""", (room_id, to_epoch)).fetchone()
    if mark is None:
        raise VerifiedTransitionError("no verified transition is marked for this room and epoch")
    for table in ("hosted_room_events", "hosted_room_replica_events"):
        if conn.execute(f"SELECT 1 FROM {table} WHERE room_id=? AND seq=? AND event_id=?",
                        (room_id, mark["seq"], mark["event_id"])).fetchone():
            raise VerifiedTransitionError("the marked transition is still in this room's log")
    archived = {
        "room_id": room_id, "branch_id": branch_id, "from_epoch": int(mark["from_epoch"]), "to_epoch": to_epoch,
        "successor_gateway_id": mark["successor_gateway_id"], "proof_kind": mark["proof_kind"],
        "proof_digest": mark["proof_digest"], "marked_at": float(mark["created_at"]), "seq": int(mark["seq"]),
        "event_id": mark["event_id"], "archived_at": time.time()}
    conn.execute(f"INSERT INTO hosted_room_branch_transitions ({', '.join(archived)}) "
                 f"VALUES ({', '.join('?' for _ in archived)})", tuple(archived.values()))
    conn.execute("DELETE FROM hosted_room_verified_transitions WHERE room_id=? AND to_epoch=?", (room_id, to_epoch))
    conn.execute("DELETE FROM hosted_room_verified_transition_uses WHERE room_id=? AND to_epoch=?", (room_id, to_epoch))
    return archived


def _quarantine_unsafe_authorities_locked(conn: sqlite3.Connection) -> None:
    """Derive missing fences after historical replay; retain original quarantine evidence.

    A demotion or transition is verified only where its own mark was used by exactly that event.
    """
    unverified = """NOT EXISTS (
        SELECT 1 FROM hosted_room_verified_transition_uses AS used
         WHERE used.room_id=event.room_id AND used.to_epoch=event.authority_epoch
           AND used.seq=event.seq AND used.event_id=event.event_id)"""
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_quarantine
           (room_id, reason, detected_at)
           SELECT room_id, 'unsafe_replica_promotion', MIN(created_at)
             FROM hosted_room_events
            WHERE kind='authority.claimed'
              AND payload_json LIKE '%"promoted_from_replica":true%'
            GROUP BY room_id"""
    )
    for kind, reason in (("authority.lost", "unsafe_authority_demotion"),
                         ("authority.transition", "unverified_authority_transition")):
        conn.execute(
            f"""INSERT OR IGNORE INTO hosted_room_quarantine
               (room_id, reason, detected_at)
               SELECT room_id, ?, MIN(created_at)
                 FROM hosted_room_events AS event
                WHERE kind=? AND {unverified}
                GROUP BY room_id""",
            (reason, kind),
        )


def initialize_safety_schema(conn: sqlite3.Connection) -> None:
    from gateway.hosted_room_replicas import _initialize_replica_schema

    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_quarantine (
            room_id TEXT PRIMARY KEY,
            reason TEXT NOT NULL,
            detected_at REAL NOT NULL
        )"""
    )
    # A use binds a mark to the one event it verified. Marks and uses are lineage evidence: never deleted.
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_verified_transition_uses (
            room_id TEXT NOT NULL,
            to_epoch INTEGER NOT NULL,
            seq INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            PRIMARY KEY (room_id, to_epoch)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_verified_transitions (
            room_id TEXT NOT NULL,
            from_epoch INTEGER NOT NULL CHECK (from_epoch >= 1),
            to_epoch INTEGER NOT NULL CHECK (to_epoch > from_epoch),
            successor_gateway_id TEXT NOT NULL,
            proof_kind TEXT NOT NULL CHECK (proof_kind IN ('attested', 'certified', 'evidence', 'handover')),
            proof_digest TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, to_epoch),
            FOREIGN KEY (room_id, to_epoch)
                REFERENCES hosted_room_verified_transition_uses (room_id, to_epoch)
                DEFERRABLE INITIALLY DEFERRED
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_branch_transitions (
            room_id TEXT NOT NULL,
            branch_id TEXT NOT NULL,
            from_epoch INTEGER NOT NULL,
            to_epoch INTEGER NOT NULL,
            successor_gateway_id TEXT NOT NULL,
            proof_kind TEXT NOT NULL,
            proof_digest TEXT NOT NULL,
            marked_at REAL NOT NULL,
            seq INTEGER NOT NULL,
            event_id TEXT NOT NULL,
            archived_at REAL NOT NULL,
            PRIMARY KEY (room_id, branch_id, to_epoch)
        )"""
    )
    _initialize_replica_schema(conn)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_event_budget (
            singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0)
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_quarantine_disbands (
            room_id TEXT PRIMARY KEY,
            confirmed_at REAL NOT NULL
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS hosted_room_id_reservations (
            room_id TEXT PRIMARY KEY,
            owner_kind TEXT NOT NULL CHECK (owner_kind IN ('authority', 'replica')),
            reserved_at REAL NOT NULL
        )"""
    )
    _quarantine_unsafe_authorities_locked(conn)
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_quarantine
           (room_id, reason, detected_at)
           SELECT rooms.room_id, 'room_namespace_collision', rooms.updated_at
             FROM hosted_rooms AS rooms
             JOIN hosted_room_replicas AS replicas
               ON replicas.room_id=rooms.room_id"""
    )
    conn.execute(
        """UPDATE hosted_room_replicas
              SET quarantined_at=COALESCE(
                      quarantined_at,
                      (SELECT updated_at FROM hosted_rooms
                        WHERE hosted_rooms.room_id=hosted_room_replicas.room_id)
                  ),
                  quarantine_reason=COALESCE(
                      quarantine_reason,
                      'room_namespace_collision'
                  )
            WHERE EXISTS (
                SELECT 1 FROM hosted_rooms
                 WHERE hosted_rooms.room_id=hosted_room_replicas.room_id
            )"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_id_reservations
           (room_id, owner_kind, reserved_at)
           SELECT room_id, 'authority', created_at FROM hosted_rooms"""
    )
    conn.execute(
        """INSERT OR IGNORE INTO hosted_room_id_reservations
           (room_id, owner_kind, reserved_at)
           SELECT room_id, 'replica', created_at FROM hosted_room_replicas"""
    )
    conn.execute(
        """INSERT INTO hosted_room_event_budget(singleton, event_bytes)
           VALUES (
               1,
               COALESCE((
                   SELECT SUM(
                       LENGTH(CAST(event_id AS BLOB)) +
                       LENGTH(CAST(kind AS BLOB)) +
                       LENGTH(CAST(actor_json AS BLOB)) +
                       LENGTH(CAST(payload_json AS BLOB))
                   ) FROM hosted_room_events
               ), 0) +
               COALESCE((
                   SELECT SUM(
                       LENGTH(CAST(event_id AS BLOB)) +
                       LENGTH(CAST(kind AS BLOB)) +
                       LENGTH(CAST(actor_json AS BLOB)) +
                       LENGTH(CAST(payload_json AS BLOB))
                   ) FROM hosted_room_replica_events
               ), 0)
           )
           ON CONFLICT(singleton) DO UPDATE SET event_bytes=excluded.event_bytes"""
    )
    from gateway import hosted_rooms as limits

    ordinary_event_budget = int(limits.MAX_GATEWAY_EVENT_BYTES)
    control_event_budget = ordinary_event_budget + int(
        limits.CONTROL_EVENT_BYTE_RESERVE
    )
    for name, marker in _REVISED_TRIGGERS.items():
        row = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND name=?", (name,)).fetchone()
        if row is not None and marker not in str(row[0]):
            conn.execute(f"DROP TRIGGER {name}")
    lineage_change = "NEW.kind IN ('authority.lost', 'authority.transition')"
    for trigger in (
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_reject_reserved_insert
           BEFORE INSERT ON hosted_rooms
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_id_reservations WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room_id is already reserved');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_reserve_insert
           AFTER INSERT ON hosted_rooms
           BEGIN
               INSERT INTO hosted_room_id_reservations
                   (room_id, owner_kind, reserved_at)
               VALUES (NEW.room_id, 'authority', NEW.created_at);
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replicas_reject_reserved_insert
           BEFORE INSERT ON hosted_room_replicas
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_id_reservations WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room_id is already reserved');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replicas_reserve_insert
           AFTER INSERT ON hosted_room_replicas
           BEGIN
               INSERT INTO hosted_room_id_reservations
                   (room_id, owner_kind, reserved_at)
               VALUES (NEW.room_id, 'replica', NEW.created_at);
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_reject_quarantined_insert
           BEFORE INSERT ON hosted_room_events
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=NEW.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'room authority is quarantined');
           END""",
        # An authority change is accepted only with its own verified-transition mark; any other
        # promotion, demotion or transition leaves the room read-only.
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_events_quarantine_unsafe_lineage
           AFTER INSERT ON hosted_room_events
           WHEN {lineage_change}
             OR (
                 NEW.kind='authority.claimed'
                 AND NEW.payload_json LIKE '%"promoted_from_replica":true%'
             )
           BEGIN
               INSERT OR IGNORE INTO hosted_room_quarantine
                   (room_id, reason, detected_at)
               SELECT
                   NEW.room_id,
                   CASE NEW.kind
                       WHEN 'authority.lost' THEN 'unsafe_authority_demotion'
                       WHEN 'authority.transition' THEN 'unverified_authority_transition'
                       ELSE 'unsafe_replica_promotion'
                   END,
                   NEW.created_at
                WHERE NOT ({lineage_change} AND {_transition_match_sql("hosted_room_events")});
               {_USE_TRANSITION_SQL.format(
                   match=f"{lineage_change} AND {_transition_match_sql('hosted_room_events')}")}
           END""",
        # A stored copy refuses an unverified transition outright; it never quarantines a copy half-written.
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_verified_lineage
           AFTER INSERT ON hosted_room_replica_events
           WHEN NEW.kind='authority.transition'
           BEGIN
               SELECT RAISE(ABORT, 'authority transition is not verified')
                WHERE NOT {_transition_match_sql("hosted_room_replica_events")};
               {_USE_TRANSITION_SQL.format(match=_transition_match_sql("hosted_room_replica_events"))}
           END""",
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_events_shared_budget
           BEFORE INSERT ON hosted_room_events
           WHEN NOT EXISTS (
               SELECT 1 FROM hosted_room_events
                WHERE room_id=NEW.room_id
                  AND (seq=NEW.seq OR event_id=NEW.event_id)
             )
             AND (
                 (SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1) +
                 LENGTH(CAST(NEW.event_id AS BLOB)) +
                 LENGTH(CAST(NEW.kind AS BLOB)) +
                 LENGTH(CAST(NEW.actor_json AS BLOB)) +
                 LENGTH(CAST(NEW.payload_json AS BLOB))
             ) > CASE
                 WHEN NEW.kind IN (
                     'authority.claimed', 'authority.lost', 'authority.transition',
                     'room.disbanded', 'room.stop_requested'
                 ) THEN {control_event_budget}
                 ELSE {ordinary_event_budget}
             END
           BEGIN
               SELECT RAISE(ABORT, 'hosted room event budget exceeded');
           END""",
        f"""CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_shared_budget
           BEFORE INSERT ON hosted_room_replica_events
           WHEN NOT EXISTS (
               SELECT 1 FROM hosted_room_replica_events
                WHERE room_id=NEW.room_id AND seq=NEW.seq
             )
             AND (
                 (SELECT event_bytes FROM hosted_room_event_budget WHERE singleton=1) +
                 LENGTH(CAST(NEW.event_id AS BLOB)) +
                 LENGTH(CAST(NEW.kind AS BLOB)) +
                 LENGTH(CAST(NEW.actor_json AS BLOB)) +
                 LENGTH(CAST(NEW.payload_json AS BLOB))
             ) > {ordinary_event_budget}
           BEGIN
               SELECT RAISE(ABORT, 'hosted room event budget exceeded');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_budget_account_insert
           AFTER INSERT ON hosted_room_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=event_bytes +
                      LENGTH(CAST(NEW.event_id AS BLOB)) +
                      LENGTH(CAST(NEW.kind AS BLOB)) +
                      LENGTH(CAST(NEW.actor_json AS BLOB)) +
                      LENGTH(CAST(NEW.payload_json AS BLOB))
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_budget_account_delete
           AFTER DELETE ON hosted_room_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=MAX(
                      0,
                      event_bytes -
                      LENGTH(CAST(OLD.event_id AS BLOB)) -
                      LENGTH(CAST(OLD.kind AS BLOB)) -
                      LENGTH(CAST(OLD.actor_json AS BLOB)) -
                      LENGTH(CAST(OLD.payload_json AS BLOB))
                  )
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_budget_account_insert
           AFTER INSERT ON hosted_room_replica_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=event_bytes +
                      LENGTH(CAST(NEW.event_id AS BLOB)) +
                      LENGTH(CAST(NEW.kind AS BLOB)) +
                      LENGTH(CAST(NEW.actor_json AS BLOB)) +
                      LENGTH(CAST(NEW.payload_json AS BLOB))
                WHERE singleton=1;
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_replica_events_budget_account_delete
           AFTER DELETE ON hosted_room_replica_events
           BEGIN
               UPDATE hosted_room_event_budget
                  SET event_bytes=MAX(
                      0,
                      event_bytes -
                      LENGTH(CAST(OLD.event_id AS BLOB)) -
                      LENGTH(CAST(OLD.kind AS BLOB)) -
                      LENGTH(CAST(OLD.actor_json AS BLOB)) -
                      LENGTH(CAST(OLD.payload_json AS BLOB))
                  )
                WHERE singleton=1;
           END""",
        # A quarantined room takes one change only: the tombstone of a Disband an operator confirmed,
        # recorded in the same transaction, which leaves its authority and history untouched.
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_quarantined_tombstone
           BEFORE UPDATE OF disbanded_at ON hosted_rooms
           WHEN NEW.disbanded_at IS NOT OLD.disbanded_at
             AND EXISTS (
                 SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
             )
             AND NOT (
                 OLD.disbanded_at IS NULL AND NEW.disbanded_at IS NOT NULL
                 AND EXISTS (
                     SELECT 1 FROM hosted_room_quarantine_disbands WHERE room_id=OLD.room_id
                 )
                 AND NEW.authority_gateway_id IS OLD.authority_gateway_id
                 AND NEW.authority_epoch IS OLD.authority_epoch
                 AND NEW.next_seq IS OLD.next_seq
                 AND NEW.event_bytes IS OLD.event_bytes
             )
           BEGIN
               SELECT RAISE(ABORT, 'room authority is quarantined');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_rooms_keep_quarantined
           BEFORE DELETE ON hosted_rooms
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'quarantined room history is kept');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_hosted_events_keep_quarantined
           BEFORE DELETE ON hosted_room_events
           WHEN EXISTS (
               SELECT 1 FROM hosted_room_quarantine WHERE room_id=OLD.room_id
           )
           BEGIN
               SELECT RAISE(ABORT, 'quarantined room history is kept');
           END""",
        # A set-aside transition's mark is evidence of its divergent branch.
        """CREATE TRIGGER IF NOT EXISTS trg_branch_transitions_keep
           BEFORE DELETE ON hosted_room_branch_transitions
           BEGIN
               SELECT RAISE(ABORT, 'a divergent branch transition is kept');
           END""",
        """CREATE TRIGGER IF NOT EXISTS trg_branch_transitions_unchanged
           BEFORE UPDATE ON hosted_room_branch_transitions
           BEGIN
               SELECT RAISE(ABORT, 'a divergent branch transition is kept');
           END""",
    ):
        conn.execute(trigger)
    # Audits every stored copy, re-deriving its byte count, before compacting any.
    _compact_over_budget_replicas_locked(conn)


def safety_schema_is_current(conn: sqlite3.Connection) -> bool:
    tables = {
        "hosted_room_quarantine": _QUARANTINE_SCHEMA_COLUMNS,
        "hosted_room_id_reservations": _ROOM_RESERVATION_SCHEMA_COLUMNS,
        "hosted_room_replicas": _REPLICA_SCHEMA_COLUMNS,
        "hosted_room_replica_events": _REPLICA_EVENT_SCHEMA_COLUMNS,
        "hosted_room_event_budget": _EVENT_BUDGET_SCHEMA_COLUMNS,
        "hosted_room_quarantine_disbands": _QUARANTINE_DISBAND_SCHEMA_COLUMNS,
        "hosted_room_verified_transitions": _VERIFIED_TRANSITION_SCHEMA_COLUMNS,
        "hosted_room_verified_transition_uses": _TRANSITION_USE_SCHEMA_COLUMNS,
        "hosted_room_branch_transitions": _BRANCH_TRANSITION_SCHEMA_COLUMNS,
    }
    triggers = {str(row[0]): str(row[1]) for row in conn.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger'")}
    return all(columns.issubset(table_columns(conn, table)) for table, columns in tables.items()) and (
        _ROOM_SAFETY_TRIGGERS.issubset(triggers)) and all(
        marker in triggers[name] for name, marker in _REVISED_TRIGGERS.items())


def _compact_over_budget_replicas_locked(conn: sqlite3.Connection) -> int:
    """Reclaim only verified terminal copies; never erase an active prefix.

    Retained excess bytes keep the shared budget full and further writes refused.
    Capacity is not invented integrity failure: authorized retirement can still
    clean a valid copy, while genuinely quarantined evidence remains protected.
    """
    if not table_exists(conn, "hosted_room_replicas"):
        return 0
    from gateway import hosted_rooms as limits

    hosted_bytes = int(conn.execute("SELECT COALESCE(SUM(event_bytes),0) FROM hosted_rooms").fetchone()[0])
    # This helper re-audits before selecting only completed, non-quarantined
    # history under the existing explicit terminal-retention policy.
    return _prune_disbanded_replicas_locked(
        conn, now=None, max_replica_event_bytes=max(0, limits.MAX_GATEWAY_EVENT_BYTES - hosted_bytes))


def _quarantine_reason_locked(conn: sqlite3.Connection, room_id: str) -> str | None:
    row = conn.execute(
        "SELECT reason FROM hosted_room_quarantine WHERE room_id=?", (room_id,)
    ).fetchone()
    return str(row["reason"]) if row is not None else None


def _raise_if_quarantined(conn: sqlite3.Connection, room_id: str, *, hint: str = "") -> None:
    from gateway.hosted_rooms import RoomQuarantinedError

    reason = _quarantine_reason_locked(conn, room_id)
    if reason is not None:
        raise RoomQuarantinedError(
            "This Group Chat has an unverified authority takeover and is read-only "
            f"until its history is reconciled ({reason}).{hint}"
        )


def _record_quarantine_disband_locked(conn: sqlite3.Connection, room_id: str, now: float) -> None:
    """Record an operator's confirmation; the tombstone trigger accepts nothing else."""
    conn.execute(
        "INSERT OR IGNORE INTO hosted_room_quarantine_disbands (room_id, confirmed_at) VALUES (?, ?)",
        (room_id, now),
    )


def _replica_reserves_room_id_locked(conn: sqlite3.Connection, room_id: str) -> bool:
    if not conn.execute(
        """SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='hosted_room_replicas'"""
    ).fetchone():
        return False
    return (
        conn.execute(
            "SELECT 1 FROM hosted_room_replicas WHERE room_id=?", (room_id,)
        ).fetchone()
        is not None
    )


def _room_id_reservation_kind_locked(
    conn: sqlite3.Connection, room_id: str
) -> str | None:
    row = conn.execute(
        "SELECT owner_kind FROM hosted_room_id_reservations WHERE room_id=?",
        (room_id,),
    ).fetchone()
    return str(row["owner_kind"]) if row is not None else None


def _replica_event_bytes_locked(conn: sqlite3.Connection) -> int:
    """Return passive-replica bytes when the optional replica table exists."""
    if not conn.execute(
        """SELECT 1 FROM sqlite_master
             WHERE type='table' AND name='hosted_room_replicas'"""
    ).fetchone():
        return 0
    return int(
        conn.execute(
            "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
        ).fetchone()[0]
    )


def _prune_disbanded_replicas_locked(
    conn: sqlite3.Connection,
    *,
    now: float | None,
    max_replica_event_bytes: int | None = None,
    max_replica_rooms: int | None = None,
) -> int:
    """Reclaim terminal replica payload while its room-ID reservation remains.

    A copy is terminal once it holds its room's complete history up to ``room.disbanded``, or once
    its owner-enrolled retirement covers exactly this copy. Quarantined copies are never reclaimed.
    """
    from gateway import hosted_rooms as limits
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    from gateway.hosted_room_replicas import (
        _audit_existing_replicas_locked, _replica_header_bounds_sql, _replica_read_envelope_locked)

    # Re-audit even if an earlier observation/ingest audited then rolled back,
    # or an old writer committed new history since the last replica read.
    _audit_existing_replicas_locked(conn)
    canonical = "(disbanded_at IS NOT NULL AND last_seq=latest_seq)"
    eligible, terminal_at = canonical, "disbanded_at"
    if table_exists(conn, RETIREMENT_TABLE):
        retired = f"""SELECT retired_at FROM {RETIREMENT_TABLE} AS retirement
            WHERE retirement.room_id=hosted_room_replicas.room_id
              AND retirement.authority_gateway_id=hosted_room_replicas.authority_gateway_id
              AND retirement.authority_epoch=hosted_room_replicas.authority_epoch"""
        eligible = f"({canonical} OR EXISTS ({retired}))"
        terminal_at = f"CASE WHEN {canonical} THEN disbanded_at ELSE ({retired}) END"
    eligible += """ AND quarantine_reason IS NULL AND NOT EXISTS (
        SELECT 1 FROM hosted_room_quarantine WHERE hosted_room_quarantine.room_id=hosted_room_replicas.room_id)"""
    eligible += f" AND ({_replica_header_bounds_sql()})"
    candidates: set[str] = set()
    if now is not None:
        cutoff = now - limits.DISBANDED_REPLICA_RETENTION_SECONDS
        candidates.update(
            str(row["room_id"])
            for row in conn.execute(
                f"SELECT room_id FROM hosted_room_replicas WHERE {eligible} AND ({terminal_at})<=?", (cutoff,),
            ).fetchall()
        )
    if max_replica_event_bytes is not None:
        retained_bytes = int(
            conn.execute(
                "SELECT COALESCE(SUM(event_bytes), 0) FROM hosted_room_replicas"
            ).fetchone()[0]
        )
        if retained_bytes > max_replica_event_bytes:
            for row in conn.execute(
                f"""SELECT room_id, event_bytes FROM hosted_room_replicas WHERE {eligible}
                     ORDER BY ({terminal_at}) ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if not _replica_read_envelope_locked(conn, room_id):
                    continue
                candidates.add(room_id)
                retained_bytes -= int(row["event_bytes"])
                if retained_bytes <= max_replica_event_bytes:
                    break
    if max_replica_rooms is not None:
        retained_rooms = int(
            conn.execute("SELECT COUNT(*) FROM hosted_room_replicas").fetchone()[0]
        )
        if retained_rooms > max_replica_rooms:
            for row in conn.execute(
                f"""SELECT room_id FROM hosted_room_replicas WHERE {eligible}
                     ORDER BY ({terminal_at}) ASC, room_id ASC"""
            ).fetchall():
                room_id = str(row["room_id"])
                if not _replica_read_envelope_locked(conn, room_id):
                    continue
                candidates.add(room_id)
                retained_rooms -= 1
                if retained_rooms <= max_replica_rooms:
                    break
    candidates = {room_id for room_id in candidates if _replica_read_envelope_locked(conn, room_id)}
    if not candidates:
        return 0
    placeholders = ",".join("?" for _ in candidates)
    room_ids = tuple(sorted(candidates))
    eligible = f"room_id IN ({placeholders}) AND {eligible}"
    conn.execute(
        f"""DELETE FROM hosted_room_replica_events WHERE room_id IN (
                SELECT room_id FROM hosted_room_replicas WHERE {eligible})""",
        room_ids,
    )
    deleted = conn.execute(
        f"DELETE FROM hosted_room_replicas WHERE {eligible}",
        room_ids,
    )
    return max(0, int(deleted.rowcount))
