"""Transactional namespace transfer for one explicit, still-inert recovery.

The temporary 'transferring' row is never committed: successful transfer leaves
'pending_reconciliation'. Other writers cannot borrow this transaction's row.
"""

TABLE = "hosted_room_manual_recoveries"

_TRANSFER_MATCH = f"""EXISTS (
    SELECT 1 FROM {TABLE} AS decision
    JOIN hosted_room_replicas AS copy ON copy.room_id=decision.room_id
    JOIN hosted_room_id_reservations AS reservation ON reservation.room_id=copy.room_id
    WHERE decision.room_id=NEW.room_id AND decision.status='transferring'
      AND reservation.owner_kind='replica'
      AND copy.authority_gateway_id=decision.source_gateway_id
      AND copy.authority_epoch=decision.source_epoch
      AND copy.last_seq=decision.history_seq AND copy.latest_seq=decision.history_seq
      AND copy.disbanded_at IS NULL AND copy.quarantine_reason IS NULL
      AND NEW.authority_gateway_id=decision.target_gateway_id
      AND NEW.authority_epoch=decision.source_epoch+1
      AND NEW.name=copy.name AND NEW.members_json=copy.members_json
      AND NEW.next_seq=decision.history_seq+1 AND NEW.disbanded_at IS NULL
      AND NOT EXISTS (SELECT 1 FROM hosted_room_quarantine WHERE room_id=NEW.room_id)
      AND NOT EXISTS (SELECT 1 FROM hosted_room_retired_ids WHERE room_id=NEW.room_id)
)"""


def initialize(conn):
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {TABLE} (
        room_id TEXT PRIMARY KEY, recovery_id TEXT NOT NULL UNIQUE, snapshot_id TEXT NOT NULL,
        source_gateway_id TEXT NOT NULL, source_epoch INTEGER NOT NULL,
        target_gateway_id TEXT NOT NULL, history_seq INTEGER NOT NULL,
        work_record_json TEXT NOT NULL, created_at REAL NOT NULL,
        status TEXT NOT NULL CHECK(status IN ('transferring','pending_reconciliation','active')),
        FOREIGN KEY(room_id) REFERENCES hosted_rooms(room_id) ON DELETE CASCADE
            DEFERRABLE INITIALLY DEFERRED)""")
    # These replace only the two reservation triggers. All other quarantine,
    # retirement, event-budget and immutable replay guards stay installed.
    conn.execute("DROP TRIGGER IF EXISTS trg_hosted_rooms_reject_reserved_insert")
    conn.execute(f"""CREATE TRIGGER trg_hosted_rooms_reject_reserved_insert
        BEFORE INSERT ON hosted_rooms
        WHEN EXISTS (SELECT 1 FROM hosted_room_id_reservations WHERE room_id=NEW.room_id)
          AND NOT ({_TRANSFER_MATCH})
        BEGIN SELECT RAISE(ABORT, 'room_id is already reserved'); END""")
    conn.execute("DROP TRIGGER IF EXISTS trg_hosted_rooms_reserve_insert")
    conn.execute(f"""CREATE TRIGGER trg_hosted_rooms_reserve_insert
        AFTER INSERT ON hosted_rooms
        BEGIN
            INSERT INTO hosted_room_id_reservations(room_id,owner_kind,reserved_at)
            VALUES(NEW.room_id,'authority',NEW.created_at)
            ON CONFLICT(room_id) DO UPDATE SET owner_kind='authority'
            WHERE {_TRANSFER_MATCH};
        END""")
