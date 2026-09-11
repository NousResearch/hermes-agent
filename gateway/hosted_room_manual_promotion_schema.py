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
      AND ((json_type(decision.work_record_json,'$.binding.enrollment')='null'
            AND decision.source_epoch=1
            AND NOT EXISTS (SELECT 1 FROM hosted_room_replica_retirement_enrollments WHERE room_id=NEW.room_id AND is_current=1))
        OR EXISTS (SELECT 1 FROM hosted_room_replica_retirement_enrollments enrolled
            WHERE enrolled.room_id=NEW.room_id AND enrolled.is_current=1 AND enrolled.state='active'
            AND enrolled.target_install_id=decision.target_gateway_id
            AND enrolled.authority_gateway_id=decision.source_gateway_id AND enrolled.authority_epoch=decision.source_epoch
            AND {" AND ".join(f"enrolled.{k} IS json_extract(decision.work_record_json,'$.binding.enrollment.{k}') AND typeof(enrolled.{k})=json_type(decision.work_record_json,'$.binding.enrollment.{k}')" for k in ("enrollment_id", "room_id", "authority_gateway_id", "authority_epoch", "target_install_id", "roster_sha256", "version", "is_current", "state", "authority_history_json", "lineage_sha256"))}))
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
    from gateway.hosted_room_replica_retirement import _initialize as initialize_retirement
    initialize_retirement(conn)
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
    immutable = ("room_id", "recovery_id", "snapshot_id", "source_gateway_id", "source_epoch",
                 "target_gateway_id", "history_seq", "work_record_json", "created_at")
    changed = " OR ".join(f"NEW.{k} IS NOT OLD.{k} OR typeof(NEW.{k})!=typeof(OLD.{k})" for k in immutable)
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_recovery_evidence_immutable
        BEFORE UPDATE ON {TABLE} WHEN {changed}
        BEGIN SELECT RAISE(ABORT, 'recovery evidence is immutable'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_recovery_evidence_replace
        BEFORE INSERT ON {TABLE} WHEN EXISTS (SELECT 1 FROM {TABLE}
            WHERE room_id=NEW.room_id OR recovery_id=NEW.recovery_id)
        BEGIN SELECT RAISE(ABORT, 'recovery evidence is immutable'); END""")
    from gateway.hosted_room_work_storage import initialize as initialize_work
    initialize_work(conn)
