"""Producer-scoped passive evidence storage and durable older-writer guards."""

import json

from gateway.hosted_rooms_common import table_exists


INVALID_TABLE = "hosted_room_work_records_invalid"


def initialize(conn):
    if not conn.in_transaction:
        # Reserve the writer before schema reads. A deferred SAVEPOINT can lose
        # its write upgrade to another publisher without honoring busy_timeout.
        conn.execute("BEGIN IMMEDIATE")
        with conn:
            _initialize_locked(conn)
        return
    # Nested initialization must never commit or roll back caller-owned work.
    conn.execute("SAVEPOINT work_record_initialize")
    try:
        _initialize_locked(conn)
    except BaseException:
        conn.execute("ROLLBACK TO work_record_initialize")
        conn.execute("RELEASE work_record_initialize")
        raise
    conn.execute("RELEASE work_record_initialize")


def _initialize_locked(conn):
    from gateway import hosted_room_work_records as work
    tables = (work.SOURCE_TABLE, work.TARGET_TABLE, work.PENDING_TABLE)
    # Orphans are opaque evidence, not snapshots. No FK or producer can honestly
    # be supplied for them. Preserve every original column and forbid mutation.
    conn.execute(f"""CREATE TABLE IF NOT EXISTS {INVALID_TABLE} (
        evidence_id INTEGER PRIMARY KEY, source_table TEXT NOT NULL,
        room_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
        target_install_id TEXT, route_generation TEXT, status TEXT,
        disposition TEXT NOT NULL CHECK(disposition='invalid'))""")
    for table in tables:
        pending = table == work.PENDING_TABLE
        parent = "hosted_room_replicas" if table == work.TARGET_TABLE else "hosted_rooms"
        columns = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        legacy = bool(columns) and "producer_gateway_id" not in columns
        # Preserve surrounding cleanup triggers by recreating the same table name
        # inside this transaction, rather than renaming its referents.
        rows = conn.execute(f"SELECT * FROM {table}").fetchall() if legacy else []
        triggers = conn.execute("SELECT sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?", (table,)).fetchall() if legacy else []
        if legacy:
            conn.execute(f"DROP TABLE {table}")
        extra = "target_install_id TEXT NOT NULL, route_generation TEXT NOT NULL, status TEXT NOT NULL," if pending else ""
        key = ",target_install_id" if pending else ""
        conn.execute(f"""CREATE TABLE IF NOT EXISTS {table} (
            room_id TEXT NOT NULL, revision INTEGER NOT NULL, digest TEXT NOT NULL, record_json TEXT NOT NULL,
            producer_gateway_id TEXT NOT NULL, producer_epoch INTEGER NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN ('current','superseded_authority','historical','invalid')),
            {extra} PRIMARY KEY(room_id,producer_gateway_id,producer_epoch{key}),
            FOREIGN KEY(room_id) REFERENCES {parent}(room_id) ON DELETE CASCADE)""")
        for row in rows:
            if not conn.execute(f"SELECT 1 FROM {parent} WHERE room_id=?", (row["room_id"],)).fetchone():
                fields = ["source_table", "room_id", "revision", "digest", "record_json", "disposition"]
                values = [table, row["room_id"], row["revision"], row["digest"], row["record_json"], "invalid"]
                if pending:
                    fields += ["target_install_id", "route_generation", "status"]
                    values += [row[k] for k in fields[-3:]]
                conn.execute(f"INSERT INTO {INVALID_TABLE} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
                continue
            producer, epoch, disposition = "", 0, "invalid"
            try:
                record = work.validate(json.loads(row["record_json"]))
                if (record["version"] != 1 or record["room_id"] != row["room_id"]
                        or record["revision"] != row["revision"] or record["digest"] != row["digest"]):
                    raise work.WorkRecordError("legacy work scope conflicts")
                producer, epoch, disposition = record["home_install_id"], 1, "current"
            except (ValueError, TypeError, KeyError):
                pass  # Explicitly invalid, with original bytes and no invented producer.
            fields = ["room_id", "revision", "digest", "record_json", "producer_gateway_id", "producer_epoch", "disposition"]
            values = [row[k] for k in fields[:4]] + [producer, epoch, disposition]
            if pending:
                fields += ["target_install_id", "route_generation", "status"]
                values += [row[k] for k in fields[-3:]]
            values[6] = scope_disposition(conn, table, dict(zip(fields, values)))
            conn.execute(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})", values)
        for trigger in triggers:
            conn.execute(trigger[0])
    _guards(conn, work, tables)
    _invalid_guards(conn)
    initialize_lineage_guards(conn)
    # Ordinary initialization owns schema/guards only. Each consuming operation
    # validates its own row; authority/enrollment triggers freeze exact-room old
    # scopes. Legacy dispositions were resolved at insertion above.
    work.initialize_target_guards(conn)
    from gateway.hosted_room_work_record_budget import install_recovery_budget_guards
    install_recovery_budget_guards(conn)


def scope_disposition(conn, table, row):
    """Read-only authority projection, also used when inserting migrated rows."""
    from gateway import hosted_room_work_records as work
    if row["disposition"] != "current":
        return row["disposition"]
    current = None
    if table == work.TARGET_TABLE:
        from gateway.hosted_room_passive_lineage import ENROLLMENTS
        if table_exists(conn, ENROLLMENTS):
            current = conn.execute(f"SELECT authority_gateway_id,authority_epoch FROM {ENROLLMENTS} "
                                   "WHERE room_id=? AND is_current=1", (row["room_id"],)).fetchone()
    else:
        current = conn.execute("SELECT authority_gateway_id,authority_epoch FROM hosted_rooms WHERE room_id=?",
                               (row["room_id"],)).fetchone()
    if current is None or tuple(current) == (row["producer_gateway_id"], row["producer_epoch"]):
        return "current"
    return "superseded_authority" if table == work.PENDING_TABLE and row["status"] != "acked" else "historical"


def _guards(conn, work, tables):
    total, count = usage_sql((*tables, INVALID_TABLE))
    for table in tables:
        for operation in ("INSERT", "UPDATE"):
            key = "room_id=NEW.room_id AND producer_gateway_id=NEW.producer_gateway_id AND producer_epoch=NEW.producer_epoch"
            if table == work.PENDING_TABLE:
                key += " AND target_install_id=NEW.target_install_id"
            old_size = (f"COALESCE((SELECT {row_size_sql(table)} FROM {table} WHERE {key}),0)"
                        if operation == "INSERT" else row_size_sql(table, "OLD."))
            new_size = row_size_sql(table, "NEW.")
            old_count = f"(SELECT COUNT(*) FROM {table} WHERE {key})" if operation == "INSERT" else "1"
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v2_{operation.lower()}")
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v3_{operation.lower()}")
            conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_budget_v4_{operation.lower()}")
            conn.execute(f"""CREATE TRIGGER trg_{table}_budget_v4_{operation.lower()}
                BEFORE {operation} ON {table} WHEN (
                    (({new_size})>({old_size}) AND ({total})-({old_size})+({new_size})>{work.MAX_STORE_BYTES})
                    OR ({old_count}=0 AND ({total}>{work.MAX_STORE_BYTES} OR {count}+1>{work.MAX_STORE_ROWS})))
                BEGIN SELECT RAISE(ABORT, 'work record storage is full'); END""")
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_immutable_v2
            BEFORE UPDATE ON {table} WHEN
                NEW.room_id!=OLD.room_id OR NEW.producer_gateway_id!=OLD.producer_gateway_id
                OR NEW.producer_epoch!=OLD.producer_epoch
                OR (OLD.disposition!='current' AND (NEW.record_json!=OLD.record_json OR NEW.revision!=OLD.revision
                    OR NEW.digest!=OLD.digest OR NEW.disposition!=OLD.disposition))
            BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")
        _historical_delete_guard(conn, work, table)
        # INSERT OR REPLACE performs a delete internally. Refuse it before the
        # uniqueness conflict can remove an immutable snapshot.
        extra = " AND p.target_install_id=NEW.target_install_id" if table == work.PENDING_TABLE else ""
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_{table}_replace_v2
            BEFORE INSERT ON {table} WHEN EXISTS (SELECT 1 FROM {table} p WHERE p.room_id=NEW.room_id
                AND p.producer_gateway_id=NEW.producer_gateway_id AND p.producer_epoch=NEW.producer_epoch
                {extra} AND p.disposition!='current')
            BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_pending_outcome_v2 BEFORE UPDATE ON {work.PENDING_TABLE}
        WHEN OLD.disposition!='current' AND (NEW.status!=OLD.status OR NEW.route_generation!=OLD.route_generation
            OR NEW.target_install_id!=OLD.target_install_id)
        BEGIN SELECT RAISE(ABORT, 'historical delivery outcome is immutable'); END""")
    # A raw authority update by an older owner must also freeze old evidence.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_authority_transition_v2
        AFTER UPDATE OF authority_gateway_id,authority_epoch ON hosted_rooms BEGIN
        UPDATE {work.SOURCE_TABLE} SET disposition='historical' WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch);
        UPDATE {work.PENDING_TABLE} SET disposition=CASE WHEN status='acked' THEN 'historical' ELSE 'superseded_authority' END
            WHERE room_id=NEW.room_id AND disposition='current'
            AND (producer_gateway_id!=NEW.authority_gateway_id OR producer_epoch!=NEW.authority_epoch);
        END""")


def _historical_delete_guard(conn, work, table):
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    parent, retired = "hosted_rooms", ""
    if table == work.TARGET_TABLE:
        # Retirement may initialize after work storage. Never install a guard
        # which cannot recognize its later-authorized partial-copy cleanup.
        if not table_exists(conn, RETIREMENT_TABLE):
            return
        parent = "hosted_room_replicas"
        retired = f" AND NOT EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=OLD.room_id)"
    from gateway.hosted_room_recovery_evidence import transfer_permission_sql
    transfer = transfer_permission_sql(conn, table) if table == work.TARGET_TABLE else "0"
    conn.execute(f"DROP TRIGGER IF EXISTS trg_{table}_delete_v2")
    conn.execute(f"""CREATE TRIGGER trg_{table}_delete_v2 BEFORE DELETE ON {table}
        WHEN OLD.disposition!='current' AND EXISTS (SELECT 1 FROM {parent} WHERE room_id=OLD.room_id
            AND disbanded_at IS NULL) {retired} AND NOT ({transfer})
        BEGIN SELECT RAISE(ABORT, 'historical work record is immutable'); END""")


def initialize_lineage_guards(conn):
    from gateway import hosted_room_work_records as work
    from gateway.hosted_room_passive_lineage import ENROLLMENTS
    if (not table_exists(conn, ENROLLMENTS) or "producer_epoch" not in
            {r["name"] for r in conn.execute(f"PRAGMA table_info({work.TARGET_TABLE})")}):
        return
    _historical_delete_guard(conn, work, work.TARGET_TABLE)
    for operation in ("INSERT", "UPDATE"):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_enrollment_v2_{operation.lower()}
            AFTER {operation} ON {ENROLLMENTS} WHEN NEW.is_current=1 BEGIN
            UPDATE {work.TARGET_TABLE} SET disposition='historical' WHERE room_id=NEW.room_id
                AND disposition='current' AND (producer_gateway_id!=NEW.authority_gateway_id
                    OR producer_epoch!=NEW.authority_epoch);
            END""")


def retained_fields(table):
    from gateway import hosted_room_work_records as work
    fields = ["room_id", "revision", "digest", "record_json"]
    fields += ["source_table"] if table == INVALID_TABLE else ["producer_gateway_id", "producer_epoch"]
    if table in (work.PENDING_TABLE, INVALID_TABLE):
        fields += ["target_install_id", "route_generation", "status"]
    return fields


def row_size_sql(table, prefix=""):
    # Reserve the longest bounded disposition up front so freezing/cleanup can
    # still run at capacity. Every original variable column is charged as bytes.
    return "20+" + "+".join(f"COALESCE(length(CAST({prefix}{k} AS BLOB)),0)" for k in retained_fields(table))


def usage_sql(tables):
    return (" + ".join(f"(SELECT COALESCE(SUM({row_size_sql(t)}),0) FROM {t})" for t in tables),
            " + ".join(f"(SELECT COUNT(*) FROM {t})" for t in tables))


def _invalid_guards(conn):
    from gateway import hosted_room_work_records as work
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_invalid_update BEFORE UPDATE ON {INVALID_TABLE}
        BEGIN SELECT RAISE(ABORT, 'invalid work evidence is immutable'); END""")
    # No ordinary writer admits orphan evidence. Migration precedes this trigger;
    # later legacy reconstruction refuses atomically rather than losing evidence.
    conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_invalid_insert BEFORE INSERT ON {INVALID_TABLE}
        BEGIN SELECT RAISE(ABORT, 'invalid work evidence is immutable'); END""")
    from gateway.hosted_room_replica_retirement import RETIREMENT_TABLE
    retired = (f"OR (OLD.source_table='{work.TARGET_TABLE}' AND EXISTS (SELECT 1 FROM {RETIREMENT_TABLE} WHERE room_id=OLD.room_id))"
               if table_exists(conn, RETIREMENT_TABLE) else "")
    from gateway.hosted_room_recovery_evidence import transfer_permission_sql
    transfer = transfer_permission_sql(conn, INVALID_TABLE)
    conn.execute("DROP TRIGGER IF EXISTS trg_work_invalid_delete")
    conn.execute(f"""CREATE TRIGGER trg_work_invalid_delete BEFORE DELETE ON {INVALID_TABLE}
        WHEN NOT ((OLD.source_table IN ('{work.SOURCE_TABLE}','{work.PENDING_TABLE}') AND (
                EXISTS (SELECT 1 FROM hosted_rooms WHERE room_id=OLD.room_id AND disbanded_at IS NOT NULL)
                OR (EXISTS (SELECT 1 FROM hosted_room_id_reservations WHERE room_id=OLD.room_id AND owner_kind='authority')
                    AND NOT EXISTS (SELECT 1 FROM hosted_rooms WHERE room_id=OLD.room_id))))
            OR (OLD.source_table='{work.TARGET_TABLE}' AND (
                EXISTS (SELECT 1 FROM hosted_room_replicas WHERE room_id=OLD.room_id AND disbanded_at IS NOT NULL)
                OR (EXISTS (SELECT 1 FROM hosted_room_id_reservations WHERE room_id=OLD.room_id AND owner_kind='replica')
                    AND NOT EXISTS (SELECT 1 FROM hosted_room_replicas WHERE room_id=OLD.room_id)))) {retired}) AND NOT ({transfer})
        BEGIN SELECT RAISE(ABORT, 'invalid work evidence is immutable'); END""")
    for parent, kind in (("hosted_rooms", f"source_table IN ('{work.SOURCE_TABLE}','{work.PENDING_TABLE}')"),
                         ("hosted_room_replicas", f"source_table='{work.TARGET_TABLE}'")):
        for operation, ref, condition in (("DELETE", "OLD", ""), ("UPDATE OF disbanded_at", "NEW", " WHEN NEW.disbanded_at IS NOT NULL")):
            name = 'delete' if operation == 'DELETE' else 'disband'
            conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_invalid_{parent}_{name}
                AFTER {operation} ON {parent}{condition} BEGIN
                DELETE FROM {INVALID_TABLE} WHERE room_id={ref}.room_id AND {kind}; END""")
    if table_exists(conn, RETIREMENT_TABLE):
        conn.execute(f"""CREATE TRIGGER IF NOT EXISTS trg_work_invalid_retired AFTER INSERT ON {RETIREMENT_TABLE}
            BEGIN DELETE FROM {INVALID_TABLE} WHERE room_id=NEW.room_id AND source_table='{work.TARGET_TABLE}'; END""")


def validate_stored(row):
    from gateway import hosted_room_work_records as work
    try:
        record = work.validate(json.loads(row["record_json"]))
        if (row["disposition"] == "invalid"
                or scope(record) != (row["room_id"], row["producer_gateway_id"], row["producer_epoch"])
                or record["revision"] != row["revision"] or record["digest"] != row["digest"]):
            raise work.WorkRecordError("stored work metadata conflicts with record scope")
    except (ValueError, TypeError, KeyError, RecursionError) as exc:
        raise work.InvalidStoredWorkRecord("stored work evidence is invalid") from exc
    return record


def validate_stored_locked(conn, table, row):
    """Validate the consumed row; invalidate only its unchanged current version.

    The savepoint owns a standalone classification or joins the caller's write
    transaction. A caller catching InvalidStoredWorkRecord must commit that
    transaction before reporting controlled failure. Historical rows are never
    rewritten, and read-only projections use validate_stored instead.
    """
    from gateway import hosted_room_work_records as work
    try:
        return validate_stored(row)
    except work.InvalidStoredWorkRecord:
        if row["disposition"] == "current":
            # Match all examined values, not merely a key that may since have
            # been replaced. IS preserves NULL/BLOB distinctions in old stores.
            fields = list(row.keys())
            where = " AND ".join(f'"{k}" IS ?' for k in fields)
            conn.execute("SAVEPOINT work_record_invalidate")
            try:
                conn.execute(f"UPDATE {table} SET disposition='invalid' WHERE {where}", [row[k] for k in fields])
            except BaseException:
                conn.execute("ROLLBACK TO work_record_invalidate")
                conn.execute("RELEASE work_record_invalidate")
                raise
            conn.execute("RELEASE work_record_invalidate")
        raise


def scope(record):
    return record["room_id"], record["authority"]["gateway_id"], record["authority"]["epoch"]


def save_locked(conn, table, record, *, target_install_id=None, route_generation=None):
    from gateway import hosted_room_work_records as work
    data = work.encode(record)
    fields = ["room_id", "producer_gateway_id", "producer_epoch", "revision", "digest", "record_json", "disposition"]
    values = [*scope(record), record["revision"], record["digest"], data, "current"]
    keys = fields[:3]
    if target_install_id is not None:
        fields += ["target_install_id", "route_generation", "status"]
        values += [target_install_id, route_generation, "pending"]
        keys += ["target_install_id"]
    work._budget(conn, table, dict(zip(fields, values)))
    updates = ','.join(f'{k}=excluded.{k}' for k in fields if k not in keys)
    conn.execute(f"INSERT INTO {table} ({','.join(fields)}) VALUES ({','.join('?' for _ in fields)}) "
                 f"ON CONFLICT({','.join(keys)}) DO UPDATE SET {updates}", values)
