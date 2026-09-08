"""Shared full-retention accounting, including copy-first recovery evidence."""

from gateway.hosted_rooms_common import table_exists

RECOVERY_TABLE = "hosted_room_manual_recoveries"
RECOVERY_FIELDS = ("room_id", "recovery_id", "snapshot_id", "source_gateway_id", "source_epoch",
                   "target_gateway_id", "history_seq", "work_record_json", "created_at")


def recovery_size_sql(prefix=""):
    # Reserve status transitions; charge the actual serialized envelope overhead.
    return "22+" + "+".join(f"COALESCE(length(CAST({prefix}{k} AS BLOB)),0)" for k in RECOVERY_FIELDS)


def recovery_count_sql(prefix=""):
    data = prefix + "work_record_json"
    return f"CASE WHEN json_valid({data}) THEN MAX(1,COALESCE((SELECT SUM(json_array_length(value)) FROM json_each({data},'$.rows')),1)) ELSE 1 END"


def duplicate_credit_sql(prefix=""):
    """Credit each still-present exact original once, not each envelope entry."""
    from gateway import hosted_room_work_records as records
    from gateway import hosted_room_work_storage as storage
    from gateway.hosted_room_recovery_evidence import exact_row_sql, OBJECT
    data = prefix + "work_record_json"
    predicate = (f"{prefix}status='transferring' AND json_valid({data}) "
                 f"AND json_extract({data},'$.object')='{OBJECT}' AND json_extract({data},'$.version')=2")
    sizes, counts = [], []
    for table in (records.TARGET_TABLE, storage.INVALID_TABLE):
        source_scope = f" AND original.source_table='{records.TARGET_TABLE}'" if table == storage.INVALID_TABLE else ""
        match = f"""original.room_id={prefix}room_id {source_scope} AND EXISTS (
            SELECT 1 FROM json_each({data},'$.rows.{table}') item
            WHERE {exact_row_sql(table, prefix='original.')})"""
        sizes.append(f"(SELECT COALESCE(SUM({storage.row_size_sql(table, 'original.')}),0) FROM {table} original WHERE {match})")
        counts.append(f"(SELECT COUNT(*) FROM {table} original WHERE {match})")
    return tuple(f"CASE WHEN {predicate} THEN ({' + '.join(parts)}) ELSE 0 END" for parts in (sizes, counts))


def usage_sql(conn):
    from gateway import hosted_room_work_records as records
    from gateway import hosted_room_work_storage as storage
    total, count = storage.usage_sql((records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE, storage.INVALID_TABLE))
    if table_exists(conn, RECOVERY_TABLE):
        byte_credit, row_credit = duplicate_credit_sql("decision.")
        total += f" + (SELECT COALESCE(SUM(({recovery_size_sql('decision.')})-({byte_credit})),0) FROM {RECOVERY_TABLE} decision)"
        count += f" + (SELECT COALESCE(SUM(({recovery_count_sql('decision.')})-({row_credit})),0) FROM {RECOVERY_TABLE} decision)"
    return total, count


def contribution_sql(table, prefix):
    """Effective bytes/rows of one value, including its exact transfer credit."""
    from gateway import hosted_room_work_records as records
    from gateway import hosted_room_work_storage as storage
    from gateway.hosted_room_recovery_evidence import exact_row_sql, OBJECT
    if table == RECOVERY_TABLE:
        credits = duplicate_credit_sql(prefix)
        return tuple(f"({size})-({credit})" for size, credit in zip(
            (recovery_size_sql(prefix), recovery_count_sql(prefix)), credits))
    size = storage.row_size_sql(table, prefix)
    if table not in (records.TARGET_TABLE, storage.INVALID_TABLE):
        return size, "1"
    scope = f"AND {prefix}source_table='{records.TARGET_TABLE}'" if table == storage.INVALID_TABLE else ""
    # EXISTS counts an original once per decision, never once per repeated entry.
    # Include disposition and storage types: even a reserved-size change may
    # invalidate an exact copy and thus increase effective retained usage.
    credit = f"""(SELECT COUNT(*) FROM {RECOVERY_TABLE} decision
        WHERE decision.room_id={prefix}room_id {scope}
        AND decision.status='transferring' AND json_valid(decision.work_record_json)
        AND json_extract(decision.work_record_json,'$.object')='{OBJECT}'
        AND json_extract(decision.work_record_json,'$.version')=2
        AND EXISTS (SELECT 1 FROM json_each(decision.work_record_json,'$.rows.{table}') item
                    WHERE {exact_row_sql(table, prefix=prefix)}))"""
    return f"({size})*(1-({credit}))", f"1-({credit})"


def install_recovery_budget_guards(conn):
    """Persist the same accountant for direct SQL and older Python writers."""
    from gateway import hosted_room_work_records as records
    from gateway import hosted_room_work_storage as storage
    if not table_exists(conn, RECOVERY_TABLE):
        return
    total, count = usage_sql(conn)
    owners = (records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE, storage.INVALID_TABLE, RECOVERY_TABLE)
    for table in owners:
        keys = [row["name"] for row in conn.execute(f"PRAGMA table_info({table})") if row["pk"]]
        key = " AND ".join(f"prior.{k}=NEW.{k}" for k in keys)
        for operation in ("INSERT", "UPDATE"):
            name = f"trg_work_budget_{table}_{operation.lower()}"
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            new_bytes, new_rows = contribution_sql(table, "NEW.")
            old = contribution_sql(table, "OLD." if operation == "UPDATE" else "prior.")
            if operation == "INSERT":
                # BEFORE sees the row an INSERT OR REPLACE would remove. After
                # conflict resolution that evidence is gone and cannot be charged
                # correctly. Immutable replacements remain separately forbidden.
                old = tuple(f"COALESCE((SELECT {part} FROM {table} prior WHERE {key}),0)" for part in old)
            old_bytes, old_rows = old
            conn.execute(f"""CREATE TRIGGER {name} BEFORE {operation} ON {table}
                WHEN (SELECT (byte_delta>0 OR row_delta>0) AND
                    (({total})+byte_delta>{int(records.MAX_STORE_BYTES)}
                     OR ({count})+row_delta>{int(records.MAX_STORE_ROWS)})
                    FROM (SELECT ({new_bytes})-({old_bytes}) AS byte_delta,
                                 ({new_rows})-({old_rows}) AS row_delta))
                BEGIN SELECT RAISE(ABORT, 'work record storage is full'); END""")
