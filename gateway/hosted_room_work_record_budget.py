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


def install_recovery_budget_guards(conn):
    """Persist the same accountant for direct SQL and older Python writers."""
    from gateway import hosted_room_work_records as records
    from gateway import hosted_room_work_storage as storage
    if not table_exists(conn, RECOVERY_TABLE):
        return
    total, count = usage_sql(conn)
    owners = (records.SOURCE_TABLE, records.TARGET_TABLE, records.PENDING_TABLE, storage.INVALID_TABLE, RECOVERY_TABLE)
    for table in owners:
        for operation in ("INSERT", "UPDATE"):
            name = f"trg_work_budget_{table}_{operation.lower()}"
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            # Updates preserving all evidence bytes may still freeze dispositions
            # or clean up outcomes in an already-full database. No metadata growth.
            fields = (*RECOVERY_FIELDS, "status") if table == RECOVERY_TABLE else storage.retained_fields(table)
            unchanged = " AND ".join(f"NEW.{k} IS OLD.{k} AND typeof(NEW.{k})=typeof(OLD.{k})" for k in fields)
            exempt = f"AND NOT ({unchanged})" if operation == "UPDATE" else ""
            conn.execute(f"""CREATE TRIGGER {name} AFTER {operation} ON {table}
                WHEN (({total})>{int(records.MAX_STORE_BYTES)} OR ({count})>{int(records.MAX_STORE_ROWS)}) {exempt}
                BEGIN SELECT RAISE(ABORT, 'work record storage is full'); END""")
