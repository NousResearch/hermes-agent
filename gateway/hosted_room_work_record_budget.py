"""Shared work-evidence limits, including records retained during recovery."""

from gateway.hosted_rooms_common import table_exists

RECOVERY_TABLE = "hosted_room_manual_recoveries"


def stores(conn):
    from gateway import hosted_room_work_records as records
    result = [(records.SOURCE_TABLE, "record_json", ("room_id",)),
              (records.TARGET_TABLE, "record_json", ("room_id",)),
              (records.PENDING_TABLE, "record_json", ("room_id", "target_install_id"))]
    if table_exists(conn, RECOVERY_TABLE):
        result.append((RECOVERY_TABLE, "work_record_json", ("room_id",)))
    return result


def install_recovery_budget_guards(conn):
    """Persist enforcement for older writers whose Python budget knows only three stores."""
    from gateway import hosted_room_work_records as records
    owners = stores(conn)
    total = " + ".join(f"(SELECT COALESCE(SUM(length(CAST({column} AS BLOB))),0) FROM {table})"
                       for table, column, _keys in owners)
    count = " + ".join(f"(SELECT COUNT(*) FROM {table})" for table, _column, _keys in owners)
    for table, column, keys in owners:
        match = " AND ".join(f"{key}=NEW.{key}" for key in keys)
        old_bytes = f"COALESCE((SELECT length(CAST({column} AS BLOB)) FROM {table} WHERE {match}),0)"
        old_count = f"(SELECT COUNT(*) FROM {table} WHERE {match})"
        for operation in ("INSERT", "UPDATE"):
            credit = old_bytes if operation == "INSERT" else f"length(CAST(OLD.{column} AS BLOB))"
            growth = f"1 - {old_count}" if operation == "INSERT" else "0"
            name = f"trg_work_budget_{table}_{operation.lower()}"
            conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            conn.execute(f"""CREATE TRIGGER {name} BEFORE {operation} ON {table}
                WHEN ({total}) - ({credit}) + length(CAST(NEW.{column} AS BLOB)) > {int(records.MAX_STORE_BYTES)}
                  OR ({count}) + ({growth}) > {int(records.MAX_STORE_ROWS)}
                BEGIN SELECT RAISE(ABORT, 'work record storage is full'); END""")
