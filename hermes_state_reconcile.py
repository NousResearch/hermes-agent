"""Explicit offline repair of additive state.db schema drift.

Browsing stays read-only. Reconcile a protected snapshot so failed DDL never
leaves the user's store partly upgraded or escalates into FTS surgery.
"""
import sqlite3
from contextlib import closing
from pathlib import Path


def schema_drift_reason(path: Path):
    """Return a missing declared table/column, not a corruption diagnosis."""
    from hermes_state_schema import schema_read_probe_statements
    from hermes_cli.sqlite_safe_read import connect_tracked

    with closing(connect_tracked(path.resolve().as_uri() + '?mode=ro', uri=True, tracking_path=path)) as conn:
        try:
            for statement in schema_read_probe_statements():
                conn.execute(statement).fetchone()
        except sqlite3.OperationalError as exc:
            from hermes_state_repair import _schema_not_built
            if _schema_not_built(exc):
                return str(exc)
            raise
    return None


def _reconcile_snapshot(path, guard):
    from hermes_state_repair import (
        _copy_database_snapshot, _db_opens_cleanly, _repair_conn,
        _repair_scratch_space_error, _unlink_db_triple,
    )
    from hermes_state_schema import SessionSchemaMixin, reconcile_state_schema

    scratch = path.with_name(path.name + '.reconcile-scratch')
    if error := _repair_scratch_space_error(path):
        raise RuntimeError(error)
    if error := _unlink_db_triple(scratch):
        raise RuntimeError(error)
    try:
        _copy_database_snapshot(path, scratch, source_connection=guard)
        with _repair_conn(scratch) as conn:
            _require_history_tables(conn)
            # Old indexed columns must exist before the canonical DDL can create
            # their indexes. Reuse the column owner; the final probe fails closed
            # if an additive change is not possible. Missing tables are created below.
            object.__new__(SessionSchemaMixin)._reconcile_columns(conn.cursor())
            reconcile_state_schema(conn)
            conn.commit()
        if reason := schema_drift_reason(scratch):
            raise RuntimeError(f'schema reconciliation incomplete: {reason}')
        if reason := _db_opens_cleanly(scratch):
            raise RuntimeError(f'reconciled snapshot is unhealthy: {reason}')
        _copy_database_snapshot(scratch, path, destination_connection=guard)
    finally:
        if error := _unlink_db_triple(scratch):
            import logging
            logging.getLogger(__name__).warning('Could not remove reconciliation snapshot: %s', error)


def _require_history_tables(conn):
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if {'sessions', 'messages'} - tables:
        raise RuntimeError('canonical history tables are missing; use sessions recover, not additive reconciliation')


def _reconcile_locked(path, backup, report):
    from hermes_state_repair import (
        _backup_db_file, _db_opens_cleanly, _exclusive_repair_db_guard,
        _live_writer_holds_db,
    )

    if _live_writer_holds_db(path):
        raise RuntimeError('a live writer still holds state.db; stop it and retry')
    from hermes_cli.sqlite_safe_read import connect_tracked
    with closing(connect_tracked(path.resolve().as_uri() + '?mode=ro', uri=True, tracking_path=path)) as conn:
        _require_history_tables(conn)
    if schema_drift_reason(path) is None:
        report.update(repaired=True, strategy='already_healthy')
        return
    # Additive repair is not a substitute for corruption recovery.
    if reason := _db_opens_cleanly(path):
        raise RuntimeError(reason)
    if backup:
        bpath, error = _backup_db_file(path)
        if bpath is None:
            raise RuntimeError(f'pre-repair backup refused: {error}')
        report['backup_path'] = str(bpath)
    with _exclusive_repair_db_guard(path) as (guard, error):
        if guard is None:
            raise RuntimeError(f'could not acquire exclusive state.db repair ownership: {error}')
        _reconcile_snapshot(path, guard)
        report.update(repaired=True, strategy='reconcile_declarative_schema')


def repair_declarative_schema(path: Path, *, backup: bool = True):
    """Reuse repair's snapshot/SQLite exclusion beneath the runtime owner lock."""
    from gateway.runtime_ownership import OwnershipConflict, exclusive_maintenance
    from hermes_state_repair import _cross_process_repair_lock

    report = {'repaired': False, 'strategy': None, 'backup_path': None, 'error': None}
    try:
        with exclusive_maintenance([path.parent]):
            with _cross_process_repair_lock(path) as holding:
                if not holding:
                    raise RuntimeError('could not obtain the state.db repair lock')
                _reconcile_locked(path, backup, report)
    except (OwnershipConflict, OSError, sqlite3.Error, RuntimeError, TimeoutError) as exc:
        report['error'] = str(exc)
    return report
