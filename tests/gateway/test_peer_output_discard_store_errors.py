"""SQLite faults are attributed to the statement's actual owning store."""
import sqlite3

import pytest
from tests.gateway.test_canonical_peer_target_setup import target  # noqa: F401
from tests.gateway.test_canonical_peer_files_target import files_target  # noqa: F401
from tests.gateway.peer_output_fixtures import peer_case
from tests.gateway.test_peer_output_discard import wire_discard
from tests.gateway.test_peer_output_fences import unretired
from gateway.session_results import admission_result
from tui_gateway.hosted_room_peer_http import PeerRunsHTTPError
from hermes_state import StateDbCorruptError


def corrupt_grant_table_in_transaction(conn):
    # Corrupt the actual store's schema only inside the already-started writer.
    # The unchanged production grant SELECT below raises SQLITE_CORRUPT; its
    # rollback repairs this disposable fixture, but must NOT clear quarantine.
    assert conn.in_transaction
    conn.execute('PRAGMA writable_schema=ON')
    conn.execute("UPDATE sqlite_master SET rootpage=999999 WHERE name='hosted_room_revoked_grant_tokens'")
    conn.execute('PRAGMA writable_schema=OFF')
    version = conn.execute('PRAGMA schema_version').fetchone()[0]
    conn.execute(f'PRAGMA schema_version={version + 1}')


@pytest.mark.asyncio
@pytest.mark.parametrize('phase', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('fault_kind', ['corrupt', 'busy', 'unavailable'])
async def test_shared_statement_failure_does_not_quarantine_owner(files_target, monkeypatch, phase, fault_kind):
    from gateway.platforms import api_server_room_artifacts as module
    async with peer_case(files_target, monkeypatch) as c:
        db = c.target.db
        require = module.require_current_grant
        execute = db._execute_write
        calls, actual_errors, boundary_errors = [], [], []
        def fail_shared(conn, claims):
            if conn is not db._conn:
                assert conn.in_transaction and db._conn.in_transaction
                calls.append(conn)
                if len(calls) == phase:
                    if fault_kind == 'corrupt':
                        corrupt_grant_table_in_transaction(conn)
                    try:
                        if fault_kind == 'busy':
                            raise sqlite3.OperationalError('database is locked')
                        if fault_kind == 'unavailable':
                            conn.execute('SELECT * FROM deliberately_unavailable_grant_page')
                        return require(conn, claims)
                    except sqlite3.Error as exc:
                        actual_errors.append(exc)
                        raise
            return require(conn, claims)
        def observe(fn, *args, **kwargs):
            try:
                return execute(fn, *args, **kwargs)
            except Exception as exc:
                boundary_errors.append(exc)
                raise
        with monkeypatch.context() as fault:
            fault.setattr(module, 'require_current_grant', fail_shared)
            fault.setattr(db, '_execute_write', observe)
            # Old SQLite busy retry would re-enter the callback; keep this
            # deterministic without waiting the owner's entire patience budget.
            fault.setattr(db, '_WRITE_PATIENCE_S', 0)
            with pytest.raises(PeerRunsHTTPError) as error:
                await wire_discard(c)
            assert error.value.status_code == 503 and error.value.retryable
        assert len(calls) == phase and len(actual_errors) == 1
        if fault_kind == 'corrupt':
            assert actual_errors[0].sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert not db._db_corrupt
        assert not isinstance(boundary_errors[-1], sqlite3.Error)
        assert boundary_errors[-1].__cause__ is actual_errors[0]
        # This is a real write through the same healthy SessionDB, not a reset
        # or a runner._draining stand-in for corruption.
        db._execute_write(lambda conn: conn.execute("INSERT INTO state_meta(key,value) VALUES('healthy-after-foreign-error','yes')"))
        assert db._conn.execute("SELECT value FROM state_meta WHERE key='healthy-after-foreign-error'").fetchone()[0] == 'yes'
        record = admission_result(db, c.row['admission_id'])
        if phase < 4:
            unretired(c)
            assert 'peer_output_discard' not in record
        else:
            assert record['peer_output_discard']['state'] == 'pending'
            assert record['peer_output_discard']['blobs']
        assert await wire_discard(c) == {'discarded': True, 'removed': 1}


@pytest.mark.asyncio
async def test_owner_origin_corrupt_select_still_quarantines_and_refuses(files_target, monkeypatch):
    from gateway.platforms import api_server_room_artifacts as module
    async with peer_case(files_target, monkeypatch) as c:
        db = c.target.db
        require = module.require_current_grant
        errors = []
        def fail_owner(conn, claims):
            if conn is db._conn:
                corrupt_grant_table_in_transaction(conn)
                try:
                    return require(conn, claims)
                except sqlite3.Error as exc:
                    errors.append(exc)
                    raise
            return require(conn, claims)
        with monkeypatch.context() as fault:
            fault.setattr(module, 'require_current_grant', fail_owner)
            with pytest.raises(PeerRunsHTTPError):
                await wire_discard(c)
        assert len(errors) == 1 and errors[0].sqlite_errorcode == sqlite3.SQLITE_CORRUPT
        assert db._db_corrupt
        with pytest.raises(StateDbCorruptError):
            db._execute_write(lambda conn: conn.execute("INSERT INTO state_meta(key,value) VALUES('must-not-land','no')"))
        assert db._conn.execute("SELECT value FROM state_meta WHERE key='must-not-land'").fetchone() is None
        unretired(c)
