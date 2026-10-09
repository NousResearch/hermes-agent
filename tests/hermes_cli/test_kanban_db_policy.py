"""Opt-in lifecycle policy validation; all databases and markers are fixtures."""
import sqlite3

import pytest


def test_required_board_cannot_claim_without_marker(tmp_path):
    from hermes_cli.kanban_db_policy import PolicyViolation, validate_before_claim
    with sqlite3.connect(tmp_path / 'board.db') as conn:
        with pytest.raises(PolicyViolation, match='policy'):
            validate_before_claim(conn, 't_fixture', 'worker', required=True)


@pytest.fixture
def guarded(tmp_path):
    import hashlib
    import json
    conn = sqlite3.connect(tmp_path / 'board.db', isolation_level=None)
    conn.execute('CREATE TABLE fixture_policy(value TEXT)')
    conn.execute("CREATE TRIGGER fixture_guard BEFORE DELETE ON fixture_policy BEGIN SELECT RAISE(ABORT,'held'); END")
    marker = tmp_path / 'board.db.policy-required.json'
    conn.execute('CREATE TABLE kanban_execution_policy_required(marker TEXT NOT NULL)')
    conn.execute('INSERT INTO kanban_execution_policy_required VALUES(?)', (str(marker),))
    objects = {r[0]: hashlib.sha256(r[1].encode()).hexdigest() for r in conn.execute("SELECT name,sql FROM sqlite_master WHERE name IN ('fixture_policy','fixture_guard','kanban_execution_policy_required')")}
    marker.write_text(json.dumps({'version': '1', 'database': str(tmp_path / 'board.db'), 'required': True,
                                  'capabilities': objects, 'protected_tables': ['fixture_policy','kanban_execution_policy_required']}))
    conn.execute('BEGIN IMMEDIATE')
    yield conn, marker
    conn.rollback()
    conn.close()


def test_valid_opt_in_delegates_receipt_validation(guarded):
    from hermes_cli.kanban_db_policy import validate_before_claim
    conn, marker = guarded
    calls = []
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return True
    assert validate_before_claim(conn, 't_fixture', 'worker', verify_receipt=verify) is True
    assert calls[0][0] == (conn, 't_fixture', 'worker', 'claim')


@pytest.mark.parametrize('damage', ['marker', 'trigger', 'table', 'malformed', 'wrong_database'])
def test_guarded_board_fails_closed_on_missing_or_malformed_capability(guarded, damage):
    import json
    from hermes_cli.kanban_db_policy import PolicyViolation, validate_before_claim
    conn, marker = guarded
    if damage == 'marker':
        marker.unlink()
    elif damage == 'trigger':
        conn.execute('DROP TRIGGER fixture_guard')
    elif damage == 'table':
        conn.execute('DROP TABLE fixture_policy')
    elif damage == 'malformed':
        marker.write_text('{}')
    else:
        data = json.loads(marker.read_text())
        data['database'] = '/not/this/board.db'
        marker.write_text(json.dumps(data))
    with pytest.raises(PolicyViolation, match='policy'):
        validate_before_claim(conn, 't_fixture', 'worker', verify_receipt=lambda *a, **k: True)


def test_missing_receipt_verifier_never_defaults_to_approval(guarded):
    from hermes_cli.kanban_db_policy import PolicyViolation, validate_before_claim
    conn, marker = guarded
    with pytest.raises(PolicyViolation, match='policy'):
        validate_before_claim(conn, 't_fixture', 'worker')


def test_spawn_requires_latest_effective_assertion_and_exact_run(guarded):
    from hermes_cli.kanban_db_policy import PolicyViolation, validate_before_spawn
    conn, marker = guarded
    calls = []
    def verify(*args, **kwargs):
        calls.append((args, kwargs))
        return True
    latest = {'tools': ['web_search'], 'revision': 'now'}
    assert validate_before_spawn(conn, 't_fixture', 'worker', 42, effective=latest, verify_receipt=verify) is True
    assert calls[-1][0][-1] == 'spawn'
    assert calls[-1][1] == {'run_id': 42, 'effective': latest}
    with pytest.raises(PolicyViolation, match='policy'):
        validate_before_spawn(conn, 't_fixture', 'worker', 42, verify_receipt=verify)


def test_ordinary_connection_cannot_remove_policy_or_import_over_it(guarded):
    from hermes_cli.kanban_db_policy import protect_connection, guard_destructive_operation, PolicyViolation
    conn, marker = guarded
    assert protect_connection(conn) is True
    for statement in ('DROP TRIGGER fixture_guard', 'DROP TABLE fixture_policy',
                      'DELETE FROM kanban_execution_policy_required', "INSERT INTO fixture_policy VALUES('forged')"):
        with pytest.raises(sqlite3.DatabaseError):
            conn.execute(statement)
    with pytest.raises(PolicyViolation, match='policy'):
        guard_destructive_operation(conn, 'import')
