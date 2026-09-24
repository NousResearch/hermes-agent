"""Real selected-PostgreSQL contract for ``hermes sessions import``."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from cli_session_store import open_cli_session_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store_runtime_readiness import trap_state_db_opens

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {
    "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
}}}
pytestmark = pytest.mark.integration


@pytest.fixture
def pg_home(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-pg-foreign-import"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    try:
        yield home
    finally:
        reset_hermes_home_override(token)


def _codex_file(tmp_path):
    path = tmp_path / ".codex" / "sessions" / "rollout-foreign.jsonl"
    path.parent.mkdir(parents=True)
    rows = [
        {"type": "session_meta", "payload": {"session_id": "foreign-codex-1", "cwd": "/tmp/foreign"}},
        {"type": "response_item", "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Import me"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Imported."}]}},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_selected_postgresql_foreign_import_is_atomic_idempotent_and_never_opens_sqlite(pg_home, monkeypatch, tmp_path, capsys):
    from hermes_cli.sessions_cmd import cmd_sessions

    path = _codex_file(tmp_path)
    args = SimpleNamespace(sessions_action="import", from_source="codex", path=str(path))
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: _CONFIG)
    with trap_state_db_opens(pg_home) as opens:
        assert cmd_sessions(args) is None
        first_output = capsys.readouterr().out
        assert cmd_sessions(args) is None
        second_output = capsys.readouterr().out
        first_id = first_output.split(" as ", 1)[1].splitlines()[0]
        second_id = second_output.split(" as ", 1)[1].splitlines()[0]
        assert first_id == second_id
        store = open_cli_session_store(_CONFIG)
        try:
            row = store.get_session(first_id)
            assert row["source"] == "codex-cli"
            assert row["cwd"] == "/tmp/foreign"
            assert row["last_activity_at"] is None
            assert row["last_activity_description"] == ""
            assert row["last_activity_provenance"] == "unknown"
            assert [message["role"] for message in store.get_messages(first_id)] == ["user", "assistant"]
            assert [message["content"] for message in store.get_messages(first_id)] == ["Import me", "Imported."]
            with store._store._connection() as connection, connection.cursor() as cursor:
                cursor.execute(f"SELECT session_id FROM {store._store._schema}.foreign_import_receipts")
                assert cursor.fetchall() == [(first_id,)]
        finally:
            store.close()
    assert opens == []
    assert not (pg_home / "state.db").exists()


def test_selected_postgresql_foreign_import_rejects_before_mutation(pg_home):
    store = open_cli_session_store(_CONFIG)
    try:
        before = (store.session_count(), store.message_count())
        with pytest.raises(ValueError, match="alternate roles"):
            store.import_foreign_history(
                {"tool": "codex-cli", "path": "/tmp/bad", "foreign_session_id": "bad"},
                [{"role": "user", "content": "one"}, {"role": "user", "content": "two"}],
                title="Bad import", cwd=None, profile=None,
            )
        assert (store.session_count(), store.message_count()) == before
    finally:
        store.close()


def test_selected_postgresql_foreign_import_rolls_back_on_storage_failure(pg_home):
    store = open_cli_session_store(_CONFIG)
    schema = store._store._schema
    try:
        before = (store.session_count(), store.message_count())
        with store._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f'''CREATE FUNCTION {schema}.reject_foreign_import_message() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN IF NEW.content = 'rollback' THEN RAISE EXCEPTION 'injected foreign import failure'; END IF; RETURN NEW; END $$''')
            cursor.execute(f"CREATE TRIGGER reject_foreign_import_message BEFORE INSERT ON {schema}.messages FOR EACH ROW EXECUTE FUNCTION {schema}.reject_foreign_import_message()")
        with pytest.raises(Exception, match="injected foreign import failure"):
            store.import_foreign_history(
                {"tool": "codex-cli", "path": "/tmp/rollback", "foreign_session_id": "rollback"},
                [{"role": "user", "content": "begin"}, {"role": "assistant", "content": "rollback"}],
                title="Rollback import", cwd=None, profile=None,
            )
        assert (store.session_count(), store.message_count()) == before
        with store._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {schema}.foreign_import_receipts WHERE origin_json->>'foreign_session_id' = 'rollback'")
            assert cursor.fetchone() == (0,)
    finally:
        with store._store._connection() as connection, connection.cursor() as cursor:
            cursor.execute(f"DROP TRIGGER IF EXISTS reject_foreign_import_message ON {schema}.messages")
            cursor.execute(f"DROP FUNCTION IF EXISTS {schema}.reject_foreign_import_message()")
        store.close()
