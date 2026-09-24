"""ACP-facing facade contracts: update_session_meta / replace_messages passthrough."""
from __future__ import annotations


import json
import uuid

import pytest

from cli_session_store import PostgreSQLCLISessionCapabilityError, open_cli_session_store
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_DSN = "postgresql://hermes_state_store_test@127.0.0.1:5432/hermes_state_store_test"
_CONFIG = {"state_store": {"backend": "postgresql", "postgresql": {
    "dsn_env": "HERMES_STATE_STORE_TEST_DSN", "connect_timeout_seconds": 5, "pool_max_size": 2,
}}}

pytestmark = pytest.mark.integration


@pytest.fixture
def pg_cli_home(tmp_path, monkeypatch, postgresql_test_target):
    home = tmp_path / ".hermes-dev-postgresql-acp-facade"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n    dsn_env: HERMES_STATE_STORE_TEST_DSN\n    connect_timeout_seconds: 5\n    pool_max_size: 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", _DSN)
    import state_store
    monkeypatch.setattr(state_store, "postgresql_tenant_schema", lambda *_args, **_kwargs: postgresql_test_target.schema)
    token = set_hermes_home_override(str(home))
    stores = []
    try:
        yield home, stores, postgresql_test_target
    finally:
        for store in stores:
            try:
                store.close()
            except Exception:
                pass
        reset_hermes_home_override(token)


def _open(stores):
    store = open_cli_session_store(_CONFIG)
    stores.append(store)
    return store


def _fresh_session_id() -> str:
    return f"acp_{uuid.uuid4().hex[:12]}"


def _rows(target, sql, params=()):
    """Read rows through the ownership-validated test target (marker-gated)."""
    with target.connect() as connection, connection.cursor() as cursor:
        cursor.execute(sql, params)
        return list(cursor.fetchall())


def test_update_session_meta_passthrough_updates_config_and_model(pg_cli_home):
    _home, stores, _target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp", model="model-old")

    facade.update_session_meta(session_id, json.dumps({"cwd": "/x", "api_mode": "chat_completions"}), "model-x")

    session = facade.get_session(session_id)
    assert session is not None
    config = session["model_config"]
    if isinstance(config, str):
        config = json.loads(config)
    assert config["cwd"] == "/x"
    assert session["model"] == "model-x"


def test_update_session_meta_none_model_preserves_existing_model(pg_cli_home):
    _home, stores, _target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp", model="model-keep")

    facade.update_session_meta(session_id, json.dumps({"cwd": "/y"}), None)

    session = facade.get_session(session_id)
    config = session["model_config"]
    if isinstance(config, str):
        config = json.loads(config)
    assert config["cwd"] == "/y"
    assert session["model"] == "model-keep"


def test_replace_messages_active_only_preserves_archived_rows(pg_cli_home):
    _home, stores, target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp")

    facade.append_message(session_id, "user", "archived q1")
    facade.append_message(session_id, "assistant", "archived a1")
    facade.append_message(session_id, "user", "live q2")
    facade.append_message(session_id, "assistant", "live a2")

    # Soft-archive the first generation (rewind-style, exactly what pre-compaction
    # turns look like to ACP's non-owning replace).
    target.execute(
        f"UPDATE {target.schema}.messages SET active = false "
        f"WHERE session_id = %s AND content IN ('archived q1', 'archived a1')",
        (session_id,),
    )

    facade.replace_messages(session_id, [
        {"role": "user", "content": "replaced q"},
        {"role": "assistant", "content": "replaced a"},
    ], active_only=True)

    # Active rows were replaced with the new content...
    active_rows = _rows(target,
        f"SELECT role, content FROM {target.schema}.messages WHERE session_id = %s AND active ORDER BY id",
        (session_id,))
    assert [(row[0], row[1]) for row in active_rows] == [
        ("user", "replaced q"), ("assistant", "replaced a")]

    # ...and the previously soft-archived rows survived the rewrite.
    archived_rows = _rows(target,
        f"SELECT role, content FROM {target.schema}.messages WHERE session_id = %s AND NOT active ORDER BY id",
        (session_id,))
    assert [(row[0], row[1]) for row in archived_rows] == [
        ("user", "archived q1"), ("assistant", "archived a1")]


def test_replace_messages_default_mode_deletes_archived_rows_too(pg_cli_home):
    _home, stores, target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp")

    facade.append_message(session_id, "user", "old q")
    facade.append_message(session_id, "assistant", "old a")
    target.execute(
        f"UPDATE {target.schema}.messages SET active = false WHERE session_id = %s",
        (session_id,),
    )

    # active_only is forwarded verbatim: the destructive default clears archived
    # rows as well, unlike the ACP mode above.
    facade.replace_messages(session_id, [
        {"role": "user", "content": "fresh q"},
    ], active_only=False)

    all_rows = _rows(target,
        f"SELECT role, content, active FROM {target.schema}.messages WHERE session_id = %s ORDER BY id",
        (session_id,))
    assert [(row[0], row[1], row[2]) for row in all_rows] == [("user", "fresh q", True)]


def test_replace_messages_rejects_unknown_control(pg_cli_home):
    _home, stores, _target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp")

    with pytest.raises(PostgreSQLCLISessionCapabilityError):
        facade.replace_messages(session_id, [], unexpected_kwarg=1)


def test_get_messages_as_conversation_repairs_alternation_on_request(pg_cli_home):
    _home, stores, _target = pg_cli_home
    facade = _open(stores)
    session_id = _fresh_session_id()
    facade.create_session(session_id, "acp")

    # A durable user;user violation — exactly the shape repair_alternation exists for.
    facade.append_message(session_id, "user", "first question")
    facade.append_message(session_id, "user", "second question")
    facade.append_message(session_id, "assistant", "answer")

    repaired = facade.get_messages_as_conversation(session_id, repair_alternation=True)
    assert [message["role"] for message in repaired] == ["user", "assistant"]
    assert "first question" in repaired[0]["content"]
    assert "second question" in repaired[0]["content"]

    # The stored transcript is never mutated by the repair pass.
    raw = facade.get_messages_as_conversation(session_id)
    assert [message["role"] for message in raw] == ["user", "user", "assistant"]
    assert [message["content"] for message in raw] == [
        "first question", "second question", "answer"]
