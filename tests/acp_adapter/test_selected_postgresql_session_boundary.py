"""ACP session persistence is backend-aware: selected PostgreSQL dispatches to
the CLI session-store facade instead of refusing; SQLite default is unchanged.

These tests deliberately do NOT assert on ``missing_capabilities`` — that tuple
is a static declaration in state_store_runtime_readiness.py owned by another
workstream; dispatch behavior here is proven by routing, not by declarations.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from acp_adapter.session import SessionManager
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from state_store_runtime_readiness import trap_state_db_opens


_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _selected_pg_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


class _FakePGStore:
    """Call-recording stand-in for the PostgreSQL CLI session-store facade."""

    def __init__(self, *, session_row=None, rich_rows=(), conversation=None):
        self.calls: list[tuple] = []
        self._session_row = session_row
        self._rich_rows = list(rich_rows)
        self._conversation = list(conversation or [])

    def _record(self, name, *args, **kwargs):
        self.calls.append((name, args, kwargs))

    def get_session(self, session_id):
        self._record("get_session", session_id)
        return self._session_row

    def create_session(self, session_id, source, **kwargs):
        self._record("create_session", session_id, source, **kwargs)
        if self._session_row is None:
            self._session_row = {"id": session_id, "source": source}
        return session_id

    ensure_session = create_session

    def update_session_meta(self, session_id, model_config_json, model=None):
        self._record("update_session_meta", session_id, model_config_json, model)

    def replace_messages(self, session_id, messages, **kwargs):
        self._record("replace_messages", session_id, list(messages), **kwargs)

    def get_messages_as_conversation(self, session_id, **kwargs):
        self._record("get_messages_as_conversation", session_id, **kwargs)
        return [dict(message) for message in self._conversation]

    def list_sessions_rich(self, **kwargs):
        self._record("list_sessions_rich", **kwargs)
        return list(self._rich_rows)

    def get_compression_tip(self, session_id):
        return session_id


def _names(fake: _FakePGStore) -> list[str]:
    return [call[0] for call in fake.calls]


def _install_fake_store(monkeypatch, fake: _FakePGStore) -> None:
    """Redirect the facade acquisition path to the recording fake.

    ``_selected_store`` imports ``open_cli_session_store`` lazily from
    ``cli_session_store``, so the patch target is the source module.
    """
    monkeypatch.setattr("cli_session_store.open_cli_session_store", lambda config: fake)


# ---------------------------------------------------------------------------
# (a) selected PG home: dispatch, not refusal; no SQLite artifacts
# ---------------------------------------------------------------------------


def test_selected_postgresql_manager_dispatches_without_sqlite_artifacts(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore()
    opened = []
    monkeypatch.setattr(
        "cli_session_store.open_cli_session_store",
        lambda config: opened.append(config) or fake,
    )
    factory_calls = []
    manager = SessionManager(agent_factory=lambda: factory_calls.append("agent") or SimpleNamespace(model="test"))

    with trap_state_db_opens(home) as opens:
        state = manager.create_session(cwd="/workspace")

    # create_session builds the agent eagerly (same contract as SQLite default)
    # and persists through the facade, never through state.db.
    assert factory_calls == ["agent"]
    assert opens == []
    assert not (home / "state.db").exists()
    assert state.cwd == "/workspace"
    assert opened, "selected PostgreSQL must open the CLI session-store facade"


def test_selected_postgresql_read_and_list_dispatch_without_sqlite_artifacts(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore()
    _install_fake_store(monkeypatch, fake)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))

    with trap_state_db_opens(home) as opens:
        assert manager.get_session("missing") is None
        assert manager.list_sessions() == []
        assert manager.update_cwd("missing", "/workspace") is None
        assert manager.fork_session("missing", "/workspace") is None

    assert opens == []
    assert not (home / "state.db").exists()
    # unknown-id lookups consulted the facade, not the refusal guard
    assert "get_session" in _names(fake)
    assert "list_sessions_rich" in _names(fake)


def test_selected_named_profile_dispatches_without_root_sqlite_session_reuse(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))
    root_state = manager.create_session()
    root_size = (root / "state.db").stat().st_size

    profile = root / "profiles" / "selected-pg"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    fake = _FakePGStore()
    _install_fake_store(monkeypatch, fake)
    # A fresh manager models a process restart under the PG profile: the root
    # SQLite session is not in memory, so restore must consult the facade.
    pg_manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))
    token = set_hermes_home_override(str(profile))
    try:
        with trap_state_db_opens(root, profile) as opens:
            restored = pg_manager.get_session(root_state.session_id)
    finally:
        reset_hermes_home_override(token)

    # The root SQLite session is NOT reused under the PG profile: the facade
    # reports no such row, so no state is restored and no root state.db open.
    assert restored is None
    assert "get_session" in _names(fake)
    assert opens == []
    assert (root / "state.db").stat().st_size == root_size
    assert not (profile / "state.db").exists()


# ---------------------------------------------------------------------------
# (b) persistence routes to the facade with SQLite-path call ordering
# ---------------------------------------------------------------------------


def test_selected_postgresql_persist_routes_create_then_replace_active_only(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore()
    _install_fake_store(monkeypatch, fake)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))
    state = manager.create_session(cwd="/workspace")

    state.history.append({"role": "user", "content": "persist me"})
    manager.save_session(state.session_id)

    names = _names(fake)
    assert "create_session" in names
    replace = [call for call in fake.calls if call[0] == "replace_messages"][-1]
    assert replace[1][0] == state.session_id
    assert replace[1][1] == [{"role": "user", "content": "persist me"}]
    assert replace[2] == {"active_only": True}
    assert names.index("create_session") < names.index("replace_messages")


def test_selected_postgresql_update_path_calls_update_session_meta(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore(session_row={"id": "known", "source": "acp",
                                     "model_config": json.dumps({"cwd": "/old"}), "model": "m1"})
    _install_fake_store(monkeypatch, fake)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="m1"))
    state = manager.get_session("known")
    assert state is not None
    assert state.cwd == "/old"

    manager.update_cwd("known", "/workspace")
    meta = [call for call in fake.calls if call[0] == "update_session_meta"][-1]
    assert meta[1][0] == "known"
    assert json.loads(meta[1][1])["cwd"] == "/workspace"


# ---------------------------------------------------------------------------
# (c) get_session/restore on PG reads through the facade
# ---------------------------------------------------------------------------


def test_selected_postgresql_restore_reads_conversation_with_repair(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch)
    conversation = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    fake = _FakePGStore(
        session_row={"id": "pg-session", "source": "acp", "model": "m",
                     "model_config": json.dumps({"cwd": "/workspace", "api_mode": "chat"})},
        conversation=conversation,
    )
    _install_fake_store(monkeypatch, fake)
    factory_calls = []
    manager = SessionManager(agent_factory=lambda: factory_calls.append("agent") or SimpleNamespace(model="rebuilt"))

    state = manager.get_session("pg-session")

    assert state is not None
    assert state.cwd == "/workspace"
    assert state.history == conversation
    assert factory_calls == ["agent"]  # restore rebuilds the agent, same as SQLite
    conv = [call for call in fake.calls if call[0] == "get_messages_as_conversation"][-1]
    assert conv[1][0] == "pg-session"
    assert conv[2].get("repair_alternation") is True


def test_selected_postgresql_restore_accepts_jsonb_mapping_model_config(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore(
        session_row={"id": "pg-jsonb", "source": "acp", "model": None,
                     "model_config": {"cwd": "/jsonb-workspace"}, "last_active": 1000.0},
        conversation=[{"role": "user", "content": "x"}],
    )
    _install_fake_store(monkeypatch, fake)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="rebuilt"))
    state = manager.get_session("pg-jsonb")
    assert state is not None
    assert state.cwd == "/jsonb-workspace"


def test_selected_postgresql_list_sessions_routes_rich_rows(tmp_path, monkeypatch):
    _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore(rich_rows=[{
        "id": "row-1", "message_count": 2, "model": "m", "model_config": {"cwd": "/workspace"},
        "title": "persisted", "preview": "hello", "last_active": 1000.0,
    }])
    _install_fake_store(monkeypatch, fake)
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))
    infos = manager.list_sessions(cwd="/workspace")
    assert [info["session_id"] for info in infos] == ["row-1"]
    rich = [call for call in fake.calls if call[0] == "list_sessions_rich"][-1]
    assert rich[2].get("source") == "acp"
    assert rich[2].get("limit") == 1000


# ---------------------------------------------------------------------------
# (d) SQLite-default regression on the same manager surface
# ---------------------------------------------------------------------------


def test_default_sqlite_acp_session_persists_and_restores(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    manager = SessionManager(agent_factory=lambda: SimpleNamespace(model="test"))
    state = manager.create_session(cwd="/workspace")
    state.history.append({"role": "user", "content": "persist me"})
    manager.save_session(state.session_id)

    restored = SessionManager(agent_factory=lambda: SimpleNamespace(model="test")).get_session(state.session_id)

    assert restored is not None
    assert restored.cwd == "/workspace"
    assert restored.history[0]["content"] == "persist me"
    assert (home / "state.db").exists()


@pytest.mark.asyncio
async def test_selected_postgresql_server_session_operations_dispatch_without_protocol_refusal(tmp_path, monkeypatch):
    pytest.importorskip("acp")
    from acp_adapter.server import HermesACPAgent

    home = _selected_pg_home(tmp_path, monkeypatch)
    fake = _FakePGStore()
    _install_fake_store(monkeypatch, fake)
    factory_calls = []
    manager = SessionManager(agent_factory=lambda: factory_calls.append("agent") or SimpleNamespace(model="test"))
    server = HermesACPAgent(session_manager=manager)
    sent = []
    server._conn = SimpleNamespace(session_update=lambda *args: sent.append(args))

    from state_store_runtime_readiness import PostgreSQLRuntimeActivationError

    with trap_state_db_opens(home) as opens:
        response = await server.new_session(cwd="/workspace")
        assert response.session_id
        assert await server.load_session(cwd="/workspace", session_id="missing") is None
        resume = await server.resume_session(cwd="/workspace", session_id="missing")
        assert resume.session_id
        assert await server.list_sessions() is not None
        await server.cancel(session_id="missing")
        assert await server.set_session_model("test", "missing") is None
        assert await server.set_session_mode("default", "missing") is None
        assert await server.set_config_option("option", "missing", "value") is None

    # new + resume (missing id falls back to create) built agents; the
    # missing-session settings paths did not.
    assert factory_calls == ["agent", "agent"]
    assert opens == []
    assert not (home / "state.db").exists()
    assert _names(fake), "server session operations must route through the facade"
