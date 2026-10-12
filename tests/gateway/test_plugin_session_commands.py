"""Real native plugin dispatch at the messaging session/profile boundary."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone, UTC
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.run_session_commands import dispatch_gateway_session_command
from gateway.session import AsyncSessionStore, SessionEntry, SessionSource, SessionStore
from hermes_cli import plugins
from hermes_state import SessionDB

POLICY_KEY = "fixture_mode_override"
FROZEN_PLUGIN = '''
from threading import get_ident
from hermes_cli.session_commands import SessionCommandRequest

KEY = "fixture_mode_override"
USAGE = "Usage: /session-mode on|off|inherit|status"

def prepare(raw):
    words = raw.lower().strip().split()
    if len(words) != 1 or words[0] not in ("on", "off", "inherit", "status"):
        raise ValueError(USAGE)
    return SessionCommandRequest(words[0] != "status", (words[0], get_ident()))

def execute(request, session):
    command, parse_thread = request.data
    config = session.get_config()
    if request.write:
        session.set_override(KEY, {"on": True, "off": False, "inherit": None}[command])
    override = session.get_override(KEY)
    enabled = config["fixture_session"]["enabled"] if override is None else override
    return (f"state={enabled}; override={override}; id={session.session_id}; "
            f"live={session.live}; home={session.profile_home}; parse_thread={parse_thread}; "
            f"run_thread={get_ident()}")

def register(ctx):
    ctx.register_session_policy(
        KEY, config_path="fixture_session.enabled",
        apply=lambda runtime, enabled: runtime.set_context_option("fixture_enabled", enabled))
    ctx.register_session_command(
        "session-mode", execute, prepare=prepare, description="Control this session",
        args_hint="[on|off|inherit|status]", subcommands=("on", "off", "inherit", "status"),
        advertise_in_gateway_menu=False)
    ctx.register_command("legacy-echo", lambda raw: "legacy:" + raw, description="Echo text")
'''


def _install_fixture(home, *, enabled=False):
    directory = home / "plugins" / "frozen-session"
    directory.mkdir(parents=True)
    (directory / "plugin.yaml").write_text(
        "name: frozen-session\nversion: 0.1.0\n", encoding="utf-8")
    (directory / "__init__.py").write_text(FROZEN_PLUGIN, encoding="utf-8")
    (home / "config.yaml").write_text(
        "plugins:\n  enabled: [frozen-session]\nfixture_session:\n  enabled: "
        + ("true" if enabled else "false") + "\n", encoding="utf-8")


@pytest.fixture
def plugin_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    _install_fixture(home)
    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    plugins._reset_plugin_managers_for_tests()
    plugins.discover_plugins(force=True)
    loaded = plugins.get_plugin_manager()._plugins["frozen-session"]
    assert loaded.error is None
    assert loaded.module is not None
    yield home
    plugins._reset_plugin_managers_for_tests()


@pytest.fixture
def db(plugin_home):
    value = SessionDB(db_path=plugin_home / "state.db")
    yield value
    value.close()


def _source(chat_id="chat-a", *, profile=None, user_id="user"):
    return SessionSource(
        platform=Platform.TELEGRAM, user_id=user_id, chat_id=chat_id,
        chat_type="dm", thread_id="topic-a", profile=profile)


def _event(args="on", *, source=None, name="session-mode"):
    return MessageEvent(
        text=f"/{name} {args}".strip(), source=source or _source(),
        message_type=MessageType.TEXT, message_id="incoming-message", internal=True)


def _key(source):
    return f"route:{source.profile or 'default'}:{source.chat_id}"


class _Store:
    def __init__(self, entries):
        self.entries = dict(entries)
        self.create_calls = []
        self.peek_calls = []
        self._store = SimpleNamespace()

    async def get_or_create_session(self, source):
        self.create_calls.append(source)
        key = _key(source)
        session_id = self.entries.setdefault(key, "canonical-" + source.chat_id)
        now = datetime.now(UTC)
        return SessionEntry(
            session_key=key, session_id=session_id, created_at=now, updated_at=now,
            origin=source, platform=source.platform, chat_type=source.chat_type)

    async def peek_session_id(self, key, *, peek_only=False):
        self.peek_calls.append((key, peek_only))
        return self.entries.get(key)


class _Cache(dict):
    def __init__(self):
        super().__init__()
        self.reads = 0

    def get(self, *args, **kwargs):
        self.reads += 1
        return super().get(*args, **kwargs)


class _Agent:
    def __init__(self, db, session_id):
        self.session_id = session_id
        self._session_db = db
        self._session_init_model_config = {"keep": "live"}
        self.ensure_calls = 0
        self.engine_calls = []
        self.context_compressor = SimpleNamespace(set_session_option=self._set_option)

    def _set_option(self, name, enabled):
        row = self._session_db.get_session(self.session_id)
        override = SessionDB.session_policy_override(row, POLICY_KEY)
        self.engine_calls.append((name, enabled, override))
        return True

    def _ensure_db_session(self):
        self.ensure_calls += 1
        if self._session_db.get_session(self.session_id) is None:
            self._session_db.create_session(
                self.session_id, source="telegram", model_config={"keep": "durable"})


@pytest.fixture
def make_runner():
    created = []

    def make(db=None, *, entries=None, extra=None, multiplex=False):
        runner = object.__new__(GatewayRunner)
        runner.config = GatewayConfig(
            multiplex_profiles=multiplex,
            platforms={Platform.TELEGRAM: PlatformConfig(enabled=True, extra=extra or {})})
        store = _Store(entries or {})
        runner.session_store = store._store
        runner._async_session_store = store
        runner._session_db = SimpleNamespace(_db=db) if db is not None else None
        runner._agent_cache = _Cache()
        runner._agent_cache_lock = threading.RLock()
        runner._session_key_for_source = _key
        runner._draining = False
        runner._profile_configs = {}
        runner._primary_profile_name = "default"
        runner.adapters = {}
        runner.hooks = SimpleNamespace(
            emit=AsyncMock(return_value=None), emit_collect=AsyncMock(return_value=[]))
        created.append(runner)
        return runner

    yield make
    for runner in created:
        executor = getattr(runner, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)


def _override(db, session_id):
    return SessionDB.session_policy_override(db.get_session(session_id), POLICY_KEY)


@pytest.mark.asyncio
async def test_real_typed_gateway_dispatch_uses_canonical_cold_row(db, make_runner):
    runner = make_runner(db)
    event = _event("on")
    result = await runner._handle_message(event)

    assert "state=True; override=True" in result
    assert "live=False" in result
    session_id = runner.async_session_store.entries[_key(event.source)]
    row = db.get_session(session_id)
    assert _override(db, session_id) is True
    assert (row["source"], row["user_id"], row["chat_id"], row["chat_type"], row["thread_id"]) == (
        "telegram", "user", "chat-a", "dm", "topic-a")
    assert runner._agent_cache == {}
    assert f"parse_thread={threading.get_ident()};" not in result
    assert f"run_thread={threading.get_ident()}" not in result


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["direct", "tuple"])
async def test_cached_agent_uses_exact_row_and_db_before_live_apply(db, make_runner, shape):
    source = _source()
    runner = make_runner(db, entries={_key(source): "live-session"})
    agent = _Agent(db, "live-session")
    runner._agent_cache[_key(source)] = agent if shape == "direct" else (agent, "runtime-key")

    result = await runner._handle_message(_event("off", source=source))

    assert "state=False; override=False" in result
    assert "live=True" in result
    assert agent.ensure_calls == 1
    assert agent.engine_calls == [("fixture_enabled", False, False)]
    assert json.loads(db.get_session("live-session")["model_config"])["keep"] == "durable"


@pytest.mark.asyncio
async def test_read_request_peeks_only_and_never_reads_cache_or_ensures(db, make_runner, monkeypatch):
    source = _source()
    db.create_session("read-session", source="telegram", model_config={POLICY_KEY: True})
    runner = make_runner(db, entries={_key(source): "read-session"})
    runner._agent_cache[_key(source)] = _Agent(db, "unrelated-live-id")
    opened = []
    original_init = SessionDB.__init__

    def record_init(self, *args, **kwargs):
        opened.append((kwargs.get("db_path"), kwargs.get("read_only")))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SessionDB, "__init__", record_init)
    result = await runner._handle_message(_event("status", source=source))

    assert "state=True; override=True" in result
    assert runner.async_session_store.peek_calls == [(_key(source), True)]
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0
    assert opened == [(db.db_path, True)]
    assert _override(db, "read-session") is True


@pytest.mark.asyncio
async def test_status_without_route_never_creates_state_files(plugin_home, make_runner):
    runner = make_runner()
    initial_paths = set(plugin_home.rglob("*"))
    result = await runner._handle_message(_event("status"))

    assert "state=False; override=None; id=;" in result
    assert not (plugin_home / "state.db").exists()
    assert set(plugin_home.rglob("*")) == initial_paths
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("args", ["", "maybe", "on off", "status extra"])
async def test_invalid_request_parses_before_route_cache_and_db(plugin_home, make_runner, args):
    runner = make_runner()
    runner._session_key_for_source = lambda source: pytest.fail("must parse before route lookup")
    handler = plugins.get_plugin_command_handler("session-mode")
    result = await dispatch_gateway_session_command(runner, _event(args), handler)

    assert result == "Usage: /session-mode on|off|inherit|status"
    assert runner.async_session_store.create_calls == []
    assert runner.async_session_store.peek_calls == []
    assert runner._agent_cache.reads == 0
    assert not (plugin_home / "state.db").exists()


@pytest.mark.asyncio
async def test_config_failure_does_not_initialize_write_route(plugin_home, make_runner, monkeypatch):
    runner = make_runner()

    def fail_config():
        raise RuntimeError("configuration unavailable")

    monkeypatch.setattr("hermes_cli.session_commands.load_session_command_config", fail_config)
    handled, result, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        _event("on"), _source(), "session-mode")

    assert handled is True
    assert "configuration unavailable" in result
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0
    assert not (plugin_home / "state.db").exists()


@pytest.mark.asyncio
async def test_db_write_failure_is_handled_without_live_apply_or_model_fallthrough(db, make_runner, monkeypatch):
    source = _source()
    db.create_session("live-session", source="telegram")
    runner = make_runner(db, entries={_key(source): "live-session"})
    agent = _Agent(db, "live-session")
    runner._agent_cache[_key(source)] = agent

    def fail_write(*args):
        raise RuntimeError("durable write refused")

    monkeypatch.setattr(db, "set_session_policy_override", fail_write)
    handled, result = await runner._hm_dispatch_idle_commands(
        _event("on", source=source), source, _key(source))
    assert handled is True
    assert "durable write refused" in result
    assert _override(db, "live-session") is None
    assert agent.engine_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mismatch", ["identity", "profile"])
async def test_cached_foreign_identity_or_db_fails_closed(db, make_runner, tmp_path, mismatch):
    source = _source()
    runner = make_runner(db, entries={_key(source): "canonical"})
    foreign = None
    try:
        if mismatch == "profile":
            foreign_home = tmp_path / "foreign-home"
            foreign_home.mkdir()
            foreign = SessionDB(db_path=foreign_home / "state.db")
            agent = _Agent(foreign, "canonical")
        else:
            agent = _Agent(db, "foreign-session")
        runner._agent_cache[_key(source)] = agent
        result = await runner._handle_message(_event("on", source=source))
        assert f"{mismatch} mismatch" in result
        assert agent.ensure_calls == 0
        assert agent.engine_calls == []
        assert db.get_session("canonical") is None
        if foreign is not None:
            assert foreign.get_session("canonical") is None
    finally:
        if foreign is not None:
            foreign.close()


@pytest.mark.asyncio
async def test_unknown_plugin_branch_applies_real_slash_access_before_parse(db, make_runner):
    runner = make_runner(db, extra={"allow_admin_from": ["admin"], "user_allowed_commands": []})
    event = _event("on", name="session_mode")
    handled, result, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        event, event.source, "session_mode")
    assert handled is True
    assert "⛔" in result
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0
    assert db.get_session("canonical-chat-a") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["admin", "allowed-user"])
async def test_plugin_admin_and_explicitly_allowed_user_dispatch(db, make_runner, user_id):
    runner = make_runner(
        db, extra={"allow_admin_from": ["admin"], "user_allowed_commands": ["session-mode"]})
    result = await runner._handle_message(_event("on", source=_source(user_id=user_id)))
    assert "state=True; override=True" in result
    assert _override(db, "canonical-chat-a") is True


@pytest.mark.asyncio
async def test_session_plugin_bypasses_adapter_queue_but_busy_guard_rejects(db, make_runner):
    from hermes_cli.commands import should_bypass_active_session

    runner = make_runner(db)
    event = _event("on", name="session_mode")
    assert should_bypass_active_session("session_mode") is True
    handled, result = await runner._hm_busy_slash_or_photo(event, event.source, _key(event.source))
    assert handled is True
    assert "can't run mid-turn" in result
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0


@pytest.mark.asyncio
async def test_busy_session_plugin_keeps_slash_access_gate(db, make_runner):
    runner = make_runner(db, extra={"allow_admin_from": ["admin"]})
    event = _event("on")
    handled, result = await runner._hm_busy_slash_or_photo(event, event.source, _key(event.source))
    assert handled is True
    assert "⛔" in result
    assert runner.async_session_store.create_calls == []


@pytest.mark.asyncio
async def test_generic_dispatch_rechecks_busy_before_route_or_db(db, make_runner):
    runner = make_runner(db)
    runner._is_session_running = lambda key: True
    handler = plugins.get_plugin_command_handler("session-mode")
    result = await dispatch_gateway_session_command(runner, _event("on"), handler)
    assert "can't run mid-turn" in result
    assert runner.async_session_store.create_calls == []
    assert runner._agent_cache.reads == 0


@pytest.mark.asyncio
async def test_routed_cold_read_and_write_use_own_profile_home(db, plugin_home, make_runner):
    satellite = plugin_home / "profiles" / "satellite"
    _install_fixture(satellite, enabled=True)
    source = _source(profile="satellite")
    runner = make_runner(db, multiplex=True)
    runner._resolve_profile_home_for_source = lambda source: satellite
    runner._profile_configs["satellite"] = runner.config

    handled, status, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        _event("status", source=source), source, "session-mode")
    assert handled is True
    assert f"home={satellite}" in status
    assert "state=True; override=None" in status
    assert not (satellite / "state.db").exists()
    handled, result, _ = await runner._hm_dispatch_quick_and_plugin_commands(
        _event("off", source=source), source, "session-mode")
    assert handled is True
    assert "state=False; override=False" in result
    assert f"home={satellite}" in result
    assert db.get_session("canonical-chat-a") is None
    own_db = SessionDB(db_path=satellite / "state.db", read_only=True)
    try:
        assert _override(own_db, "canonical-chat-a") is False
        assert own_db.get_session("canonical-chat-a")["chat_id"] == "chat-a"
    finally:
        own_db.close()


@pytest.mark.asyncio
async def test_real_store_cold_status_peek_never_loads_or_repairs_index(plugin_home, make_runner):
    config = GatewayConfig(multiplex_profiles=False)
    store = SessionStore(sessions_dir=plugin_home / "sessions", config=config)
    store.close_all_db_handles()
    for path in (plugin_home / "state.db").parent.glob("state.db*"):
        path.unlink()
    runner = make_runner()
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_key_for_source = store._generate_session_key
    initial_paths = set(plugin_home.rglob("*"))
    result = await runner._handle_message(_event("status"))
    assert "state=False; override=None" in result
    assert store._loaded is False
    assert store._entries == {}
    assert set(plugin_home.rglob("*")) == initial_paths
    assert not (plugin_home / "state.db").exists()


@pytest.mark.asyncio
async def test_raw_plugin_handler_keeps_argument_contract(db, make_runner):
    runner = make_runner(db)
    result = await runner._handle_message(_event("Some RAW args", name="legacy-echo"))
    assert result == "legacy:Some RAW args"
    assert runner.async_session_store.create_calls == []
