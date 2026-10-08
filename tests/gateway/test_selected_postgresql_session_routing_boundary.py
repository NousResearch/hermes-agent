"""Selected PostgreSQL must not create gateway SQLite/JSON routing artifacts."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionSource, SessionStore
from state_store import StateStoreConfigurationError
from state_store_runtime_readiness import PostgreSQLRuntimeActivationError, trap_state_db_opens


_PG_CONFIG = (
    "state_store:\n"
    "  backend: postgresql\n"
    "  postgresql:\n"
    "    dsn_env: HERMES_STATE_STORE_TEST_DSN\n"
)


def _source() -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="safe-route-chat",
        chat_name="Safe route",
        chat_type="dm",
        user_id="safe-route-user",
    )


def _selected_pg_home(tmp_path: Path, monkeypatch) -> Path:
    home = tmp_path / ".hermes" / "profiles" / "selected-pg"
    home.mkdir(parents=True)
    (home / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    return home


def _assert_no_gateway_artifacts(home: Path, sessions_dir: Path) -> None:
    assert not (home / "state.db").exists()
    assert not (sessions_dir / "sessions.json").exists()
    assert list(sessions_dir.glob("*.jsonl")) == [] if sessions_dir.exists() else True


def test_malformed_backend_config_fails_before_legacy_gateway_runtime(tmp_path, monkeypatch):
    """An invalid selection cannot be silently retried as a legacy route."""
    import state_store_runtime_readiness

    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("state_store:\n  backend: invalid\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    legacy_guard = Mock(side_effect=AssertionError("entered legacy routing after invalid config"))
    monkeypatch.setattr(state_store_runtime_readiness, "require_legacy_state_db_runtime", legacy_guard)

    with trap_state_db_opens(home) as opens:
        with pytest.raises(StateStoreConfigurationError, match="state_store.backend must be one of"):
            SessionStore(home / "sessions", GatewayConfig())

    legacy_guard.assert_not_called()
    assert opens == []
    _assert_no_gateway_artifacts(home, home / "sessions")


def test_constructor_does_not_demote_to_sqlite_when_second_config_read_fails(tmp_path, monkeypatch):
    """A successful initial guard cannot turn a later config failure into a SQLite open."""
    from hermes_cli import config as config_module

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    sessions_dir = home / "sessions"
    original_load = config_module.load_config
    calls = 0

    def load_then_fail():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("second config read failed")
        return original_load()

    monkeypatch.setattr(config_module, "load_config", load_then_fail)
    with trap_state_db_opens(home) as opens:
        with pytest.raises(RuntimeError, match="second config read failed"):
            SessionStore(sessions_dir, GatewayConfig())

    assert calls == 2
    assert opens == []
    _assert_no_gateway_artifacts(home, sessions_dir)


def test_selected_postgresql_index_load_does_not_demote_on_second_config_failure(tmp_path, monkeypatch):
    """A failed re-probe must not initialize the legacy routing index or its directory."""
    from hermes_cli import config as config_module

    home = _selected_pg_home(tmp_path, monkeypatch)
    sessions_dir = home / "routing-index-probe-sessions"
    with trap_state_db_opens(home) as opens:
        store = SessionStore(sessions_dir, GatewayConfig())
    assert opens == []
    assert not sessions_dir.exists()

    original_load = config_module.load_config
    failure = RuntimeError("index load second PostgreSQL config probe failed")
    calls = 0

    def load_then_fail():
        nonlocal calls
        calls += 1
        if calls == 2:
            raise failure
        return original_load()

    monkeypatch.setattr(config_module, "load_config", load_then_fail)
    with trap_state_db_opens(home) as opens:
        with pytest.raises(RuntimeError) as error:
            store._ensure_loaded()

    assert error.value is failure
    assert calls == 2
    assert opens == []
    assert not sessions_dir.exists()
    _assert_no_gateway_artifacts(home, sessions_dir)


def test_selected_postgresql_session_store_succeeds_then_fails_closed_on_route(tmp_path, monkeypatch):
    home = _selected_pg_home(tmp_path, monkeypatch)
    sessions_dir = home / "sessions"

    # The constructor must succeed under selected PG without probing or opening
    # a root/profile state.db: the legacy SessionDB open is skipped.
    with trap_state_db_opens(home) as opens:
        store = SessionStore(sessions_dir, GatewayConfig())
        assert opens == []

    # Route resolution connects to the selected PG store; an unreachable server
    # fails closed with the typed connection error instead of a SQLite fallback.
    from psycopg import OperationalError

    with trap_state_db_opens(home) as opens:
        with pytest.raises(OperationalError):
            store.get_or_create_session(_source())

    assert opens == []
    _assert_no_gateway_artifacts(home, sessions_dir)


def test_selected_postgresql_gateway_runner_refuses_before_transport_or_route_artifacts(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.platforms.base import BasePlatformAdapter

    home = _selected_pg_home(tmp_path, monkeypatch)
    sender_calls = []
    sessions_dir = home / "sessions"

    async def fake_send(*args, **kwargs):
        sender_calls.append((args, kwargs))

    monkeypatch.setattr(BasePlatformAdapter, "send", fake_send)

    with trap_state_db_opens(home) as opens:
        with pytest.raises(PostgreSQLRuntimeActivationError):
            GatewayRunner(GatewayConfig(sessions_dir=sessions_dir))

    assert sender_calls == []
    assert opens == []
    _assert_no_gateway_artifacts(home, sessions_dir)


def test_selected_named_profile_never_falls_back_to_root_sqlite_when_postgresql_unavailable(tmp_path, monkeypatch):
    """A store created at root must re-check the active named profile per route."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    root = tmp_path / ".hermes"
    root.mkdir()
    profile = root / "profiles" / "selected-pg"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(_PG_CONFIG, encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_STATE_STORE_TEST_DSN", "postgresql://fixture/only")
    # The conftest sandbox pins DEFAULT_DB_PATH to its own hermes_test home;
    # this test's root store must open root/state.db instead (re-pin like the
    # conftest does, so batch ordering cannot divert the constructor's open).
    import hermes_state as _hermes_state
    monkeypatch.setattr(_hermes_state, "DEFAULT_DB_PATH", root / "state.db")
    sessions_dir = root / "sessions"
    store = SessionStore(sessions_dir, GatewayConfig())
    root_db_size = (root / "state.db").stat().st_size

    token = set_hermes_home_override(str(profile))
    try:
        from psycopg import OperationalError
        with trap_state_db_opens(root, profile) as opens:
            # Connection failure is propagated; selected PG must never demote to the
            # root SessionDB when a named profile has no reachable server.
            with pytest.raises(OperationalError):
                store.get_or_create_session(_source())
            with pytest.raises(PostgreSQLRuntimeActivationError):
                store.append_to_transcript("must-not-spool", {"role": "user", "content": "blocked"})
            with pytest.raises(PostgreSQLRuntimeActivationError):
                store.rewrite_transcript("must-not-rewrite", [])
            with pytest.raises(PostgreSQLRuntimeActivationError):
                store.rewind_session("must-not-rewind")
            # Lifecycle recovery walks the legacy in-memory index, which is empty
            # under selected PG (routes live in the PG table): a clean no-op with
            # zero state.db/sessions.json side effects.
            assert store.recover_interrupted_turns() == 0
    finally:
        reset_hermes_home_override(token)

    assert opens == []
    assert (root / "state.db").stat().st_size == root_db_size
    assert not (sessions_dir / "sessions.json").exists()
    assert not (profile / "state.db").exists()


def test_default_sqlite_session_store_still_routes_and_persists(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    sessions_dir = home / "sessions"

    store = SessionStore(sessions_dir, GatewayConfig())
    entry = store.get_or_create_session(_source())

    assert entry.session_id
    assert (home / "state.db").exists()
    assert (sessions_dir / "sessions.json").exists()
