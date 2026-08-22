"""Named-profile turns must not persist into the launch/root state.db.

Desktop Bot Mode often talks to the default TUI process while stamping
``profile_name=worker``. The agent still holds the launch SessionDB handle, so
the row and messages land in the root store. The named profile's own state.db
never sees them, and opening that profile looks blank.

#88532 made SessionStore follow HERMES_HOME. This covers the TUI/agent handle
that is constructed once against the launch home and then reused.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import hermes_state
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB, session_db_for_named_profile


@pytest.fixture
def homes(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    profile = root / "profiles" / "worker"
    root.mkdir(parents=True)
    profile.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", hermes_state._IMPORT_DEFAULT_DB_PATH)
    return root, profile


def _ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    conn = sqlite3.connect(str(db_path))
    try:
        try:
            return {row[0] for row in conn.execute("SELECT id FROM sessions")}
        except sqlite3.OperationalError:
            return set()
    finally:
        conn.close()


def test_named_profile_create_does_not_land_in_launch_store(homes):
    root, profile = homes
    launch = SessionDB(db_path=root / "state.db")
    token = set_hermes_home_override(str(profile))
    try:
        db = session_db_for_named_profile(launch, "worker")
        db.create_session("20260820_tui_worker", "tui", profile_name="worker")
        if db is not launch:
            db.close()
    finally:
        reset_hermes_home_override(token)
        launch.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_default_profile_keeps_the_launch_handle(homes):
    root, _profile = homes
    launch = SessionDB(db_path=root / "state.db")
    try:
        assert session_db_for_named_profile(launch, None) is launch
        assert session_db_for_named_profile(launch, "default") is launch
    finally:
        launch.close()


def _stub_make_agent_runtime(monkeypatch, launch):
    from tui_gateway import server

    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_resolve_startup_runtime", lambda: ("test-model", None))
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": None,
            "base_url": None,
            "api_key": None,
            "api_mode": None,
            "command": None,
            "args": None,
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_reasoning_config", lambda model="": None)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda *_a, **_kw: None)
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    monkeypatch.setattr(server, "_agent_cbs", lambda _sid: {})
    return server


def test_agent_session_db_writes_to_profile_home_store(homes, monkeypatch):
    """Bot Mode rebuilds call _make_agent without a db; profile_home is the bind."""
    root, profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_home": str(profile)}
    try:
        db = server._agent_session_db(sid)
        db.create_session("20260820_tui_worker", "tui", profile_name="worker")
        if db is not launch:
            db.close()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_agent_session_db_without_profile_home_keeps_launch_handle(homes, monkeypatch):
    root, _profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-default"
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    with server._sessions_lock:
        server._sessions[sid] = {}
    try:
        assert server._agent_session_db(sid) is launch
        assert server._agent_session_db(sid, launch) is launch
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()


def test_make_agent_binds_named_profile_home_store(homes, monkeypatch):
    """_make_agent must not default a profile_home session to the launch store."""
    from unittest.mock import patch

    root, profile = homes
    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    server = _stub_make_agent_runtime(monkeypatch, launch)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_home": str(profile)}
    try:
        with patch("run_agent.AIAgent") as mock_agent:
            server._make_agent(sid, "key-worker")
        db = mock_agent.call_args.kwargs["session_db"]
        try:
            db.create_session("20260820_tui_worker", "tui", profile_name="worker")
        finally:
            if db is not launch:
                db.close()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_agent_session_db_retargets_explicit_launch_handle(homes, monkeypatch):
    root, profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_home": str(profile)}
    try:
        db = server._agent_session_db(sid, launch)
        assert db is not launch
        db.create_session("20260820_tui_worker", "tui", profile_name="worker")
        db.close()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_agent_session_db_binds_from_profile_name_without_home(homes, monkeypatch):
    root, profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_name": "worker"}
    try:
        db = server._agent_session_db(sid, launch)
        assert db is not launch
        db.create_session("20260820_tui_worker", "tui", profile_name="worker")
        db.close()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_agent_session_db_does_not_fallback_to_launch_on_open_failure(homes, monkeypatch):
    root, _profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)

    class Boom(Exception):
        pass

    def boom_db(**_kwargs):
        raise Boom("profile store unavailable")

    monkeypatch.setattr("hermes_state.SessionDB", boom_db)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_home": str(root / "profiles" / "worker")}
    try:
        with pytest.raises(Boom):
            server._agent_session_db(sid)
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()


def test_agent_session_db_name_only_does_not_fallback_to_launch_on_open_failure(
    homes, monkeypatch
):
    """Named-profile bind with only profile_name must raise, not reuse launch."""
    root, _profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)

    class Boom(Exception):
        pass

    def boom_db(**_kwargs):
        raise Boom("profile store unavailable")

    monkeypatch.setattr("hermes_state.SessionDB", boom_db)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_name": "worker"}
    try:
        with pytest.raises(Boom):
            server._agent_session_db(sid, launch)
        assert _ids(root / "state.db") == set()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()


def test_agent_session_db_name_only_does_not_fallback_on_profile_dir_failure(
    homes, monkeypatch
):
    """get_profile_dir / resolve failure must not bind the agent to launch."""
    root, _profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)

    class Boom(Exception):
        pass

    def boom_dir(_name):
        raise Boom("profile dir unavailable")

    monkeypatch.setattr("hermes_cli.profiles.get_profile_dir", boom_dir)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_name": "worker"}
    try:
        with pytest.raises(Boom):
            server._agent_session_db(sid, launch)
        assert _ids(root / "state.db") == set()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()


def test_agent_session_db_reuses_existing_handle_for_profile_home(homes, monkeypatch):
    """Deferred builds must keep the dedicated handle they already opened."""
    root, profile = homes
    from tui_gateway import server

    launch = SessionDB(db_path=root / "state.db")
    existing = SessionDB(db_path=profile / "state.db")
    sid = "sid-worker"
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    with server._sessions_lock:
        server._sessions[sid] = {"profile_home": str(profile)}
    db = None
    try:
        db = server._agent_session_db(sid, existing)
        assert db is existing
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        if db is not None and db is not existing:
            db.close()
        existing.close()
        launch.close()
