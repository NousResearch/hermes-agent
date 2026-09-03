"""Named-profile agents must FAIL CLOSED when their profile state.db won't open.

Desktop Bot Mode / app-global remote mode talks to a single TUI backend while
stamping ``profile_home`` for a named profile. The deferred agent build opens
that profile's ``state.db`` and hands it to ``_make_agent``. If that open used
to fail, the build silently fell back to the launch handle (``session_db =
None`` → ``_make_agent``'s ``_get_db()`` default), so the session's rows and
messages bled into the wrong profile's store exactly when the profile store
was briefly unopenable — and opening the named profile looked blank.

These tests pin the fail-closed contract: an unopenable profile store raises a
clear error (no agent turn), and NO path — deferred build or the
``_init_session`` cwd hydration — touches the launch ``state.db`` instead.

#88532 made SessionStore follow HERMES_HOME; this covers the TUI/agent
SessionDB handle. Related to #87723 and #89789.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_state
from hermes_state import SessionDB


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


def _make_store_unopenable(profile: Path) -> Path:
    """A directory named state.db — SessionDB cannot open it. Real, no mocks."""
    store = profile / "state.db"
    store.mkdir()
    return store


def test_open_profile_session_db_returns_profile_handle(homes):
    root, profile = homes
    from tui_gateway import server

    db = server._open_profile_session_db(str(profile))
    try:
        db.create_session("20260820_tui_worker", "tui", profile_name="worker")
    finally:
        db.close()

    assert _ids(profile / "state.db") == {"20260820_tui_worker"}
    assert _ids(root / "state.db") == set()


def test_open_profile_session_db_raises_when_store_unopenable(homes):
    """A genuinely unopenable state.db (directory in its place) must raise."""
    root, profile = homes
    from tui_gateway import server

    _make_store_unopenable(profile)
    with pytest.raises(RuntimeError, match="profile session store unavailable"):
        server._open_profile_session_db(str(profile))
    assert _ids(root / "state.db") == set()


def test_open_profile_session_db_does_not_fallback_on_open_failure(homes, monkeypatch):
    """SessionDB constructor failure must raise, not reuse/return launch."""
    root, profile = homes
    from tui_gateway import server

    class Boom(Exception):
        pass

    def boom_db(**_kwargs):
        raise Boom("profile store unavailable")

    monkeypatch.setattr("hermes_state.SessionDB", boom_db)
    with pytest.raises(RuntimeError, match="profile session store unavailable") as ei:
        server._open_profile_session_db(str(profile))
    assert isinstance(ei.value.__cause__, Boom)
    assert _ids(root / "state.db") == set()


def test_deferred_build_fails_closed_when_profile_store_unopenable(homes, monkeypatch):
    """_start_agent_build must error out — never build against the launch DB.

    Before the fix, the deferred build swallowed the open failure
    (``except Exception: session_db = None``) and _make_agent bound the
    launch ``_get_db()`` handle: the agent ran, and every turn landed in the
    wrong profile's state.db. Now the build must record a clear agent_error,
    emit an error event, and never reach _make_agent.
    """
    root, profile = homes
    from tui_gateway import server

    _make_store_unopenable(profile)
    launch = SessionDB(db_path=root / "state.db")
    sid = "sid-worker-build"
    make_agent_calls: list[dict] = []
    events: list[tuple[str, str, dict | None]] = []

    monkeypatch.setattr(server, "_get_db", lambda: launch)
    monkeypatch.setattr(server, "_set_session_context", lambda *a, **kw: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda tokens: None)
    monkeypatch.setattr(
        server, "_emit", lambda event, _sid, payload=None: events.append((event, _sid, payload))
    )

    def fake_make_agent(_sid, _key, **kw):
        make_agent_calls.append(kw)
        return SimpleNamespace()

    monkeypatch.setattr(server, "_make_agent", fake_make_agent)

    session = {
        "session_key": "key-worker-build",
        "agent_ready": threading.Event(),
        "profile_home": str(profile),
    }
    with server._sessions_lock:
        server._sessions[sid] = session
    try:
        server._start_agent_build(sid, session)
        thread = session.get("_agent_build_thread")
        assert thread is not None
        thread.join(timeout=30)
        assert not thread.is_alive()
        assert session["agent_ready"].is_set()

        # FAIL CLOSED: no agent was built at all — especially not one bound
        # to the launch handle.
        assert make_agent_calls == []
        assert session.get("agent") is None
        assert "profile session store unavailable" in str(session.get("agent_error"))
        assert any(
            evt == "error" and "agent init failed" in str((payload or {}).get("message"))
            for evt, _s, payload in events
        )
        # And nothing bled into the launch store.
        assert _ids(root / "state.db") == set()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
        launch.close()


def test_init_session_skips_launch_db_when_profile_store_unopenable(homes, monkeypatch):
    """Sibling site: _init_session's cwd hydration must not fall back either.

    Before the fix, ``except Exception: db = _get_db()`` hydrated/persisted a
    named-profile session's cwd row against the launch state.db. _get_db is
    patched to a tripwire: any call proves the launch store was touched.
    """
    root, profile = homes
    from tui_gateway import server

    _make_store_unopenable(profile)
    sid = "sid-worker-init"

    def tripwire():
        raise AssertionError("launch _get_db() must not be touched for a named-profile session")

    monkeypatch.setattr(server, "_get_db", tripwire)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_start_notification_poller", lambda _sid, _s: threading.Event())
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **kw: None)
    monkeypatch.setattr(server, "_session_info", lambda _agent, _s=None: {})
    monkeypatch.setattr(server, "_schedule_mcp_late_refresh", lambda _sid, _agent: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "off")
    monkeypatch.setattr(server, "_load_memory_notifications", lambda: "off")

    try:
        server._init_session(
            sid,
            "key-worker-init",
            SimpleNamespace(),
            [],
            cwd=str(root),
            profile_home=str(profile),
        )
        with server._sessions_lock:
            assert sid in server._sessions
        assert _ids(root / "state.db") == set()
    finally:
        with server._sessions_lock:
            server._sessions.pop(sid, None)
