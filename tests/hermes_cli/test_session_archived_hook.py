"""``on_session_archived``: a deliberate archive notifies plugins once, after the flag commits.

Exercised end to end against a real ``SessionDB`` under the isolated test ``HERMES_HOME`` and the
real plugin hook registry, through every deliberate archive surface (Desktop/dashboard PATCH, TUI
``session.archive``, ``hermes sessions archive``). Housekeeping archives (idle sweep) stay silent,
as do repeats and un-archives, so a consumer can treat the event as the user's "done" signal.
"""

from __future__ import annotations

import argparse

import pytest

from hermes_cli.plugins import VALID_HOOKS, get_plugin_manager


@pytest.fixture
def archived_events():
    """Capture ``on_session_archived`` callbacks via the registry ``invoke_hook`` reads."""
    manager = get_plugin_manager()
    events: list[dict] = []
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}

    def _capture(**kwargs):
        # The flag is committed before observers run: a consumer may read it back.
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            kwargs["archived_at_dispatch"] = db.get_session(kwargs["session_id"])["archived"]
        finally:
            db.close()
        events.append(kwargs)

    manager._hooks.setdefault("on_session_archived", []).append(_capture)
    try:
        yield events
    finally:
        manager._hooks = saved


@pytest.fixture
def store(monkeypatch, _isolate_hermes_home):
    import hermes_state
    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    monkeypatch.setattr(hermes_state, "DEFAULT_DB_PATH", get_hermes_home() / "state.db")
    db = SessionDB()
    for sid in ("done-chat", "other-chat"):
        db.create_session(sid, source="desktop")
        db.append_message(sid, "user", content="hi")
    yield db
    db.close()


def test_hook_is_registered():
    assert "on_session_archived" in VALID_HOOKS


def test_dashboard_patch_fires_once_per_real_archive(store, archived_events):
    try:
        from starlette.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi/starlette not installed")
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    assert client.patch("/api/sessions/done-chat", json={"archived": True}).status_code == 200
    # Repeating the archive, un-archiving and unrelated flags are not new "done" events.
    assert client.patch("/api/sessions/done-chat", json={"archived": True}).status_code == 200
    assert client.patch("/api/sessions/other-chat", json={"pinned": True}).status_code == 200
    assert client.patch("/api/sessions/done-chat", json={"archived": False}).status_code == 200

    assert [(e["session_id"], e["surface"], e["archived_at_dispatch"]) for e in archived_events] == [
        ("done-chat", "dashboard", 1)]
    assert store.get_session("done-chat")["archived"] == 0  # the un-archive still applied

    # Archiving again after an un-archive is a new transition.
    client.patch("/api/sessions/done-chat", json={"archived": True})
    assert len(archived_events) == 2


def test_tui_archive_rpc_fires(store, archived_events, monkeypatch):
    import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)
    import tui_gateway.server as srv

    monkeypatch.setattr(srv, "_get_db", lambda: store)
    envelope = srv._methods["session.archive"](1, {"session_id": "done-chat", "archived": True})
    assert "error" not in envelope, envelope
    srv._methods["session.archive"](2, {"session_id": "done-chat", "archived": True})

    assert [(e["session_id"], e["surface"]) for e in archived_events] == [("done-chat", "tui")]


def test_cli_bulk_archive_fires_per_session(store, archived_events):
    from hermes_cli.sessions_cmd import _cmd_prune_or_archive

    for sid in ("done-chat", "other-chat"):
        store.end_session(sid, "user_exit")
    args = argparse.Namespace(
        older_than=None, newer_than=None, source="desktop", title=None, dry_run=False, yes=True,
        include_pinned=False, include_archived=False, never_active=False)
    _cmd_prune_or_archive(store, args, "archive")

    assert sorted((e["session_id"], e["surface"]) for e in archived_events) == [
        ("done-chat", "cli"), ("other-chat", "cli")]


def test_idle_sweep_does_not_fire(store, archived_events):
    # Idle = freshest of last activity, latest message and start: age all three.
    store._conn.execute("UPDATE sessions SET started_at = 0, last_activity_at = 0")
    store._conn.execute("UPDATE messages SET timestamp = 0")
    store._conn.commit()
    assert store.archive_stale_sessions(1) == 2
    assert store.get_session("done-chat")["archived"] == 1
    assert archived_events == []


def test_failing_observer_never_fails_the_archive(store):
    from hermes_cli.lifecycle import archive_session

    manager = get_plugin_manager()
    saved = {name: list(callbacks) for name, callbacks in manager._hooks.items()}

    def _boom(**_kwargs):
        raise RuntimeError("plugin exploded")

    manager._hooks.setdefault("on_session_archived", []).append(_boom)
    try:
        assert archive_session(store, "done-chat", True, surface="dashboard")
    finally:
        manager._hooks = saved
    assert store.get_session("done-chat")["archived"] == 1
