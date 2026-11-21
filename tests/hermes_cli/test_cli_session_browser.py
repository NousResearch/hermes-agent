"""Classic CLI browser contracts using the real session store and picker."""

import os
import sys
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages


@pytest.fixture
def store():
    """A writable handle on the test home's state.db (the autouse fixture pins HERMES_HOME) for seeding."""
    from hermes_constants import get_hermes_home

    handle = open_persistence(profile_scope(home=get_hermes_home()))
    yield handle
    handle.close()


def _seed(store, session_id, source="cli", *, message=None):
    store.sessions.create(NewSession(session_id=session_id, source=source))
    if message is not None:
        seed_messages(store, session_id, {"role": "user", "content": message})


@pytest.fixture
def browser_cli(store):
    """The CLI on its gateway session store, as main runs it: reads go through the read-only reader on
    the same state.db; the gateway double admits every ``session.resume``."""
    from hermes_cli.gateway_connection import GatewayConnection
    from hermes_cli.gateway_session_store import GatewaySessionStore

    _seed(store, "current")

    def gateway(method, **params):
        if method == "session.resume":
            sid = params["session_id"]
            return {"stored_session_id": sid, "subscription_id": f"sub-{sid}", "info": {}}
        return {}

    conn = GatewayConnection()
    conn.start = lambda *_a, **_k: None
    session_store = GatewaySessionStore(conn)
    session_store.call = gateway
    session_store.attach("current")
    obj = HermesCLI.__new__(HermesCLI)
    obj._session_db = session_store
    obj.session_id = "current"
    obj.agent = None
    obj.conversation_history = []
    obj._pending_resume_sessions = None
    obj._pending_title = None
    obj._resumed = False
    obj._agent_running = False
    obj._app = None
    obj.resume_display = "minimal"
    yield obj
    session_store.close_reader()


def test_browser_listing_has_no_recency_cap_and_keeps_cli_policy(browser_cli, store):
    eligible = {f"session_{i:04d}" for i in range(1015)}
    for sid in sorted(eligible):
        _seed(store, sid)
    for source in ("telegram", "tool", "oneshot", "kanban", "subagent"):
        _seed(store, f"excluded_{source}", source)
    _seed(store, "hidden")
    store.sessions.set_hidden("hidden", True)
    _seed(store, "archived")
    store.sessions.set_archived("archived", True)

    rows = browser_cli._list_recent_sessions(limit=None)

    assert {row["id"] for row in rows} == eligible
    assert len(rows) == len(eligible)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", ["/sessions", "/resume", "/sessions list", "/resume list", "/resume ls", "/resume browse"])
def test_slash_browser_resumes_exact_selected_id_after_listing_changes(browser_cli, store, monkeypatch, command):
    import curses

    for i in range(15):
        _seed(store, f"saved_{i:02d}", message=f"message {i}")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(curses, "curs_set", lambda *_: None)
    monkeypatch.setattr(curses, "has_colors", lambda: False)
    screen = MagicMock()
    screen.getmaxyx.return_value = (14, 64)
    keys = iter([curses.KEY_END, 10])

    def read_key():
        key = next(keys)
        if key == 10:
            _seed(store, "arrived_during_browse")
        return key

    screen.get_wch.side_effect = read_key
    wrappers = []

    def wrapper(run):
        wrappers.append(run)
        run(screen)

    monkeypatch.setattr(curses, "wrapper", wrapper)
    for method in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        monkeypatch.setattr(browser_cli, method, lambda *_: None)

    browser_cli.process_command(command)

    assert len(wrappers) == 1
    assert browser_cli.session_id == "saved_00"
    assert browser_cli.conversation_history[0]["content"] == "message 0"
    assert browser_cli._pending_resume_sessions is None


@pytest.mark.asyncio
@pytest.mark.parametrize("caller", ["worker", "ui_loop"])
async def test_worker_browser_suspends_real_prompt_toolkit_and_restores_on_cancel(browser_cli, store, monkeypatch, caller):
    import asyncio
    import threading

    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.patch_stdout import patch_stdout

    from hermes_cli import cli_session_browser, sessions_cmd_browse

    _seed(store, "other")
    observed = []
    picked = asyncio.Event()
    with create_pipe_input() as pipe:
        app = Application(input=pipe, output=DummyOutput())
        browser_cli._app = app
        ready = asyncio.Event()
        task = asyncio.create_task(app.run_async(pre_run=ready.set))
        await asyncio.wait_for(ready.wait(), timeout=5)

        def picker(*args, **kwargs):
            observed.append((app._running_in_terminal, threading.current_thread() is threading.main_thread()))
            picked.set()

        monkeypatch.setattr(cli_session_browser, "can_browse_sessions", lambda: True)
        monkeypatch.setattr(sessions_cmd_browse, "_session_browse_picker", picker)
        try:
            with patch_stdout():
                if caller == "worker":
                    await asyncio.wait_for(asyncio.to_thread(browser_cli.process_command, "/sessions"), timeout=5)
                else:
                    browser_cli.process_command("/sessions")
                    await asyncio.wait_for(picked.wait(), timeout=5)
            assert observed == [(True, True)]
            assert app._running_in_terminal is False
            assert browser_cli.session_id == "current"
            assert browser_cli._session_db.get_session("current")["end_reason"] is None
        finally:
            app.exit()
            await asyncio.wait_for(task, timeout=5)


@pytest.mark.parametrize("command", ["/sessions", "/resume"])
def test_noninteractive_listing_is_nonblocking_and_cell_width_bounded(browser_cli, store, monkeypatch, capsys, command):
    import shutil
    from wcwidth import wcswidth

    _seed(store, "other", message="Long preview " + "界🙂" * 30)
    store.sessions.set_title("other", "界🙂" * 25)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda *_, **kwargs: os.terminal_size((40, 24)))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("noninteractive commands must not prompt"))

    browser_cli.process_command(command)

    lines = capsys.readouterr().out.splitlines()
    assert any("other" in line for line in lines)
    assert all(wcswidth(line) <= 40 for line in lines)
    assert browser_cli.session_id == "current"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", ["/sessions", "/resume"])
@pytest.mark.parametrize("failure", ["missing_curses", "bad_terminal"])
def test_unavailable_picker_renders_static_choices_without_reading_input(
    browser_cli, store, monkeypatch, capsys, command, failure
):
    import curses

    _seed(store, "fallback_target")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("classic fallback must not read input"))
    if failure == "missing_curses":
        monkeypatch.setitem(sys.modules, "curses", None)
    else:
        def unavailable(_run):
            raise curses.error("unsupported terminal")
        monkeypatch.setattr(curses, "wrapper", unavailable)

    browser_cli.process_command(command)

    output = capsys.readouterr().out
    assert "fallback_target" in output
    assert "/resume" in output
    assert browser_cli.session_id == "current"


@pytest.mark.platforms("posix")
def test_picker_cancel_is_not_an_unavailable_terminal(browser_cli, store, monkeypatch, capsys):
    import curses

    _seed(store, "cancel_target")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda _run: None)  # initialized picker, no selection
    browser_cli.process_command("/sessions")
    assert "cancel_target" not in capsys.readouterr().out
    assert browser_cli.session_id == "current"


@pytest.mark.parametrize("title", ["list", "ls", "browse"])
def test_resume_alias_keeps_existing_title_resolution(browser_cli, store, monkeypatch, title):
    _seed(store, "named_session")
    store.sessions.set_title("named_session", title)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    from hermes_cli import sessions_cmd_browse
    monkeypatch.setattr(sessions_cmd_browse, "_session_browse_picker", lambda *a, **k: pytest.fail("title must win"))
    for method in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        monkeypatch.setattr(browser_cli, method, lambda *_: None)

    browser_cli.process_command(f"/resume {title}")

    assert browser_cli.session_id == "named_session"


@pytest.mark.parametrize("command", ["/sessions", "/resume"])
def test_numeric_resume_uses_the_last_displayed_snapshot_beyond_ten(browser_cli, store, monkeypatch, command):
    from hermes_cli import cli_session_browser, sessions_cmd_browse

    db = browser_cli._session_db
    for i in range(25):
        _seed(store, f"numbered_{i:02d}", message=f"message {i}")
    snapshots = []

    def cancelled_picker(sessions, **kwargs):
        snapshots.append(list(sessions))

    monkeypatch.setattr(cli_session_browser, "can_browse_sessions", lambda: True)
    monkeypatch.setattr(sessions_cmd_browse, "_session_browse_picker", cancelled_picker)
    for method in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        monkeypatch.setattr(browser_cli, method, lambda *_: None)
    browser_cli.process_command(command)
    expected_id = snapshots[0][22]["id"]
    _seed(store, "newer_arrival", message="arrived after the browser closed")

    browser_cli.process_command("/resume 23")

    assert browser_cli.session_id == expected_id
    assert browser_cli.conversation_history[0]["content"] == db.get_resume_conversations(expected_id)[0][0]["content"]

    # A new static listing replaces the numbering rather than retaining the old snapshot.
    monkeypatch.setattr(cli_session_browser, "can_browse_sessions", lambda: False)
    expected_first = browser_cli._list_recent_sessions(limit=10)[0]["id"]
    browser_cli.process_command("/sessions")
    browser_cli.process_command("/resume 1")
    assert browser_cli.session_id == expected_first
