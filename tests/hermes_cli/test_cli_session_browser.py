"""Classic CLI browser contracts using the real session store and picker."""

import os
import sys
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI
from hermes_state import SessionDB


@pytest.fixture
def browser_cli():
    # The autouse fixture pins both HERMES_HOME and SessionDB's default path.
    db = SessionDB()
    obj = HermesCLI.__new__(HermesCLI)
    obj._session_db = db
    obj.session_id = "current"
    obj.agent = None
    obj.conversation_history = []
    obj._pending_resume_sessions = None
    obj._pending_title = None
    obj._resumed = False
    obj._agent_running = False
    obj._app = None
    obj.resume_display = "minimal"
    db.create_session("current", "cli")
    yield obj
    db.close()


def test_browser_listing_has_no_recency_cap_and_keeps_cli_policy(browser_cli):
    db = browser_cli._session_db
    eligible = {f"session_{i:04d}" for i in range(1015)}
    for sid in sorted(eligible):
        db.create_session(sid, "cli")
    for source in ("telegram", "tool", "oneshot", "kanban", "subagent"):
        db.create_session(f"excluded_{source}", source)
    db.create_session("hidden", "cli")
    db.set_session_hidden("hidden", True)
    db.create_session("archived", "cli")
    db.set_session_archived("archived", True)

    rows = browser_cli._list_recent_sessions(limit=None)

    assert {row["id"] for row in rows} == eligible
    assert len(rows) == len(eligible)


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("command", ["/sessions", "/resume", "/sessions list", "/resume list", "/resume ls", "/resume browse"])
def test_slash_browser_resumes_exact_selected_id_after_listing_changes(browser_cli, monkeypatch, command):
    import curses

    db = browser_cli._session_db
    for i in range(15):
        db.create_session(f"saved_{i:02d}", "cli")
        db.append_message(f"saved_{i:02d}", "user", f"message {i}")
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
            db.create_session("arrived_during_browse", "cli")
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
async def test_worker_browser_suspends_real_prompt_toolkit_and_restores_on_cancel(browser_cli, monkeypatch, caller):
    import asyncio
    import threading

    from prompt_toolkit.application import Application
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.patch_stdout import patch_stdout

    from hermes_cli import cli_session_browser, sessions_cmd_browse

    browser_cli._session_db.create_session("other", "cli")
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
def test_noninteractive_listing_is_nonblocking_and_cell_width_bounded(browser_cli, monkeypatch, capsys, command):
    import shutil
    from wcwidth import wcswidth

    db = browser_cli._session_db
    db.create_session("other", "cli")
    db.set_session_title("other", "界🙂" * 25)
    db.append_message("other", "user", "Long preview " + "界🙂" * 30)
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
    browser_cli, monkeypatch, capsys, command, failure
):
    import curses

    browser_cli._session_db.create_session("fallback_target", "cli")
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
def test_picker_cancel_is_not_an_unavailable_terminal(browser_cli, monkeypatch, capsys):
    import curses

    browser_cli._session_db.create_session("cancel_target", "cli")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(curses, "wrapper", lambda _run: None)  # initialized picker, no selection
    browser_cli.process_command("/sessions")
    assert "cancel_target" not in capsys.readouterr().out
    assert browser_cli.session_id == "current"


@pytest.mark.parametrize("title", ["list", "ls", "browse"])
def test_resume_alias_keeps_existing_title_resolution(browser_cli, monkeypatch, title):
    db = browser_cli._session_db
    db.create_session("named_session", "cli")
    db.set_session_title("named_session", title)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    from hermes_cli import sessions_cmd_browse
    monkeypatch.setattr(sessions_cmd_browse, "_session_browse_picker", lambda *a, **k: pytest.fail("title must win"))
    for method in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        monkeypatch.setattr(browser_cli, method, lambda *_: None)

    browser_cli.process_command(f"/resume {title}")

    assert browser_cli.session_id == "named_session"


@pytest.mark.parametrize("command", ["/sessions", "/resume"])
def test_numeric_resume_uses_the_last_displayed_snapshot_beyond_ten(browser_cli, monkeypatch, command):
    from hermes_cli import cli_session_browser, sessions_cmd_browse

    db = browser_cli._session_db
    for i in range(25):
        db.create_session(f"numbered_{i:02d}", "cli")
        db.append_message(f"numbered_{i:02d}", "user", f"message {i}")
    snapshots = []

    def cancelled_picker(sessions, **kwargs):
        snapshots.append(list(sessions))

    monkeypatch.setattr(cli_session_browser, "can_browse_sessions", lambda: True)
    monkeypatch.setattr(sessions_cmd_browse, "_session_browse_picker", cancelled_picker)
    for method in ("_restore_session_cwd", "_restore_session_yolo", "_restore_session_model"):
        monkeypatch.setattr(browser_cli, method, lambda *_: None)
    browser_cli.process_command(command)
    expected_id = snapshots[0][22]["id"]
    db.create_session("newer_arrival", "cli")
    db.append_message("newer_arrival", "user", "arrived after the browser closed")

    browser_cli.process_command("/resume 23")

    assert browser_cli.session_id == expected_id
    assert browser_cli.conversation_history[0]["content"] == db.get_resume_conversations(expected_id)[0][0]["content"]

    # A new static listing replaces the numbering rather than retaining the old snapshot.
    monkeypatch.setattr(cli_session_browser, "can_browse_sessions", lambda: False)
    expected_first = browser_cli._list_recent_sessions(limit=10)[0]["id"]
    browser_cli.process_command("/sessions")
    browser_cli.process_command("/resume 1")
    assert browser_cli.session_id == expected_first
