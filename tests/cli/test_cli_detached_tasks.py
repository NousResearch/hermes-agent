import time
import queue

from agent.background_task import SessionOrigin, background_tasks
from cli import HermesCLI
from gateway.session_context import get_session_env


def _make_cli():
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.session_id = "cli-session-123"
    cli_obj._background_tasks = {}
    cli_obj._app = None
    cli_obj._agent_running = False
    cli_obj._spinner_text = ""
    cli_obj._pending_background_wakes = queue.Queue()
    cli_obj._pending_input = queue.Queue()
    return cli_obj


def _clear_background_registry():
    with background_tasks._lock:
        background_tasks._pending.clear()
        background_tasks._active.clear()


def test_cli_session_env_uses_contextvars(monkeypatch):
    cli_obj = _make_cli()
    monkeypatch.delenv("HERMES_SESSION_KEY", raising=False)

    tokens = cli_obj._set_cli_session_env()
    try:
        assert get_session_env("HERMES_SESSION_PLATFORM") == "cli"
        assert get_session_env("HERMES_SESSION_CHAT_ID") == "cli-session-123"
        assert get_session_env("HERMES_SESSION_KEY") == "cli-session-123"
    finally:
        cli_obj._clear_cli_session_env(tokens)

    assert get_session_env("HERMES_SESSION_PLATFORM") == ""
    assert get_session_env("HERMES_SESSION_CHAT_ID") == ""
    assert get_session_env("HERMES_SESSION_KEY") == ""


def test_cli_starts_pending_detached_tasks():
    cli_obj = _make_cli()
    _clear_background_registry()

    async def _done():
        return "Music generation completed."

    handle = background_tasks.create(
        coro=_done(),
        session_key=cli_obj.session_id,
        origin=SessionOrigin(
            session_key=cli_obj.session_id,
            platform="cli",
            chat_id=cli_obj.session_id,
        ),
        label="music generation",
    )

    assert handle is not None
    cli_obj._start_pending_detached_tasks()

    deadline = time.time() + 2
    while time.time() < deadline and handle.task_id in cli_obj._background_tasks:
        time.sleep(0.01)

    assert background_tasks.get_active(cli_obj.session_id) is None
    assert handle.task_id not in cli_obj._background_tasks
    followup = cli_obj._pending_background_wakes.get_nowait()
    assert "Music generation completed." in followup


def test_cli_queues_detached_task_followup_for_agent_reply():
    cli_obj = _make_cli()
    _clear_background_registry()

    async def _done():
        return "Music generation completed.\nSaved: /tmp/night-drive.mp3\nMEDIA:/tmp/night-drive.mp3"

    handle = background_tasks.create(
        coro=_done(),
        session_key=cli_obj.session_id,
        origin=SessionOrigin(
            session_key=cli_obj.session_id,
            platform="cli",
            chat_id=cli_obj.session_id,
        ),
        label="music generation",
    )

    assert handle is not None
    cli_obj._start_pending_detached_tasks()

    deadline = time.time() + 2
    while time.time() < deadline and handle.task_id in cli_obj._background_tasks:
        time.sleep(0.01)

    followup = cli_obj._pending_background_wakes.get_nowait()
    assert "Reply to the user now with a brief update" in followup
    assert "Saved: /tmp/night-drive.mp3" in followup
    assert "MEDIA:/tmp/night-drive.mp3" in followup


def test_cli_drains_detached_followups_into_pending_input():
    cli_obj = _make_cli()
    cli_obj._pending_background_wakes.put("[SYSTEM: Background task completed.]")

    cli_obj._drain_detached_task_followups()

    assert cli_obj._pending_background_wakes.empty()
    assert cli_obj._pending_input.get_nowait() == "[SYSTEM: Background task completed.]"
