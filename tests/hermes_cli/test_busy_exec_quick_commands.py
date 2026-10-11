"""Busy classic-CLI exec commands bypass the blocked chat worker, not the UI."""
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
import threading

import pytest
from prompt_toolkit.buffer import Buffer


def make_cli(monkeypatch, command="echo busy-quick-receipt"):
    from cli import HermesCLI
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {"quick_commands": {"receipt": {"type": "exec", "command": command}}}
    cli._agent_running = True
    cli._attached_images = []
    cli._tui_multiline_shortcuts = False
    cli._tui_last_text_change = 0
    cli._pending_input = Queue()
    cli._interrupt_queue = Queue()
    cli.agent = Mock()
    cli.history = []
    cli._tui_enter_overlay = Mock(return_value=False)
    cli._inline_pastes = Mock()
    cli._console_print = Mock()
    return cli


def enter(cli, text):
    buf = Buffer()
    buf.text = text
    event = SimpleNamespace(app=SimpleNamespace(current_buffer=buf, invalidate=Mock()))
    cli._tui_handle_enter(event)
    return event


@pytest.mark.parametrize("mode", ["queue", "steer", "interrupt"])
def test_busy_exec_reaches_real_subprocess_before_foreground_finishes(monkeypatch, mode):
    cli = make_cli(monkeypatch)
    cli.busy_input_mode = mode
    delivered = threading.Event()
    outputs = []
    def output(value):
        outputs.append(str(value))
        delivered.set()
    cli._console_print = output
    event = enter(cli, "/receipt")
    assert delivered.wait(3), "exec must run while foreground chat is still busy"
    assert outputs == ["busy-quick-receipt"]
    assert cli._agent_running is True
    assert cli._pending_input.empty()
    assert cli._interrupt_queue.empty()
    assert cli.agent.mock_calls == []
    assert cli.history == []
    assert event.app.current_buffer.text == ""


@pytest.mark.parametrize("text,qcmd,busy,images", [
    ("/receipt", {"type": "exec", "command": "echo no"}, False, []),
    ("/receipt", {"type": "exec", "command": "echo no"}, True, ["image.png"]),
    ("/receipt", {"type": "alias", "target": "help"}, True, []),
    ("/receipt", {"type": "other"}, True, []),
    ("/unknown", {"type": "exec", "command": "echo no"}, True, []),
    ("/update", {"type": "exec", "command": "echo no"}, True, []),
])
def test_non_exec_paths_keep_normal_dispatch(monkeypatch, text, qcmd, busy, images):
    cli = make_cli(monkeypatch)
    cli.config["quick_commands"] = {text[1:] if text == "/update" else "receipt": qcmd}
    cli._agent_running = busy
    cli._attached_images = list(images)
    cli._run_quick_command = Mock()
    enter(cli, text)
    assert cli._pending_input.get_nowait() == ((text, images) if images else text)
    cli._run_quick_command.assert_not_called()


def test_slow_exec_does_not_block_ui_and_preserves_context(monkeypatch):
    from contextvars import ContextVar
    cli = make_cli(monkeypatch)
    scope = ContextVar("quick_exec_test_scope", default="wrong")
    token = scope.set("caller")
    started, release, finished = (threading.Event() for _ in range(3))
    captured = []
    def execute(base, qcmd, args):
        started.set()
        try:
            assert release.wait(5)
            captured.append((base, qcmd["command"], args, scope.get()))
        finally:
            finished.set()
    cli._run_quick_command = execute
    try:
        event = enter(cli, "/RECEIPT Mixed Case")
        assert started.wait(3)
        assert event.app.current_buffer.text == ""
        # The UI returned although execution is still blocked, and mutation cannot
        # change the selected snippet after submission.
        cli.config["quick_commands"]["receipt"]["command"] = "changed"
        scope.set("new-context")
    finally:
        release.set()
        scope.reset(token)
        assert finished.wait(3)
    assert captured == [("/RECEIPT", "echo busy-quick-receipt", "Mixed Case", "caller")]


@pytest.mark.parametrize("command,side_effect,stdout,stderr", [
    ("", None, "", ""),
    ("fixture", None, "", ""),
    ("fixture", OSError("fixture spawn failure"), "", ""),
    ("fixture", None, "", "fixture stderr"),
])
def test_exec_preserves_empty_error_and_stderr_handling(monkeypatch, command, side_effect, stdout, stderr):
    import subprocess
    cli = make_cli(monkeypatch, command)
    run = Mock(side_effect=side_effect, return_value=SimpleNamespace(stdout=stdout, stderr=stderr))
    monkeypatch.setattr(subprocess, "run", run)
    delivered = threading.Event()
    outputs = []
    cli._console_print = lambda value: (outputs.append(str(value)), delivered.set())
    enter(cli, "/receipt")
    assert delivered.wait(3)
    assert len(outputs) == 1
    assert cli._pending_input.empty()
    if not command:
        run.assert_not_called()
    elif side_effect:
        assert "fixture spawn failure" in outputs[0]
    elif stderr:
        assert outputs == [stderr]
    else:
        assert outputs[0]
