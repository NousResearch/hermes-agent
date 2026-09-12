"""Sudo dismissal must reach execution as cancellation, while empty Enter remains a skip."""

import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from cli import HermesCLI
from tools.environments.base import BaseEnvironment
from tools.interrupt import set_interrupt
import tools.terminal_tool as terminal
import tools.terminal_tool_sudo as sudo


def _cli():
    cli = HermesCLI.__new__(HermesCLI)
    cli._approval_state = cli._clarify_state = cli._secret_state = None
    cli._slash_confirm_state = cli._sudo_state = cli._modal_input_snapshot = None
    cli._sudo_deadline = 0
    cli._sudo_lock = threading.Lock()
    cli._sudo_state_lock = threading.Lock()
    cli._sudo_interrupt_generation = 0
    cli._paint_now = cli._ring_bell = MagicMock()
    buffer = SimpleNamespace(text="draft", cursor_position=5)
    buffer.reset = lambda: setattr(buffer, "text", "")
    cli._app = SimpleNamespace(current_buffer=buffer, invalidate=MagicMock())
    return cli


def _wait(predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.002)
    assert predicate(), "worker did not reach expected state"


@pytest.mark.parametrize("response", ["escape", "interrupt", "empty", "password", "timeout", "post-submit-interrupt"])
def test_sudo_ui_decision_controls_execution(monkeypatch, tmp_path, response):
    cli = _cli()
    calls = []

    class Env(BaseEnvironment):
        def _before_execute(self):
            pass

        def _prepare_command(self, command):
            result = super()._prepare_command(command)
            if response == "post-submit-interrupt":
                set_interrupt(True)
            return result

        def _run_bash(self, command, **kwargs):
            calls.append((command, kwargs.get("stdin_data")))
            return object()

        def _wait_for_process(self, *_args, **_kwargs):
            return {"output": "completed", "returncode": 0}

        def cleanup(self):
            pass

    env = Env(cwd=str(tmp_path), timeout=5)
    monkeypatch.delenv("SUDO_PASSWORD", raising=False)
    # Current BaseEnvironment owns the backend-scoped NOPASSWD probe; keep this
    # fake environment explicitly password-requiring without probing the host.
    monkeypatch.setattr(Env, "_sudo_nopasswd_works", lambda _self: False)
    monkeypatch.setattr(terminal, "_resolve_container_task_id", lambda _: "sudo-test")
    monkeypatch.setattr(terminal, "resolve_task_overrides", lambda _: {})
    monkeypatch.setattr(terminal, "_start_cleanup_thread", lambda: None)
    monkeypatch.setattr(terminal, "_get_env_config", lambda: {
        "env_type": "local", "cwd": str(tmp_path), "timeout": 5})
    monkeypatch.setitem(terminal._active_environments, "sudo-test", env)
    monkeypatch.setitem(terminal._last_activity, "sudo-test", 0)
    sudo._reset_cached_sudo_passwords()
    result = {}

    def run():
        terminal.set_sudo_password_callback(cli._sudo_password_callback)
        try:
            result.update(json.loads(terminal.terminal_tool("sudo true", force=True)))
        finally:
            terminal.set_sudo_password_callback(None)
            set_interrupt(False)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    try:
        _wait(lambda: cli._sudo_state is not None or not worker.is_alive())
        assert cli._sudo_state is not None
        if response == "escape":
            cli._tui_handle_escape_modal(SimpleNamespace(app=cli._app))
        elif response == "interrupt":
            cli._clear_active_overlays_for_interrupt()
        elif response == "timeout":
            cli._sudo_deadline = time.monotonic() - 1
            # The prompt owns its deadline; shortening its published state simulates expiry.
            cli._sudo_state["deadline"] = cli._sudo_deadline
        else:
            cli._app.current_buffer.text = "secret" if response in ("password", "post-submit-interrupt") else ""
            assert cli._tui_enter_overlay(SimpleNamespace(app=cli._app))
        worker.join(3)
        assert not worker.is_alive()
        if response in ("escape", "interrupt", "post-submit-interrupt"):
            assert result.get("status") == "cancelled", result
            assert result["exit_code"] == 130
            assert calls == []
        else:
            assert result["exit_code"] == 0, result
            assert len(calls) == 1
            assert calls[0][1] == ("secret\n" if response == "password" else None)
        assert cli._sudo_state is None
        assert cli._app.current_buffer.text == "draft"
    finally:
        if cli._sudo_state:
            cli._sudo_state["response_queue"].put(None)
        worker.join(3)
        sudo._reset_cached_sudo_passwords()


def test_concurrent_prompt_ownership_and_interrupt_generation():
    cli = _cli()
    results = []
    waiting = threading.Event()
    real_lock = cli._sudo_lock

    class ObservedLock:
        def __enter__(self):
            waiting.set()
            real_lock.acquire()

        def __exit__(self, *_args):
            real_lock.release()

    workers = [threading.Thread(target=lambda: results.append(cli._sudo_password_callback()), daemon=True)
               for _ in range(2)]
    workers[0].start()
    _wait(lambda: cli._sudo_state is not None)
    first = cli._sudo_state
    cli._sudo_lock = ObservedLock()
    workers[1].start()
    try:
        # Main's unsynchronized callback replaces the first state; either event exposes it.
        _wait(lambda: waiting.is_set() or cli._sudo_state is not first)
        assert cli._sudo_state is first
        cli._clear_active_overlays_for_interrupt()
        for worker in workers:
            worker.join(3)
            assert not worker.is_alive()
        assert results == [None, None]
        assert cli._sudo_state is None
        # A callback beginning after the interrupt remains usable.
        fresh = threading.Thread(target=lambda: results.append(cli._sudo_password_callback()), daemon=True)
        fresh.start()
        _wait(lambda: cli._sudo_state is not None)
        cli._app.current_buffer.text = "fresh"
        cli._tui_enter_overlay(SimpleNamespace(app=cli._app))
        fresh.join(3)
        assert results[-1] == "fresh"
        assert cli._app.current_buffer.text == "draft"
    finally:
        for state in (first, cli._sudo_state):
            if state:
                state["response_queue"].put(None)
        for worker in workers:
            worker.join(3)
