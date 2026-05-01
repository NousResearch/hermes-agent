"""Regression tests for prompt text input from CLI worker threads."""

from __future__ import annotations

import builtins
import threading
import types

from cli import HermesCLI


class FakeApp:
    def __init__(self) -> None:
        self.invalidations = 0

    def invalidate(self) -> None:
        self.invalidations += 1


def _make_cli() -> HermesCLI:
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj._app = FakeApp()
    cli_obj._status_bar_visible = True
    return cli_obj


def test_prompt_text_input_from_worker_thread_does_not_call_run_in_terminal(monkeypatch):
    """Regression for /reload-mcp confirmation in process_loop.

    process_loop runs slash commands in a background thread. prompt_toolkit's
    run_in_terminal needs the app's event loop and raises
    "There is no current event loop" there. _prompt_text_input must fall back
    to direct input outside the main thread, like the curses picker already
    does.
    """
    cli_obj = _make_cli()
    run_in_terminal_called = []

    fake_pt_app = types.ModuleType("prompt_toolkit.application")

    def fake_run_in_terminal(_func):
        run_in_terminal_called.append(True)
        raise RuntimeError("There is no current event loop in thread 'Thread-3 (process_loop)'.")

    fake_pt_app.run_in_terminal = fake_run_in_terminal
    monkeypatch.setitem(__import__("sys").modules, "prompt_toolkit.application", fake_pt_app)
    monkeypatch.setattr(builtins, "input", lambda prompt: "1")

    result = {}
    errors = []

    def call_prompt() -> None:
        try:
            result["value"] = cli_obj._prompt_text_input("Choice [1/2/3]: ")
        except Exception as exc:  # pragma: no cover - failure detail
            errors.append(exc)

    thread = threading.Thread(target=call_prompt, name="process_loop")
    thread.start()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []
    assert result == {"value": "1"}
    assert run_in_terminal_called == []
    assert cli_obj._status_bar_visible is True
