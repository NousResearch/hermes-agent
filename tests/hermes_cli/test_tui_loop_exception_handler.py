"""The classic TUI must own the event loop's exception handler from its first frame to its last.

Regression for the shutdown traceback in errors.log 2026-11-18 09:56:03::

    ERROR asyncio: Unhandled error in exception handler
    context: {'message': 'Task was destroyed but it is pending!', 'task': <Task ...
              coro=<Application._handle_exception.<locals>.in_term() ...>>}
    RuntimeError: There is no current event loop in thread 'MainThread'.

prompt_toolkit's ``Application.run()`` installs ``Application._handle_exception`` on the loop and
restores whatever it found on the way out. The subagent dock runs a *nested* Application
(``cli_subagent_monitor.open_monitor``) on the same loop; when that task is cancelled during
``asyncio.run`` shutdown it restores the outer Application's handler -- after the outer app had
already restored the original one. The loop that is being closed is then owned by prompt_toolkit,
so a task destroyed while still pending reaches ``ensure_future(in_term())`` with no current event
loop and Python logs the traceback above.

``HermesCLI.run()`` installs ``_tui_suppress_closed_loop_errors`` for exactly this purpose, but it
did so with ``get_running_loop()`` *before* ``app.run()`` started a loop, so the install raised and
was swallowed. These tests pin both halves: the handler is installed inside the running loop, and it
is still the loop's handler once the app and ``asyncio.run`` have torn everything else down.
"""
from __future__ import annotations

import asyncio
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.input.base import DummyInput, _dummy_context_manager
from prompt_toolkit.layout import Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.output import DummyOutput

import cli as cli_mod


class _IdleInput(DummyInput):
    """An input that never reports EOF, so a nested Application stays running until cancelled."""

    @property
    def closed(self) -> bool:
        return False

    def attach(self, input_ready_callback):
        return _dummy_context_manager()


def _application(label, input_obj=None):
    return Application(
        layout=Layout(Window(FormattedTextControl(label))),
        input=input_obj if input_obj is not None else _IdleInput(),
        output=DummyOutput(),
    )


def _stub_cli(host, app):
    """Everything ``HermesCLI.run`` does before/after ``app.run()``, minus the terminal work."""
    host._claim_active_session = lambda *a, **k: True
    host._tui_print_startup = lambda: None
    host._tui_init_run_state = lambda: None
    host._tui_build_key_bindings = lambda: object()
    host._tui_build_layout = lambda kb: (object(), object())
    host._tui_build_application = lambda layout, kb, style: app
    host._pet_flush_kitty_frame = lambda *a, **k: None
    host._install_resize_recovery = lambda app: None
    host._tui_spinner_loop = lambda: None
    host._tui_process_loop = lambda: None
    host._tui_wake_startup = lambda: None
    host._tui_install_signal_handlers = lambda: None
    host._tui_stdin_usable = lambda: True
    host._tui_shutdown = lambda: None
    host._pet_start_anim = lambda: None
    host._tui_multiline_shortcuts = False
    host._pending_relaunch = None
    host._should_exit = False


def _run_cli_until_shutdown():
    """Run the real ``HermesCLI.run()`` frame with a live app, a pending nested app, then shut down."""
    pipe = create_pipe_input()
    outer = _application("outer", pipe.__enter__())
    captured: dict[str, Any] = {"spawned": False}

    def before_render(app):
        if captured["spawned"]:
            return
        captured["spawned"] = True
        loop = asyncio.get_running_loop()
        captured["loop"] = loop
        captured["handler_during_run"] = loop.get_exception_handler()

        async def nested():
            # The shape of cli_subagent_monitor.open_monitor(): a second Application on the
            # same loop, still running when the main app exits.
            await _application("nested").run_async()

        captured["nested_task"] = loop.create_task(nested())
        loop.call_later(0.05, app.exit)

    outer.before_render += before_render

    host = object.__new__(cli_mod.HermesCLI)
    _stub_cli(host, outer)

    saved = {
        name: getattr(cli_mod, name)
        for name in (
            "_disable_prompt_toolkit_cpr_warning",
            "_apply_bracketed_paste_timeout_patch",
            "_mark_tui_input_modes_active",
            "_release_paints",
            "_run_cleanup",
        )
    }
    from hermes_cli.observability import shared_metrics_startup

    saved_ready = shared_metrics_startup.cli_prompt_ready_handler
    cli_mod._disable_prompt_toolkit_cpr_warning = lambda app: None
    cli_mod._apply_bracketed_paste_timeout_patch = lambda: None
    cli_mod._mark_tui_input_modes_active = lambda: None
    cli_mod._release_paints = lambda: None
    cli_mod._run_cleanup = lambda: None
    shared_metrics_startup.cli_prompt_ready_handler = lambda: (lambda app: None)
    try:
        host.run()
    finally:
        for name, value in saved.items():
            setattr(cli_mod, name, value)
        shared_metrics_startup.cli_prompt_ready_handler = saved_ready
        pipe.__exit__(None, None, None)

    return captured


def test_tui_run_keeps_hermes_handler_through_prompt_toolkit_shutdown():
    captured = _run_cli_until_shutdown()
    loop = captured["loop"]
    suppress = cli_mod.HermesCLI._tui_suppress_closed_loop_errors

    # What the interpreter does with a task that is still pending at shutdown: the report goes
    # through the loop's remaining handler, now that the thread has no current event loop.
    seen = []
    loop.default_exception_handler = seen.append
    loop.call_exception_handler(
        {"message": "Task was destroyed but it is pending!", "task": object()}
    )

    assert seen, "the destroyed-task report never reached the loop's default handler"
    assert [c["message"] for c in seen] == ["Task was destroyed but it is pending!"], (
        f"prompt_toolkit's handler ran against a closed loop: {seen!r}"
    )
    assert getattr(captured["handler_during_run"], "__func__", None) is suppress, (
        "the CLI's loop exception handler must be installed for the whole app run; prompt_toolkit "
        f"owned it instead: {captured['handler_during_run']!r}"
    )
    assert captured["nested_task"].cancelled(), (
        "the nested Application must outlive the main app, as the subagent dock does"
    )
    assert loop.is_closed()
    assert getattr(loop.get_exception_handler(), "__func__", None) is suppress, (
        "a nested Application restored a stale prompt_toolkit handler onto the loop being closed; "
        f"the loop ended up owned by {loop.get_exception_handler()!r}"
    )


def test_tui_run_application_keeps_prompt_toolkit_out_of_the_handler_slot():
    host = object.__new__(cli_mod.HermesCLI)

    class _FakeApp:
        kwargs = None

        def run(self, **kwargs):
            type(self).kwargs = kwargs

    app = _FakeApp()
    host._tui_run_application(app)

    assert app.kwargs["set_exception_handler"] is False, (
        "app.run() must not let prompt_toolkit replace the CLI's loop exception handler"
    )
    assert callable(app.kwargs["pre_run"]), (
        "the CLI's handler has to be installed from inside the running loop (pre_run)"
    )
