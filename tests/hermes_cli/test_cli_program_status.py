"""Herdr OSC 7501 lifecycle reports from the interactive Hermes CLI."""

from types import SimpleNamespace

import pytest

import cli as cli_module
import hermes_cli.cli_tui_runtime_mixin as runtime_mixin
from cli import HermesCLI


def _runtime_cli(events):
    cli = HermesCLI.__new__(HermesCLI)
    cli._report_program_status = lambda state, **kwargs: events.append((state, kwargs))
    cli._tui_dock_drop_dequeued = lambda: None
    cli._tui_unwrap_input = lambda value: (value, False, False)
    cli._typed_voice_stop = lambda value: False
    cli.handle_bang_shell = lambda value: False
    cli._pending_resume_sessions = None
    cli._expand_paste_references = lambda value: value
    cli._print_user_message_preview = lambda value: None
    cli._turn_summary_begin = lambda: None
    cli._app = SimpleNamespace(invalidate=lambda: None)
    cli._pending_tool_info = {}
    cli._pet_react_turn_end = lambda: None
    cli._turn_summary_emit = lambda: None
    cli._drain_interrupt_queue_to_pending_input = lambda: None
    cli._recover_terminal_after_interrupt = lambda: None
    cli._maybe_continue_goal_after_turn = lambda: None
    cli._maybe_complete_loop_tick_after_turn = lambda: None
    cli._drain_process_notifications = lambda where: None
    cli._voice_mode = False
    cli._voice_continuous = False
    cli._voice_recording = False
    cli._last_turn_interrupted = False
    cli._last_turn_failed = False
    cli._agent_running = False
    cli._command_running = False
    cli._interactive_turn = True
    return cli


def test_submitted_prompt_reports_working_before_chat():
    events = []
    cli = _runtime_cli(events)
    cli.chat = lambda *args, **kwargs: events.append(("chat", {})) or "ok"
    cli._tui_after_turn = lambda: events.append(("after", {}))

    cli._tui_process_one_input("hello")

    assert events[:2] == [("working", {}), ("chat", {})]
    assert events[-1] == ("after", {})


@pytest.mark.parametrize("chat_result", [None, RuntimeError("boom")])
def test_failed_or_raised_chat_reports_error(chat_result):
    events = []
    cli = _runtime_cli(events)

    def chat(*args, **kwargs):
        if isinstance(chat_result, BaseException):
            raise chat_result
        return chat_result

    cli.chat = chat
    if isinstance(chat_result, BaseException):
        with pytest.raises(RuntimeError, match="boom"):
            cli._tui_process_one_input("hello")
    else:
        cli._tui_process_one_input("hello")

    assert events == [("working", {}), ("error", {})]


@pytest.mark.parametrize(
    ("failed", "interrupted", "expected"),
    [
        (False, False, "done"),
        (True, False, "error"),
        (False, True, "idle"),
    ],
)
def test_after_turn_reports_terminal_outcome(failed, interrupted, expected):
    events = []
    cli = _runtime_cli(events)
    cli._last_turn_failed = failed
    cli._last_turn_interrupted = interrupted

    cli._tui_after_turn()

    assert events == [(expected, {})]


def test_tui_startup_reports_idle(monkeypatch):
    events = []
    cli = _runtime_cli(events)
    cli._vim_mode = False
    monkeypatch.setattr(cli_module, "_select_classic_cli_pt_output", lambda stdout: None)
    monkeypatch.setattr(runtime_mixin.terminal_image, "supports_kitty_placeholders", lambda: False)
    created = {}
    monkeypatch.setattr(runtime_mixin, "Application", lambda **kwargs: created.update(kwargs) or object())

    runtime_mixin.CLITuiRuntimeMixin._tui_build_application(cli, "layout", "keys", "style")

    assert events == [("idle", {})]
    assert created["layout"] == "layout"


def test_tui_shutdown_clears_status_before_cleanup():
    events = []
    cli = _runtime_cli(events)

    def stop_cleanup():
        raise RuntimeError("cleanup reached")

    cli._pet_stop_anim = stop_cleanup
    with pytest.raises(RuntimeError, match="cleanup reached"):
        cli._tui_shutdown()

    assert events == [("clear", {})]


def test_approval_reports_blocked_then_working(monkeypatch):
    events = []
    cli = _runtime_cli(events)
    cli._approval_lock = __import__("threading").Lock()
    cli._approval_choices = lambda *args, **kwargs: ["once", "deny"]
    cli._ring_bell = lambda **kwargs: None
    cli._paint_now = lambda: None
    cli._poll_modal_queue = lambda *args, **kwargs: "once"
    cli._persist_prompt_summary = lambda *args, **kwargs: None
    monkeypatch.setitem(__import__("cli").CLI_CONFIG, "approvals", {"timeout": 1})

    assert cli._approval_callback("echo ok", "test") == "once"
    assert events == [
        ("blocked", {"kind": "permission"}),
        ("working", {}),
    ]


def test_clarify_reports_blocked_then_working(monkeypatch):
    events = []
    cli = _runtime_cli(events)
    cli._clarify_batch_set_active = lambda state, index: None
    cli._ring_bell = lambda **kwargs: None
    cli._paint_now = lambda: None
    cli._poll_modal_queue = lambda *args, **kwargs: {"q1": "yes"}
    monkeypatch.setattr("tools.clarify_gateway.resolve_clarify_timeout", lambda config: 1)

    result = cli._clarify_callback([{"qid": "q1", "question": "Continue?", "choices": ["yes"]}])

    assert result == {"answers": {}, "outcome": "submitted"}
    assert events == [
        ("blocked", {"kind": "question"}),
        ("working", {}),
    ]


def test_auth_modal_reports_blocked_then_working():
    events = []
    cli = _runtime_cli(events)
    cli._poll_modal_queue = lambda *args, **kwargs: "secret"

    result = cli._poll_modal_queue_with_status(object(), "_deadline", kind="auth", refresh=0)

    assert result == "secret"
    assert events == [
        ("blocked", {"kind": "auth"}),
        ("working", {}),
    ]


def test_non_turn_modal_restores_idle():
    events = []
    cli = _runtime_cli(events)
    cli._interactive_turn = False
    cli._poll_modal_queue = lambda *args, **kwargs: "once"

    result = cli._poll_modal_queue_with_status(object(), "_deadline", kind="question")

    assert result == "once"
    assert events == [
        ("blocked", {"kind": "question"}),
        ("idle", {}),
    ]


def test_slash_confirmation_reports_question_then_idle():
    events = []
    cli = _runtime_cli(events)
    cli._interactive_turn = False
    cli._app = SimpleNamespace(loop=None)
    cli._slash_confirm_state = None
    cli._slash_confirm_deadline = 0
    cli._capture_modal_input_snapshot = lambda: None
    cli._restore_modal_input_snapshot = lambda: None
    cli._invalidate = lambda min_interval=0.25: None
    cli._poll_modal_queue = lambda *args, **kwargs: "once"

    result = cli._prompt_text_input_modal(
        title="Confirm", detail="Proceed?", choices=[("once", "Once", "Proceed once")]
    )

    assert result == "once"
    assert events == [
        ("blocked", {"kind": "question"}),
        ("idle", {}),
    ]