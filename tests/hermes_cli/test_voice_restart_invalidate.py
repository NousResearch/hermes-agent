"""Regression for #85278: continuous voice restart uses the CLI repaint seam."""

from types import SimpleNamespace


def test_continuous_voice_restart_uses_invalidate_helper(monkeypatch):
    """The background recorder restart must retain terminal repaint guards."""
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self.target = target

        def start(self):
            self.target()

    calls = []
    cli = CLITuiRuntimeMixin()
    cli._agent_running = cli._pet_reasoning = cli._interactive_turn = True
    cli._spinner_text = cli._last_scrollback_tool = "busy"
    cli._tool_start_time = 1.0
    cli._pending_tool_info = {}
    cli._voice_mode = cli._voice_continuous = True
    cli._voice_recording = cli._voice_tts = False
    cli._voice_barge_capture = SimpleNamespace(is_set=lambda: False)
    cli._last_turn_interrupted = False
    cli._app = SimpleNamespace(invalidate=lambda: calls.append("direct"))
    cli._pet_react_turn_end = lambda: None
    cli._turn_summary_emit = lambda: None
    cli._drain_interrupt_queue_to_pending_input = lambda: None
    cli._maybe_continue_goal_after_turn = lambda: None
    cli._maybe_complete_loop_tick_after_turn = lambda: None
    cli._voice_start_recording = lambda: calls.append("recording")
    cli._invalidate = lambda *, min_interval: calls.append(("invalidate", min_interval))
    cli._drain_process_notifications = lambda _source: None
    monkeypatch.setattr("hermes_cli.cli_tui_runtime_mixin.threading.Thread", _ImmediateThread)

    cli._tui_after_turn()

    assert calls == ["direct", "recording", ("invalidate", 0.0)]
