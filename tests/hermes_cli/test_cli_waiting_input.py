import queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.mark.parametrize("interrupt,steer", [("late interrupt", ""), ("", "revised scope"), ("late interrupt", "revised scope")])
def test_waiting_turn_preserves_pending_input_without_display(monkeypatch, interrupt, steer):
    import cli as cli_module
    from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin

    cli = CLIChatTurnMixin()
    cli._secret_capture_callback = Mock()
    cli._ensure_runtime_credentials = lambda: True
    cli._resolve_turn_agent_config = lambda _message: {"signature": "same", "model": "fake", "runtime": {}}
    cli._active_agent_route_signature = "same"
    cli.agent = SimpleNamespace(_interrupt_requested=True, clear_interrupt=Mock())
    cli._init_agent = lambda **_kwargs: True
    cli._chat_route_images = lambda message, _images: message
    cli._chat_expand_context_references = lambda message: (message, None)
    cli._chat_stage_user_message = lambda *_args: None
    cli._reset_stream_state = lambda: None
    cli._chat_setup_turn_audio = lambda *_args: None
    cli._chat_release_turn_audio = lambda *_args: None
    cli._chat_settle_turn = lambda *_args: None
    cli._pending_input = queue.Queue()
    cli._interrupt_queue = queue.Queue()

    def worker(turn, _message):
        turn.result = {"waiting_on_delegates": True, "pending_steer": steer,
                       "completed": False, "final_response": "hidden waiting text"}

    def monitor(turn, thread):
        thread.join()
        return interrupt

    cli._chat_run_agent = worker
    cli._chat_monitor_agent_thread = monitor
    cli._chat_print_reasoning_box = Mock()
    cli._chat_print_response_panel = Mock()
    cli._ring_bell = Mock()
    cli._emit_focus_recovery_line = Mock()
    cli._voice_tts = True
    cli._voice_speak_response_async = Mock()
    monkeypatch.setattr(cli_module, "set_secret_capture_callback", lambda *_a: None)
    monkeypatch.setattr(cli_module, "ChatConsole", lambda: SimpleNamespace(print=lambda *_a: None))

    assert cli.chat("initial task") is None
    assert list(cli._pending_input.queue) == [text for text in (interrupt, steer) if text]
    assert cli._last_turn_interrupted is False
    if interrupt:
        cli.agent.clear_interrupt.assert_called_once()
    cli._chat_print_reasoning_box.assert_not_called()
    cli._chat_print_response_panel.assert_not_called()
    cli._ring_bell.assert_not_called()
    cli._voice_speak_response_async.assert_not_called()
