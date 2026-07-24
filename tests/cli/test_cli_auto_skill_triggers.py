from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _ImmediateThread:
    def __init__(self, *, target=None, daemon=None):
        self._target = target
        self._alive = False
        self.ident = 1

    def start(self):
        self._alive = True
        try:
            if self._target is not None:
                self._target()
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        return None


def _make_cli_stub():
    import cli as cli_mod

    shell = cli_mod.HermesCLI.__new__(cli_mod.HermesCLI)
    shell._secret_capture_callback = MagicMock()
    shell._sudo_password_callback = MagicMock()
    shell._approval_callback = MagicMock()
    shell._ensure_runtime_credentials = MagicMock(return_value=True)
    shell._resolve_turn_agent_config = MagicMock(
        return_value={
            "signature": "same-route",
            "model": None,
            "runtime": None,
            "request_overrides": None,
        }
    )
    shell._active_agent_route_signature = "same-route"
    shell._init_agent = MagicMock(return_value=True)
    shell._reset_stream_state = MagicMock(
        side_effect=lambda: setattr(shell, "_stream_started", False)
    )
    shell._flush_stream = MagicMock()
    shell._flush_credit_notices = MagicMock()
    shell._invalidate = MagicMock()
    shell._transfer_session_yolo = MagicMock()
    shell._voice_speak_response_async = MagicMock()
    shell._pending_model_switch_note = None
    shell._pending_skills_reload_note = None
    shell._pending_moa_config = None
    shell._pending_moa_disable_after_turn = False
    shell._clarify_state = None
    shell._clarify_freetext = None
    shell._voice_tts = False
    shell._voice_mode = False
    shell._voice_continuous = False
    shell.show_reasoning = False
    shell.bell_on_complete = False
    shell.show_timestamps = False
    shell.final_response_markdown = "auto"
    shell.session_id = "session-123"
    shell.provider = "test-provider"
    shell.model = "test-model"
    shell.base_url = ""
    shell.api_key = ""
    shell.api_mode = "chat_completions"
    shell._session_db = None
    shell.conversation_history = []
    shell.console = SimpleNamespace(width=80)
    shell._stream_started = False
    shell._stream_box_opened = False
    shell._reasoning_shown_this_turn = False
    shell.agent = SimpleNamespace(
        session_id="session-123",
        max_iterations=90,
        run_conversation=MagicMock(
            return_value={
                "final_response": "",
                "messages": [],
                "completed": True,
                "response_previewed": True,
            }
        ),
    )
    return shell


def test_chat_auto_loads_triggered_skill_and_persists_original_message():
    import cli as cli_mod

    shell = _make_cli_stub()

    with (
        patch.object(cli_mod.threading, "Thread", _ImmediateThread),
        patch.object(cli_mod, "ChatConsole", return_value=SimpleNamespace(print=MagicMock())),
        patch.object(cli_mod, "_cprint"),
        patch.object(
            cli_mod,
            "find_triggered_skill_command",
            return_value=("/market-watch", {"name": "market-watch"}, "market"),
        ),
        patch.object(
            cli_mod,
            "build_skill_invocation_message",
            return_value="[expanded skill payload]",
        ),
    ):
        shell.chat("What is the market doing today?")

    shell.agent.run_conversation.assert_called_once()
    kwargs = shell.agent.run_conversation.call_args.kwargs
    assert kwargs["user_message"] == "[expanded skill payload]"
    assert kwargs["persist_user_message"] == "What is the market doing today?"


def test_chat_leaves_plain_messages_unchanged_when_no_trigger_matches():
    import cli as cli_mod

    shell = _make_cli_stub()

    with (
        patch.object(cli_mod.threading, "Thread", _ImmediateThread),
        patch.object(cli_mod, "ChatConsole", return_value=SimpleNamespace(print=MagicMock())),
        patch.object(cli_mod, "_cprint"),
        patch.object(cli_mod, "find_triggered_skill_command", return_value=None),
    ):
        shell.chat("Tell me a joke.")

    kwargs = shell.agent.run_conversation.call_args.kwargs
    assert kwargs["user_message"] == "Tell me a joke."
    assert kwargs["persist_user_message"] is None
