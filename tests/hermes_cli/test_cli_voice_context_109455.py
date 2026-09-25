"""Structured voice_context wiring in CLIChatTurnMixin (#109455).

The existing ``voice_prefix`` (issue #65827) bakes a prose note into the API-only message so
the model speaks concisely; these tests cover the newer, structured counterpart that lets a
``pre_llm_call`` hook branch on the signal instead of pattern-matching that prose.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cli import _ChatTurn
from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


class _VoiceAudioHost(CLIChatTurnMixin):
    """Minimal stand-in covering only what ``_chat_setup_turn_audio`` touches."""

    def __init__(self, *, voice_mode: bool):
        self._voice_mode = voice_mode
        self._voice_continuous = False
        self._voice_tts = False


class TestChatSetupTurnAudioVoiceContext:
    def test_voice_input_sets_structured_voice_context(self):
        host = _VoiceAudioHost(voice_mode=True)
        turn = _ChatTurn()

        host._chat_setup_turn_audio(turn, "what's the weather", voice_input=True)

        assert turn.voice_context == {
            "input_modality": "voice", "voice_session_active": True, "client_surface": "cli",
        }
        assert turn.voice_prefix  # existing #65827 prose behavior is untouched

    def test_voice_input_outside_continuous_mode_reports_session_inactive(self):
        # A one-off dictated utterance (push-to-talk), not standing voice mode.
        host = _VoiceAudioHost(voice_mode=False)
        turn = _ChatTurn()

        host._chat_setup_turn_audio(turn, "hi", voice_input=True)

        assert turn.voice_context["voice_session_active"] is False

    def test_typed_input_leaves_voice_context_unset(self):
        host = _VoiceAudioHost(voice_mode=False)
        turn = _ChatTurn()

        host._chat_setup_turn_audio(turn, "hello", voice_input=False)

        assert turn.voice_context is None
        assert turn.voice_prefix == ""


class _RunAgentHost(CLIChatTurnMixin):
    """Minimal stand-in covering only what ``_chat_run_agent`` touches."""

    def __init__(self):
        self.session_id = "sess-1"
        self.conversation_history = [{"role": "user", "content": "hi"}]
        self.agent = MagicMock()
        self.agent.run_conversation.return_value = {
            "final_response": "ok", "messages": [], "api_calls": 1, "completed": True,
        }
        for _cb in ("_sudo_password_callback", "_approval_callback", "_secret_capture_callback",
                    "_vault_unlock_callback", "_vault_save_login_callback", "_vault_code_callback"):
            setattr(self, _cb, MagicMock())
        self._pending_model_switch_note = None
        self._pending_skills_reload_note = None
        self._pending_moa_config = None
        self._pending_moa_disable_after_turn = False
        self._pending_one_turn_model_restore = None
        self._flush_credit_notices = MagicMock()


@pytest.fixture(autouse=True)
def _no_speech_interrupt():
    with patch("tools.tts_streaming.take_speech_interrupted", return_value=False):
        yield


class TestChatRunAgentForwardsVoiceContext:
    def test_voice_turn_context_reaches_run_conversation(self):
        host = _RunAgentHost()
        turn = _ChatTurn()
        turn.voice_context = {
            "input_modality": "voice", "voice_session_active": True, "client_surface": "cli",
        }

        host._chat_run_agent(turn, "what's the weather")

        _, kwargs = host.agent.run_conversation.call_args
        assert kwargs["voice_context"] == turn.voice_context

    def test_typed_turn_forwards_none(self):
        host = _RunAgentHost()
        turn = _ChatTurn()  # voice_context defaults to None

        host._chat_run_agent(turn, "hello")

        _, kwargs = host.agent.run_conversation.call_args
        assert kwargs["voice_context"] is None
