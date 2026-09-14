from types import SimpleNamespace
from unittest.mock import patch

from cli import HermesCLI


def _cli_stub():
    cli = HermesCLI.__new__(HermesCLI)
    cli._sudo_state = None
    cli._secret_state = None
    cli._approval_state = None
    cli._clarify_state = None
    cli._clarify_freetext = False
    cli._command_running = False
    cli._agent_running = False
    cli._voice_recording = False
    cli._voice_processing = False
    cli._voice_mode = False
    cli._command_spinner_frame = lambda: "⟳"
    cli._app = SimpleNamespace()
    return cli


def test_prompt_render_uses_cached_peer_presence():
    cli = _cli_stub()

    with patch("hermes_cli.peer_presence.peer_presence_pill", return_value="●2") as pill:
        cli._get_tui_prompt_fragments()

    pill.assert_called_once_with()


def test_peer_bar_render_uses_cached_peer_presence():
    cli = _cli_stub()
    summary = {
        "live_count": 2,
        "active_count": 1,
        "idle_count": 1,
        "offline_count": 0,
    }

    with patch("hermes_cli.peer_presence.peer_presence_summary", return_value=summary) as peer:
        fragments = cli._get_peer_presence_fragments()

    peer.assert_called_once_with()
    assert "● 2 Live" in "".join(text for _style, text in fragments)


def test_turn_settlement_invalidates_peer_presence_cache():
    cli = _cli_stub()
    cli._prompt_start_time = None
    cli._last_turn_finished_at = None
    cli._flush_stream = lambda: None
    cli.conversation_history = []
    cli.agent = None
    cli.session_id = "session-a"
    turn = SimpleNamespace(use_streaming_tts=False, text_queue=None, tts_thread=None, result=None)

    with patch("hermes_cli.peer_presence.clear_peer_presence_cache") as clear, \
            patch("hermes_cli.cli_chat_turn_mixin.time.sleep"):
        cli._chat_settle_turn(turn)

    clear.assert_called_once_with()
