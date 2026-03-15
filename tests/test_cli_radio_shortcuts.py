from unittest.mock import patch

from cli import HermesCLI


class _FakeBuffer:
    def __init__(self, text=""):
        self.text = text


class _FakeApp:
    def __init__(self, text=""):
        self.current_buffer = _FakeBuffer(text)


def _make_cli_stub(buffer_text=""):
    cli = HermesCLI.__new__(HermesCLI)
    cli._app = _FakeApp(buffer_text)
    cli._radio_menu_state = None
    cli._clarify_state = None
    cli._approval_state = None
    cli._sudo_state = None
    cli._secret_state = None
    return cli


def test_can_toggle_radio_expanded_requires_active_radio_and_empty_buffer():
    cli = _make_cli_stub(buffer_text="")

    with patch("radio.player.HermesRadio.active", return_value=True):
        assert cli._can_toggle_radio_expanded() is True

    with patch("radio.player.HermesRadio.active", return_value=False):
        assert cli._can_toggle_radio_expanded() is False


def test_can_toggle_radio_expanded_blocks_when_user_is_typing():
    cli = _make_cli_stub(buffer_text="vibe check")

    with patch("radio.player.HermesRadio.active", return_value=True):
        assert cli._can_toggle_radio_expanded() is False


def test_can_toggle_radio_expanded_blocks_during_modal_states():
    cli = _make_cli_stub()
    cli._approval_state = {"choices": ["yes", "no"]}

    with patch("radio.player.HermesRadio.active", return_value=True):
        assert cli._can_toggle_radio_expanded() is False
