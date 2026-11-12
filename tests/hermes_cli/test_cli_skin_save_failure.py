from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI


def _make_cli_stub():
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
    cli._tui_style_base = {
        "prompt": "#fff",
        "input-area": "#fff",
        "input-rule": "#aaa",
        "prompt-working": "#888 italic",
    }
    cli._app = SimpleNamespace(style=None)
    cli._invalidate = MagicMock()
    return cli


class TestSkinSaveFailure:
    def test_save_failure_reports_session_only_not_saved(self, capsys):
        cli = _make_cli_stub()
        with patch("cli.save_config_value", return_value=False):
            cli._handle_skin_command("/skin ares")
        out = capsys.readouterr().out
        assert "(saved)" not in out
        assert "session only" in out.lower()
        assert "save failed" in out.lower()

    def test_save_success_still_reports_saved(self, capsys):
        cli = _make_cli_stub()
        with patch("cli.save_config_value", return_value=True):
            cli._handle_skin_command("/skin ares")
        out = capsys.readouterr().out
        assert "(saved)" in out
