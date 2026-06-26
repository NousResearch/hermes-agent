"""Tests for /config set, /config get, and enhanced show_config display."""
from unittest.mock import MagicMock, patch
import pytest

from cli import HermesCLI


def _make_cli(**config_overrides):
    """Create a minimal HermesCLI stub for testing slash commands."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.config = {
        "display": {"personality": "", "show_reasoning": False, "bell_on_complete": False, "compact": False, "tool_progress": "all"},
        "compression": {"enabled": True, "threshold": 0.50},
        "terminal": {"backend": "local", "cwd": ".", "timeout": 60},
        "model": {"default": "test-model", "provider": "auto"},
        "agent": {"max_turns": 90},
    }
    cli_obj.config.update(config_overrides)
    cli_obj.console = MagicMock()
    cli_obj.agent = None
    cli_obj.conversation_history = []
    cli_obj.session_id = "test-session"
    cli_obj._pending_input = MagicMock()
    cli_obj.compact = False
    cli_obj.tool_progress_mode = "all"
    cli_obj.show_reasoning = False
    cli_obj.bell_on_complete = False
    cli_obj.api_key = "sk-test-dummy-key-12345678"
    cli_obj.model = "test-model"
    cli_obj.base_url = ""
    cli_obj.max_turns = 90
    cli_obj.enabled_toolsets = []
    cli_obj.verbose = False
    cli_obj.session_start = MagicMock()
    cli_obj.session_start.strftime.return_value = "2026-01-01 00:00:00"
    return cli_obj


class TestConfigSetValidation:
    """Key path validation rejects typos and unknown sections."""

    def test_unknown_section_rejected(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config set disploy.bell true")
        combined = " ".join(printed)
        assert "Unknown config section" in combined
        assert "disploy" in combined

    def test_unknown_parent_rejected(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config set display.nonexistent.key true")
        # "display.nonexistent" is not in config
        combined = " ".join(printed)
        assert "Unknown config section" in combined

    def test_valid_key_accepted(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set display.bell_on_complete true")
        combined = " ".join(printed)
        assert "✓" in combined
        assert "Unknown" not in combined


class TestConfigSetCoercion:
    """Values are coerced to bool/int/float where appropriate."""

    def test_bool_true_values(self):
        cli_obj = _make_cli()
        for val in ("true", "yes", "on", "True", "YES"):
            printed = []
            with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
                 patch("cli.save_config_value", return_value=True):
                cli_obj.process_command(f"/config set display.bell_on_complete {val}")
            combined = " ".join(printed)
            assert "True" in combined, f"Failed for {val}"

    def test_bool_false_values(self):
        cli_obj = _make_cli()
        for val in ("false", "no", "off", "False", "NO"):
            printed = []
            with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
                 patch("cli.save_config_value", return_value=True):
                cli_obj.process_command(f"/config set display.bell_on_complete {val}")
            combined = " ".join(printed)
            assert "False" in combined, f"Failed for {val}"

    def test_integer_coercion(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set agent.max_turns 50")
        combined = " ".join(printed)
        assert "50" in combined

    def test_float_coercion(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set compression.threshold 0.75")
        combined = " ".join(printed)
        assert "0.75" in combined


class TestConfigSetRestartHints:
    """Restart-required keys are flagged; live-apply keys are not."""

    def test_terminal_key_shows_restart(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set terminal.timeout 120")
        combined = " ".join(printed)
        assert "requires restart" in combined

    def test_display_key_no_restart_hint(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set display.bell_on_complete true")
        combined = " ".join(printed)
        assert "requires restart" not in combined
        assert "✓" in combined

    def test_display_key_live_applied(self):
        cli_obj = _make_cli()
        with patch("cli._cprint"), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set display.bell_on_complete true")
        assert cli_obj.bell_on_complete is True
        assert cli_obj.config["display"]["bell_on_complete"] is True


class TestConfigGet:
    """Test /config get <key>."""

    def test_get_existing_value(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get compression.threshold")
        combined = " ".join(printed)
        assert "0.5" in combined

    def test_get_section(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get compression")
        combined = " ".join(printed)
        assert "enabled" in combined
        assert "threshold" in combined

    def test_get_missing_value(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get display.nonexistent")
        combined = " ".join(printed)
        assert "not set" in combined


class TestConfigUsageHints:
    """Test usage messages for malformed /config commands."""

    def test_set_no_args(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config set")
        combined = " ".join(printed)
        assert "Usage" in combined

    def test_set_missing_value(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config set display.bell_on_complete")
        combined = " ".join(printed)
        assert "Usage" in combined

    def test_get_no_args(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get")
        combined = " ".join(printed)
        assert "Usage" in combined

    def test_unknown_subcommand(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config frobnicate key value")
        combined = " ".join(printed)
        assert "Unknown" in combined


class TestShowConfigDisplay:
    """Verify show_config renders the new sections."""

    def test_display_section_present(self):
        cli_obj = _make_cli()
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(" ".join(str(x) for x in a))):
            cli_obj.show_config()
        combined = "\n".join(printed)
        assert "Display" in combined
        assert "Personality" in combined
        assert "Compression" in combined
        assert "Timezone" in combined

    def test_tip_line_present(self):
        cli_obj = _make_cli()
        printed = []
        with patch("builtins.print", side_effect=lambda *a, **kw: printed.append(" ".join(str(x) for x in a))):
            cli_obj.show_config()
        combined = "\n".join(printed)
        assert "/config set" in combined
        assert "/config get" in combined
