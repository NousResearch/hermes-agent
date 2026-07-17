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


class TestConfigSetCredentialRedaction:
    """Regression tests for maintainer review #pullrequestreview-4701297956
    finding 1: ``model.api_key`` and credential-shaped leaves must NOT be
    echoed in raw form when ``/config set`` or ``/config get`` prints them.
    Without the redaction wrapper the raw API key lands in terminal
    scrollback.
    """

    def test_set_masks_top_level_api_key(self):
        cli_obj = _make_cli(**{
            "model": {"default": "test-model", "provider": "auto", "api_key": "sk-supersecret-1234567890abcdef"},
        })
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set model.api_key sk-newsecret-1234567890abcdef")
        combined = " ".join(printed)
        assert "sk-newsecret-1234567890abcdef" not in combined, (
            "Raw api_key value must be masked in /config set output"
        )
        # The canonical mask preserves head/tail: 'sk-n...cdef'
        assert "sk-n" in combined
        assert "cdef" in combined

    def test_set_masks_top_level_token(self):
        cli_obj = _make_cli(**{
            "model": {"default": "test-model", "provider": "auto"},
        })
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set model.token mysecrettoken123456")
        combined = " ".join(printed)
        assert "mysecrettoken123456" not in combined
        # Sanity: a mask shape is still present
        assert "***" in combined or "..." in combined

    def test_set_does_not_mask_non_credential_leaf(self):
        # display.bell_on_complete is not credential-shaped — must pass through
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set display.bell_on_complete True")
        combined = " ".join(printed)
        assert "True" in combined


class TestConfigGetCredentialRedaction:
    """Regression tests for maintainer review finding 1 on /config get:
    requesting a credential-shaped leaf or a section containing one must
    print masked values, not the raw secret.
    """

    def test_get_masks_scalar_api_key(self):
        cli_obj = _make_cli(**{
            "model": {"default": "test-model", "provider": "auto", "api_key": "sk-supersecret-1234567890abcdef"},
        })
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get model.api_key")
        combined = " ".join(printed)
        assert "sk-supersecret-1234567890abcdef" not in combined
        # Mask head/tail preserved
        assert "sk-s" in combined
        assert "cdef" in combined

    def test_get_masks_api_key_inside_section(self):
        cli_obj = _make_cli(**{
            "model": {"default": "test-model", "provider": "auto", "api_key": "sk-supersecret-1234567890abcdef"},
        })
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get model")
        combined = " ".join(printed)
        # The raw value must not appear anywhere in the section dump
        assert "sk-supersecret-1234567890abcdef" not in combined
        # But other (non-credential) fields must still appear
        assert "test-model" in combined
        assert "auto" in combined

    def test_get_passes_through_non_credential_scalar(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)):
            cli_obj.process_command("/config get compression.threshold")
        combined = " ".join(printed)
        # 0.5 is the default and is not a credential
        assert "0.5" in combined


class TestCompressionRestartRequired:
    """Regression tests for maintainer review finding 2: ``compression.*``
    is documented as live-apply but ``_propagate_config_live()`` has no
    compression branch, so a ``/config set compression.threshold 0.75``
    silently does not affect the active agent's ContextCompressor.

    Marked restart-required so the user is told to restart. The active
    compressor is constructed from these values at agent init with
    constructor-time parameters (``threshold_percent``, ``protect_first_n``,
    ``protect_last_n``, ``target_ratio``) and runtime state
    (``threshold_tokens``) that are not safe to live-mutate.
    """

    def test_compression_threshold_shows_restart_hint(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set compression.threshold 0.75")
        combined = " ".join(printed)
        assert "requires restart" in combined, (
            "compression.* must surface the restart hint because the "
            "active ContextCompressor cannot live-apply the change"
        )

    def test_compression_enabled_shows_restart_hint(self):
        cli_obj = _make_cli()
        printed = []
        with patch("cli._cprint", side_effect=lambda t: printed.append(t)), \
             patch("cli.save_config_value", return_value=True):
            cli_obj.process_command("/config set compression.enabled false")
        combined = " ".join(printed)
        assert "requires restart" in combined

    def test_compression_set_does_not_call_propagate_live(self):
        # Regression guard: compression.* must NOT trigger the live-apply
        # path even if ``_propagate_config_live`` happens to learn how to
        # handle it in the future — keep restart-required authoritative
        # unless a real live-apply implementation lands.
        cli_obj = _make_cli()
        with patch("cli._cprint"), \
             patch("cli.save_config_value", return_value=True), \
             patch.object(HermesCLI, "_propagate_config_live") as mock_propagate:
            cli_obj.process_command("/config set compression.threshold 0.75")
        mock_propagate.assert_not_called()
