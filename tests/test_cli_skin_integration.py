from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import HermesCLI, _build_compact_banner, _rich_text_from_ansi
from hermes_cli.skin_engine import get_active_skin, set_active_skin


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


class TestCliSkinPromptIntegration:
    def test_default_prompt_fragments_use_default_symbol(self):
        cli = _make_cli_stub()

        set_active_skin("default")
        assert cli._get_tui_prompt_fragments() == [("class:prompt", "❯ ")]

    def test_ares_prompt_fragments_use_skin_symbol(self):
        cli = _make_cli_stub()

        set_active_skin("ares")
        assert cli._get_tui_prompt_fragments() == [("class:prompt", "⚔ ❯ ")]

    def test_secret_prompt_fragments_preserve_secret_state(self):
        cli = _make_cli_stub()
        cli._secret_state = {"response_queue": object()}

        set_active_skin("ares")
        assert cli._get_tui_prompt_fragments() == [("class:sudo-prompt", "🔑 ❯ ")]

    def test_icon_only_skin_symbol_still_visible_in_special_states(self):
        cli = _make_cli_stub()
        cli._secret_state = {"response_queue": object()}

        with patch("hermes_cli.skin_engine.get_active_prompt_symbol", return_value="⚔ "):
            assert cli._get_tui_prompt_fragments() == [("class:sudo-prompt", "🔑 ⚔ ")]

    def test_build_tui_style_dict_uses_skin_overrides(self):
        cli = _make_cli_stub()

        set_active_skin("ares")
        skin = get_active_skin()
        style_dict = cli._build_tui_style_dict()

        assert style_dict["prompt"] == skin.get_color("prompt")
        assert style_dict["input-rule"] == skin.get_color("input_rule")
        assert style_dict["prompt-working"] == f"{skin.get_color('banner_dim')} italic"
        assert style_dict["approval-title"] == f"{skin.get_color('ui_warn')} bold"

    def test_apply_tui_skin_style_updates_running_app(self):
        cli = _make_cli_stub()

        set_active_skin("ares")
        assert cli._apply_tui_skin_style() is True
        assert cli._app.style is not None
        cli._invalidate.assert_called_once_with(min_interval=0.0)

    def test_handle_skin_command_refreshes_live_tui(self, capsys):
        cli = _make_cli_stub()

        with patch("cli.save_config_value", return_value=True):
            cli._handle_skin_command("/skin ares")

        output = capsys.readouterr().out
        assert "Skin set to: ares (saved)" in output
        assert "Prompt + TUI colors updated." in output
        assert cli._app.style is not None


class TestCompactBannerSkinIntegration:
    def test_default_compact_banner_keeps_legacy_nous_hermes_branding(self):
        set_active_skin("default")

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch("cli.format_banner_version_label", return_value="Hermes Agent v0.1.0 (test)"):
            banner = _build_compact_banner()

        assert "NOUS HERMES" in banner

    def test_poseidon_compact_banner_uses_skin_branding_instead_of_nous_hermes(self):
        set_active_skin("poseidon")

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch("cli.format_banner_version_label", return_value="Hermes Agent v0.1.0 (test)"):
            banner = _build_compact_banner()

        assert "Poseidon Agent" in banner
        assert "NOUS HERMES" not in banner

    def test_poseidon_compact_banner_uses_skin_colors(self):
        set_active_skin("poseidon")
        skin = get_active_skin()

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch("cli.format_banner_version_label", return_value="Hermes Agent v0.1.0 (test)"):
            banner = _build_compact_banner()

        assert skin.get_color("banner_border") in banner
        assert skin.get_color("banner_title") in banner
        assert skin.get_color("banner_dim") in banner

    def test_compact_banner_shows_version_label(self):
        set_active_skin("default")

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch("cli.format_banner_version_label", return_value="Hermes Agent v1.0 (test) · upstream abc12345"):
            banner = _build_compact_banner()

        assert "upstream abc12345" in banner


class TestAnsiRichTextHelper:
    def test_preserves_literal_brackets(self):
        text = _rich_text_from_ansi("[notatag] literal")
        assert text.plain == "[notatag] literal"

    def test_strips_ansi_but_keeps_plain_text(self):
        text = _rich_text_from_ansi("\x1b[31mred\x1b[0m")
        assert text.plain == "red"
