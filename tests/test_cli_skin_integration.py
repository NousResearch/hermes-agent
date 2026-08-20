from io import StringIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from cli import (
    ChatConsole,
    HermesCLI,
    _build_compact_banner,
    _rich_text_from_ansi,
    _skin_markdown_theme,
)
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

    def test_ares_prompt_fragments_use_skin_symbol(self):
        cli = _make_cli_stub()

        set_active_skin("ares")
        assert cli._get_tui_prompt_fragments() == [("class:prompt", "⚔ ")]

    def test_secret_prompt_fragments_preserve_secret_state(self):
        cli = _make_cli_stub()
        cli._secret_state = {"response_queue": object()}

        set_active_skin("ares")
        assert cli._get_tui_prompt_fragments() == [("class:sudo-prompt", "🔑 ⚔ ")]


    def test_build_tui_style_dict_uses_skin_overrides(self):
        cli = _make_cli_stub()

        set_active_skin("ares")
        skin = get_active_skin()
        style_dict = cli._build_tui_style_dict()

        assert style_dict["prompt"] == skin.get_color("prompt")
        assert style_dict["input-rule"] == skin.get_color("input_rule")
        assert style_dict["prompt-working"] == f"{skin.get_color('banner_dim')} italic"
        assert style_dict["status-bar"] == (
            f"bg:{skin.get_color('status_bar_bg')} {skin.get_color('status_bar_text')}"
        )
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

    def test_poseidon_compact_banner_uses_skin_branding_instead_of_nous_hermes(self):
        set_active_skin("poseidon")

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch.dict(_build_compact_banner.__globals__, {"format_banner_version_label": lambda: "Hermes Agent v0.1.0 (test)"}):
            banner = _build_compact_banner()

        assert "Poseidon Agent" in banner
        assert "NOUS HERMES" not in banner

    def test_poseidon_compact_banner_uses_skin_colors(self):
        set_active_skin("poseidon")
        skin = get_active_skin()

        with patch("cli.shutil.get_terminal_size", return_value=SimpleNamespace(columns=90)), \
             patch.dict(_build_compact_banner.__globals__, {"format_banner_version_label": lambda: "Hermes Agent v0.1.0 (test)"}):
            banner = _build_compact_banner()

        assert skin.get_color("banner_border") in banner
        assert skin.get_color("banner_title") in banner
        assert skin.get_color("banner_dim") in banner


class TestMarkdownThemeSkinIntegration:

    def test_markdown_theme_maps_skin_accent_onto_rich_styles(self):
        set_active_skin("poseidon")
        skin = get_active_skin()
        accent = skin.get_color("ui_accent") or skin.get_color("banner_accent")

        theme = _skin_markdown_theme()

        # Rich normalises hex colors to lowercase when parsing a Style.
        assert theme is not None
        assert accent.lower() in str(theme.styles["markdown.h2"]).lower()
        assert accent.lower() in str(theme.styles["markdown.item.bullet"]).lower()

    def test_rendered_markdown_emits_skin_accent(self):
        from rich.console import Console
        from rich.markdown import Markdown

        set_active_skin("poseidon")
        skin = get_active_skin()
        accent = skin.get_color("ui_accent") or skin.get_color("banner_accent")
        red, green, blue = (int(accent[i:i + 2], 16) for i in (1, 3, 5))

        buffer = StringIO()
        Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            width=60,
            theme=_skin_markdown_theme(),
        ).print(Markdown("## Heading\n"))

        assert f"38;2;{red};{green};{blue}" in buffer.getvalue()

    def test_chat_console_attaches_the_skin_markdown_theme(self):
        set_active_skin("poseidon")
        skin = get_active_skin()
        accent = skin.get_color("ui_accent") or skin.get_color("banner_accent")

        assert accent.lower() in str(ChatConsole()._inner.get_style("markdown.h2")).lower()

    def test_markdown_theme_falls_back_to_rich_defaults_when_skin_unavailable(self):
        with patch("hermes_cli.skin_engine.get_active_skin", side_effect=RuntimeError("boom")):
            assert _skin_markdown_theme() is None



class TestAnsiRichTextHelper:
    def test_preserves_literal_brackets(self):
        text = _rich_text_from_ansi("[notatag] literal")
        assert text.plain == "[notatag] literal"

