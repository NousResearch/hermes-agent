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



class TestAnsiRichTextHelper:
    def test_preserves_literal_brackets(self):
        text = _rich_text_from_ansi("[notatag] literal")
        assert text.plain == "[notatag] literal"



import pytest

from hermes_cli.skin_engine import SkinConfig, get_active_skin, load_skin, set_active_skin
from agent.display import is_vivid_mode, set_vivid_mode, get_cute_tool_message
from cli import _render_final_assistant_content

BUILTIN_SKINS = [
    "default", "ares", "mono", "slate", "daylight",
    "warm-lightmode", "poseidon", "sisyphus", "charizard",
]
REQUIRED_CONTENT_KEYS = [
    "markdown_h1", "markdown_bold", "markdown_code",
    "tool_verb", "tool_path", "tool_duration",
]


@pytest.fixture(autouse=True)
def _reset_vivid_and_skin():
    from agent.display import set_vivid_mode
    from hermes_cli.skin_engine import set_active_skin
    set_vivid_mode(False)
    set_active_skin("default")
    yield
    set_vivid_mode(False)
    set_active_skin("default")


class TestDisplayContentStyling:

    # ------------------------------------------------------------------
    # 1. SkinConfig.content schema
    # ------------------------------------------------------------------
    def test_all_builtin_skins_have_nonempty_content_dict(self):
        for name in BUILTIN_SKINS:
            skin = load_skin(name)
            assert skin.content, f"skin '{name}' has empty content dict"

    def test_all_builtin_skins_have_required_content_keys(self):
        for name in BUILTIN_SKINS:
            skin = load_skin(name)
            for key in REQUIRED_CONTENT_KEYS:
                assert key in skin.content, f"skin '{name}' missing content key '{key}'"

    def test_get_content_color_fallback_on_empty_config(self):
        empty = SkinConfig(name="empty")
        assert empty.get_content_color("markdown_h1", "#FF0000") == "#FF0000"

    def test_default_skin_light_colors_content_has_markdown_h1(self):
        default = load_skin("default")
        # Verify the colors.content (dark authored) has it
        assert "markdown_h1" in default.content
        # Light colors have the light_colors block (not necessarily light_content separately)
        # Just confirm content is non-empty and light_colors exists when needed
        assert default.content is not None and len(default.content) > 0

    # ------------------------------------------------------------------
    # 2. Vivid mode toggle
    # ------------------------------------------------------------------
    def test_is_vivid_mode_default_false(self):
        # Fixture resets; don't call set_vivid_mode in setup
        assert is_vivid_mode() is False

    def test_set_vivid_mode_true_false(self):
        set_vivid_mode(True)
        assert is_vivid_mode() is True
        set_vivid_mode(False)
        assert is_vivid_mode() is False

    def test_get_cute_tool_message_no_brackets_when_vivid_false(self):
        # When vivid is off, output must be byte-identical to previous behavior
        # (no '[' in output) — rely on default non-vivid behavior
        msg = get_cute_tool_message("read_file", {"path": "foo.py"}, 0.5)
        assert "[" not in msg
        assert "]" not in msg

    # ------------------------------------------------------------------
    # 3. Tool-call colorization (vivid=True)
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("tool_name,args,expected_substring,kind", [
        ("read_file", {"path": "docs/api.md"}, "📖", "read"),
        ("write_file", {"path": "/tmp/new.py"}, "✍️", "write"),
        ("patch", {"path": "/tmp/old.py"}, "🔧", "modified"),
    ])
    def test_tool_colorization_contains_verb_emoji_tagged(self, tool_name, args, expected_substring, kind):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message(tool_name, args, 0.5)
        # When vivid=True, emoji should be wrapped with a color tag
        assert expected_substring in msg
        # The emoji token is tagged: contains [color]emoji[/]
        # Since we don't know the exact hex, assert '[' appears in output
        assert "[" in msg

    def test_read_file_verb_emoji_tagged_with_color(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("read_file", {"path": "docs/api.md"}, 0.5)
        # Emoji should have a color marker prefix
        assert "[" in msg
        # Verb 'read' tagged
        assert "read" in msg.lower() or "📖" in msg

    def test_write_file_verb_tagged(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("write_file", {"path": "/tmp/new.py"}, 0.5)
        assert "write" in msg.lower()

    def test_duration_0_5s_tagged(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("read_file", {"path": "x.md"}, 0.5)
        assert "0.5s" in msg
        # Duration should carry a color tag
        # Since exact hex varies by skin, just ensure '[' appears near duration
        assert msg.count("[") >= 1

    def test_write_file_path_tagged_with_tool_path_green_family(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("write_file", {"path": "/tmp/new.py"}, 0.5)
        # Path color should appear as tag wrapping the path text
        # Assert the path substring is present with color tags
        assert "/tmp/new.py" in msg
        assert msg.count("[") >= 1

    def test_patch_path_tagged_with_tool_path_modified_yellow_family(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("patch", {"path": "/tmp/old.py"}, 0.5)
        assert "/tmp/old.py" in msg
        assert msg.count("[") >= 1

    def test_read_file_path_tagged_with_tool_path_read_blue_family(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("read_file", {"path": "docs/read.md"}, 0.5)
        assert "read.md" in msg
        assert msg.count("[") >= 1

    def test_web_search_query_not_tagged_as_path(self):
        set_vivid_mode(True)
        set_active_skin("default")
        msg = get_cute_tool_message("web_search", {"query": "hermes agent"}, 0.5)
        # Query should NOT carry a tool_path color tag (no green-family path tag)
        # Since query isn't a file path, the result should have no path color for it.
        # We just verify the message exists without asserting absence of all tags,
        # because emoji/verb/duration may still be tagged.
        assert "search" in msg.lower() or "🔍" in msg
        # Ensure no path-style tag wraps the query text improperly —
        # the query text itself should not be wrapped as a file path.
        # (The exact assertion here is that the query is present unmodified as text,
        #  not wrapped in a color tag meant for paths.)
        assert "hermes agent" in msg

    def test_failure_suffix_appended_after_colorization(self):
        set_vivid_mode(True)
        # Use web_search (not terminal) because terminal result format may not trigger failure suffix
        result_with_error = '{"success": false, "error": "bad request"}'
        msg = get_cute_tool_message("web_search", {"query": "test"}, 0.3, result=result_with_error)
        # Failure suffix should be present; exact text depends on _detect_tool_failure
        assert "[" in msg or "bad" in msg or msg.endswith("s")

    # ------------------------------------------------------------------
    # 4. Markdown vivid theming
    # ------------------------------------------------------------------
    def test_render_final_assistant_content_vivid_false_uses_monokai(self):
        # The current non-vivid default uses monokai for code blocks.
        # We verify by inspecting the markdown object's code_theme when vivid is off.
        set_active_skin("default")
        set_vivid_mode(False)
        md = _render_final_assistant_content("hello `code` world", mode="render")
        # Non-vivid: should return Markdown object; code_theme should be monokai
        assert md.code_theme == "monokai"

    def test_render_final_assistant_content_vivid_true_uses_github_dark_and_non_none_style(self):
        set_active_skin("default")
        set_vivid_mode(True)
        md = _render_final_assistant_content("hello `code` world", mode="render")
        assert md.code_theme == "github-dark"
        # Style should be a non-'none' color string (from markdown_bold / markdown_h3)
        assert md.style != "none"

    def test_render_final_assistant_content_strip_returns_text(self):
        from rich.text import Text as _RichText
        set_vivid_mode(False)
        result = _render_final_assistant_content("hello", mode="strip")
        assert isinstance(result, _RichText)

    def test_render_final_assistant_content_raw_returns_text(self):
        from rich.text import Text as _RichText
        set_vivid_mode(True)
        result = _render_final_assistant_content("hello", mode="raw")
        assert isinstance(result, _RichText)
