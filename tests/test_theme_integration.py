"""Quick integration smoke-test for the full theme pipeline.

Run with:
    pytest tests/test_theme_integration.py -v

Covers:
  - All 10 syntax schemes highlight Python without crashing
  - Skin switch updates syntax colors live (via callback)
  - Markdown cache rebuilds on skin switch
  - Diff colors come from active skin
  - display.py hex helpers and diff ANSI functions work
  - Context pressure bar uses skin hex colors
  - _pt_style() returns tuple from skin
  - skills_hub helpers read skin ui_ext
"""

import pytest


@pytest.fixture(autouse=True)
def reset_skin():
    from hermes_cli import skin_engine
    from agent import rich_output
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    skin_engine._invalidation_callbacks.clear()
    rich_output._MD_ANSI_CACHE = None
    rich_output._MD_VAL_CACHE = None
    yield
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"
    skin_engine._invalidation_callbacks.clear()
    rich_output._MD_ANSI_CACHE = None
    rich_output._MD_VAL_CACHE = None


# ---------------------------------------------------------------------------
# Syntax schemes
# ---------------------------------------------------------------------------

SCHEMES = [
    "hermes", "monokai", "dracula", "one-dark", "github-dark",
    "nord", "catppuccin", "tokyo-night", "gruvbox", "solarized-dark",
]

PYTHON_SNIPPET = "def hello(name: str) -> str:\n    # greet\n    return f'hi {name}'\n"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_syntax_scheme_highlights_python(scheme):
    from hermes_cli.skin_engine import load_skin, set_active_skin
    from agent.rich_output import SyntaxHighlighter

    set_active_skin("default")
    # Patch syntax_scheme on the fly via a user skin override
    from hermes_cli import skin_engine
    skin_engine._active_skin = None  # force reload
    skin = load_skin("default")
    skin.syntax_scheme = scheme
    skin_engine._active_skin = skin

    hl = SyntaxHighlighter()
    result = hl.to_ansi(PYTHON_SNIPPET, "python")
    assert "\033[" in result, f"scheme {scheme!r} produced no ANSI output"


@pytest.mark.parametrize("scheme", SCHEMES)
def test_syntax_scheme_has_diff_tokens_styled(scheme):
    """diff_deleted / diff_inserted must never be unstyled in any scheme."""
    from hermes_cli.skin_engine import SYNTAX_SCHEMES
    styles = SYNTAX_SCHEMES[scheme]
    assert "diff_deleted" in styles, f"{scheme}: missing diff_deleted"
    assert "diff_inserted" in styles, f"{scheme}: missing diff_inserted"


def test_syntax_refresh_on_skin_switch():
    """SyntaxHighlighter.refresh() must change output when skin changes scheme."""
    from agent.rich_output import SyntaxHighlighter
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("default")   # hermes scheme
    hl = SyntaxHighlighter()
    default_out = hl.to_ansi(PYTHON_SNIPPET, "python")

    set_active_skin("charizard")  # monokai scheme — different keyword color
    hl.refresh()                  # refresh() is what the callback calls
    monokai_out = hl.to_ansi(PYTHON_SNIPPET, "python")

    assert default_out != monokai_out


def test_syntax_bold_toggle_strips_syntax_token_bold(monkeypatch):
    from hermes_cli import skin_engine

    monkeypatch.setattr(skin_engine, "_syntax_bold_enabled", lambda: False)
    styles = skin_engine.get_active_skin().get_syntax_styles()

    assert styles["keyword"] == "blue"
    assert styles["keyword_type"] == "cyan"
    assert styles["name_class"] == "yellow"
    assert styles["name_function"] == "yellow"
    assert styles["operator_word"] == "blue"


def test_diff_filename_style_uses_active_skin():
    from agent.rich_output import DiffRenderer
    from hermes_cli.skin_engine import get_active_skin, set_active_skin

    set_active_skin("default")
    skin = get_active_skin()
    skin.diff["filename"] = "bold #123456"

    diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
    header = DiffRenderer().to_lines(diff)[0]

    assert "\x1b[1;" in header or ";1;" in header
    assert "38;2;18;52;86" in header


# ---------------------------------------------------------------------------
# Markdown cache
# ---------------------------------------------------------------------------

def test_md_cache_builds_on_first_access():
    from agent.rich_output import _md_ansi, _MD_ANSI_CACHE
    result = _md_ansi("link")
    assert isinstance(result, str)
    assert len(result) > 0


def test_md_cache_rebuilds_after_skin_switch():
    from agent import display  # register callback
    from agent.rich_output import _md_ansi, _rebuild_md_cache
    from hermes_cli.skin_engine import set_active_skin, get_active_skin

    set_active_skin("default")
    _rebuild_md_cache()
    before = _md_ansi("link")

    # Switch to a skin with custom link color
    set_active_skin("default")
    skin = get_active_skin()
    skin.markdown["link"] = "#FF0000"  # red
    _rebuild_md_cache()
    after = _md_ansi("link")

    assert before != after


def test_md_val_returns_bullets_list():
    from agent.rich_output import _md_val, _rebuild_md_cache
    _rebuild_md_cache()
    bullets = _md_val("bullets")
    assert isinstance(bullets, list)
    assert len(bullets) >= 1


def test_md_val_returns_blockquote_marker():
    from agent.rich_output import _md_val, _rebuild_md_cache
    _rebuild_md_cache()
    marker = _md_val("blockquote_marker")
    assert isinstance(marker, str)
    assert len(marker) == 1


def test_apply_block_line_heading():
    from agent.rich_output import apply_block_line, _rebuild_md_cache
    _rebuild_md_cache()
    result = apply_block_line("## Section")
    assert "\033[" in result


def test_apply_block_line_blockquote_uses_skin_marker():
    from agent.rich_output import apply_block_line, _md_val, _rebuild_md_cache
    _rebuild_md_cache()
    result = apply_block_line("> some quote")
    marker = _md_val("blockquote_marker") or "▌"
    assert marker in result


def test_apply_inline_markdown_link_uses_skin_color():
    from agent.rich_output import apply_inline_markdown, _md_ansi, _rebuild_md_cache
    _rebuild_md_cache()
    result = apply_inline_markdown("[click](https://example.com)")
    link_ansi = _md_ansi("link")
    assert link_ansi in result


# ---------------------------------------------------------------------------
# Diff colors
# ---------------------------------------------------------------------------

def test_diff_cfg_returns_default_hex():
    from agent.rich_output import _diff_cfg
    assert _diff_cfg("deletion_bg") == "#781414"
    assert _diff_cfg("addition_bg") == "#145a14"
    assert _diff_cfg("deletion_marker_fg") == "#FF7B72"
    assert _diff_cfg("addition_marker_fg") == "#56D364"


def test_diff_cfg_reflects_skin_override():
    from agent.rich_output import _diff_cfg
    from hermes_cli.skin_engine import get_active_skin

    skin = get_active_skin()
    skin.diff["deletion_bg"] = "#FF0000"
    skin.diff["deletion_marker_fg"] = "#AA0000"
    assert _diff_cfg("deletion_bg") == "#FF0000"
    assert _diff_cfg("deletion_marker_fg") == "#AA0000"


def test_builtin_skin_diff_palette_overrides_defaults():
    from hermes_cli.skin_engine import set_active_skin, get_active_skin

    set_active_skin("mono")
    mono = get_active_skin()
    assert mono.get_diff("deletion_bg") == "#3A3030"
    assert mono.get_diff("addition_bg") == "#2F3A30"
    assert mono.get_diff("deletion_marker_fg") == "#D0D0D0"
    assert mono.get_diff("addition_marker_fg") == "#F0F0F0"

    set_active_skin("poseidon")
    poseidon = get_active_skin()
    assert poseidon.get_diff("intra_del_bg") == "#5A4060"
    assert poseidon.get_diff("intra_add_bg") == "#2F6259"

    set_active_skin("sisyphus")
    sisyphus = get_active_skin()
    assert sisyphus.get_diff("deletion_marker_fg") == "#D6D6D6"
    assert sisyphus.get_diff("addition_marker_fg") == "#F5F5F5"


def test_hermes_scheme_styles_operator_words():
    from hermes_cli.skin_engine import SYNTAX_SCHEMES

    assert SYNTAX_SCHEMES["hermes"]["operator_word"] == "bold blue"


def test_diff_renderer_produces_ansi():
    from agent.rich_output import DiffRenderer
    renderer = DiffRenderer()
    lines = renderer.to_lines("--- a/f\n+++ b/f\n@@ -1 +1 @@\n-old\n+new\n")
    combined = "\n".join(lines)
    assert "\033[" in combined


# ---------------------------------------------------------------------------
# display.py hex helpers + inline diff
# ---------------------------------------------------------------------------

def test_hex_to_ansi_fg():
    from agent.display import _hex_to_ansi_fg
    assert _hex_to_ansi_fg("#FF7B72") == "\033[38;2;255;123;114m"


def test_hex_to_ansi_bg():
    from agent.display import _hex_to_ansi_bg
    assert _hex_to_ansi_bg("#145a14") == "\033[48;2;20;90;20m"


def test_hex_to_ansi_fg_bad_input_returns_empty():
    from agent.display import _hex_to_ansi_fg
    assert _hex_to_ansi_fg("not-a-color") == ""


def test_ansi_minus_contains_deletion_bg():
    from agent.display import _ansi_minus
    result = _ansi_minus()
    # Should contain fg + bg components
    assert "\033[38;2;" in result
    assert "\033[48;2;" in result


def test_context_pressure_bar_uses_hex_color():
    from agent.display import format_context_pressure
    result = format_context_pressure(0.5, 100000, 0.7)
    # Should contain truecolor escape (38;2;R;G;B) from _ctx_color
    assert "\033[38;2;" in result


def test_context_pressure_bar_crit_color():
    """At 97% the crit color should be used."""
    from agent.display import format_context_pressure
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    crit_hex = skin.get_ui_ext("context_bar_crit", "#ef5350")
    result = format_context_pressure(0.97, 100000, 0.7)
    # Parse expected RGB from crit hex
    h = crit_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    assert f"\033[38;2;{r};{g};{b}m" in result


# ---------------------------------------------------------------------------
# UI chrome helpers
# ---------------------------------------------------------------------------

def test_pt_style_returns_tuple():
    from hermes_cli.main import _pt_style
    result = _pt_style("menu_cursor", ["fg_green", "bold"])
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_pt_style_string_input_splits():
    from hermes_cli.main import _pt_style
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    skin.ui_ext["menu_cursor"] = "fg_blue bold"
    result = _pt_style("menu_cursor", ["fg_green", "bold"])
    assert result == ("fg_blue", "bold")


def test_skills_hub_col_accent_returns_string():
    from hermes_cli.skills_hub import _col_accent, _col_dim, _panel_border
    assert isinstance(_col_accent(), str)
    assert isinstance(_col_dim(), str)
    assert isinstance(_panel_border(), str)


def test_skills_hub_col_accent_reflects_skin():
    from hermes_cli.skills_hub import _col_accent
    from hermes_cli.skin_engine import get_active_skin
    skin = get_active_skin()
    skin.ui_ext["table_col_accent"] = "bold magenta"
    assert _col_accent() == "bold magenta"


# ---------------------------------------------------------------------------
# Skin validation
# ---------------------------------------------------------------------------

def test_unknown_syntax_scheme_falls_back_to_hermes():
    from hermes_cli.skin_engine import _build_skin_config
    skin = _build_skin_config({"name": "t", "syntax_scheme": "nonexistent"})
    assert skin.syntax_scheme == "hermes"


def test_invalid_hex_in_diff_falls_back_to_default():
    from hermes_cli.skin_engine import _build_skin_config, _DIFF_DEFAULTS
    skin = _build_skin_config({"name": "t", "diff": {"deletion_bg": "notahex"}})
    assert skin.diff["deletion_bg"] == _DIFF_DEFAULTS["deletion_bg"]


def test_menu_cursor_string_splits_on_load():
    from hermes_cli.skin_engine import _build_skin_config
    skin = _build_skin_config({"name": "t", "ui_ext": {"menu_cursor": "fg_blue bold"}})
    assert skin.ui_ext["menu_cursor"] == ["fg_blue", "bold"]


def test_get_syntax_styles_merges_overrides():
    from hermes_cli.skin_engine import _build_skin_config
    skin = _build_skin_config({
        "name": "t",
        "syntax_scheme": "monokai",
        "syntax": {"keyword": "bold #123456"},
    })
    styles = skin.get_syntax_styles()
    assert styles["keyword"] == "bold #123456"   # override wins
    assert styles["string"] == "#E6DB74"          # monokai base unchanged


def test_callback_fires_on_skin_switch():
    from hermes_cli.skin_engine import set_active_skin, register_skin_callback
    fired = []
    register_skin_callback(lambda: fired.append(1))
    set_active_skin("ares")
    assert len(fired) == 1


def test_all_builtin_skins_have_syntax_scheme():
    from hermes_cli.skin_engine import _BUILTIN_SKINS, SYNTAX_SCHEMES
    for name, data in _BUILTIN_SKINS.items():
        scheme = data.get("syntax_scheme", "hermes")
        assert scheme in SYNTAX_SCHEMES, f"{name}: syntax_scheme {scheme!r} not in SYNTAX_SCHEMES"
