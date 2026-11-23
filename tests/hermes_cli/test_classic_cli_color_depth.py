"""Classic-CLI output color depth: the hand-built CPR-disabled output must not clamp skin colors to 256."""

from __future__ import annotations

import io

import pytest
from prompt_toolkit.output import ColorDepth

from cli import _build_cpr_disabled_output
from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides


@pytest.fixture(autouse=True)
def _clear_color_env(monkeypatch):
    for var in ("COLORTERM", "PROMPT_TOOLKIT_COLOR_DEPTH", "NO_COLOR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")


@pytest.mark.parametrize("value", ["truecolor", "24bit", "TrueColor"])
def test_truecolor_terminal_gets_exact_rgb(monkeypatch, value):
    monkeypatch.setenv("COLORTERM", value)
    out = _build_cpr_disabled_output(io.StringIO())
    assert out is not None
    assert out.get_default_color_depth() is ColorDepth.DEPTH_24_BIT
    assert out.enable_cpr is False


def test_without_colorterm_keeps_prompt_toolkit_default():
    out = _build_cpr_disabled_output(io.StringIO())
    assert out.get_default_color_depth() is ColorDepth.DEFAULT


def test_explicit_prompt_toolkit_depth_wins(monkeypatch):
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("PROMPT_TOOLKIT_COLOR_DEPTH", "DEPTH_4_BIT")
    out = _build_cpr_disabled_output(io.StringIO())
    assert out.get_default_color_depth() is ColorDepth.DEPTH_4_BIT


@pytest.mark.parametrize("term", ["dumb", "linux"])
def test_colorterm_does_not_promote_dumb_or_console_term(monkeypatch, term):
    # COLORTERM is inherited (e.g. Emacs M-x shell); TERM=dumb/linux must not be promoted to 24-bit.
    monkeypatch.setenv("TERM", term)
    monkeypatch.setenv("COLORTERM", "truecolor")
    out = _build_cpr_disabled_output(io.StringIO())
    assert out.get_default_color_depth() is not ColorDepth.DEPTH_24_BIT


def test_dumb_term_is_still_respected(monkeypatch):
    # term is passed through, so prompt_toolkit's TERM=dumb handling keeps working.
    monkeypatch.setenv("TERM", "dumb")
    monkeypatch.setenv("COLORTERM", "truecolor")
    out = _build_cpr_disabled_output(io.StringIO())
    assert out.get_default_color_depth() is ColorDepth.DEPTH_1_BIT


def test_menu_scrollbar_follows_skin(monkeypatch):
    from hermes_cli import skin_engine

    skin = skin_engine.SkinConfig(name="t", colors={
        "completion_menu_bg": "#142620", "completion_menu_meta_bg": "#0F1E19", "banner_dim": "#91AD9F",
    })
    monkeypatch.setattr(skin_engine, "get_active_skin", lambda: skin)
    styles = get_prompt_toolkit_style_overrides()
    assert styles["scrollbar.background"] == "bg:#0F1E19"
    assert styles["scrollbar.button"] == "bg:#91AD9F"


def test_menu_scrollbar_ignores_light_mode_remap(monkeypatch):
    # get_color is remapped darker in light mode; the thumb sits on a dark track, so it must use the raw skin color.
    from hermes_cli import skin_engine

    skin = skin_engine.SkinConfig(name="t", colors={"banner_dim": "#B8860B"})
    monkeypatch.setattr(skin, "get_color", lambda key, default="": "#5C4500")
    monkeypatch.setattr(skin_engine, "get_active_skin", lambda: skin)
    assert get_prompt_toolkit_style_overrides()["scrollbar.button"] == "bg:#B8860B"


# --- prints without an explicit output (banner, _cprint, history replay) ---------------------

_SKIN_TEXT = "\x1b[38;2;224;222;244;48;2;39;103;54mhi\x1b[0m"


def _emitted(monkeypatch, term, colorterm, paint):
    """Escape output of ``paint()`` through a bare Vt100_Output (no depth): the one pt's default session would use."""
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.output.vt100 import Vt100_Output

    monkeypatch.setenv("TERM", term)
    monkeypatch.setenv("COLORTERM", colorterm)
    buf = io.StringIO()
    with create_app_session(output=Vt100_Output(buf, lambda: Size(24, 80), term=term)):
        paint()
    return buf.getvalue()


def _banner():
    from hermes_cli.banner import cprint
    cprint(_SKIN_TEXT)


def _cprint_idle():
    import cli
    cli._cprint(_SKIN_TEXT)


def _replay():
    import cli
    saved = list(cli._OUTPUT_HISTORY)
    cli._OUTPUT_HISTORY.clear()
    try:
        cli._record_output_history(_SKIN_TEXT, force=True)
        cli._replay_output_history()
    finally:
        cli._OUTPUT_HISTORY.clear()
        cli._OUTPUT_HISTORY.extend(saved)


_PATHS = [pytest.param(_banner, id="banner.cprint"), pytest.param(_cprint_idle, id="_cprint"),
          pytest.param(_replay, id="history-replay")]


@pytest.mark.parametrize("paint", _PATHS)
def test_unbound_prints_use_24bit_on_truecolor(monkeypatch, paint):
    out = _emitted(monkeypatch, "xterm-256color", "truecolor", paint)
    assert "38;2;224;222;244" in out and "48;2;39;103;54" in out
    assert "38;5;" not in out and "48;5;" not in out


@pytest.mark.parametrize("paint", _PATHS)
def test_unbound_prints_stay_off_24bit_on_dumb_term(monkeypatch, paint):
    out = _emitted(monkeypatch, "dumb", "truecolor", paint)
    assert "38;2;" not in out and "48;2;" not in out
