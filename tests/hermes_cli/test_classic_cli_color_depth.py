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


def test_dumb_term_is_still_respected(monkeypatch):
    # term is passed through, so prompt_toolkit's TERM=dumb handling keeps working.
    monkeypatch.setenv("TERM", "dumb")
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
