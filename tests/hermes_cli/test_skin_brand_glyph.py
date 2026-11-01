"""Skin-driven brand glyphs: a skin's ``branding.icon`` reaches every surface that
renders the agent's mark, and stock skins keep the caduceus.

Three surfaces read the mark independently — the status bar, the response-box
label, and the TUI's brand slot — so a skin that recolors the agent and renames
it could still be preceded by ``☤`` on all three.
"""

import re

import pytest


def _plain(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s or "")


@pytest.fixture
def user_skin(tmp_path, monkeypatch):
    """A user skin whose glyph is not the caduceus, active for the test."""
    from hermes_cli import skin_engine

    skins_dir = tmp_path / "skins"
    skins_dir.mkdir()
    (skins_dir / "dingir.yaml").write_text(
        "name: dingir\n"
        "branding:\n"
        "  agent_name: Arkad\n"
        "  icon: '𒀭'\n"
        "  response_label: ' 𒀭 Arkad '\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(skin_engine, "_skins_dir", lambda: skins_dir)
    skin_engine.set_active_skin("dingir")
    yield skin_engine
    skin_engine._active_skin = None
    skin_engine._active_skin_name = "default"


def test_status_mark_follows_the_active_skin(user_skin):
    from hermes_cli.cli_status_bar_mixin import _status_mark

    assert _status_mark() == "𒀭"


def test_status_bar_falls_back_to_the_skin_mark_not_the_caduceus(user_skin):
    """The one-line bar's exception fallback carries the skin's mark rather
    than a literal caduceus."""
    from hermes_cli.cli_status_bar_mixin import CLIStatusBarMixin

    class Bar(CLIStatusBarMixin):
        model = "deepseek-flash"

        def _get_status_bar_snapshot(self):
            raise RuntimeError("snapshot unavailable")

    assert Bar()._build_status_bar_text(80) == "𒀭 deepseek-flash"


def test_response_label_follows_the_active_skin(user_skin):
    from hermes_cli import cli_modern_layout

    assert cli_modern_layout._brand_glyph() == ("𒀭", "Arkad")
    assert "𒀭 Arkad" in _plain(cli_modern_layout.answer_label())


def test_glyph_only_label_is_the_mark_not_a_caduceus_fallback(monkeypatch):
    """A symbol-only ``response_label`` is a deliberate choice; falling back to
    the built-in caduceus would silently discard it."""
    from hermes_cli import skin_engine
    from hermes_cli.cli_modern_layout import _brand_glyph

    monkeypatch.setitem(
        skin_engine._BUILTIN_SKINS["default"], "branding",
        {**skin_engine._HERMES_BRANDING, "response_label": " 𒀭 "},
    )
    skin_engine._active_skin = None

    assert _brand_glyph() == ("𒀭", "")


def test_stock_skins_keep_the_caduceus():
    """No behavior change for anything shipped today."""
    from hermes_cli.cli_modern_layout import _brand_glyph
    from hermes_cli.cli_status_bar_mixin import _status_mark
    from hermes_cli.skin_engine import set_active_skin

    set_active_skin("default")
    assert _status_mark() == "☤"
    assert _brand_glyph() == ("☤", "Hermes")

    set_active_skin("ares")
    assert _status_mark() == "⚔"


def test_every_builtin_skin_declares_its_icon():
    """``icon`` is part of the skin schema; a built-in without it only renders
    the mark by accident of the caduceus fallback."""
    from hermes_cli.skin_engine import _BUILTIN_SKINS, _build_skin_config

    missing = [name for name, data in _BUILTIN_SKINS.items()
               if not _build_skin_config(data).branding.get("icon")]
    assert missing == []
