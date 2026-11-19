"""Tests for the semantic color palette in hermes_cli.colors."""

from hermes_cli import colors
from hermes_cli.colors import Colors, Palette, color, semantic


def _force_color(monkeypatch, enabled: bool = True) -> None:
    monkeypatch.setattr(colors, "should_use_color", lambda: enabled)


def test_color_flattens_palette_role_tuple(monkeypatch):
    _force_color(monkeypatch, True)

    out = color("ok", Palette.SUCCESS)

    assert out.startswith(Colors.GREEN)
    assert out.endswith(Colors.RESET)
    assert "ok" in out
    assert "(" not in out


def test_color_accepts_mixed_raw_and_role(monkeypatch):
    _force_color(monkeypatch, True)

    out = color("hi", Colors.BOLD, Palette.WARNING)

    assert out == Colors.BOLD + Colors.YELLOW + "hi" + Colors.RESET


def test_color_flattens_nested_role_groups(monkeypatch):
    """A nested group must not leave a tuple for ``"".join``."""
    _force_color(monkeypatch, True)

    out = color("title", (Palette.HEADING, Colors.DIM))

    assert out == (Colors.CYAN + Colors.BOLD + Colors.DIM + "title" + Colors.RESET)


def test_color_disabled_returns_plain_text(monkeypatch):
    _force_color(monkeypatch, False)

    assert color("plain", Palette.ERROR) == "plain"
    assert semantic("plain", Palette.HEADING) == "plain"


def test_semantic_matches_color(monkeypatch):
    _force_color(monkeypatch, True)

    assert semantic("x", Palette.INFO) == color("x", Palette.INFO)


def test_heading_role_is_multi_code(monkeypatch):
    _force_color(monkeypatch, True)

    out = semantic("Section", Palette.HEADING)

    assert Colors.CYAN in out
    assert Colors.BOLD in out
    assert out.endswith(Colors.RESET)


def test_every_palette_role_is_a_tuple_of_known_codes():
    known = {getattr(Colors, name) for name in dir(Colors) if not name.startswith("_")}
    roles = [n for n in dir(Palette) if not n.startswith("_")]
    assert roles, "Palette must define at least one semantic role"
    for name in roles:
        value = getattr(Palette, name)
        assert isinstance(value, tuple), f"Palette.{name} must be a tuple"
        assert value, f"Palette.{name} must not be empty"
        for code in value:
            assert code in known, (
                f"Palette.{name} references unknown ANSI code {code!r}"
            )
