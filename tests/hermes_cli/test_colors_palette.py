"""Tests for the semantic color design-system primitives in hermes_cli.colors.

Covers the Palette semantic roles, the color() flattening of role tuples,
and the semantic() convenience wrapper. These are the design-system
foundation requested in issue #34566 (standardize CLI color usage so the
same semantic role renders consistently everywhere).
"""

from hermes_cli import colors
from hermes_cli.colors import Colors, Palette, color, semantic


def _force_color(monkeypatch, enabled: bool = True) -> None:
    """Make should_use_color() deterministic regardless of the test TTY."""
    monkeypatch.setattr(colors, "should_use_color", lambda: enabled)


def test_color_flattens_palette_role_tuple(monkeypatch):
    """A Palette role (a tuple of codes) wraps text just like raw codes."""
    _force_color(monkeypatch, True)

    out = color("ok", Palette.SUCCESS)

    assert out.startswith(Colors.GREEN)
    assert out.endswith(Colors.RESET)
    assert "ok" in out
    # The literal tuple must not leak into the output.
    assert "(" not in out


def test_color_accepts_mixed_raw_and_role(monkeypatch):
    """color() flattens a mix of raw codes and a role tuple in order."""
    _force_color(monkeypatch, True)

    out = color("hi", Colors.BOLD, Palette.WARNING)

    assert out == Colors.BOLD + Colors.YELLOW + "hi" + Colors.RESET


def test_color_disabled_returns_plain_text(monkeypatch):
    """With color disabled (NO_COLOR / not a TTY), text is returned bare."""
    _force_color(monkeypatch, False)

    assert color("plain", Palette.ERROR) == "plain"
    assert semantic("plain", Palette.HEADING) == "plain"


def test_semantic_matches_color(monkeypatch):
    """semantic() is a thin, equivalent wrapper over color()."""
    _force_color(monkeypatch, True)

    assert semantic("x", Palette.INFO) == color("x", Palette.INFO)


def test_heading_role_is_multi_code(monkeypatch):
    """HEADING combines cyan + bold so titles read consistently."""
    _force_color(monkeypatch, True)

    out = semantic("Section", Palette.HEADING)

    assert Colors.CYAN in out
    assert Colors.BOLD in out
    assert out.endswith(Colors.RESET)


def test_every_palette_role_is_a_tuple_of_known_codes():
    """Guardrail: each semantic role is a tuple of real ANSI code strings.

    Prevents a future edit from assigning a bare string (which color()
    would still accept but which breaks the role-tuple contract used by
    callers that splat with ``*Palette.ROLE``).
    """
    known = {
        getattr(Colors, name)
        for name in dir(Colors)
        if not name.startswith("_")
    }
    roles = [
        n for n in dir(Palette) if not n.startswith("_")
    ]
    assert roles, "Palette must define at least one semantic role"
    for name in roles:
        value = getattr(Palette, name)
        assert isinstance(value, tuple), f"Palette.{name} must be a tuple"
        assert value, f"Palette.{name} must not be empty"
        for code in value:
            assert code in known, (
                f"Palette.{name} references unknown ANSI code {code!r}"
            )
