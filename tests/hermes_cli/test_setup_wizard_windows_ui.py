"""End-to-end tests for the Windows setup-wizard UI fix (#24345).

Drives the setup wizard's banner and the ``curses_ui`` numbered
fallbacks under a simulated legacy Windows Console Host (no VT
processing, no ``curses`` stdlib module) and asserts the user-visible
output is clean -- no literal ``\\x1b[`` escape codes, no mojibake-prone
exotic Unicode, and a single ``pip install windows-curses`` hint when
the curses fallback fires.

Complements ``tests/tools/test_windows_setup_wizard_ui.py`` (which
pins the individual helpers in isolation) with full-stack behaviour
tests that catch wiring regressions between modules.
"""
from __future__ import annotations

import re
import sys
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simulate_legacy_windows_console(monkeypatch):
    """Make ``hermes_cli.stdio.is_legacy_windows_console()`` return True
    and ``hermes_cli.curses_ui._is_windows()`` return True, mirroring a
    fresh `irm install.ps1 | iex` install into the default PowerShell
    5.1 console host."""
    sys.modules.pop("hermes_cli.stdio", None)
    sys.modules.pop("hermes_cli.colors", None)
    sys.modules.pop("hermes_cli.curses_ui", None)

    import hermes_cli.stdio as stdio
    monkeypatch.setattr(stdio, "is_windows", lambda: True)
    monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: True)

    import hermes_cli.curses_ui as cu
    monkeypatch.setattr(cu, "_is_windows", lambda: True)
    cu._WINDOWS_CURSES_HINT_SHOWN = False

    # The colours helper recomputes on each call; nothing else to do.
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("HERMES_ASCII_GLYPHS", raising=False)

    yield stdio, cu

    sys.modules.pop("hermes_cli.stdio", None)
    sys.modules.pop("hermes_cli.colors", None)
    sys.modules.pop("hermes_cli.curses_ui", None)


@pytest.fixture
def simulate_modern_console(monkeypatch):
    """Modern host: VT processing on, curses available.  Used as the
    baseline so we can prove the fix is a no-op for users who weren't
    affected by the bug."""
    sys.modules.pop("hermes_cli.stdio", None)
    sys.modules.pop("hermes_cli.colors", None)
    sys.modules.pop("hermes_cli.curses_ui", None)

    import hermes_cli.stdio as stdio
    monkeypatch.setattr(stdio, "is_legacy_windows_console", lambda: False)

    import hermes_cli.curses_ui as cu
    monkeypatch.setattr(cu, "_is_windows", lambda: False)
    cu._WINDOWS_CURSES_HINT_SHOWN = False

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("HERMES_ASCII_GLYPHS", raising=False)

    yield stdio, cu

    sys.modules.pop("hermes_cli.stdio", None)
    sys.modules.pop("hermes_cli.colors", None)
    sys.modules.pop("hermes_cli.curses_ui", None)


# ---------------------------------------------------------------------------
# colours helper
# ---------------------------------------------------------------------------


# Regex matching any ANSI CSI sequence ("\x1b[" + parameters + final byte).
# We assert it's NOT present in legacy-host output -- that's exactly the
# garbage the user reported in #24345.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class TestColorOutputOnLegacyConsole:
    """``hermes_cli.colors.color()`` must strip ANSI on a legacy host."""

    def test_color_strips_ansi_on_legacy_windows(
        self, simulate_legacy_windows_console, monkeypatch,
    ):
        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)

        result = colors.color(
            "Hermes Setup Wizard", colors.Colors.MAGENTA, colors.Colors.BOLD,
        )

        assert _ANSI_CSI_RE.search(result) is None, (
            "color() must not emit ANSI on a legacy Windows console -- "
            f"got {result!r}"
        )
        assert "Hermes Setup Wizard" in result

    def test_color_preserves_ansi_on_modern_console(
        self, simulate_modern_console, monkeypatch,
    ):
        """Baseline: the bug fix must not regress users who weren't
        affected."""
        import hermes_cli.colors as colors
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)

        result = colors.color("Hermes Setup Wizard", colors.Colors.MAGENTA)
        assert _ANSI_CSI_RE.search(result) is not None
        assert "Hermes Setup Wizard" in result


# ---------------------------------------------------------------------------
# Numbered fallback rendering
# ---------------------------------------------------------------------------


class TestNumberedFallbackOnLegacyConsole:
    """The ``curses_ui`` fallback paths must output:

      * clear "type a number" instructions (#24345's third complaint
        was 'I can't use the keys to change/toggle' -- users didn't
        realise they were supposed to type a digit);
      * ASCII-safe glyphs (``x``, ``*``, ``( )`` instead of ``✓``,
        ``●``, ``○``) so they don't mojibake under the default raster
        font;
      * a single ``pip install windows-curses`` tip so users know how
        to get the arrow-key UX back.
    """

    def test_radio_fallback_uses_ascii_glyphs(
        self, simulate_legacy_windows_console, monkeypatch, capsys,
    ):
        import hermes_cli.curses_ui as cu
        # Auto-confirm the default to keep the test deterministic.
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

        result = cu._radio_numbered_fallback(
            "Pick your provider",
            ["openrouter", "anthropic", "openai"],
            selected=1,
            cancel_returns=1,
        )

        assert result == 1
        out = capsys.readouterr().out
        # Unicode markers stripped.
        assert "\u25cf" not in out and "\u25cb" not in out
        # ASCII markers used instead.
        assert "(*)" in out  # the selected one
        # User-friendly instruction copy.
        assert "Select by number" in out
        assert "Enter to confirm" in out

    def test_toggle_fallback_uses_ascii_check_mark(
        self, simulate_legacy_windows_console, monkeypatch, capsys,
    ):
        import hermes_cli.curses_ui as cu
        # Confirm on first prompt.
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

        result = cu._numbered_fallback(
            "Pick tools",
            ["search", "edit", "browse"],
            selected={0, 2},
            cancel_returns=set(),
        )

        assert result == {0, 2}
        out = capsys.readouterr().out
        # ✓ (U+2713) downgraded to x.
        assert "\u2713" not in out
        assert "[x]" in out
        # User instruction is clearer than the pre-fix
        # "Toggle by number, Enter to confirm" one-liner.
        assert "type the item number" in out.lower()

    def test_radio_fallback_keeps_unicode_on_modern_host(
        self, simulate_modern_console, monkeypatch, capsys,
    ):
        """Baseline: modern hosts keep the prettier rendering."""
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

        cu._radio_numbered_fallback(
            "Pick your provider",
            ["openrouter", "anthropic"],
            selected=0,
            cancel_returns=0,
        )
        out = capsys.readouterr().out
        assert "\u25cf" in out  # ● selected
        assert "\u25cb" in out  # ○ empty


# ---------------------------------------------------------------------------
# windows-curses install hint
# ---------------------------------------------------------------------------


class TestWindowsCursesHintWiring:
    """The hint must appear exactly once per process when the fallback
    fires on Windows, regardless of which selector hit the fallback."""

    def test_hint_appears_on_first_fallback_only(
        self, simulate_legacy_windows_console, monkeypatch, capsys,
    ):
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

        # Trip three different fallback paths in a row.
        cu._radio_numbered_fallback(
            "First", ["a", "b"], selected=0, cancel_returns=0,
        )
        cu._numbered_fallback(
            "Second", ["x", "y"], selected=set(), cancel_returns=set(),
        )
        cu._numbered_single_fallback("Third", ["m", "n"], cancel_idx=1)

        out = capsys.readouterr().out
        # Exactly one nudge across the three fallbacks.
        assert out.count("pip install windows-curses") == 1

    def test_no_hint_on_modern_host(
        self, simulate_modern_console, monkeypatch, capsys,
    ):
        import hermes_cli.curses_ui as cu
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")

        cu._radio_numbered_fallback(
            "Linux user", ["a", "b"], selected=0, cancel_returns=0,
        )
        out = capsys.readouterr().out
        assert "windows-curses" not in out


# ---------------------------------------------------------------------------
# Regression anchor — full stack
# ---------------------------------------------------------------------------


class TestBug24345EndToEnd:
    """Full-stack anchor for #24345.

    Pre-fix Windows behaviour reproduced verbatim from the issue's
    screenshots:

      1. The wizard banner contains literal ``\\x1b[35m`` sequences
         (the user sees ``←[35m┌───...``).
      2. The numbered fallback prints ``●``/``○``/``✓`` glyphs that
         mojibake under the default raster font.
      3. No nudge anywhere tells users how to get arrow-key
         navigation back.

    This anchor asserts the post-fix invariants in a single test so
    the bug shape is pinned end-to-end.  Reverting any of the three
    changes (VT processing, ASCII glyph fallback, install hint)
    causes one of the assertions to fail with an explicit
    "#24345 regression" message.
    """

    def test_legacy_console_setup_wizard_renders_cleanly(
        self, simulate_legacy_windows_console, monkeypatch, capsys,
    ):
        import hermes_cli.colors as colors
        import hermes_cli.curses_ui as cu

        # Simulate the wizard's first banner line + a checklist prompt.
        monkeypatch.setattr(colors.sys.stdout, "isatty", lambda: True)
        banner = colors.color(
            "┌───────────────────────────────────────┐", colors.Colors.MAGENTA,
        )
        # Drive a checklist that downgrades to the fallback (no curses
        # on a legacy host).  Confirm immediately.
        monkeypatch.setattr("builtins.input", lambda *a, **kw: "")
        cu._numbered_fallback(
            "Pick your tools",
            ["search", "edit"],
            selected={0},
            cancel_returns=set(),
        )
        out = capsys.readouterr().out
        full_output = banner + "\n" + out

        # 1. No literal ANSI escapes leaked through.
        assert _ANSI_CSI_RE.search(full_output) is None, (
            "#24345 regression: ANSI escape leaked into wizard output on "
            f"legacy console host. Output snippet: {full_output!r}"
        )

        # 2. Exotic Unicode glyphs replaced with ASCII.
        for bad in ("\u2713", "\u25cf", "\u25cb", "\u2192"):
            assert bad not in full_output, (
                f"#24345 regression: legacy console got Unicode glyph "
                f"{bad!r} that mojibakes under the default Windows font."
            )

        # 3. windows-curses hint was emitted exactly once.
        assert full_output.count("pip install windows-curses") == 1, (
            "#24345 regression: the one-time windows-curses install hint "
            "did not fire -- users won't know how to get arrow-key "
            "navigation back."
        )
