"""Tests for CLI redraw helpers used to recover from terminal buffer drift.

Covers:
  - _force_full_redraw (#8688 cmux tab switch, /redraw, Ctrl+L)
  - the resize handler we install over prompt_toolkit's _on_resize (#5474)

Both behaviors are exercised against fake prompt_toolkit renderer/output
objects — we're asserting the escape sequences the CLI sends, not that
the terminal physically repainted.
"""

from unittest.mock import MagicMock

import pytest

import cli as cli_mod
from cli import HermesCLI


@pytest.fixture
def bare_cli():
    """A HermesCLI with no __init__ — we only exercise the redraw helper."""
    cli = object.__new__(HermesCLI)
    return cli


def _fake_app(*, rows, columns, chrome, painted=None, cursor_y=0):
    """MagicMock app whose output reports a real size and whose layout is ``chrome`` rows tall;
    ``painted`` lists the row widths of the renderer's last paint (``None``: nothing yet), its
    cursor on row ``cursor_y``."""
    from collections import defaultdict

    from prompt_toolkit.data_structures import Point, Size
    from prompt_toolkit.layout.screen import Char, Screen

    app = MagicMock()
    app.renderer.output.get_size.return_value = Size(rows=rows, columns=columns)
    app.output = app.renderer.output
    screen = None
    if painted is not None:
        screen = Screen()
        for y, width in enumerate(painted):
            for x in range(width):
                screen.data_buffer[y][x] = Char("─")
        screen.height = len(painted)
    app.renderer._last_screen = screen
    app.renderer._cursor_pos = Point(x=0, y=cursor_y)
    app.renderer._style_string_has_style = defaultdict(bool)
    app.renderer._min_available_height = 0
    app.layout.container.preferred_height.return_value.preferred = chrome
    return app


class TestForceFullRedraw:
    def test_no_app_is_safe(self, bare_cli):
        # _force_full_redraw must be a no-op when the TUI isn't running.
        bare_cli._app = None
        bare_cli._force_full_redraw()  # must not raise




    @pytest.mark.parametrize("reflows,new_width,history_lines,paint", [
        (True, 90, 40, "settled"), (True, 90, 40, "late"), (True, 90, 40, "clipped"), (False, 90, 40, "settled"),
        (None, 90, 40, "settled"), (None, 250, 40, "settled"), (False, 90, 2, "settled")])
    def test_resize_never_loses_or_reprints_a_row(self, bare_cli, monkeypatch, reflows, new_width, history_lines,
                                                  paint):
        """#95375: on a shrink a reflowing terminal has pushed the rows that grew into
        scrollback, so nothing is erased and replayed there — prompt_toolkit's erase is only
        aimed at the chrome's re-wrapped top. A chrome paint the terminal may have got after
        it narrowed (painted right before the width change was seen, or while the width moved)
        was clipped, not re-wrapped: its rows count as one each, so the erase never reaches the
        transcript above. A terminal that does not reflow (or may not)
        keeps every visible row in place, truncated: the viewport is erased row by row (no
        CSI 2J, no 3J) and refilled from its top row with what it held, counted at the width it
        was painted at, before prompt_toolkit repaints. When the viewport holds the whole
        history, only its rows and the chrome are erased: rows above it (the startup banner)
        were never recorded. A widen truncates nothing and is left alone."""
        import time
        from collections import deque

        from prompt_toolkit.layout.screen import Char
        app = _fake_app(rows=30, columns=new_width, chrome=5, painted=[0, 200, 50, 200, 30], cursor_y=4)
        for x in range(200):  # blanks without a colour are never written: one row, not three
            app.renderer._last_screen.data_buffer[0][x] = Char(" ")
        out = app.renderer.output
        events = []
        out.erase_end_of_line.side_effect = lambda: events.append("erase_row")
        out.erase_screen.side_effect = lambda: events.append("clear")
        out.erase_down.side_effect = lambda: events.append("erase_down")
        out.cursor_up.side_effect = lambda n: events.append(("up", n))
        out.write_raw.side_effect = lambda raw: events.append(("raw", raw))

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 200
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: new_width)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(cli_mod, "_OUTPUT_HISTORY", deque("x" * 180 for _ in range(history_lines)))
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda *a: events.append(("replay", *a)))
        monkeypatch.setattr(cli_mod, "_terminal_reflows", lambda: reflows)
        monkeypatch.setattr(cli_mod, "CLI_CONFIG", {"display": {"cli_rebuild_scrollback_on_redraw": False}})
        cli_mod._set_chrome_floor(None)
        from prompt_toolkit.data_structures import Size
        app.renderer._last_size = Size(rows=30, columns=200)  # the paint: at the old width
        out.get_size.return_value = Size(rows=30, columns=new_width if paint == "clipped" else 200)
        bare_cli._note_chrome_paint(app, True)
        from hermes_cli.cli_terminal_mixin import _RESIZE_PAINT_MARGIN
        bare_cli._resize_seen_at = time.monotonic() + (_RESIZE_PAINT_MARGIN / 2 if paint == "late" else 1.0)
        out.get_size.return_value = Size(rows=30, columns=new_width)

        bare_cli._recover_after_resize(
            app, lambda: events.append(("original_resize", app.renderer._cursor_pos.y)))

        if new_width > 200:
            assert events == [("original_resize", 4)]
        elif reflows and paint == "settled":
            assert events == [("original_resize", 1 + 3 + 1 + 3)]
        elif reflows:
            assert events == [("original_resize", 4)]
        elif history_lines == 2:  # 2 lines x 2 rows, right above the chrome
            assert events[:4] == [("raw", "\r"), ("up", 4 + 4), "erase_down", ("replay", (25, 90, True, None), out)]
            assert events[4][0] == "original_resize" and len(events) == 5
        else:
            assert events[:31] == ["erase_row"] * 30 + [("replay", (25, 90, True, 0), out)]
            assert events[31][0] == "original_resize" and len(events) == 32
        assert bare_cli._last_resize_width == new_width
        assert bare_cli._status_bar_suppressed_after_resize is True

    def test_force_redraw_erases_the_viewport_without_scrollback_clear(self, bare_cli, monkeypatch):
        app = _fake_app(rows=30, columns=100, chrome=6)
        bare_cli._app = app
        monkeypatch.setattr(
            cli_mod,
            "CLI_CONFIG",
            {"display": {"cli_rebuild_scrollback_on_redraw": False}},
        )

        bare_cli._force_full_redraw()

        # Row by row, not CSI 2J: scroll-on-clear terminals (tmux, VTE) copy a 2J'd screen
        # into scrollback, stacking a duplicate of the transcript (#95375).
        app.renderer.output.erase_screen.assert_not_called()
        assert app.renderer.output.erase_end_of_line.call_count == 30
        app.renderer.output.write_raw.assert_not_called()

    def test_force_redraw_can_clear_scrollback_when_configured(self, bare_cli, monkeypatch):
        app = MagicMock()
        bare_cli._app = app
        monkeypatch.setattr(
            cli_mod,
            "CLI_CONFIG",
            {"display": {"cli_rebuild_scrollback_on_redraw": True}},
        )

        bare_cli._force_full_redraw()

        app.renderer.output.erase_screen.assert_called_once()
        app.renderer.output.write_raw.assert_called_once_with("\x1b[3J")

    def test_resize_recovery_can_clear_scrollback_when_configured(self, bare_cli, monkeypatch):
        app = MagicMock()
        events = []
        app.renderer.output.erase_screen.side_effect = lambda: events.append("erase")
        app.renderer.output.write_raw.side_effect = lambda *_: events.append("scrollback_wipe")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 200
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 90)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda *_: events.append("replay"))
        monkeypatch.setattr(
            cli_mod,
            "CLI_CONFIG",
            {"display": {"cli_rebuild_scrollback_on_redraw": "true"}},
        )

        bare_cli._recover_after_resize(app, original_on_resize)

        assert events[:3] == ["erase", "scrollback_wipe", "replay"]
        assert events.index("scrollback_wipe") < events.index("original_resize")

    def test_same_width_sigwinch_is_left_untouched(self, bare_cli, monkeypatch):
        """Same-width SIGWINCH (tmux attach, benign focus/tab signals) must not
        clear the viewport or replay: a 2J without replay erases the visible
        transcript, and a replay duplicates it (#65293). The tmux-attach
        stale-paint crash is handled by _hermes_call_output_screen_diff's
        retry instead (#83874)."""
        app = MagicMock()
        events = []
        app.renderer.output.erase_screen.side_effect = lambda: events.append("erase")
        app.renderer.output.write_raw.side_effect = lambda *_: events.append("scrollback_wipe")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 120
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 120)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda: events.append("replay"))

        bare_cli._recover_after_resize(app, original_on_resize)

        assert "erase" not in events
        assert "replay" not in events
        assert "scrollback_wipe" not in events
        assert events == ["original_resize"]
        assert bare_cli._last_resize_width == 120
        assert bare_cli._status_bar_suppressed_after_resize is True

    def test_output_screen_diff_retries_on_corrupt_previous_screen(self, bare_cli):
        """Corrupt previous_screen must not wedge the paint loop.

        After tmux attach, _output_screen_diff can raise AttributeError
        ('cell' object has no attribute 'char'). Retry with previous_screen=None.
        """
        calls = []

        def fake_osd(
            app, output, screen, current_pos, color_depth,
            previous_screen, last_style, is_done, full_screen,
            attrs_for_style_string, style_string_has_style,
            size, previous_width,
        ):
            calls.append((previous_screen, previous_width, last_style))
            if previous_screen is not None:
                # Exact failure mode from the classic CLI event loop.
                raise AttributeError("'cell' object has no attribute 'char'")
            return ("ok", current_pos, last_style)

        screen = MagicMock()
        screen.height = 10
        previous = MagicMock()
        previous.height = 8

        result = cli_mod._hermes_call_output_screen_diff(
            fake_osd,
            app=None,
            output=None,
            screen=screen,
            current_pos=None,
            color_depth=None,
            previous_screen=previous,
            last_style="style",
            is_done=False,
            full_screen=False,
            attrs_for_style_string=None,
            style_string_has_style=None,
            size=None,
            previous_width=80,
        )

        assert result[0] == "ok"
        assert len(calls) == 2
        assert calls[0][0] is previous
        assert previous.height == 10  # height inflate still applied first
        assert calls[1] == (None, 0, None)

    def test_resize_recovery_is_debounced(self, bare_cli, monkeypatch):
        timers = []
        calls = []

        class FakeTimer:
            def __init__(self, delay, callback):
                self.delay = delay
                self.callback = callback
                self.cancelled = False
                self.daemon = False
                timers.append(self)

            def start(self):
                calls.append(("start", self.delay))

            def cancel(self):
                self.cancelled = True
                calls.append(("cancel", self.delay))

            def fire(self):
                self.callback()

        app = MagicMock()
        app.loop.call_soon_threadsafe.side_effect = lambda cb: cb()
        monkeypatch.setattr(cli_mod.threading, "Timer", FakeTimer)
        monkeypatch.setattr(
            bare_cli,
            "_recover_after_resize",
            lambda _app, _orig: calls.append(("recover", _orig())),
        )

        original_one = lambda: "first"
        original_two = lambda: "second"

        bare_cli._schedule_resize_recovery(app, original_one, delay=0.25)
        assert bare_cli._resize_recovery_pending is True
        bare_cli._schedule_resize_recovery(app, original_two, delay=0.25)

        assert len(timers) == 2
        assert timers[0].cancelled is True
        timers[0].fire()
        assert ("recover", "first") not in calls

        timers[1].fire()
        assert ("recover", "second") in calls
        assert bare_cli._resize_recovery_pending is False

    def test_invalidate_is_suppressed_while_resize_recovery_is_pending(self, bare_cli):
        app = MagicMock()
        bare_cli._app = app
        bare_cli._last_invalidate = 0.0
        bare_cli._resize_recovery_pending = True

        bare_cli._invalidate(min_interval=0)

        app.invalidate.assert_not_called()

    def test_swallows_renderer_exceptions(self, bare_cli):
        # If the renderer blows up for any reason, the helper must not
        # propagate — otherwise a stray Ctrl+L would crash the CLI.
        app = MagicMock()
        app.renderer.output.erase_screen.side_effect = RuntimeError("boom")
        bare_cli._app = app

        bare_cli._force_full_redraw()  # must not raise

        # invalidate() is still attempted after a renderer failure.
        app.invalidate.assert_called_once()

    def test_swallows_invalidate_exceptions(self, bare_cli):
        app = MagicMock()
        app.invalidate.side_effect = RuntimeError("boom")
        bare_cli._app = app

        bare_cli._force_full_redraw()  # must not raise


class TestFirstSigwinchBaseline:
    """Bug #65293: the session's FIRST SIGWINCH used to be force-treated as a
    width change (no prior width to compare against), so a benign resize
    signal — GNOME Terminal tab bar appearing, monitor-scale change, focus
    events — cleared the viewport and replayed ``_OUTPUT_HISTORY``.  After a
    resume that deque holds the whole "Previous Conversation" recap plus the
    first live exchange, so everything reprinted as a duplicate.  A replay
    must require an OBSERVED width change against a recorded baseline.
    """

    def test_first_sigwinch_with_unchanged_width_does_not_replay(
        self, bare_cli, monkeypatch
    ):
        app = MagicMock()
        events = []
        app.renderer.output.erase_screen.side_effect = lambda: events.append("erase")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        # No baseline recorded yet — the pre-fix code forced width_changed=True.
        assert getattr(bare_cli, "_last_resize_width", None) is None
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 120)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(
            cli_mod, "_replay_output_history", lambda: events.append("replay")
        )

        bare_cli._recover_after_resize(app, original_on_resize)

        # Width did not change — no clear, no replay, straight to prompt_toolkit.
        assert events == ["original_resize"]
        app.renderer.output.erase_screen.assert_not_called()
        # The signal still records the baseline for the next comparison.
        assert bare_cli._last_resize_width == 120

    def test_real_width_change_after_baseline_still_replays(
        self, bare_cli, monkeypatch
    ):
        """The #49120 recovery (viewport erase + replay) must still fire on a real change."""
        app = _fake_app(rows=30, columns=90, chrome=3, painted=[120, 120, 30], cursor_y=2)
        events = []
        app.renderer.output.erase_end_of_line.side_effect = lambda: events.append("erase")
        app.renderer.output.erase_down.side_effect = lambda: events.append("erase")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 120
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 90)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(cli_mod, "_terminal_reflows", lambda: None)
        monkeypatch.setattr(
            cli_mod, "_replay_output_history", lambda *_: events.append("replay")
        )

        bare_cli._recover_after_resize(app, original_on_resize)

        assert "erase" in events and "replay" in events
        assert bare_cli._last_resize_width == 90

    def test_install_resize_recovery_seeds_width_baseline(self, bare_cli):
        """Hook installation records the CURRENT width as the baseline, so an
        initial maximize/restore (a real change vs that baseline) is still
        recovered while a same-size first signal is not.

        The baseline must come from ``app.output`` — the same object the
        running app measures on SIGWINCH — not from ``get_app()``, which
        before ``app.run()`` is a DummyApplication reporting a fake 80 cols.
        """
        app = MagicMock()
        app.output.get_size.return_value.columns = 132
        scheduled = []
        bare_cli._schedule_resize_recovery = lambda *a, **k: scheduled.append(a)

        original = app._on_resize
        bare_cli._install_resize_recovery(app)

        assert bare_cli._last_resize_width == 132
        assert app._on_resize is not original  # hook installed
        app._on_resize()  # simulated SIGWINCH → routes to the debouncer
        assert len(scheduled) == 1
        assert scheduled[0][0] is app
        assert scheduled[0][1] is original

    def test_install_resize_recovery_falls_back_to_shutil(
        self, bare_cli, monkeypatch
    ):
        """A dead app.output probe falls back to shutil, never to the
        DummyApplication's fake width."""
        import os as os_mod

        app = MagicMock()
        app.output.get_size.side_effect = RuntimeError("not attached")
        monkeypatch.setattr(
            cli_mod.shutil,
            "get_terminal_size",
            lambda _default: os_mod.terminal_size((97, 40)),
        )

        bare_cli._install_resize_recovery(app)

        assert bare_cli._last_resize_width == 97

    def test_install_resize_recovery_survives_width_probe_failure(
        self, bare_cli, monkeypatch
    ):
        app = MagicMock()
        app.output.get_size.side_effect = RuntimeError("not attached")

        def _boom(_default):
            raise RuntimeError("no tty")

        monkeypatch.setattr(cli_mod.shutil, "get_terminal_size", _boom)

        bare_cli._install_resize_recovery(app)  # must not raise

        assert getattr(bare_cli, "_last_resize_width", None) is None


class TestFocusRegainRedraw:
    """Focus-in (CSI I) routes through the same recovery as Ctrl+L, rate-limited.

    While the tab/window is hidden the emulator may coalesce output or repaint
    the surface; on regain prompt_toolkit's incremental diff stacks a fresh
    copy of the prompt chrome on top of the stale one (#60920 focus-regain
    variant, #25337).
    """

    def test_focus_regain_triggers_full_redraw(self, bare_cli):
        calls = []
        bare_cli._force_full_redraw = lambda: calls.append("redraw")

        bare_cli._schedule_focus_regain_redraw()

        assert calls == ["redraw"]

    def test_focus_regain_redraw_is_rate_limited(self, bare_cli):
        calls = []
        bare_cli._force_full_redraw = lambda: calls.append("redraw")

        bare_cli._schedule_focus_regain_redraw(min_interval=60.0)
        bare_cli._schedule_focus_regain_redraw(min_interval=60.0)
        bare_cli._schedule_focus_regain_redraw(min_interval=60.0)

        assert calls == ["redraw"]

    def test_focus_regain_redraw_fires_again_after_interval(self, bare_cli):
        calls = []
        bare_cli._force_full_redraw = lambda: calls.append("redraw")

        bare_cli._schedule_focus_regain_redraw(min_interval=0.0)
        bare_cli._schedule_focus_regain_redraw(min_interval=0.0)

        assert calls == ["redraw", "redraw"]

    def test_first_redraw_fires_even_with_small_monotonic_clock(
        self, bare_cli, monkeypatch
    ):
        """The first-ever redraw must fire regardless of monotonic's epoch.

        time.monotonic() starts from an arbitrary point (boot on Linux); on a
        fresh CI VM it can be smaller than min_interval. The old 0.0 sentinel
        turned that into ``now - 0.0 < min_interval`` and silently swallowed
        the FIRST redraw (CI run 32494557030 — both retry attempts failed).
        """
        calls = []
        bare_cli._force_full_redraw = lambda: calls.append("redraw")
        monkeypatch.setattr(cli_mod.time, "monotonic", lambda: 3.0)

        bare_cli._schedule_focus_regain_redraw(min_interval=60.0)

        assert calls == ["redraw"]

    def test_first_invalidate_fires_even_with_small_monotonic_clock(
        self, bare_cli, monkeypatch
    ):
        """Same class as above for the streaming/spinner repaint throttle."""
        app = MagicMock()
        bare_cli._app = app
        bare_cli._last_invalidate = None
        monkeypatch.setattr(cli_mod.time, "monotonic", lambda: 0.1)

        bare_cli._invalidate(min_interval=0.25)

        app.invalidate.assert_called_once()
