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


class TestForceFullRedraw:
    def test_no_app_is_safe(self, bare_cli):
        # _force_full_redraw must be a no-op when the TUI isn't running.
        bare_cli._app = None
        bare_cli._force_full_redraw()  # must not raise




    def test_resize_recovery_clears_viewport_on_width_change(self, bare_cli, monkeypatch):
        """A WIDTH change must wipe the visible viewport (CSI 2J) and replay.

        On column shrink the terminal reflows the old full-width chrome into
        extra rows that prompt_toolkit's stale-cursor erase cannot reach,
        leaving a duplicated status bar (#19280/#5474 class). We route through
        the same recovery as Ctrl+L: erase_screen (2J) + replay transcript.
        It must be banner-safe — CSI 3J (write_raw) must NOT fire.
        """
        app = MagicMock()
        events = []
        app.renderer.output.erase_screen.side_effect = lambda: events.append("erase")
        app.renderer.output.write_raw.side_effect = lambda *_: events.append("scrollback_wipe")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 200
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 90)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda: events.append("replay"))
        monkeypatch.setattr(
            cli_mod,
            "CLI_CONFIG",
            {"display": {"cli_rebuild_scrollback_on_redraw": False}},
        )

        bare_cli._recover_after_resize(app, original_on_resize)

        # Viewport cleared and transcript replayed BEFORE prompt_toolkit's resize.
        assert "erase" in events
        assert "replay" in events
        assert events.index("erase") < events.index("original_resize")
        # Banner-safe: scrollback (CSI 3J) must never be wiped on a resize.
        assert "scrollback_wipe" not in events
        # New width recorded for the next comparison.
        assert bare_cli._last_resize_width == 90
        assert bare_cli._status_bar_suppressed_after_resize is True

    def test_force_redraw_uses_full_screen_clear_without_scrollback_clear(self, bare_cli, monkeypatch):
        app = MagicMock()
        bare_cli._app = app
        monkeypatch.setattr(
            cli_mod,
            "CLI_CONFIG",
            {"display": {"cli_rebuild_scrollback_on_redraw": False}},
        )

        bare_cli._force_full_redraw()

        app.renderer.output.erase_screen.assert_called_once()
        app.renderer.output.cursor_goto.assert_called_once_with(0, 0)
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
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda: events.append("replay"))
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
        """The #49120 recovery (2J + replay) must still fire on a real change."""
        app = MagicMock()
        events = []
        app.renderer.output.erase_screen.side_effect = lambda: events.append("erase")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._status_bar_suppressed_after_resize = False
        bare_cli._last_resize_width = 120
        monkeypatch.setattr(bare_cli, "_get_tui_terminal_width", lambda: 90)
        monkeypatch.setattr(bare_cli, "_schedule_status_bar_unsuppress", lambda *_: None)
        monkeypatch.setattr(
            cli_mod, "_replay_output_history", lambda: events.append("replay")
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
