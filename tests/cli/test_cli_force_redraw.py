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

    def test_missing_app_attr_is_safe(self, bare_cli):
        # Simulate HermesCLI before the TUI has ever been constructed.
        bare_cli._force_full_redraw()  # must not raise

    def test_sends_full_clear_replays_then_invalidates(self, bare_cli, monkeypatch):
        app = MagicMock()
        out = app.renderer.output
        bare_cli._app = app
        events = []
        out.reset_attributes.side_effect = lambda: events.append("reset_attrs")
        out.erase_screen.side_effect = lambda: events.append("erase")
        out.cursor_goto.side_effect = lambda *_: events.append("home")
        out.flush.side_effect = lambda: events.append("flush")
        app.renderer.reset.side_effect = lambda **_: events.append("renderer_reset")
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda: events.append("replay"))
        app.invalidate.side_effect = lambda: events.append("invalidate")

        bare_cli._force_full_redraw()

        # Must erase screen, home cursor, and flush — in that order.
        out.reset_attributes.assert_called_once()
        out.erase_screen.assert_called_once()
        out.cursor_goto.assert_called_once_with(0, 0)
        out.flush.assert_called_once()

        # Must reset prompt_toolkit's tracked screen/cursor state so the
        # next incremental redraw starts from a clean (0, 0) origin.
        app.renderer.reset.assert_called_once_with(leave_alternate_screen=False)

        # Must schedule a repaint.
        app.invalidate.assert_called_once()
        assert events == [
            "reset_attrs",
            "erase",
            "home",
            "flush",
            "renderer_reset",
            "replay",
            "invalidate",
        ]

    def test_resize_preserves_scrollback_and_resets_renderer(self, bare_cli, monkeypatch):
        """Resize recovery must NOT erase screen or scrollback.

        The startup banner lives in normal terminal scrollback (printed
        before prompt_toolkit owns the chrome).  Clearing scrollback on
        SIGWINCH removes it and ``_replay_output_history`` cannot
        reconstruct it.  The fix is to only reset the renderer cache and
        let ``original_on_resize`` recalculate layout.
        """
        app = MagicMock()
        events = []
        app.renderer.reset.side_effect = lambda **_: events.append("renderer_reset")
        app.invalidate.side_effect = lambda: events.append("invalidate")
        original_on_resize = lambda: events.append("original_resize")

        bare_cli._recover_after_resize(app, original_on_resize)

        assert events == [
            "renderer_reset",
            "invalidate",
            "original_resize",
        ]
        # Must NOT clear the screen or scrollback — those destroy the banner.
        app.renderer.output.erase_screen.assert_not_called()
        app.renderer.output.write_raw.assert_not_called()
        app.renderer.output.cursor_goto.assert_not_called()

    def test_force_redraw_uses_full_screen_clear_without_scrollback_clear(self, bare_cli):
        app = MagicMock()
        bare_cli._app = app

        bare_cli._force_full_redraw()

        app.renderer.output.erase_screen.assert_called_once()
        app.renderer.output.cursor_goto.assert_called_once_with(0, 0)
        app.renderer.output.write_raw.assert_not_called()

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
