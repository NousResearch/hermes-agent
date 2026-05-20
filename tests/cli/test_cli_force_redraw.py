"""Tests for CLI redraw helpers used to recover from terminal buffer drift.

Covers:
  - _force_full_redraw (#8688 cmux tab switch, /redraw, Ctrl+L)
  - the resize handler we install over prompt_toolkit's _on_resize (#5474)

Both behaviors are exercised against fake prompt_toolkit renderer/output
objects — we're asserting the escape sequences the CLI sends, not that
the terminal physically repainted.
"""

import types
from unittest.mock import MagicMock, patch

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

    def test_preserves_scrollback_and_redraws_running_app_inline(self, bare_cli, monkeypatch):
        app = MagicMock()
        app._is_running = True
        fake_loop = types.SimpleNamespace(is_running=lambda: True)
        app.loop = fake_loop
        bare_cli._app = app

        fake_asyncio = types.ModuleType("asyncio")

        class _Policy:
            def get_event_loop(self):
                return fake_loop

        fake_asyncio.get_event_loop_policy = lambda: _Policy()
        monkeypatch.setitem(__import__("sys").modules, "asyncio", fake_asyncio)

        bare_cli._force_full_redraw()

        app.renderer.erase.assert_called_once_with(leave_alternate_screen=False)
        app.renderer.clear.assert_not_called()
        app._request_absolute_cursor_position.assert_called_once_with()
        app._redraw.assert_called_once_with()
        app.invalidate.assert_not_called()

    def test_cross_thread_redraw_hops_to_app_loop(self, bare_cli, monkeypatch):
        class FakeLoop:
            def __init__(self):
                self.call_count = 0

            def is_running(self):
                return True

            def call_soon_threadsafe(self, cb, *args):
                self.call_count += 1
                cb(*args)

        fake_loop = FakeLoop()
        app = MagicMock()
        app._is_running = True
        app.loop = fake_loop
        bare_cli._app = app

        fake_current_loop = types.SimpleNamespace(is_running=lambda: True)
        fake_asyncio = types.ModuleType("asyncio")

        class _Policy:
            def get_event_loop(self):
                return fake_current_loop

        fake_asyncio.get_event_loop_policy = lambda: _Policy()
        monkeypatch.setitem(__import__("sys").modules, "asyncio", fake_asyncio)

        bare_cli._force_full_redraw()

        assert fake_loop.call_count == 1
        app.renderer.erase.assert_called_once_with(leave_alternate_screen=False)
        app.renderer.clear.assert_not_called()
        app._request_absolute_cursor_position.assert_called_once_with()
        app._redraw.assert_called_once_with()
        app.invalidate.assert_not_called()

    def test_falls_back_to_invalidate_when_app_not_running(self, bare_cli, monkeypatch):
        app = MagicMock()
        app._is_running = False
        app.loop = None
        bare_cli._app = app
        events = []
        out = app.renderer.output
        out.reset_attributes.side_effect = lambda: events.append("reset_attrs")
        out.erase_screen.side_effect = lambda: events.append("erase")
        out.cursor_goto.side_effect = lambda *_: events.append("home")
        out.flush.side_effect = lambda: events.append("flush")
        app.renderer.reset.side_effect = lambda **_: events.append("renderer_reset")
        monkeypatch.setattr(cli_mod, "_replay_output_history", lambda: events.append("replay"))
        app.invalidate.side_effect = lambda: events.append("invalidate")

        bare_cli._force_full_redraw()

        app.renderer.erase.assert_called_once_with(leave_alternate_screen=False)
        app.renderer.clear.assert_not_called()
        app._request_absolute_cursor_position.assert_called_once_with()
        app._redraw.assert_not_called()
        app.invalidate.assert_called_once()
        assert events == ["invalidate"]

    def test_resize_uses_non_destructive_redraw_after_prompt_toolkit_resize(self, bare_cli):
        events = []
        original_on_resize = lambda: events.append("original_resize")
        bare_cli._force_full_redraw = lambda: events.append("force_redraw")

        bare_cli._recover_after_resize(MagicMock(), original_on_resize)

        assert events == ["original_resize", "force_redraw"]

    def test_resize_still_attempts_redraw_when_prompt_toolkit_resize_raises(self, bare_cli):
        events = []

        def _raise_resize():
            events.append("original_resize")
            raise RuntimeError("boom")

        bare_cli._force_full_redraw = lambda: events.append("force_redraw")

        with pytest.raises(RuntimeError, match="boom"):
            bare_cli._recover_after_resize(MagicMock(), _raise_resize)

        assert events == ["original_resize", "force_redraw"]

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
        app._is_running = False
        app.renderer.erase.side_effect = RuntimeError("boom")
        bare_cli._app = app

        bare_cli._force_full_redraw()  # must not raise

        # invalidate() is still attempted after a renderer failure.
        app.invalidate.assert_called_once()

    def test_falls_back_to_invalidate_when_redraw_raises(self, bare_cli):
        app = MagicMock()
        app._is_running = True
        app.loop = None
        app._redraw.side_effect = RuntimeError("boom")
        bare_cli._app = app

        bare_cli._force_full_redraw()  # must not raise

        app.renderer.erase.assert_called_once_with(leave_alternate_screen=False)
        app.renderer.clear.assert_not_called()
        app._request_absolute_cursor_position.assert_called_once_with()
        app._redraw.assert_called_once_with()
        app.invalidate.assert_called_once()

    def test_swallows_invalidate_exceptions(self, bare_cli):
        app = MagicMock()
        app._is_running = False
        app.invalidate.side_effect = RuntimeError("boom")
        bare_cli._app = app

        bare_cli._force_full_redraw()  # must not raise


class TestFocusReporting:
    def test_enable_focus_reporting_writes_escape_sequence(self, bare_cli):
        app = MagicMock()
        bare_cli._app = app
        bare_cli._focus_reporting_enabled = False

        with patch("cli.time.monotonic", return_value=123.0):
            bare_cli._set_terminal_focus_reporting(True)

        app.renderer.output.write_raw.assert_called_once_with("\x1b[?1004h")
        app.renderer.output.flush.assert_called_once()
        assert bare_cli._focus_reporting_enabled is True
        assert bare_cli._focus_reporting_started_at == 123.0

    def test_disable_focus_reporting_writes_escape_sequence(self, bare_cli):
        app = MagicMock()
        bare_cli._app = app
        bare_cli._focus_reporting_enabled = True
        bare_cli._focus_redraw_pending = True

        bare_cli._set_terminal_focus_reporting(False)

        app.renderer.output.write_raw.assert_called_once_with("\x1b[?1004l")
        app.renderer.output.flush.assert_called_once()
        assert bare_cli._focus_reporting_enabled is False
        assert bare_cli._focus_redraw_pending is False

    def test_focus_in_after_focus_out_forces_redraw(self, bare_cli):
        bare_cli._focus_redraw_pending = False
        bare_cli._focus_reporting_started_at = 0.0
        bare_cli._last_focus_redraw = 0.0
        bare_cli._force_full_redraw = MagicMock()

        with patch("cli.time.monotonic", return_value=10.0):
            bare_cli._handle_terminal_focus_out()
        assert bare_cli._focus_redraw_pending is True

        with patch("cli.time.monotonic", return_value=11.0):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_called_once()
        assert bare_cli._focus_redraw_pending is False
        assert bare_cli._last_focus_redraw == 11.0

    def test_focus_in_without_pending_focus_out_is_noop(self, bare_cli):
        bare_cli._focus_redraw_pending = False
        bare_cli._focus_reporting_enabled = False
        bare_cli._focus_reporting_started_at = 0.0
        bare_cli._last_focus_redraw = 0.0
        bare_cli._force_full_redraw = MagicMock()

        with patch("cli.time.monotonic", return_value=11.0):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_not_called()

    def test_focus_in_only_terminal_refreshes_prompt_after_grace(self, bare_cli):
        bare_cli._focus_redraw_pending = False
        bare_cli._focus_reporting_enabled = True
        bare_cli._focus_reporting_started_at = 10.0
        bare_cli._last_focus_redraw = 0.0
        bare_cli._force_full_redraw = MagicMock()
        bare_cli._invalidate = MagicMock()

        with patch("cli.time.monotonic", return_value=11.2):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_not_called()
        bare_cli._invalidate.assert_called_once_with(min_interval=0.0)
        assert bare_cli._focus_redraw_pending is False
        assert bare_cli._last_focus_redraw == 11.2

    def test_focus_in_only_startup_handshake_still_noops(self, bare_cli):
        bare_cli._focus_redraw_pending = False
        bare_cli._focus_reporting_enabled = True
        bare_cli._focus_reporting_started_at = 10.0
        bare_cli._last_focus_redraw = 0.0
        bare_cli._force_full_redraw = MagicMock()
        bare_cli._invalidate = MagicMock()

        with patch("cli.time.monotonic", return_value=10.5):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_not_called()
        bare_cli._invalidate.assert_not_called()
        assert bare_cli._focus_redraw_pending is False

    def test_focus_in_only_bursts_are_throttled(self, bare_cli):
        bare_cli._focus_redraw_pending = False
        bare_cli._focus_reporting_enabled = True
        bare_cli._focus_reporting_started_at = 0.0
        bare_cli._last_focus_redraw = 11.2
        bare_cli._force_full_redraw = MagicMock()
        bare_cli._invalidate = MagicMock()

        with patch("cli.time.monotonic", return_value=11.8):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_not_called()
        bare_cli._invalidate.assert_not_called()
        assert bare_cli._focus_redraw_pending is False

    def test_startup_focus_handshake_does_not_redraw(self, bare_cli):
        bare_cli._focus_redraw_pending = True
        bare_cli._focus_reporting_started_at = 10.0
        bare_cli._last_focus_redraw = 0.0
        bare_cli._force_full_redraw = MagicMock()

        with patch("cli.time.monotonic", return_value=10.1):
            bare_cli._handle_terminal_focus_in()

        bare_cli._force_full_redraw.assert_not_called()
        assert bare_cli._focus_redraw_pending is True
