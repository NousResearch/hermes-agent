"""Terminal repaint/resize recovery, input-mode healing, and clipboard helpers for the interactive
CLI. Mixin on ``HermesCLI``; cli.py symbols are imported lazily inside methods (import cycle)."""

from __future__ import annotations

import base64
import errno
import os
import shutil
import sys
import threading
import time

from hermes_constants import get_hermes_home


def _is_eio(exc: BaseException) -> bool:
    return getattr(exc, "errno", None) == errno.EIO


def _run_on_app_loop(app, fn) -> None:
    """Run *fn* on the app's asyncio loop when one exists, else inline (fail-open)."""
    try:
        loop = getattr(app, "loop", None)
    except Exception:
        loop = None
    if loop is not None:
        try:
            loop.call_soon_threadsafe(fn)
            return
        except Exception:
            pass
    fn()


def _write_terminal_sequence(app, seq: str) -> None:
    """Write a raw escape *seq* via the app output (write_raw > write) or stdout."""
    output = getattr(app, "output", None) if app else None
    if output and hasattr(output, "write_raw"):
        output.write_raw(seq)
        output.flush()
    elif output and hasattr(output, "write"):
        output.write(seq)
        output.flush()
    else:
        sys.stdout.write(seq)
        sys.stdout.flush()


class CLITerminalMixin:
    """Terminal repaint/resize recovery, input-mode healing, and clipboard helpers for the interactive CLI"""

    def _mark_terminal_io_broken(self, reason: str = "") -> None:
        """Stop UI paints after the PTY/stdout becomes unusable (#81521)."""
        from cli import logger
        if getattr(self, "_terminal_io_broken", False):
            return
        self._terminal_io_broken = True
        try:
            self._pet_stop_anim()
        except Exception:
            pass
        logger.warning(
            "Terminal I/O broken%s — freezing UI paints to avoid redraw storm (#81521)",
            f" ({reason})" if reason else "")

    def _app_invalidate(self, app, where: str, *, swallow: bool) -> None:
        """``app.invalidate()``: EIO freezes paints, other OSErrors re-raise, and any
        other exception re-raises unless *swallow*."""
        try:
            app.invalidate()
        except OSError as exc:
            if _is_eio(exc):
                self._mark_terminal_io_broken(where)
                return
            raise
        except Exception:
            if not swallow:
                raise

    def _invalidate(self, min_interval: float = 0.25) -> None:
        """Throttled UI repaint for high-frequency background updates (spinner frames,
        streaming flushes): the throttle prevents blinking on slow/SSH links and the
        resize-recovery guard keeps footer chrome out of scrollback mid-SIGWINCH.

        NOT for user-blocking modals (approval / clarify / sudo) — use ``_paint_now``: a
        throttled or resize-gated entry paint is silently dropped, so the prompt never
        renders and times out unseen (#41098).
        """
        if getattr(self, "_terminal_io_broken", False) or getattr(self, "_resize_recovery_pending", False):
            return
        now = time.monotonic()
        if hasattr(self, "_app") and self._app and (now - getattr(self, "_last_invalidate", 0.0)) >= min_interval:
            self._last_invalidate = now
            self._app_invalidate(self._app, "invalidate", swallow=False)

    def _paint_now(self) -> None:
        """Immediate, unthrottled repaint for user-blocking modal prompts.

        Deliberately bypasses the ``_invalidate`` throttle and resize-recovery guard —
        a modal the user is waiting on must never be dropped (#41098) — mirroring the
        direct ``event.app.invalidate()`` the modal key-binding handlers use.
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if app is not None:
            self._app_invalidate(app, "paint_now", swallow=True)

    def _force_full_redraw(self) -> None:
        """Force a clean full-screen repaint of the prompt_toolkit UI (Ctrl+L, ``/redraw``).

        Recovers from terminal buffer drift caused by external redraws we can't detect
        (cmux/tmux tab switches, ``clear`` from a subshell, SSH window restores): they
        repaint without SIGWINCH, so prompt_toolkit's tracked ``_cursor_pos`` is stale
        and the next incremental redraw stacks on old content (ghost status bars).
        """
        from cli import _replay_output_history
        if getattr(self, "_terminal_io_broken", False):
            return
        app = getattr(self, "_app", None)
        if not app:
            return
        self._clear_prompt_toolkit_screen(app, rebuild_scrollback=self._redraw_rebuilds_scrollback())
        if getattr(self, "_terminal_io_broken", False):
            return
        _replay_output_history()
        self._pet_queue_kitty_frame()
        self._app_invalidate(app, "force_full_redraw", swallow=True)

    def _schedule_focus_regain_redraw(self, min_interval: float = 1.0) -> None:
        """Repaint after a terminal focus-in report (``CSI I``), at most once per
        ``min_interval`` (rapid Alt+Tab / pane-hop bursts). Emulators with focus tracking
        (DECSET 1004) may coalesce hidden-tab output, so on regain the incremental diff
        stacks on stale content (#60920, #25337); terminals without it never emit ``CSI I``.
        """
        now = time.monotonic()
        if now - getattr(self, "_last_focus_regain_redraw", 0.0) < min_interval:
            return
        self._last_focus_regain_redraw = now
        self._force_full_redraw()

    @staticmethod
    def _redraw_rebuilds_scrollback() -> bool:
        """Whether redraw/resize recovery should also clear scrollback (CSI 3J).

        Some terminal/tmux stacks move prompt_toolkit's bottom chrome into scrollback on
        maximize/restore; CSI 2J cannot remove those rows, so affected users opt in to 3J
        followed by the bounded output-history replay.
        """
        from cli import CLI_CONFIG
        display_config = CLI_CONFIG.get("display") if isinstance(CLI_CONFIG, dict) else {}
        raw = (display_config.get("cli_rebuild_scrollback_on_redraw", False)
               if isinstance(display_config, dict) else False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on", "always"}
        return bool(raw)

    def _recover_terminal_after_interrupt(self) -> None:
        """Recover the terminal after an interrupted agent turn (#33271): an in-flight
        ``CSI 6n`` reply arriving after the input parser tore down leaks as literal text
        and can stall the VT100 parser mid-escape. Drain stdin, then full redraw; each step
        self-guards. A dead PTY (EIO) skips the redraw (#81521 redraw storm). Never clear
        output history here — the interruption marker is printed under
        ``_suspend_output_history``, so replay already excludes it (#60920).
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        try:
            from hermes_cli.curses_ui import flush_stdin
            flush_stdin()
        except Exception:
            pass
        self._force_full_redraw()

    def _clear_prompt_toolkit_screen(self, app, *, rebuild_scrollback: bool = False) -> None:
        """Clear the terminal and reset prompt_toolkit renderer state."""
        if getattr(self, "_terminal_io_broken", False):
            return
        try:
            renderer = app.renderer
            out = renderer.output
            out.reset_attributes()
            out.erase_screen()
            if rebuild_scrollback:
                try:
                    out.write_raw("\x1b[3J")
                except Exception:
                    pass
            out.cursor_goto(0, 0)
            out.flush()
            # Drop cached screen + cursor state so the next _redraw() starts from a
            # known (0, 0) origin and re-renders every cell instead of diffing stale.
            renderer.reset(leave_alternate_screen=False)
        except OSError as exc:
            if _is_eio(exc):
                self._mark_terminal_io_broken("clear_screen")
        except Exception:
            pass

    def _recover_after_resize(self, app, original_on_resize) -> None:
        """Recover a resized classic CLI without desynchronizing cursor state.

        Never clears scrollback (the startup banner lives there and replay cannot rebuild
        it) and never resets the renderer before prompt_toolkit's own ``_on_resize``,
        which erases via the cached cursor position. The status bar / input rules are
        suppressed while the reflow settles: on column shrink the terminal reflows
        already-painted rows into scrollback first, so a fresh bar looks duplicated
        (#19280, #22976). Suppression cannot erase the already-reflowed OLD bar
        (``renderer.erase()`` uses ``_cursor_pos.y`` cached at the OLD width), so on an
        OBSERVED width change we wipe the viewport (CSI 2J, banner-safe; 3J only via
        ``display.cli_rebuild_scrollback_on_redraw``) and replay the transcript first.
        Same-width SIGWINCH (tmux attach, GNOME tab bar, focus) and the first signal
        without a seeded baseline are left alone — 2J+replay against preserved scrollback
        duplicates ``_OUTPUT_HISTORY`` (#65293). tmux-attach's stale previous_screen is
        handled by ``_hermes_call_output_screen_diff`` (#83874). Suppression is cleared by
        a debounced timer so the bar returns during idle; next-submit stays a fast path.
        """
        from cli import _replay_output_history
        self._status_bar_suppressed_after_resize = True
        try:
            new_width = self._get_tui_terminal_width()
        except Exception:
            new_width = None
        prev_width = getattr(self, "_last_resize_width", None)
        width_changed = new_width is not None and prev_width is not None and new_width != prev_width
        if width_changed:
            try:
                self._clear_prompt_toolkit_screen(
                    app, rebuild_scrollback=self._redraw_rebuilds_scrollback())
                _replay_output_history()
            except Exception:
                pass
        if new_width is not None:
            self._last_resize_width = new_width
        if width_changed:
            self._pet_queue_kitty_frame()
        original_on_resize()
        self._schedule_status_bar_unsuppress(app)

    def _restart_debounce_timer(self, attr: str, delay: float, fn) -> None:
        """Cancel the daemon Timer stored on ``self.<attr>`` (if any) and start a new one.

        ``fn`` receives the new Timer so it can detect being superseded."""
        old_timer = getattr(self, attr, None)
        if old_timer is not None:
            try:
                old_timer.cancel()
            except Exception:
                pass
        timer = threading.Timer(delay, lambda: fn(timer))
        timer.daemon = True
        setattr(self, attr, timer)
        timer.start()

    def _schedule_status_bar_unsuppress(self, app, delay: float = 0.35) -> None:
        """Clear the post-resize status-bar suppression after the reflow settles.

        Debounced: a fresh resize cancels the pending timer, so a resize storm repaints
        the bar only once it stops.
        """
        try:
            def _clear():
                self._status_bar_suppressed_after_resize = False
                try:
                    app.invalidate()
                except Exception:
                    pass
            self._restart_debounce_timer(
                "_status_bar_unsuppress_timer", delay, lambda _t: _run_on_app_loop(app, _clear))
        except Exception:
            # Fail open: never leave the bar stuck hidden.
            self._status_bar_suppressed_after_resize = False

    def _schedule_resize_recovery(self, app, original_on_resize, delay: float = 0.12) -> None:
        """Debounce resize redraws so footer chrome is not stamped into scrollback."""
        try:
            lock = getattr(self, "_resize_recovery_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._resize_recovery_lock = lock

            def _timer_fired(timer_ref):
                def _run_recovery():
                    with lock:
                        if getattr(self, "_resize_recovery_timer", None) is not timer_ref:
                            return  # superseded by a newer resize
                        self._resize_recovery_timer = None
                        self._resize_recovery_pending = False
                    self._recover_after_resize(app, original_on_resize)
                _run_on_app_loop(app, _run_recovery)
            with lock:
                self._resize_recovery_pending = True
                self._restart_debounce_timer("_resize_recovery_timer", delay, _timer_fired)
        except Exception:
            self._resize_recovery_pending = False
            self._recover_after_resize(app, original_on_resize)

    def _install_resize_recovery(self, app) -> None:
        """Route ``app._on_resize`` through the debounced ghost-clearing recovery
        (#5474/#49120) and seed the width baseline so the FIRST SIGWINCH can tell a benign
        signal from a real resize (#65293). Reads ``app.output`` directly, NOT
        ``_get_tui_terminal_width``: before ``app.run()`` the DummyApplication's output
        reports a hardcoded 80 columns, which would make the first real signal look like a
        width change. ``app.output`` is what the running resize handler measures.
        """
        width = None
        for probe in (lambda: app.output.get_size().columns,
                      lambda: shutil.get_terminal_size((80, 24)).columns):
            try:
                width = probe()
            except Exception:
                width = None
            if width and width > 0:
                break
        self._last_resize_width = width
        original_on_resize = app._on_resize
        app._on_resize = lambda: self._schedule_resize_recovery(app, original_on_resize)

    def _try_attach_clipboard_image(self) -> bool:
        """Save a clipboard image to ~/.hermes/images/ and attach it; True if attached."""
        from cli import datetime
        from hermes_cli.clipboard import save_clipboard_image
        self._image_counter += 1
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = get_hermes_home() / "images" / f"clip_{ts}_{self._image_counter}.png"
        if save_clipboard_image(img_path):
            self._attached_images.append(img_path)
            return True
        self._image_counter -= 1
        return False

    def _write_osc52_clipboard(self, text: str) -> None:
        """Copy *text* to the terminal clipboard via OSC 52.

        Wrapped for tmux/screen passthrough (mirrors ui-tui/src/lib/osc52.ts) — without
        the DCS wrapper the multiplexer consumes the sequence and the copy is lost.
        """
        payload = base64.b64encode(text.encode("utf-8")).decode("ascii")
        seq = f"\x1b]52;c;{payload}\x07"
        if os.environ.get("TMUX"):
            seq = "\x1bPtmux;" + seq.replace("\x1b", "\x1b\x1b") + "\x1b\\"
        elif os.environ.get("STY"):
            seq = "\x1bP" + seq + "\x1b\\"
        _write_terminal_sequence(getattr(self, "_app", None), seq)

    def _recover_terminal_input_modes(self, *, reason: str) -> None:
        """Best-effort reset when leaked mouse reports indicate mode drift."""
        from cli import (
            CLI_CONFIG, _DIM, _RST, _TERMINAL_INPUT_MODE_RESET_SEQ,
            _cli_multiline_shortcuts_enabled, _cprint, _enable_extended_enter_keys, logger)
        now = time.monotonic()
        # Rate-limit to avoid thrashing if a terminal floods reports.
        if now - self._last_input_mode_recovery < 0.5:
            return
        self._last_input_mode_recovery = now
        app = getattr(self, "_app", None)
        output = getattr(app, "output", None) if app else None
        try:
            _write_terminal_sequence(app, _TERMINAL_INPUT_MODE_RESET_SEQ)
        except Exception:
            return

        # The reset pops kitty keyboard mode and resets modifyOtherKeys too — re-request
        # extended keys so Shift+Enter isn't silently dead for the rest of the session.
        try:
            if _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG):
                _enable_extended_enter_keys(output)
        except Exception:
            pass
        logger.warning("Recovered terminal input modes after leak: %s", reason)
        if not self._input_mode_recovery_notice_shown:
            self._input_mode_recovery_notice_shown = True
            _cprint(
                f"  {_DIM}Recovered terminal input modes after leaked mouse reports. "
                f"If this repeats, run /new or restart this tab.{_RST}")

    def _check_termios_drift(self) -> None:
        """Idle watchdog: heal a tty that drifted back to cooked mode (a lost
        ``run_in_terminal`` cooked→raw restore leaves it line-buffering while prompt_toolkit
        believes it owns raw mode — CLI looks dead, process is healthy). Skipped while
        ``run_in_terminal`` legitimately holds cooked mode, while the agent runs
        (approval/sudo prompts touch the tty), and on Windows (no termios).
        """
        from cli import _DIM, _RST, _cprint, _heal_cooked_mode_drift, logger
        if os.name == "nt":
            return
        app = getattr(self, "_app", None)
        if app is None or not getattr(app, "_is_running", False):
            return
        if getattr(app, "_running_in_terminal", False):
            return
        now = time.monotonic()
        if now - self._last_termios_drift_check < 1.0:
            return
        self._last_termios_drift_check = now
        try:
            if not sys.stdin.isatty():
                return
            fd = sys.stdin.fileno()
        except Exception:
            return
        if _heal_cooked_mode_drift(fd):
            logger.warning(
                "Healed cooked-mode termios drift on stdin — a "
                "run_in_terminal cooked→raw restore was lost.")
            try:
                self._invalidate()  # so the prompt is visibly alive again
            except Exception:
                pass
            if not self._termios_drift_notice_shown:
                self._termios_drift_notice_shown = True
                _cprint(
                    f"  {_DIM}Recovered terminal from cooked-mode drift "
                    f"(input should respond normally again).{_RST}")
