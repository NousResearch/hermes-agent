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

from hermes_cli.cli_render import (
    _chrome_floor,
    _hold_paints,
    _output_history_rows,
    _release_paints,
    _set_chrome_floor,
    _set_paint_gate,
    _take_suspect_rows,
)
from hermes_constants import get_hermes_home


# A replay ``fit`` with no room: nothing is replayed.
_NO_REPLAY = (0, 0, False, None)
# How long a resize drag holds output before its next width change runs a recovery anyway,
# that long after the change (seconds): output freezes about their sum plus one signal interval.
_RESIZE_HOLD_MAX = 0.7
_RESIZE_SETTLE = 0.03
# How long before a width change was first seen a chrome paint may still have reached the
# terminal after it narrowed (seconds; a paint the width moved under is always suspect): the
# signal lands a few ms after the multiplexer re-wraps, and it reads our output promptly — in
# tmux, paints that ended up to 4 ms before the signal were re-wrapped, never clipped (#95375).
_RESIZE_PAINT_MARGIN = 0.01


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
        # None sentinel, not 0.0 — monotonic's epoch is arbitrary (boot on Linux), so on
        # a fresh VM ``now`` can be < min_interval and a 0.0 sentinel would swallow the
        # first repaint (same class as _schedule_focus_regain_redraw below).
        last = getattr(self, "_last_invalidate", None)
        if hasattr(self, "_app") and self._app and (last is None or now - last >= min_interval):
            self._last_invalidate = now
            self._app_invalidate(self._app, "invalidate", swallow=False)

    def _paint_now(self) -> None:
        """Unthrottled repaint for user-blocking modal prompts.

        Bypasses the ``_invalidate`` throttle — a modal the user is waiting on must never be
        dropped (#41098) — mirroring the direct ``event.app.invalidate()`` the modal key-binding
        handlers use. While a width change awaits its recovery the app's redraw is gated
        (``_install_resize_recovery``), so the paint waits for that recovery, which repaints the
        whole app about every ``_RESIZE_HOLD_MAX`` seconds even while a drag keeps signalling.
        """
        if getattr(self, "_terminal_io_broken", False):
            return
        monitor = getattr(self, "_subagent_monitor", None)
        if monitor is not None:
            monitor.invalidate()
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
        fit = self._clear_prompt_toolkit_screen(app, rebuild_scrollback=self._redraw_rebuilds_scrollback())
        if getattr(self, "_terminal_io_broken", False):
            return
        # A viewport refill paints at once, before prompt_toolkit's next erase; after CSI 3J
        # the whole history goes through the usual print (the screen was cleared anyway).
        _replay_output_history(fit, None if fit is None else app.renderer.output)
        self._pet_queue_kitty_frame()
        self._app_invalidate(app, "force_full_redraw", swallow=True)

    def _schedule_focus_regain_redraw(self, min_interval: float = 1.0) -> None:
        """Repaint after a terminal focus-in report (``CSI I``), at most once per
        ``min_interval`` (rapid Alt+Tab / pane-hop bursts). Emulators with focus tracking
        (DECSET 1004) may coalesce hidden-tab output, so on regain the incremental diff
        stacks on stale content (#60920, #25337); terminals without it never emit ``CSI I``.
        """
        now = time.monotonic()
        # Sentinel is None, not 0.0: time.monotonic() counts from an arbitrary epoch
        # (boot on Linux). On a fresh VM (CI runners, containers) ``now`` can be smaller
        # than ``min_interval``, and ``now - 0.0 < min_interval`` would suppress the
        # FIRST redraw ever requested (CI run 32494557030: uptime < 60s made the
        # min_interval=60 rate-limit test fail both attempts).
        last = getattr(self, "_last_focus_regain_redraw", None)
        if last is not None and now - last < min_interval:
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

    def _clear_prompt_toolkit_screen(self, app, *, rebuild_scrollback: bool = False, keep_above: bool = False):
        """Clear the terminal and reset prompt_toolkit renderer state.

        Returns the ``fit`` for the replay that follows (``_transcript_room`` plus where to
        paint it). ``None`` means replay the whole history (scrollback wiped by CSI 3J). Without
        3J the older transcript stays in scrollback, so the replay may only repaint what the
        viewport held (#95375); the viewport is erased row by row because CSI 2J makes
        scroll-on-clear terminals (tmux, VTE) copy the whole screen into scrollback first,
        stacking a duplicate. With ``keep_above`` (a resize), on a terminal that keeps rows in
        place, and the whole history on screen, only its rows and the chrome are erased — what
        sits above them (the startup banner) was never recorded and a replay could not restore
        it. Counted as painted, the replay leaves the chrome's top at a known row: the chrome is
        drawn from there to the bottom (``_set_chrome_floor``), where the next count assumes it.
        A clear that fails replays nothing: the history is already on screen or in scrollback.
        """
        from cli import _terminal_reflows
        if getattr(self, "_terminal_io_broken", False):
            return _NO_REPLAY
        try:
            renderer = app.renderer
            out = renderer.output
            size = out.get_size()
            fit = None
            out.reset_attributes()
            if rebuild_scrollback:
                out.erase_screen()
                out.write_raw("\x1b[3J")
                out.cursor_goto(0, 0)
                _set_chrome_floor(None)
            else:
                room, columns, painted = self._transcript_room(app)
                room += _take_suspect_rows()
                # Only where rows stay in place: the erase below counts up from the cursor.
                keep_above = keep_above and _terminal_reflows() is False
                history_rows = _output_history_rows(room, columns, painted) if keep_above else None
                if keep_above and history_rows is not None:
                    # Rows stay where prompt_toolkit's cursor has them: its oldest row is
                    # ``history_rows`` above the chrome's top — row ``room`` once that is known.
                    out.write_raw("\r")
                    out.cursor_up(renderer._cursor_pos.y + history_rows)
                    out.erase_down()
                    known = painted and _chrome_floor() is not None
                    fit = (room, columns, painted, room - history_rows if known else None)
                else:
                    for row in range(1, size.rows + 1):  # CUP rows count from 1
                        out.cursor_goto(row, 0)
                        out.erase_end_of_line()
                    out.cursor_goto(0, 0)
                    fit = (room, columns, painted, 0 if painted else None)
            out.flush()
            # Drop cached screen + cursor state so the next _redraw() starts from a
            # known (0, 0) origin and re-renders every cell instead of diffing stale.
            renderer.reset(leave_alternate_screen=False)
            return fit
        except OSError as exc:
            if _is_eio(exc):
                self._mark_terminal_io_broken("clear_screen")
        except Exception:
            pass
        return _NO_REPLAY

    @staticmethod
    def _transcript_room(app):
        """``(rows, columns, painted)``: the viewport rows the transcript fills above the
        prompt chrome, and how to count them — at the current ``columns``, or with ``painted``
        at the width each line was painted at, as a terminal that does not reflow keeps them.
        Unless the terminal is known to reflow, count as painted: that replays at least what
        the viewport held, never less.

        The chrome is the renderer's last paint: prompt_toolkit may draw the app taller than
        its preferred height (it fills the rows below the cursor it measured, or down to the
        bottom row once a refill left its top at a known row).
        """
        from cli import _terminal_reflows
        renderer = app.renderer
        size = renderer.output.get_size()
        screen = renderer._last_screen
        if screen is None:
            drawn = app.layout.container.preferred_height(size.columns, size.rows).preferred
        else:
            drawn = screen.height
        rows = max(0, size.rows - max(renderer._min_available_height, drawn, _chrome_floor() or 0))
        return rows, size.columns, _terminal_reflows() is not True

    @staticmethod
    def _painted_row_widths(renderer) -> list[int]:
        """Cells each row of the renderer's last paint reaches: prompt_toolkit writes a row up to
        its last cell that shows something (the blanks after it are erase-to-end-of-line)."""
        screen = renderer._last_screen
        if screen is None:
            return []
        has_style = renderer._style_string_has_style
        return [1 + max((x for x, ch in screen.data_buffer[y].items() if ch.char != " " or has_style[ch.style]),
                        default=0) for y in range(screen.height)]

    def _note_chrome_paint(self, app, full: bool) -> None:
        """Remember when each row of the chrome prompt_toolkit just drew was written, and how
        wide — ``full``: the whole chrome was written anew (for ``_aim_erase_at_reflowed_chrome``).
        A row only changes height when its width does, so only those changes are kept."""
        renderer = app.renderer
        size = renderer._last_size
        try:  # the width moved while it rendered: the terminal may have got the rows clipped
            suspect = size is None or app.output.get_size().columns != size.columns
        except Exception:
            suspect = True
        now = time.monotonic()
        previous = [] if full else getattr(self, "_chrome_paints", [])
        paints = []
        for y, width in enumerate(self._painted_row_widths(renderer)):
            history = previous[y] if y < len(previous) else []
            if not history or history[-1][1] != width:
                history = [*history[-3:], (now, width, suspect)]
            paints.append(history)
        self._chrome_paints = paints
        self._chrome_paints_screen = renderer._last_screen

    def _aim_erase_at_reflowed_chrome(self, app, columns: int) -> None:
        """Point prompt_toolkit's erase at the top of its last paint as a reflowing terminal
        re-wrapped it to ``columns`` (#95375).

        ``renderer.erase()`` moves up ``_cursor_pos.y`` rows — right where rows stay in place,
        short of the top on a reflowing terminal, whose re-wrapped chrome rows would stay
        above the new paint. A row's width, re-wrapped, is its height now — if the terminal got
        the row before it narrowed. prompt_toolkit writes rows with autowrap off, so one it got
        after narrowing (painted just before the width change was seen, the signal still on its
        way) was clipped to one row instead, and counting it as re-wrapped would erase
        transcript rows above the chrome. Rows painted within ``_RESIZE_PAINT_MARGIN`` of the
        width change, or unrecorded, count as one row: a stale chrome row left above the new
        paint is benign, an erased transcript row is lost for good.
        """
        renderer = app.renderer
        screen, cursor = renderer._last_screen, renderer._cursor_pos
        if screen is None or columns <= 0:
            return
        paints = getattr(self, "_chrome_paints", []) if getattr(self, "_chrome_paints_screen", None) is screen else []
        cutoff = (getattr(self, "_resize_seen_at", None) or time.monotonic()) - _RESIZE_PAINT_MARGIN
        rows = cursor.x // columns
        for y in range(cursor.y):
            history = paints[y] if y < len(paints) else []
            settled = [i for i, (at, _width, suspect) in enumerate(history) if at < cutoff and not suspect]
            if settled:  # re-wrapped as it was then, or as a later, narrower paint reached it
                rows += -(-min(width for _at, width, _suspect in history[settled[-1]:]) // columns)
            else:
                rows += 1
        renderer._cursor_pos = cursor._replace(y=rows)

    def _recover_after_resize(self, app, original_on_resize) -> None:
        """Recover a resized classic CLI without desynchronizing cursor state.

        Never clears scrollback (the startup banner lives there and replay cannot rebuild
        it) and never resets the renderer before prompt_toolkit's own ``_on_resize``,
        which erases the chrome from the cached cursor position. The status bar / input
        rules are suppressed while the reflow settles: on column shrink the terminal reflows
        already-painted rows into scrollback first, so a fresh bar looks duplicated
        (#19280, #22976). On a column shrink a reflowing terminal has re-wrapped the old
        chrome into more rows than that cached cursor reaches, so the erase is aimed at the
        re-wrapped top instead; the transcript above is left exactly as the terminal
        re-wrapped it, never erased and replayed (a replay cannot know which rows the
        terminal kept on screen and which it pushed into scrollback, #95375). A terminal that
        does not reflow truncates every visible row at the new width instead and keeps it in
        place, out of scrollback: the viewport is refilled as Ctrl+L does — also after a drag
        that narrowed and widened back, which truncated rows all the same. When nothing says
        whether the terminal reflows it is refilled too: a reflowing terminal had already
        pushed some of those rows into scrollback and now shows them twice, which beats
        truncating them for good.
        With ``display.cli_rebuild_scrollback_on_redraw`` (3J + whole-history replay) every
        width change rebuilds.
        Same-width SIGWINCH (tmux attach, GNOME tab bar, focus) and the first signal
        without a seeded baseline are left alone — 2J+replay against preserved scrollback
        duplicates ``_OUTPUT_HISTORY`` (#65293). tmux-attach's stale previous_screen is
        handled by ``_hermes_call_output_screen_diff`` (#83874). Suppression is cleared by
        a debounced timer so the bar returns during idle; next-submit stays a fast path.
        """
        try:
            # A debounced composer resize can arrive after the alternate screen opens.
            # Its erase/replay belongs to the suspended composer, not the monitor.
            if getattr(getattr(self, '_subagent_monitor', None), 'opening', False):
                return
            from cli import _replay_output_history, _terminal_reflows
            self._status_bar_suppressed_after_resize = True
            try:
                new_width = self._get_tui_terminal_width()
            except Exception:
                new_width = None
            prev_width = getattr(self, "_last_resize_width", None)
            narrowest = min(filter(None, (new_width, getattr(self, "_resize_narrowest", None))), default=None)
            self._resize_narrowest = None
            width_changed = new_width is not None and prev_width is not None and new_width != prev_width
            narrowed = narrowest is not None and prev_width is not None and narrowest < prev_width
            reflows = _terminal_reflows()
            if width_changed and self._redraw_rebuilds_scrollback():
                _replay_output_history(self._clear_prompt_toolkit_screen(app, rebuild_scrollback=True))
            elif reflows and width_changed and new_width < prev_width:
                self._aim_erase_at_reflowed_chrome(app, new_width)
            elif not reflows and narrowed:
                fit = self._clear_prompt_toolkit_screen(app, keep_above=True)
                _replay_output_history(fit, app.renderer.output)
            if new_width is not None:
                self._last_resize_width = new_width
            if width_changed:
                self._pet_queue_kitty_frame()
            # Its redraw is skipped, and the next recovery scheduled, if the width moved again
            # since it was read above (``_output_waits_for_resize``).
            original_on_resize()
            self._schedule_status_bar_unsuppress(app)
        finally:
            # Output held since the signal paints now, against the settled screen — unless
            # another width change is already waiting for its own recovery.
            if not getattr(self, "_resize_recovery_pending", False):
                self._resize_hold_since = None
                _release_paints()

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

    def _schedule_resize_recovery(self, app, original_on_resize, delay: float = 0.3) -> None:
        """Debounce resize redraws so footer chrome is not stamped into scrollback.

        The delay outwaits tmux, which re-wraps a pane on every resize but signals it at most
        every 250 ms, so mid-drag the width a recovery reads is stale — also over ssh from a
        tmux pane, where nothing says tmux is there (#95375). A width change also holds output
        paints and redraws until its recovery; while a drag keeps signalling, a recovery still
        runs ``_RESIZE_SETTLE`` after the first width change once output was held for
        ``_RESIZE_HOLD_MAX`` seconds, so output never freezes for the whole drag. A signal that
        leaves the width alone holds nothing.
        """
        try:
            lock = getattr(self, "_resize_recovery_lock", None)
            if lock is None:
                lock = threading.Lock()
                self._resize_recovery_lock = lock
            try:
                width = app.output.get_size().columns
            except Exception:
                width = None

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
                now = time.monotonic()
                pending = getattr(self, "_resize_recovery_pending", False)
                if pending or width is None or width != getattr(self, "_last_resize_width", None):
                    if not pending:
                        self._resize_recovery_pending = True
                        self._resize_seen_at = now
                        # A hold its last recovery could not release keeps its start: the
                        # next recovery is due at once.
                        if getattr(self, "_resize_hold_since", None) is None:
                            self._resize_hold_since = now
                        self._resize_narrowest = None
                        _hold_paints()
                    if isinstance(width, int) and width > 0:  # truncated rows at the narrowest
                        self._resize_narrowest = min(width, getattr(self, "_resize_narrowest", None) or width)
                    if now - self._resize_hold_since >= _RESIZE_HOLD_MAX:
                        # Shortly after a width change, not on a timer: between two of a drag,
                        # once the terminal (a multiplexer re-wrapping its rows) has settled;
                        # one landing mid-recovery would move rows under its erase and replay.
                        delay = _RESIZE_SETTLE
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
        self._resize_original_on_resize = original_on_resize
        app._on_resize = lambda: self._schedule_resize_recovery(app, original_on_resize)
        # A render before the recovery lands at the new width from the old geometry and strands
        # chrome rows; the recovery repaints everything anyway (#95375). The final render at
        # exit is never held.
        original_redraw = app._redraw

        def _redraw_unless_resizing(render_as_done: bool = False) -> None:
            if render_as_done or not self._output_waits_for_resize(app):
                renderer = app.renderer
                floor = _chrome_floor()
                if floor:  # what CPR would tell prompt_toolkit: the rows down to the bottom
                    renderer._min_available_height = max(renderer._min_available_height, floor)
                before = renderer._last_screen, renderer._last_size
                original_redraw(render_as_done=render_as_done)
                # prompt_toolkit repaints every row when it has no previous paint or a new size
                self._note_chrome_paint(app, before[0] is None or renderer._last_size != before[1])

        app._redraw = _redraw_unless_resizing
        _set_paint_gate(app, lambda: self._output_waits_for_resize(app))

    def _output_waits_for_resize(self, app) -> bool:
        """Whether output must wait for a resize recovery: one is pending, or the terminal's
        width no longer matches the one the app last settled at — its SIGWINCH is not handled
        yet, and a paint or chrome render now would land in geometry that no recovery accounts
        for: on a terminal that does not reflow it scrolls rows the resize truncated into
        scrollback before any refill (#95375). Noticing that schedules the recovery."""
        if getattr(self, "_resize_recovery_pending", False):
            return True
        try:
            width = app.output.get_size().columns
        except Exception:
            return False
        if width == getattr(self, "_last_resize_width", None):
            return False
        self._schedule_resize_recovery(app, self._resize_original_on_resize)
        return bool(getattr(self, "_resize_recovery_pending", False))

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
        # Rate-limit to avoid thrashing if a terminal floods reports. None = never
        # (monotonic epoch is arbitrary, see _invalidate).
        last = self._last_input_mode_recovery
        if last is not None and now - last < 0.5:
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
        last = self._last_termios_drift_check  # None = never (monotonic epoch is arbitrary)
        if last is not None and now - last < 1.0:
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
