"""Modal prompt / slash-confirm machinery for the interactive CLI (god-file decomposition).

This module hosts the modal text-input and slash-confirm handlers lifted out of
``cli.py``'s ``HermesCLI`` class. ``HermesCLI`` inherits ``CLIModalConfirmMixin``
so every ``self.<handler>`` call resolves unchanged via the MRO — behavior-neutral.

Import discipline mirrors the other ``hermes_cli.cli_*_mixin`` modules:
  * Only stdlib, cycle-free dependencies are imported at module top level below.
  * Everything else (``threading``, ``time``, ``hermes_cli.curses_ui``,
    ``prompt_toolkit``) is imported LAZILY inside each method, so this module
    never imports ``cli`` at top level (no cycle).
"""

from __future__ import annotations

import queue
import shutil
import sys
import textwrap


class CLIModalConfirmMixin:
    """Mixin holding interactive-CLI modal prompt and slash-confirm handlers."""

    def _run_curses_picker(self, title: str, items: list[str], default_index: int = 0) -> int | None:
        """Run curses_single_select via run_in_terminal so prompt_toolkit handles terminal ownership cleanly."""
        import threading
        from hermes_cli.curses_ui import curses_single_select

        result = [None]

        def _pick():
            result[0] = curses_single_select(title, items, default_index=default_index)

        # run_in_terminal requires an asyncio event loop — only exists in the
        # main prompt_toolkit thread.  If we're in a background thread (e.g.
        # process_loop), fall back to direct curses call.
        in_main_thread = threading.current_thread() is threading.main_thread()

        if self._app and in_main_thread:
            from prompt_toolkit.application import run_in_terminal
            was_visible = self._status_bar_visible
            self._status_bar_visible = False
            self._app.invalidate()
            try:
                run_in_terminal(_pick)
            finally:
                self._status_bar_visible = was_visible
                self._app.invalidate()
        else:
            _pick()

        return result[0]

    def _prompt_text_input(self, prompt_text: str) -> str | None:
        """Prompt for free-text input safely inside or outside prompt_toolkit.

        Mirrors the thread-aware guard in ``_run_curses_picker``: ``run_in_terminal``
        returns a coroutine that must be awaited by the prompt_toolkit event loop,
        which only exists on the main thread.  Slash commands are dispatched from
        the ``process_loop`` daemon thread (see issue #23185), so calling
        ``run_in_terminal`` from there orphans the coroutine — ``_ask`` never runs,
        and user keystrokes leak into the composer instead.  Fall back to a direct
        ``input()`` when we're off the main thread.
        """
        import threading
        result = [None]

        def _ask():
            try:
                result[0] = input(prompt_text).strip() or None
            except (KeyboardInterrupt, EOFError):
                pass

        in_main_thread = threading.current_thread() is threading.main_thread()

        # Slash-worker guard (#23185 / billing auto-reload hang): when a
        # prompt_toolkit app is running but we're on a non-main thread (the
        # process_loop / TUI slash-worker daemon thread), stdin is owned by the
        # event loop / JSON-RPC pipe.  A bare input() there blocks forever until
        # the worker's 45s timeout fires.  We cannot safely prompt off the main
        # thread, so cancel cleanly (None) instead of hanging — mirrors the
        # _stdin_fallback discipline in _prompt_text_input_modal.
        if self._app and not in_main_thread:
            self._invalidate()
            return None

        if self._app and in_main_thread:
            from prompt_toolkit.application import run_in_terminal
            was_visible = self._status_bar_visible
            self._status_bar_visible = False
            self._app.invalidate()
            try:
                run_in_terminal(_ask)
            except Exception:
                # WSL / Warp / certain terminal emulators silently drop the
                # scheduled coroutine.  Fall back to a direct input() so the
                # user's keystrokes don't leak into the agent buffer.
                try:
                    _ask()
                except Exception:
                    pass
            finally:
                self._status_bar_visible = was_visible
                self._app.invalidate()
        else:
            _ask()
        return result[0]

    def _prompt_text_input_modal(
        self,
        *,
        title: str,
        detail: str,
        choices: list[tuple[str, str, str]],
        timeout: float = 120,
    ) -> str | None:
        """Prompt through the prompt_toolkit composer instead of raw input().

        This is for CLI slash-command confirmations.  The old raw input() path
        fought prompt_toolkit's active stdin ownership: in some terminals the
        prompt appeared above the TUI, choices were redrawn later, and Enter
        could be interpreted as EOF/exit.  A first-class modal state keeps the
        choices visible and lets the normal Enter key binding submit the typed
        or highlighted choice.

        **Platform note (Windows — issue #33961):**
        Earlier code bypassed the modal on ``sys.platform == "win32"`` and fell
        back to a raw ``input()`` prompt.  When the confirm was triggered from the
        ``process_loop`` daemon thread (the normal case) that ``input()`` ran off
        the main thread and deadlocked against prompt_toolkit's stdin ownership —
        the user saw a frozen cursor and Ctrl-C was swallowed (bare ``/reset``
        froze; ``/reset now`` worked only because it skips the prompt entirely).

        Native Windows now uses the same path as Linux/macOS: the modal is set up
        on ``self._app.loop`` via ``call_soon_threadsafe`` and answered by the
        normal prompt_toolkit key bindings (the same input channel that already
        handles ordinary typing on Windows).  The raw ``input()`` fallback is kept
        only for the genuinely safe cases: no running app (unit tests /
        non-interactive), no resolvable event loop, or a scheduling failure.
        """
        import threading
        import time as _time

        if not choices:
            return None

        # If prompt_toolkit is not running (unit tests / non-interactive calls),
        # keep the simple stdin fallback.
        if not getattr(self, "_app", None):
            return self._prompt_text_input("Choice [1/2/3]: ")

        try:
            app_loop = self._app.loop
        except Exception:
            app_loop = None

        in_main_thread = threading.current_thread() is threading.main_thread()

        def _stdin_fallback() -> str | None:
            # On native Windows a raw input() from a non-main thread deadlocks
            # against prompt_toolkit's stdin ownership (#33961).  With an app
            # running we cannot safely prompt off the main thread, so cancel
            # cleanly (None) rather than hang the terminal.
            if sys.platform == "win32" and not in_main_thread:
                self._invalidate()
                return None
            return self._prompt_text_input("Choice [1/2/3]: ")

        if not in_main_thread and app_loop is None:
            return _stdin_fallback()

        response_queue = queue.Queue()

        def _setup_modal() -> None:
            self._capture_modal_input_snapshot()
            self._slash_confirm_state = {
                "title": title,
                "detail": detail,
                "choices": choices,
                "selected": 0,
                "response_queue": response_queue,
            }
            self._slash_confirm_deadline = _time.monotonic() + timeout
            self._invalidate()

        def _teardown_modal() -> None:
            self._slash_confirm_state = None
            self._slash_confirm_deadline = 0
            self._restore_modal_input_snapshot()
            self._invalidate()

        def _run_on_app_loop(fn) -> bool:
            if in_main_thread or app_loop is None:
                fn()
                return True
            ready = threading.Event()

            def _wrapped() -> None:
                try:
                    fn()
                finally:
                    ready.set()

            try:
                app_loop.call_soon_threadsafe(_wrapped)
            except Exception:
                return False
            return ready.wait(timeout=5)

        if not _run_on_app_loop(_setup_modal):
            return _stdin_fallback()

        _last_countdown_refresh = _time.monotonic()
        try:
            while True:
                try:
                    result = response_queue.get(timeout=1)
                    _run_on_app_loop(_teardown_modal)
                    return result
                except queue.Empty:
                    remaining = self._slash_confirm_deadline - _time.monotonic()
                    if remaining <= 0:
                        break
                    now = _time.monotonic()
                    if now - _last_countdown_refresh >= 5.0:
                        _last_countdown_refresh = now
                        self._invalidate()
        finally:
            if self._slash_confirm_state is not None:
                _run_on_app_loop(_teardown_modal)
        return None

    def _submit_slash_confirm_response(self, value: str | None) -> None:
        state = self._slash_confirm_state
        if not state:
            return
        state["response_queue"].put(value)
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._invalidate()

    def _normalize_slash_confirm_choice(
        self,
        raw: str | None,
        choices: list[tuple[str, str, str]],
    ) -> str | None:
        if raw is None:
            return None
        choice_raw = raw.strip().lower()
        if not choice_raw:
            return None
        aliases = {
            "1": "once",
            "once": "once",
            "approve": "once",
            "yes": "once",
            "y": "once",
            "ok": "once",
            "2": "always",
            "always": "always",
            "remember": "always",
            "3": "cancel",
            "cancel": "cancel",
            "nevermind": "cancel",
            "no": "cancel",
            "n": "cancel",
        }
        allowed = {choice[0] for choice in choices}
        normalized = aliases.get(choice_raw)
        if normalized in allowed:
            return normalized
        if choice_raw in allowed:
            return choice_raw
        return None

    def _get_slash_confirm_display_fragments(self):
        """Render the /new-/clear-style confirmation panel."""
        state = self._slash_confirm_state
        if not state:
            return []

        title = state.get("title") or "Confirm action"
        detail = state.get("detail") or ""
        choices = state.get("choices") or []
        selected = state.get("selected", 0)

        def _panel_box_width(title_text: str, content_lines: list[str], min_width: int = 56, max_width: int = 86) -> int:
            term_cols = shutil.get_terminal_size((100, 20)).columns
            longest = max([len(title_text)] + [len(line) for line in content_lines] + [min_width - 4])
            inner = min(max(longest + 4, min_width - 2), max_width - 2, max(24, term_cols - 6))
            return inner + 2

        def _wrap_panel_text(text: str, width: int, subsequent_indent: str = "") -> list[str]:
            wrapped = textwrap.wrap(
                text,
                width=max(8, width),
                replace_whitespace=False,
                drop_whitespace=False,
                subsequent_indent=subsequent_indent,
            )
            return wrapped or [""]

        def _append_panel_line(lines, border_style: str, content_style: str, text: str, box_width: int) -> None:
            inner_width = max(0, box_width - 2)
            lines.append((border_style, "│ "))
            lines.append((content_style, text.ljust(inner_width)))
            lines.append((border_style, " │\n"))

        def _append_blank_panel_line(lines, border_style: str, box_width: int) -> None:
            lines.append((border_style, "│" + (" " * box_width) + "│\n"))

        preview_lines = []
        for line in detail.splitlines():
            preview_lines.extend(_wrap_panel_text(line, 72))
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            preview_lines.extend(_wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", 72, subsequent_indent="    "))
        preview_lines.append("Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.")

        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)
        detail_wrapped = []
        for line in detail.splitlines():
            detail_wrapped.extend(_wrap_panel_text(line, inner_text_width))
        choice_wrapped: list[tuple[int, str]] = []
        for idx, (_value, label, desc) in enumerate(choices):
            marker = "❯" if idx == selected else " "
            for wrapped in _wrap_panel_text(f"{marker} [{idx + 1}] {label} — {desc}", inner_text_width, subsequent_indent="    "):
                choice_wrapped.append((idx, wrapped))

        term_rows = shutil.get_terminal_size((100, 24)).lines
        reserved_below = 6
        chrome_full = 6
        available = max(0, term_rows - reserved_below)
        max_detail_rows = max(1, available - chrome_full - len(choice_wrapped))
        max_detail_rows = min(max_detail_rows, 8)
        if len(detail_wrapped) > max_detail_rows:
            keep = max(1, max_detail_rows - 1)
            detail_wrapped = detail_wrapped[:keep] + ["… (detail truncated)"]

        lines = []
        lines.append(('class:approval-border', '╭' + ('─' * box_width) + '╮\n'))
        _append_panel_line(lines, 'class:approval-border', 'class:approval-title', title, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for wrapped in detail_wrapped:
            _append_panel_line(lines, 'class:approval-border', 'class:approval-desc', wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        for idx, wrapped in choice_wrapped:
            style = 'class:approval-selected' if idx == selected else 'class:approval-choice'
            _append_panel_line(lines, 'class:approval-border', style, wrapped, box_width)
        _append_blank_panel_line(lines, 'class:approval-border', box_width)
        _append_panel_line(lines, 'class:approval-border', 'class:approval-cmd', 'Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels.', box_width)
        lines.append(('class:approval-border', '╰' + ('─' * box_width) + '╯\n'))
        return lines
