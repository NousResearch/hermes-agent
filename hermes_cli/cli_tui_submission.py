"""prompt_toolkit TUI topic sibling; cli.py-internal symbols remain lazy."""

from __future__ import annotations

import errno

import json

import os

import queue

import shutil

import string

import sys

import threading

import time

from agent.interrupt_compat import request_hard_interrupt

from hermes_cli.commands_completion import SlashCommandAutoSuggest, SlashCommandCompleter

from pathlib import Path

from prompt_toolkit.filters import Condition

from prompt_toolkit.history import FileHistory

from prompt_toolkit.key_binding import KeyBindings

from prompt_toolkit.layout import (
    ConditionalContainer,
    FormattedTextControl,
    HSplit,
    Layout,
    Window,
    WindowAlign)

from prompt_toolkit.layout.dimension import Dimension

from prompt_toolkit.layout.menus import CompletionsMenu

from prompt_toolkit.layout.processors import (
    ConditionalProcessor,
    PasswordProcessor,
    Processor,
    Transformation)

from prompt_toolkit.styles import Style as PTStyle

from prompt_toolkit.widgets import TextArea

from typing import Optional

_PANEL_RESERVED_BELOW = 6

_TYPING_CHARS = string.digits + string.ascii_letters + "-_.:/ "

_APPROVAL_CHOICE_LABELS = {
    "once": "Allow once",
    "session": "Allow for this session",
    "always": "Add to permanent allowlist",
    "deny": "Deny",
    "view": "Show full command"}

def _num_prefix(i: int) -> str:
    """Quick-select key label: 1-9 for items 1-9, 0 for the 10th, blank beyond."""
    return str(i + 1) if i < 9 else ("0" if i == 9 else " ")

def _term_rows() -> int:
    return shutil.get_terminal_size((100, 24)).lines

class _Panel:
    """Fragment accumulator for one bordered overlay panel (``(style, text)`` tuples)."""

    def __init__(self, border: str, box_width: int, title: str = "", title_style: str = ""):
        from cli import _append_blank_panel_line, _append_panel_line
        self.lines, self.border, self.width = [], border, box_width
        self._row, self._blank = _append_panel_line, _append_blank_panel_line
        if title:
            # Title inlined into the top rule: ``╭─ Title ───╮``.
            self.lines.append((border, "╭─ "))
            self.lines.append((title_style, title))
            self.lines.append((border, " " + ("─" * max(0, box_width - len(title) - 3)) + "╮\n"))
        else:
            self.lines.append((border, "╭" + ("─" * box_width) + "╮\n"))

    def row(self, style: str, text: str) -> None:
        self._row(self.lines, self.border, style, text, self.width)

    def blank(self) -> None:
        self._blank(self.lines, self.border, self.width)

    def close(self) -> list:
        self.lines.append((self.border, "╰" + ("─" * self.width) + "╯\n"))
        return self.lines

def _wrap_rows(wrap, items, width, indent) -> list[tuple[int, str]]:
    """``(index, wrapped_line)`` pairs so selected styling can be re-applied per row."""
    return [(i, w) for i, label in enumerate(items) for w in wrap(label, width, subsequent_indent=indent)]


class CLITuiSubmissionMixin:
    def _tui_handle_enter(self, event):
        """Enter: submit input.

        Modal overlays (sudo/secret/approval/slash-confirm/picker/clarify) are answered first via
        ``_tui_enter_overlay``. Otherwise: agent running → busy_input_mode routing (steer /
        redirect / interrupt queue / next-turn queue); idle → ``_pending_input``. Slash and bang
        commands always take the local-dispatch path (never steer/interrupt text to the model).
        """
        from cli import (
            _apply_backslash_line_continuation,
            _is_backslash_line_continuation,
            _looks_like_slash_command)
        if self._tui_enter_overlay(event):
            return
        buf = event.app.current_buffer
        raw_text = buf.text
        if (
            self._tui_multiline_shortcuts
            and buf.cursor_position == len(raw_text)
            and _is_backslash_line_continuation(raw_text)):
            continued = _apply_backslash_line_continuation(raw_text)
            buf.text = continued
            buf.cursor_position = len(continued)
            event.app.invalidate()
            return
        text = raw_text.strip()
        has_images = bool(self._attached_images)
        if not (text or has_images):
            return
        if self._tui_enter_inline_command(event, text, has_images):
            return
        # Snapshot and clear attached images; bundle text + images as a tuple when present.
        images = list(self._attached_images)
        self._attached_images.clear()
        event.app.invalidate()
        payload = (text, images) if images else text
        # A bang command is treated like a slash command while the agent is busy: it must never
        # be routed into steer/redirect (injecting `!git status` into the model's context as a
        # prompt). It queues and runs locally once the loop drains.
        _is_local_dispatch = bool(text) and (_looks_like_slash_command(text) or text.strip().startswith("!"))
        if self._agent_running and not _is_local_dispatch:
            self._tui_enter_while_busy(text, images, payload)
        else:
            self._pending_input.put(payload)
        # History stores real pasted content, not the placeholder, so up-arrow recall restores it.
        self._inline_pastes(buf)
        buf.reset(append_to_history=True)

    def _tui_enter_inline_command(self, event, text: str, has_images: bool) -> bool:
        """Run /model, /steer, /bg, /btw directly on the UI thread; True when handled.

        /model needs the prompt_toolkit terminal-handoff helpers of the interactive pickers.
        /steer, /bg and /btw while the agent runs must not queue through _pending_input: the
        process loop is blocked inside self.chat(), so they would only run after the foreground
        turn — turning /steer into a next-turn message (defeating mid-run injection, #34569) and
        starting the /bg side task after the turn it should run alongside (#75221). The
        foreground turn is left alone: no interrupt, no steer. agent.steer() is thread-safe.

        Every branch invalidates after reset: process_command() prints through patch_stdout
        and never invalidates the app, so the just-cleared input area would keep showing the
        submitted text (looking unsent, inviting a re-submit) until some unrelated redraw.
        """
        if self._should_handle_model_command_inline(text, has_images=has_images):
            if not self.process_command(text):
                self._should_exit = True
                if event.app.is_running:
                    event.app.exit()
        elif (
            self._should_handle_steer_command_inline(text, has_images=has_images)
            or self._should_handle_background_command_inline(text, has_images=has_images)):
            self.process_command(text)
        else:
            return False
        event.app.current_buffer.reset(append_to_history=True)
        event.app.invalidate()
        return True

    def _tui_enter_while_busy(self, text: str, images: list, payload) -> None:
        """Route a submission typed while the agent runs, per ``busy_input_mode``.

        steer → agent.steer(text) mid-run (images can't ride along and a missing/rejecting
        steer() falls back to queue so nothing is lost). interrupt → agent.redirect() when the
        agent supports active-turn redirect, else the legacy interrupt queue (older agents,
        multimodal follow-ups, or a turn that finished in the race). queue → next turn.
        """
        from cli import CLI_CONFIG, _ACCENT, _DIM, _RST, _cprint, _hermes_home
        _effective_mode = self.busy_input_mode
        redirected = False
        if _effective_mode == "steer":
            if images or not text:
                _effective_mode = "queue"
            else:
                accepted = False
                try:
                    if self.agent is not None and hasattr(self.agent, "steer"):
                        accepted = bool(self.agent.steer(text))
                except Exception as exc:
                    _cprint(f"  {_DIM}Steer failed ({exc}) — queued for next turn.{_RST}")
                    accepted = False
                if accepted:
                    preview = text[:80] + ("..." if len(text) > 80 else "")
                    _cprint(f"  {_ACCENT}⏩ Steered: '{preview}'{_RST}")
                else:
                    _effective_mode = "queue"
        if _effective_mode == "queue":
            self._pending_input.put(payload)
            preview = text if text else f"[{len(images)} image{'s' if len(images) != 1 else ''} attached]"
            _cprint(f"  Queued for the next turn: {preview[:80]}{'...' if len(preview) > 80 else ''}")
        elif _effective_mode == "interrupt":
            if not images and text:
                try:
                    if (
                        self.agent is not None
                        and getattr(self.agent, "_supports_active_turn_redirect", False) is True
                        and hasattr(self.agent, "redirect")):
                        redirected = bool(self.agent.redirect(text))
                except Exception:
                    redirected = False
            if redirected:
                preview = text[:80] + ("..." if len(text) > 80 else "")
                _cprint(f"  {_ACCENT}↪ Redirected current turn: '{preview}'{_RST}")
            else:
                self._interrupt_queue.put(payload)
                try:
                    with open(_hermes_home / "interrupt_debug.log", "a", encoding="utf-8") as _f:
                        _f.write(
                            f"{time.strftime('%H:%M:%S')} ENTER: queued interrupt msg={str(payload)[:60]!r}, "
                            f"agent_running={self._agent_running}\n")
                except Exception:
                    pass
        # First-touch onboarding: one-line tip about the /busy knob on the first busy-while-
        # running event for this install; the flag persists to config.yaml. Guarded so
        # onboarding can never break the input loop.
        try:
            from agent.onboarding import BUSY_INPUT_FLAG, busy_input_hint_cli, is_seen, mark_seen
            if not is_seen(CLI_CONFIG, BUSY_INPUT_FLAG):
                _hint_mode = "redirect" if redirected else _effective_mode
                _cprint(f"  {_DIM}{busy_input_hint_cli(_hint_mode)}{_RST}")
                mark_seen(_hermes_home / "config.yaml", BUSY_INPUT_FLAG)
                CLI_CONFIG.setdefault("onboarding", {}).setdefault("seen", {})[BUSY_INPUT_FLAG] = True
        except Exception:
            pass

    def _tui_enter_overlay(self, event) -> bool:
        """Enter while a modal overlay is up: submit it. True when handled."""
        from cli import _cprint
        buf = event.app.current_buffer
        if self._sudo_state:
            self._sudo_state["response_queue"].put(buf.text)
            self._sudo_state = None
            event.app.invalidate()
            return True
        if self._secret_state:
            value = buf.text
            buf.reset()
            self._submit_secret_response(value)
            event.app.invalidate()
            return True
        if self._approval_state:
            self._handle_approval_selection()
            event.app.invalidate()
            return True
        if self._slash_confirm_state:
            # Typed choice wins over the highlighted one.
            text = buf.text.strip()
            choices = self._slash_confirm_state.get("choices") or []
            choice = self._normalize_slash_confirm_choice(text, choices) if text else None
            if choice is None:
                selected = self._slash_confirm_state.get("selected", 0)
                if 0 <= selected < len(choices):
                    choice = choices[selected][0]
            self._submit_slash_confirm_response(choice or "cancel")
            buf.reset()
            event.app.invalidate()
            return True
        if self._model_picker_state:
            try:
                # Picker selections follow the same session-scoped default as /model <name>
                # (model.persist_switch_by_default).
                from hermes_cli.model_switch import resolve_persist_behavior
                self._handle_model_picker_selection(persist_global=resolve_persist_behavior(False, False))
            except Exception as _exc:
                _cprint(f"  ✗ Model selection failed: {_exc}")
                self._close_model_picker()
            buf.reset()
            event.app.invalidate()
            return True
        if self._clarify_state and self._clarify_freetext:
            self._tui_enter_clarify_freetext(event)
            return True
        if self._clarify_state:
            self._tui_enter_clarify_choice(event)
            return True
        return False

    def _tui_enter_clarify_freetext(self, event) -> None:
        """Clarify "Other": submit the typed answer (empty input is ignored)."""
        buf = event.app.current_buffer
        text = buf.text.strip()
        if not text:
            return
        state = self._clarify_state
        base = getattr(self, '_clarify_multi_base', None)
        if state.get("questions"):
            # Batch mode: lock the typed answer for the active question. Multi-select "Other"
            # appends the typed answer to the checked labels as a JSON array string.
            if base is not None:
                answer = json.dumps(base + [text], ensure_ascii=False)
                meta = {"kind": "multi", "choices": list(base), "other_text": text}
                self._clarify_multi_base = None
            else:
                answer = text
                meta = {"kind": "other", "other_text": text}
            self._clarify_freetext = False
            self._clarify_prefill = ""
            self._clarify_batch_lock(state, answer, meta=meta)
        else:
            # Multi-select: prepend the previously checked real choices.
            if base:
                text = ", ".join(base) + ", " + text
                self._clarify_multi_base = None
            state["response_queue"].put(text)
            self._clarify_state = None
            self._clarify_freetext = False
        buf.reset()
        event.app.invalidate()

    def _tui_enter_clarify_choice(self, event) -> None:
        """Clarify choice mode: confirm the highlighted selection."""
        state = self._clarify_state
        if state.get("questions"):
            # Batch mode: lock the active question's answer and advance to the next unanswered.
            self._clarify_batch_enter(state)
            # Editing an earlier "Other" answer: prefill the composer with the previous text.
            if self._clarify_freetext and self._clarify_prefill:
                event.app.current_buffer.text = self._clarify_prefill
                event.app.current_buffer.cursor_position = len(self._clarify_prefill)
                self._clarify_prefill = ""
            event.app.invalidate()
            return
        selected = state["selected"]
        choices = state.get("choices") or []
        if state.get("multi_select"):
            indices = state.get("selected_indices")
            if not indices:
                # Nothing checked → submit empty string (parses to []).
                state["response_queue"].put("")
                self._clarify_state = None
            else:
                sorted_idx = sorted(indices)
                selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
                if len(choices) in sorted_idx and selected_choices:
                    # "Other" + real choices: remember the base, switch to freetext so the typed
                    # custom answer gets appended.
                    self._clarify_multi_base = selected_choices
                    self._clarify_freetext = True
                elif selected_choices:
                    state["response_queue"].put(", ".join(selected_choices))
                    self._clarify_state = None
                else:
                    self._clarify_freetext = True  # only "Other" checked
        elif selected < len(choices):
            state["response_queue"].put(choices[selected])
            self._clarify_state = None
        else:
            self._clarify_freetext = True  # "Other" selected
        event.app.invalidate()

    def _tui_collapse_paste(self, text: str, line_count: int, *, fallback: bool) -> str:
        """Save a large paste under ~/.hermes/pastes and return the placeholder for the buffer."""
        from cli import _hermes_home, datetime, logger
        self._tui_paste_counter += 1
        paste_dir = _hermes_home / "pastes"
        paste_dir.mkdir(parents=True, exist_ok=True)
        paste_file = paste_dir / f"paste_{self._tui_paste_counter}_{datetime.now().strftime('%H%M%S')}.txt"
        paste_file.write_text(text, encoding="utf-8")
        logger.info(
            "Collapsed paste #%d: %d lines, %d chars -> %s" + (" (fallback)" if fallback else ""),
            self._tui_paste_counter, line_count + 1, len(text), paste_file)
        self._tui_paste_just_collapsed = True
        return f"[Pasted text #{self._tui_paste_counter}: {line_count + 1} lines \u2192 {paste_file}]"

    def _tui_paste_over_threshold(self, text: str, line_count: int, threshold_key: str) -> bool:
        threshold = self.config.get(threshold_key, 5)
        char_threshold = self.config.get("paste_collapse_char_threshold", 2000)
        lines_hit = threshold > 0 and line_count >= threshold
        chars_hit = char_threshold > 0 and len(text) >= char_threshold
        return lines_hit or chars_hit

    def _tui_handle_paste(self, event):
        """Bracketed paste: strip leaked terminal responses, auto-attach a clipboard image only
        for image-only/empty gestures (so text pastes and dictation never attach stale images),
        and collapse large pastes to a file-reference placeholder, preserving existing text."""
        from cli import (
            _should_auto_attach_clipboard_image_on_paste,
            _strip_leaked_bracketed_paste_wrappers,
            _strip_leaked_terminal_responses_with_meta,
            logger)
        # Diagnostic canary: log when the handler blocks the event loop >500ms so recurring
        # "CLI freezes on paste" reports (#16263, macOS Tahoe + iTerm2/Ghostty) arrive with data.
        _paste_handler_start = time.perf_counter()
        _paste_raw_size = len(event.data or "")
        # Normalise line endings so the collapse threshold and display are consistent.
        pasted_text = (event.data or "").replace('\r\n', '\n').replace('\r', '\n')
        pasted_text = _strip_leaked_bracketed_paste_wrappers(pasted_text)
        pasted_text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(pasted_text)
        if _had_mouse_reports:
            self._recover_terminal_input_modes(reason="mouse reports leaked into bracketed paste payload")
        if _should_auto_attach_clipboard_image_on_paste(pasted_text) and self._try_attach_clipboard_image():
            event.app.invalidate()
        if pasted_text:
            # Sanitize surrogates (Word/Google Docs paste) before writing.
            from agent.message_sanitization import _sanitize_surrogates
            pasted_text = _sanitize_surrogates(pasted_text)
            line_count = pasted_text.count('\n')
            buf = event.current_buffer
            if (
                self._tui_paste_over_threshold(pasted_text, line_count, "paste_collapse_threshold")
                and not buf.text.strip().startswith('/')):
                placeholder = self._tui_collapse_paste(pasted_text, line_count, fallback=False)
                prefix = "\n" if buf.cursor_position > 0 and buf.text[buf.cursor_position - 1] != '\n' else ""
                buf.insert_text(prefix + placeholder)
            else:
                buf.insert_text(pasted_text)
        _paste_handler_elapsed_ms = (time.perf_counter() - _paste_handler_start) * 1000.0
        if _paste_handler_elapsed_ms > 500.0:
            logger.warning(
                "Slow bracketed-paste handler: %.1fms to process %d bytes "
                "(%d lines) on %s. If the input becomes unresponsive after "
                "this, attach this log line to the bug report.",
                _paste_handler_elapsed_ms,
                _paste_raw_size,
                pasted_text.count('\n') + 1 if pasted_text else 0,
                sys.platform)

    def _tui_on_text_changed(self, buf):
        """Fallback paste collapse for terminals without bracketed paste.

        Either heuristic triggers: many characters added in one event (paste delivered in one
        tick), or the newline count jumped by 4+ (terminals that feed characters individually
        but batch newlines; Alt+Enter adds 1 newline per event so never trips it).
        """
        from cli import _strip_leaked_bracketed_paste_wrappers, _strip_leaked_terminal_responses_with_meta
        text = _strip_leaked_bracketed_paste_wrappers(buf.text)
        text, _had_mouse_reports = _strip_leaked_terminal_responses_with_meta(text)
        if _had_mouse_reports:
            self._recover_terminal_input_modes(reason="mouse reports leaked into prompt buffer")
        if text != buf.text:
            cursor = min(buf.cursor_position, len(text))
            self._tui_paste_just_collapsed = True
            buf.text = text
            buf.cursor_position = cursor
            self._tui_prev_text_len = len(text)
            self._tui_prev_newline_count = text.count('\n')
            return
        chars_added = len(text) - self._tui_prev_text_len
        self._tui_prev_text_len = len(text)
        if self._tui_paste_just_collapsed or self._skip_paste_collapse:
            self._tui_paste_just_collapsed = False
            self._skip_paste_collapse = False
            self._tui_prev_newline_count = text.count('\n')
            return
        line_count = text.count('\n')
        newlines_added = line_count - self._tui_prev_newline_count
        self._tui_prev_newline_count = line_count
        is_paste = chars_added > 1 or newlines_added >= 4
        if (
            self._tui_paste_over_threshold(text, line_count, "paste_collapse_threshold_fallback")
            and is_paste
            and not text.startswith('/')):
            buf.text = self._tui_collapse_paste(text, line_count, fallback=True)
            buf.cursor_position = len(buf.text)

    def _tui_handle_prompt_stash(self, event):
        """Ctrl+S: composer has content → push onto the stash and clear; empty + one stashed →
        pop it back; empty + several → open the browse panel; panel open → close it.

        A stack (not a single slot) is what makes repeated Ctrl+S safe: a second stash never
        silently overwrites the first, both stay reachable in the panel.
        """
        from hermes_cli.prompt_stash import ACTION_RESTORED, ACTION_STASHED, resolve_ctrl_s
        buf = event.app.current_buffer
        action, payload = resolve_ctrl_s(self._prompt_stash, buf.text, self._attached_images)
        if action == ACTION_STASHED:
            # reset() (not `text = ""`) also clears completion state, selection and the undo stack.
            buf.reset()
            self._attached_images.clear()
        elif action == ACTION_RESTORED:
            self._tui_restore_stash_payload(event, payload)
        # ACTION_OPEN_PANEL: resolve_ctrl_s already flipped panel_open.
        event.app.invalidate()

    def _tui_handle_stash_panel_restore(self, event):
        """Enter in the browse panel restores the highlighted draft."""
        self._tui_restore_stash_payload(event, self._prompt_stash.restore_at_cursor())
        event.app.invalidate()

    def _tui_history_up(self, event):
        """Up: browse history when on the first line, else move the cursor up."""
        buf = event.app.current_buffer
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_up(count=event.arg))

    def _tui_history_down(self, event):
        buf = event.app.current_buffer
        self._tui_recall_without_recollapse(buf, lambda: buf.auto_down(count=event.arg))


