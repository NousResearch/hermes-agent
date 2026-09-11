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


class CLITuiHandlersMixin:
    def _tui_handle_voice_record(self, event):
        """Toggle voice recording when voice mode is active.

        Runs on prompt_toolkit's event-loop thread: any blocking call here (locks, sd.wait,
        disk I/O) freezes the whole UI, so all heavy work goes to daemon threads.
        """
        from cli import _DIM, _RST, _cprint, logger
        if not self._voice_mode:
            return
        if self._voice_recording:
            # Always allow STOPPING (even while the agent runs); manual stop ends continuous
            # mode. Flag clearing happens atomically inside _voice_stop_and_transcribe.
            with self._voice_lock:
                self._voice_continuous = False
            event.app.invalidate()
            threading.Thread(target=self._voice_stop_and_transcribe, daemon=True).start()
            return
        # Allow disarming continuous mode while the agent runs or transcribes — otherwise the
        # user is stuck in an auto-restart loop until /voice off.
        if self._agent_running or self._voice_processing:
            with self._voice_lock:
                self._voice_continuous = False
            event.app.invalidate()
            return
        # Don't START recording during interactive prompts.
        if self._clarify_state or self._sudo_state or self._approval_state or self._slash_confirm_state:
            return
        # Cut TTS so the user can start talking: stop_playback() just terminates a subprocess;
        # the stop event drains the streaming pipeline if one is live.
        if not self._voice_tts_done.is_set():
            try:
                logger.info("TTS CUT: record key handler cutting TTS")
                from tools.tts_streaming import mark_speech_interrupted
                mark_speech_interrupted()
                if self._voice_tts_stop is not None:
                    self._voice_tts_stop.set()
                from tools.voice_mode import stop_playback
                stop_playback()
                self._voice_tts_done.set()
            except Exception:
                pass
        with self._voice_lock:
            self._voice_continuous = True

        # play_beep(sd.wait), AudioRecorder.start(lock) and config I/O must never block the loop.
        def _start_recording():
            try:
                self._voice_start_recording()
                if hasattr(self, '_app') and self._app:
                    self._app.invalidate()
            except Exception as e:
                _cprint(f"\n{_DIM}Voice recording failed: {e}{_RST}")

        threading.Thread(target=_start_recording, daemon=True).start()
        event.app.invalidate()

    def _tui_cancel_voice_recording(self, event) -> bool:
        """Cancel an active recording; True when one was cancelled (caller stops there)."""
        from cli import _DIM, _RST, _cprint
        _recorder_ref = None
        with self._voice_lock:
            if self._voice_recording and self._voice_recorder:
                _recorder_ref = self._voice_recorder
                self._voice_recording = False
                self._voice_continuous = False
        if _recorder_ref is None:
            return False
        _cprint(f"\n{_DIM}Recording cancelled.{_RST}")
        # cancel() may block on AudioRecorder._lock / CoreAudio — keep it off the event loop.
        threading.Thread(target=_recorder_ref.cancel, daemon=True).start()
        event.app.invalidate()
        return True

    def _tui_cancel_foreground_ui(self, event, *, closers) -> bool:
        """Close the first active foreground UI (slash-confirm / picker / palette); True if any."""
        for state_attr, close in closers:
            if getattr(self, state_attr):
                close()
                event.app.current_buffer.reset()
                event.app.invalidate()
                return True
        return False

    def _tui_clear_blocking_overlays(self, event) -> bool:
        """Clear every agent-blocking overlay (approval/clarify/sudo/secret) in one shot.

        Callers must NOT return on True alone: they fall through so a stale/orphaned overlay
        (left by an earlier interrupt) can't swallow the press before the agent-interrupt
        branch, leaving the chat frozen (#14026).
        """
        if not (self._sudo_state or self._secret_state or self._approval_state or self._clarify_state):
            return False
        self._clear_active_overlays_for_interrupt()
        event.app.current_buffer.reset()
        event.app.invalidate()
        return True

    def _tui_clear_or_exit(self, event) -> None:
        """Idle press: clear text/images like bash; exit when everything is already empty."""
        if event.app.current_buffer.text or self._attached_images:
            event.app.current_buffer.reset()
            self._attached_images.clear()
            event.app.invalidate()
        else:
            self._should_exit = True
            event.app.exit()

    def _tui_handle_ctrl_c(self, event):
        """Ctrl+C priority: cancel voice recording → cancel foreground UI/overlay prompt →
        interrupt the running agent (first press) → force exit (second press within 2s) → when
        idle clear the draft or exit."""
        now = time.time()
        if self._tui_cancel_voice_recording(event):
            return
        if self._tui_cancel_foreground_ui(event, closers=(
            ("_slash_confirm_state", lambda: self._submit_slash_confirm_response("cancel")),
            ("_model_picker_state", self._close_model_picker),
            ("_command_palette_state", self._close_command_palette))):
            return
        overlay_cleared = self._tui_clear_blocking_overlays(event)
        if overlay_cleared and not (self._agent_running and self.agent):
            return
        if self._agent_running and self.agent:
            if now - self._last_ctrl_c_time < 2.0:
                print("\n⚡ Force exiting...")
                self._should_exit = True
                event.app.exit()
                return
            self._last_ctrl_c_time = now
            print("\n⚡ Interrupting agent... (press Ctrl+C again to force exit)")
            request_hard_interrupt(self.agent)
        else:
            self._tui_clear_or_exit(event)

    def _tui_handle_ctrl_q(self, event):
        """Ctrl+Q: like Ctrl+C minus the double-press force exit (and it leaves the palette)."""
        if self._tui_cancel_voice_recording(event):
            return
        if self._tui_cancel_foreground_ui(event, closers=(
            ("_slash_confirm_state", lambda: self._submit_slash_confirm_response("cancel")),
            ("_model_picker_state", self._close_model_picker))):
            return
        overlay_cleared = self._tui_clear_blocking_overlays(event)
        if overlay_cleared and not (self._agent_running and self.agent):
            return
        if self._agent_running and self.agent:
            print("\n⚡ Interrupting agent...")
            request_hard_interrupt(self.agent)
        else:
            self._tui_clear_or_exit(event)

    def _tui_make_clarify_number_handler(self, idx):
        def handler(event):
            state = self._clarify_state
            if not state or self._clarify_freetext:
                return
            choices = state.get("choices") or []
            if idx > len(choices):
                return
            # Multi-select: number keys toggle checkboxes (incl. "Other") instead of submitting.
            if state.get("multi_select"):
                indices = state.get("selected_indices", set())
                indices.symmetric_difference_update({idx})
                event.app.invalidate()
                return
            if idx == len(choices):
                # "Other" → freetext
                self._clarify_freetext = True
            elif state.get("questions"):
                # Batch mode: lock the numbered choice for the active question only.
                self._clarify_batch_lock(state, choices[idx])
            else:
                state["response_queue"].put(choices[idx])
                self._clarify_state = None
                self._clarify_freetext = False
            event.app.invalidate()
        return handler

    def _tui_restore_stash_payload(self, event, payload) -> None:
        """Put a popped (text, images) payload back into the composer."""
        if not payload:
            return
        text, images = payload
        buf = event.app.current_buffer
        buf.text = text
        buf.cursor_position = len(text)
        # Extend rather than replace attachments: the user may have attached something new since
        # the stash was taken and dropping it silently would be data loss.
        for img in images or ():
            if img not in self._attached_images:
                self._attached_images.append(img)

    def _tui_handle_stash_panel_up(self, event):
        self._prompt_stash.move_cursor(-1)
        event.app.invalidate()

    def _tui_handle_stash_panel_down(self, event):
        self._prompt_stash.move_cursor(1)
        event.app.invalidate()

    def _tui_handle_stash_panel_delete(self, event):
        """D in the browse panel discards the highlighted draft."""
        self._prompt_stash.delete_at_cursor()
        event.app.invalidate()

    def _tui_handle_stash_panel_close(self, event):
        self._prompt_stash.close_panel()
        event.app.invalidate()

    def _tui_handle_tab(self, event):
        """Tab: accept the open completion, else the ghost auto-suggestion, else start completions.

        After accepting a provider like 'anthropic:' the menu closes and complete_while_typing
        doesn't fire (no keystroke); re-triggering here makes stage-2 models appear immediately.
        """
        buf = event.current_buffer
        if buf.complete_state:
            completion = buf.complete_state.current_completion
            if completion is None:
                # Menu open but nothing selected — select first then grab it
                buf.go_to_completion(0)
                completion = buf.complete_state and buf.complete_state.current_completion
            if completion is None:
                return
            buf.apply_completion(completion)
        elif buf.suggestion and buf.suggestion.text:
            buf.insert_text(buf.suggestion.text)
        else:
            buf.start_completion()

    def _tui_handle_double_escape(self, event):
        """Double ESC discards the draft and attached images (Claude Code / Gemini CLI gesture).

        Works while the agent streams — the gap Ctrl+C leaves (it interrupts the turn and only
        clears the draft when idle). The draft is appended to history first so Up recalls it,
        which is what makes a reflex key safe. Single ESC is the Alt-sequence prefix
        (escape+enter/g/v) so the escape-timeout keeps those distinct; modal prompts bind ESC
        eagerly and are excluded so cancel still wins.
        """
        buf = event.app.current_buffer
        if not (buf.text or self._attached_images):
            return
        buf.reset(append_to_history=bool(buf.text))
        self._attached_images.clear()
        event.app.invalidate()

    def _tui_handle_ignored_terminal_sequence(self, event):
        """Consume parser-level ignored terminal sequences before self-insert.

        hermes_cli.pt_input_extras registers focus reports (CSI I / CSI O) as Keys.Ignore at the
        VT100 parser; without this no-op binding the default self-insert would still land the
        bytes in the buffer. Focus-in additionally schedules a rate-limited full repaint: while
        hidden, the emulator may have coalesced output or repainted, so prompt_toolkit's
        incremental diff would stack a fresh prompt chrome on the stale one (#60920, #25337).
        """
        try:
            for press in getattr(event, "key_sequence", None) or ():
                if getattr(press, "data", None) == "\x1b[I":
                    self._schedule_focus_regain_redraw()
                    break
        except Exception:
            pass
        return None

    def _tui_handle_escape_modal(self, event):
        """ESC cancels active secret/sudo/slash-confirm prompts."""
        if self._secret_state:
            self._cancel_secret_capture()
            event.app.current_buffer.reset()
            event.app.invalidate()
        elif self._sudo_state:
            self._sudo_state["response_queue"].put("")
            self._sudo_state = None
            event.app.invalidate()
        elif self._slash_confirm_state:
            self._submit_slash_confirm_response("cancel")
            event.app.current_buffer.reset()
            event.app.invalidate()

    def _tui_handle_ctrl_z(self, event):
        """Ctrl+Z suspends the process (Unix only)."""
        from cli import _DIM, _RST, _cprint
        if sys.platform == 'win32':
            _cprint(f"\n{_DIM}Suspend (Ctrl+Z) is not supported on Windows.{_RST}")
            event.app.invalidate()
            return
        import signal as _sig
        from prompt_toolkit.application import run_in_terminal
        from hermes_cli.skin_engine import get_active_skin
        agent_name = get_active_skin().get_branding("agent_name", "Hermes Agent")
        msg = f"\n{agent_name} has been suspended. Run `fg` to bring {agent_name} back."

        def _suspend():
            os.write(1, msg.encode())
            os.kill(0, _sig.SIGTSTP)
        run_in_terminal(_suspend)

    def _tui_handle_ctrl_d(self, event):
        """Ctrl+D deletes under the cursor (readline); exits only on empty input, like bash/zsh.
        Pending attached images count as input so the user doesn't lose them silently."""
        buf = event.app.current_buffer
        if buf.text:
            buf.delete()
        elif not self._attached_images:
            self._should_exit = True
            event.app.exit()

    def _tui_recall_without_recollapse(self, buf, move):
        """Run a history move with paste-collapse suppressed.

        Recalled history can hold the full text of a paste collapsed at submit time; loading it
        back looks like a fresh large paste to ``_on_text_changed``. If the move didn't change the
        text (plain cursor movement) the flag is cleared so a later real paste still collapses.
        """
        before = buf.text
        self._skip_paste_collapse = True
        move()
        if buf.text == before:
            self._skip_paste_collapse = False

    def _tui_handle_alt_v(self, event):
        """Alt+V pastes an image from the clipboard. Alt combos pass through every terminal
        (ESC + key), unlike Ctrl+V which terminals intercept — reliable on WSL2/VSCode/SSH.
        Silent when no image (avoid noise on accidental press)."""
        if self._try_attach_clipboard_image():
            event.app.invalidate()

    def _tui_handle_ctrl_v(self, event):
        """Image paste for terminals without bracketed paste: GNOME Terminal/Konsole send raw
        0x16. Terminals that intercept Ctrl+V (macOS Terminal, iTerm2, VSCode, Windows Terminal)
        fire the bracketed-paste handler instead and never reach this."""
        if self._try_attach_clipboard_image():
            event.app.invalidate()

    def _tui_handle_ctrl_l(self, event):
        """Ctrl+L forces a clean repaint after terminal buffer drift (tmux/cmux tab switches,
        ``clear`` from a subshell, SSH restores) that prompt_toolkit can't detect."""
        self._force_full_redraw()

    def _tui_insert_newline(self, event):
        """Newline for multi-line input (Alt+Enter; Ctrl+J/Ctrl+Enter with multiline shortcuts).
        Windows Terminal intercepts Alt+Enter (fullscreen) and delivers Ctrl+Enter as c-j."""
        event.current_buffer.insert_text('\n')

    def _tui_handle_open_in_editor(self, event):
        """Ctrl+G (or Alt+G in VSCode/Cursor) opens the draft in an external editor."""
        self._open_external_editor(event.current_buffer)

    def _tui_model_picker_down(self, event):
        state = self._model_picker_state
        if not state:
            return
        if state.get("stage") == "provider":
            max_idx = len(state.get("providers") or [])
        else:
            # +1 for "← Back" and Cancel over the filtered visible rows.
            _fp = state.get("_filtered_pairs")
            max_idx = (len(_fp) if _fp is not None else len(state.get("model_list") or [])) + 1
        state["selected"] = min(max_idx, state.get("selected", 0) + 1)
        event.app.invalidate()

    def _tui_model_picker_up(self, event):
        if self._model_picker_state:
            self._model_picker_state["selected"] = max(0, self._model_picker_state.get("selected", 0) - 1)
            event.app.invalidate()

    @staticmethod
    def _tui_set_filter(st, value: str) -> None:
        """Replace a picker/palette filter and rewind the selection + viewport."""
        st["filter"] = value
        st["selected"] = 0
        st["_scroll_offset"] = 0

    def _tui_model_picker_escape(self, event):
        """ESC clears an active filter first, else closes the picker."""
        st = self._model_picker_state
        if st and st.get("stage") == "model" and (st.get("filter") or ""):
            self._tui_set_filter(st, "")
            event.app.invalidate()
            return
        self._close_model_picker()
        event.app.current_buffer.reset()
        event.app.invalidate()

    def _tui_model_picker_filter_backspace(self, event):
        st = self._model_picker_state
        if st:
            self._tui_set_filter(st, (st.get("filter", "") or "")[:-1])
            event.app.invalidate()

    def _tui_make_model_filter_char_handler(self, ch: str):
        def handler(event):
            st = self._model_picker_state
            if not st or st.get("stage") != "model":
                return
            self._tui_set_filter(st, (st.get("filter", "") or "") + ch)
            event.app.invalidate()
        return handler

    def _tui_make_palette_char_handler(self, ch: str):
        def handler(event):
            st = self._command_palette_state
            if st:
                self._tui_set_filter(st, (st.get("filter", "") or "") + ch)
                event.app.invalidate()
        return handler

    def _tui_make_approval_number_handler(self, idx):
        def handler(event):
            if self._approval_state and idx < len(self._approval_state["choices"]):
                self._approval_state["selected"] = idx
                self._handle_approval_selection()
                event.app.invalidate()
        return handler

    def _tui_make_slash_confirm_number_handler(self, idx):
        def handler(event):
            if self._slash_confirm_state and idx < len(self._slash_confirm_state.get("choices") or []):
                self._submit_slash_confirm_response(self._slash_confirm_state["choices"][idx][0])
                event.app.current_buffer.reset()
                event.app.invalidate()
        return handler

    def _tui_clarify_toggle(self, event):
        if self._clarify_state:
            indices = self._clarify_state.get("selected_indices", set())
            indices.symmetric_difference_update({self._clarify_state["selected"]})
            event.app.invalidate()

    def _tui_clarify_down(self, event):
        if self._clarify_state:
            max_idx = len(self._clarify_state.get("choices") or [])  # last index is "Other"
            self._clarify_state["selected"] = min(max_idx, self._clarify_state["selected"] + 1)
            event.app.invalidate()

    def _tui_clarify_up(self, event):
        if self._clarify_state:
            self._clarify_state["selected"] = max(0, self._clarify_state["selected"] - 1)
            event.app.invalidate()

    def _tui_clarify_batch_step(self, event, delta: int):
        state = self._clarify_state
        if state and state.get("questions"):
            self._clarify_batch_set_active(state, (state["active"] + delta) % len(state["questions"]))
            event.app.invalidate()

    def _tui_clarify_batch_tab(self, event):
        self._tui_clarify_batch_step(event, 1)

    def _tui_clarify_batch_backtab(self, event):
        self._tui_clarify_batch_step(event, -1)

    def _tui_command_palette_backspace(self, event):
        st = self._command_palette_state
        if st:
            self._tui_set_filter(st, (st.get("filter", "") or "")[:-1])
            event.app.invalidate()

    def _tui_command_palette_down(self, event):
        st = self._command_palette_state
        if st:
            n = st.get("_visible_count", len(self._command_palette_visible_entries()))
            st["selected"] = min(max(0, n - 1), st.get("selected", 0) + 1)
            event.app.invalidate()

    def _tui_command_palette_up(self, event):
        st = self._command_palette_state
        if st:
            st["selected"] = max(0, st.get("selected", 0) - 1)
            event.app.invalidate()

    def _tui_command_palette_enter(self, event):
        self._handle_command_palette_selection()
        event.app.invalidate()

    def _tui_command_palette_escape(self, event):
        self._close_command_palette()
        event.app.invalidate()

    def _tui_open_command_palette(self, event):
        self._open_command_palette()
        event.app.invalidate()

    def _tui_slash_confirm_down(self, event):
        st = self._slash_confirm_state
        if st:
            st["selected"] = min(len(st.get("choices") or []) - 1, st.get("selected", 0) + 1)
            event.app.invalidate()

    def _tui_slash_confirm_up(self, event):
        st = self._slash_confirm_state
        if st:
            st["selected"] = max(0, st.get("selected", 0) - 1)
            event.app.invalidate()

    def _tui_approval_down(self, event):
        st = self._approval_state
        if st:
            st["selected"] = min(len(st["choices"]) - 1, st["selected"] + 1)
            event.app.invalidate()

    def _tui_approval_up(self, event):
        st = self._approval_state
        if st:
            st["selected"] = max(0, st["selected"] - 1)
            event.app.invalidate()


