"""prompt_toolkit TUI construction, key-binding handlers, and overlay display fragments for the
interactive CLI. Mixin bound onto ``HermesCLI`` via the MRO; cli.py-internal symbols are imported
LAZILY inside each method (``from cli import ...``) — never at module load (import cycle)."""

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

# Rows below an overlay panel taken by spinner/tool-progress, status bar, input, separators and
# prompt symbol (measured ~6 during live PTY approval prompts) — shared by every panel budget.
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


class CLITuiMixin:
    """prompt_toolkit TUI construction, key-binding handlers, and overlay display fragments."""

    def _tui_input_rule_height(self, position: str, width: Optional[int] = None) -> int:
        """Visible height for the top/bottom input separator rules."""
        if position not in {"top", "bottom"}:
            raise ValueError(f"Unknown input rule position: {position}")
        if getattr(self, "_status_bar_suppressed_after_resize", False):
            return 0
        if position == "top":
            return 1
        return 0 if self._use_minimal_tui_chrome(width=width) else 1

    def _get_slash_confirm_display_fragments(self):
        """Render the /new-/clear-style confirmation panel."""
        from cli import _panel_box_width, _wrap_panel_text_keep_ws
        state = self._slash_confirm_state
        if not state:
            return []
        wrap = _wrap_panel_text_keep_ws
        title = state.get("title") or "Confirm action"
        detail = state.get("detail") or ""
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        footer = "Type 1/2/3 or use ↑/↓ then Enter. ESC/Ctrl+C cancels."
        choice_labels = [
            f"{'❯' if idx == selected else ' '} [{idx + 1}] {label} — {desc}"
            for idx, (_value, label, desc) in enumerate(choices)]

        preview_lines = [w for line in detail.splitlines() for w in wrap(line, 72)]
        preview_lines.extend(w for _i, w in _wrap_rows(wrap, choice_labels, 72, "    "))
        preview_lines.append(footer)
        box_width = _panel_box_width(title, preview_lines, min_width=56, max_width=86)
        inner_text_width = max(8, box_width - 2)
        detail_wrapped = [w for line in detail.splitlines() for w in wrap(line, inner_text_width)]
        choice_wrapped = _wrap_rows(wrap, choice_labels, inner_text_width, "    ")

        chrome_full = 6
        available = max(0, _term_rows() - _PANEL_RESERVED_BELOW)
        max_detail_rows = min(8, max(1, available - chrome_full - len(choice_wrapped)))
        if len(detail_wrapped) > max_detail_rows:
            detail_wrapped = detail_wrapped[:max(1, max_detail_rows - 1)] + ["… (detail truncated)"]

        panel = _Panel('class:approval-border', box_width)
        panel.row('class:approval-title', title)
        panel.blank()
        for wrapped in detail_wrapped:
            panel.row('class:approval-desc', wrapped)
        panel.blank()
        for idx, wrapped in choice_wrapped:
            style = 'class:approval-selected' if idx == selected else 'class:approval-choice'
            panel.row(style, wrapped)
        panel.blank()
        panel.row('class:approval-cmd', footer)
        return panel.close()

    def _get_approval_display_fragments(self):
        """Render the dangerous-command approval panel.

        Layout priority: title + command + choices must always render, even in a short terminal
        or with a long (tirith multi-paragraph) description. The description sits at the bottom
        and is truncated to the remaining row budget, so HSplit never clips approve/deny off-screen.
        """
        from cli import _panel_box_width, _wrap_panel_text_keep_ws
        state = self._approval_state
        if not state:
            return []
        wrap = _wrap_panel_text_keep_ws
        command = state["command"]
        description = state["description"]
        choices = state["choices"]
        selected = state.get("selected", 0)
        show_full = state.get("show_full", False)
        title = "⚠️  Dangerous Command"

        preview_lines = wrap(description, 60)
        preview_lines.extend(wrap(command, 60))
        for i, choice in enumerate(choices):
            prefix = '❯ ' if i == selected else '  '
            label = _APPROVAL_CHOICE_LABELS.get(choice, choice)
            preview_lines.extend(wrap(f"{prefix}{label}", 60, subsequent_indent="  "))
        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)

        # Pre-wrap the mandatory content — command + choices must always render.
        cmd_wrapped = wrap(command, inner_text_width)
        if not show_full and "view" in choices and len(cmd_wrapped) > 4:
            cmd_wrapped = cmd_wrapped[:3] + wrap("… (choose Show full command)", inner_text_width)
        choice_labels = [
            f"{'❯' if i == selected else ' '} {_num_prefix(i)}. {_APPROVAL_CHOICE_LABELS.get(choice, choice)}"
            for i, choice in enumerate(choices)]
        choice_wrapped = _wrap_rows(wrap, choice_labels, inner_text_width, "    ")

        # Row budget so HSplit never clips the command or choices. Full chrome = top border +
        # title + blank + blank-between-cmd/choices + bottom border (5); when that doesn't fit,
        # drop the separator blanks (3) so every choice stays on-screen in compact terminals.
        available = max(0, _term_rows() - _PANEL_RESERVED_BELOW)
        use_compact_chrome = 5 + len(cmd_wrapped) + len(choice_wrapped) > available
        chrome_rows = 3 if use_compact_chrome else 5

        # A command too long to leave room for the choices (e.g. "view" on a multi-hundred-char
        # command) is truncated so approve/deny still render; keep at least 1 command row.
        max_cmd_rows = max(1, available - chrome_rows - len(choice_wrapped))
        if len(cmd_wrapped) > max_cmd_rows:
            keep = max(1, max_cmd_rows - 1) if max_cmd_rows > 1 else 1
            cmd_wrapped = cmd_wrapped[:keep] + wrap(
                "… (command truncated — use /logs or /debug for full text)", inner_text_width)

        # Remaining rows go to the description (minus the blank separator in full mode), capped
        # at 10 so the panel stays compact even on huge terminals.
        mandatory_no_desc = chrome_rows + len(cmd_wrapped) + len(choice_wrapped)
        available_for_desc = available - mandatory_no_desc - (0 if use_compact_chrome else 1)
        available_for_desc = max(0, min(available_for_desc, 10))
        desc_wrapped = wrap(description, inner_text_width) if description else []
        if available_for_desc < 1 or not desc_wrapped:
            desc_wrapped = []
        elif len(desc_wrapped) > available_for_desc:
            desc_wrapped = desc_wrapped[:max(1, available_for_desc - 1)] + ["… (description truncated)"]

        # Render title → command → choices → description; description last so any overflow
        # clips the least-critical content, never the command or choices.
        panel = _Panel('class:approval-border', box_width)
        panel.row('class:approval-title', title)
        if not use_compact_chrome:
            panel.blank()
        for wrapped in cmd_wrapped:
            panel.row('class:approval-cmd', wrapped)
        if not use_compact_chrome:
            panel.blank()
        for i, wrapped in choice_wrapped:
            style = 'class:approval-selected' if i == selected else 'class:approval-choice'
            panel.row(style, wrapped)
        if desc_wrapped:
            if not use_compact_chrome:
                panel.blank()
            for wrapped in desc_wrapped:
                panel.row('class:approval-desc', wrapped)
        return panel.close()

    def _get_tui_prompt_symbols(self) -> tuple[str, str]:
        """Return ``(normal_prompt, state_suffix)`` for the active skin.

        ``state_suffix`` is what special states (sudo/secret/approval/agent) render after their
        leading icon. A non-default profile name is prepended (``coder ❯``).
        """
        try:
            from hermes_cli.skin_engine import get_active_prompt_symbol
            symbol = get_active_prompt_symbol("❯ ")
        except Exception:
            symbol = "❯ "
        symbol = (symbol or "❯ ").rstrip() + " "
        try:
            from hermes_cli.profiles import get_active_profile_name
            profile = get_active_profile_name()
            if profile not in {"default", "custom"}:
                symbol = f"{profile} {symbol}"
        except Exception:
            pass
        stripped = symbol.rstrip()
        if not stripped:
            return "❯ ", "❯ "
        parts = stripped.split()
        candidate = parts[-1] if parts else ""
        if any(ch in candidate for ch in ("❯", ">", "$", "#", "›", "»", "→")):
            return symbol, candidate.rstrip() + " "
        # Icon-only custom prompts should still remain visible in special states.
        return symbol, symbol

    def _audio_level_bar(self) -> str:
        """One-char audio level indicator from the recorder's current RMS."""
        rec = getattr(self, "_voice_recorder", None)
        if rec is None:
            return ""
        # RMS 0-32767 → index 0-7; typical speech is 500-5000, display caps at ~8000.
        return " ▁▂▃▄▅▆▇"[min(rec.current_rms, 8000) * 7 // 8000]

    def _get_tui_prompt_fragments(self):
        """prompt_toolkit fragments for the current interactive state."""
        symbol, state_suffix = self._get_tui_prompt_symbols()
        compact = self._use_minimal_tui_chrome(width=self._get_tui_terminal_width())

        def _state_fragment(style: str, icon: str, extra: str = ""):
            if compact:
                text = icon
                if extra:
                    text = f"{text} {extra.strip()}".rstrip()
                return [(style, text + " ")]
            if extra:
                return [(style, f"{icon} {extra} {state_suffix}")]
            return [(style, f"{icon} {state_suffix}")]

        if self._voice_recording:
            return _state_fragment("class:voice-recording", "●", self._audio_level_bar())
        if self._voice_processing:
            return _state_fragment("class:voice-processing", "◉")
        if self._sudo_state:
            return _state_fragment("class:sudo-prompt", "🔐")
        if self._secret_state:
            return _state_fragment("class:sudo-prompt", "🔑")
        if self._approval_state or getattr(self, "_slash_confirm_state", None):
            return _state_fragment("class:prompt-working", "⚠")
        if self._clarify_freetext:
            return _state_fragment("class:clarify-selected", "✎")
        if self._clarify_state:
            return _state_fragment("class:prompt-working", "?")
        if self._command_running:
            return _state_fragment("class:prompt-working", self._command_spinner_frame())
        if self._agent_running:
            return _state_fragment("class:prompt-working", "⚕")
        if self._voice_mode:
            return _state_fragment("class:voice-prompt", "🎤")
        return [("class:prompt", symbol)]

    def _get_tui_prompt_text(self) -> str:
        """Visible prompt text for width calculations."""
        return "".join(text for _, text in self._get_tui_prompt_fragments())

    def _build_tui_style_dict(self) -> dict[str, str]:
        """Layer the active skin's prompt_toolkit colors over the base TUI style.

        On a light terminal, hex tokens in each style string are rewritten through the light-mode
        remap so the chrome stays readable on cream Terminal.app backgrounds. CRITICAL: a style
        that paints its own ``bg:`` (status bar, completion menu) is left alone — its fg was tuned
        for that dark bg and remapping would give dark-on-dark; the terminal's mode is irrelevant.
        """
        from cli import _detect_light_mode, _maybe_remap_for_light_mode
        style_dict = dict(getattr(self, "_tui_style_base", {}) or {})
        try:
            from hermes_cli.skin_engine import get_prompt_toolkit_style_overrides
            style_dict.update(get_prompt_toolkit_style_overrides())
        except Exception:
            pass
        try:
            if _detect_light_mode():
                def _remap_value(v: str) -> str:
                    if not v:
                        return v
                    tokens = v.split()
                    if any(t.startswith("bg:") for t in tokens):
                        return v
                    return " ".join(_maybe_remap_for_light_mode(t) if t.startswith("#") else t for t in tokens)
                style_dict = {k: _remap_value(v or "") for k, v in style_dict.items()}
        except Exception:
            pass
        return style_dict

    def _apply_tui_skin_style(self) -> bool:
        """Refresh prompt_toolkit styling for a running interactive TUI."""
        if not getattr(self, "_app", None) or not getattr(self, "_tui_style_base", None):
            return False
        self._app.style = PTStyle.from_dict(self._build_tui_style_dict())
        self._invalidate(min_interval=0.0)
        return True

    def _get_extra_tui_widgets(self) -> list:
        """Extension hook: wrapper CLIs return widgets inserted between the spacer and status bar."""
        return []

    def _register_extra_tui_keybindings(self, kb, *, input_area) -> None:
        """Extension hook: wrapper CLIs add bindings to ``kb`` (``input_area`` is the main TextArea)."""

    def _build_tui_layout_children(
        self,
        *,
        sudo_widget,
        secret_widget,
        approval_widget,
        slash_confirm_widget=None,
        clarify_widget,
        model_picker_widget=None,
        command_palette_widget=None,
        spinner_widget=None,
        spacer,
        status_bar,
        input_rule_top,
        image_bar,
        input_area,
        input_rule_bot,
        voice_status_bar,
        completions_menu) -> list:
        """Ordered children of the root ``HSplit``; override only for full control over ordering
        (wrappers normally override ``_get_extra_tui_widgets`` instead)."""
        ordered = [
            Window(height=0),
            sudo_widget,
            secret_widget,
            approval_widget,
            slash_confirm_widget,
            clarify_widget,
            model_picker_widget,
            command_palette_widget,
            spinner_widget,
            spacer,
            *self._get_extra_tui_widgets(),
            getattr(self, "_pet_widget", None),
            getattr(self, "_stash_panel_widget", None),
            status_bar,
            input_rule_top,
            image_bar,
            input_area,
            input_rule_bot,
            voice_status_bar,
            completions_menu]
        return [item for item in ordered if item is not None]

    def _tui_spinner_loop(self):
        while not self._should_exit:
            if not self._app:
                time.sleep(0.1)
                continue
            if self._command_running:
                self._invalidate(min_interval=0.1)
                time.sleep(0.1)
            else:
                # Never repaint the idle prompt on a timer: in non-full-screen mode background
                # redraws fight tmux/Ghostty/cmux viewport restoration after focus changes and
                # visually move the input area. Input/agent events invalidate explicitly.
                time.sleep(0.2)

    def _get_clarify_batch_display_fragments(self, state):
        """Batch (multi-question) clarify panel: "N questions" header, one status line per question
        (✓ answered → answer / ▸ active / · pending), and the active question's numbered choices
        (+ Other) expanded beneath its status line."""
        from cli import _panel_box_width, _wrap_panel_text
        questions_list = state.get("questions") or []
        answers = state.get("answers") or {}
        answer_meta = state.get("answer_meta") or {}
        active = state.get("active", 0)
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        multi_select = state.get("multi_select", False)
        selected_indices = state.get("selected_indices", set()) if multi_select else set()
        freetext = self._clarify_freetext
        title = "Hermes needs your input"
        header = f"{len(questions_list)} questions"

        def _status_rows(width):
            rows = []
            for idx, entry in enumerate(questions_list):
                answered = entry["qid"] in answers
                marker = "✓" if answered else ("▸" if idx == active else "·")
                row_style = 'class:clarify-selected' if idx == active else 'class:clarify-choice'
                for wrapped in _wrap_panel_text(f"{marker} {entry['question']}", width, subsequent_indent="  "):
                    rows.append((row_style, wrapped))
                if answered:
                    # Locked answer on its own line/color so it stays readable while Tab-walking.
                    answer = f"    {answers[entry['qid']]}"
                    for wrapped in _wrap_panel_text(answer, width, subsequent_indent="    "):
                        rows.append(('class:clarify-answer', wrapped))
                if idx != active:
                    continue
                for i, choice in enumerate(choices):
                    cursor = "❯" if i == selected and not freetext else " "
                    cb = ("[x] " if i in selected_indices else "[ ] ") if multi_select else ""
                    style = 'class:clarify-selected' if i == selected and not freetext else 'class:clarify-choice'
                    label = f"  {cursor} {cb}{_num_prefix(i)}. {choice}"
                    for wrapped in _wrap_panel_text(label, width, subsequent_indent="      "):
                        rows.append((style, wrapped))
                if choices:
                    other_idx = len(choices)
                    mid = _num_prefix(other_idx)
                    if multi_select:
                        mid = f"{'[x]' if other_idx in selected_indices else '[ ]'} {mid}"
                    # An earlier typed answer stays visible next to Other; Enter on it edits
                    # (the composer is prefilled).
                    other_text = (answer_meta.get(entry["qid"]) or {}).get("other_text") or ""
                    other_suffix = f"Other: {other_text}" if other_text else None
                    if freetext:
                        other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type below)")
                        other_style = 'class:clarify-active-other'
                    elif selected == other_idx:
                        other_label = f"  ❯ {mid}. " + (other_suffix or "Other (type your answer)")
                        other_style = 'class:clarify-selected'
                    else:
                        other_label = f"    {mid}. " + (other_suffix or "Other (type your answer)")
                        other_style = 'class:clarify-choice'
                    for wrapped in _wrap_panel_text(other_label, width, subsequent_indent="      "):
                        rows.append((other_style, wrapped))
                elif freetext:
                    guidance = "  Type your answer in the prompt below, then press Enter."
                    for wrapped in _wrap_panel_text(guidance, width):
                        rows.append(('class:clarify-active-other', wrapped))
            return rows

        preview_rows = _status_rows(60)
        box_width = _panel_box_width(title, [header] + [text for _, text in preview_rows])
        rows = _status_rows(max(8, box_width - 2))

        panel = _Panel('class:clarify-border', box_width, title, 'class:clarify-title')
        panel.row('class:clarify-question', header)
        for style, text in rows:
            panel.row(style, text)
        return panel.close()

    def _get_clarify_display_fragments(self):
        """Clarify question/choices panel.

        Layout priority: choices + the Other option must always render even for a very long
        question; the question is budgeted to the rows left over and truncated with a marker.
        """
        from cli import _panel_box_width, _wrap_panel_text
        state = self._clarify_state
        if not state:
            return []
        if state.get("questions"):
            return self._get_clarify_batch_display_fragments(state)
        wrap = _wrap_panel_text
        question = state["question"]
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        multi_select = state.get("multi_select", False)
        selected_indices = state.get("selected_indices", set()) if multi_select else set()
        freetext = self._clarify_freetext
        title = "Hermes needs your input"
        other_idx = len(choices)

        def _label(i, text):
            cursor = "❯" if (i == selected and not freetext) or (freetext and i == other_idx) else " "
            cb = ("[x] " if i in selected_indices else "[ ] ") if multi_select else ""
            return f"{cursor} {cb}{_num_prefix(i)}. {text}"

        choice_labels = [_label(i, c) for i, c in enumerate(choices)]
        other_label = _label(other_idx, "Other (type below)" if freetext else "Other (type your answer)")

        preview_lines = wrap(question, 60)
        preview_lines.extend(w for _i, w in _wrap_rows(wrap, choice_labels + [other_label], 60, "    "))
        box_width = _panel_box_width(title, preview_lines)
        inner_text_width = max(8, box_width - 2)

        # Mandatory rows: choices + Other (or the freetext guidance line when there are no choices).
        choice_wrapped = _wrap_rows(wrap, choice_labels, inner_text_width, "    ")
        if choices:
            other_wrapped = wrap(other_label, inner_text_width, subsequent_indent="    ")
        elif freetext:
            other_wrapped = wrap("Type your answer in the prompt below, then press Enter.", inner_text_width)
        else:
            other_wrapped = []

        # Row budget so the mandatory rows always render. Full chrome = top border + blank after
        # title + blank after question + blank before bottom + bottom border (5); tight = the two
        # borders (2). The compact decision reserves 1 question row on top of the choices —
        # otherwise full chrome is kept when there is no room for it, the panel overflows and
        # HSplit silently clips the choices.
        available = max(0, _term_rows() - _PANEL_RESERVED_BELOW)
        mandatory = len(choice_wrapped) + len(other_wrapped)
        use_compact_chrome = 5 + 1 + mandatory > available
        chrome_rows = 2 if use_compact_chrome else 5
        max_question_rows = min(12, max(1, available - chrome_rows - mandatory))  # soft cap on huge terminals
        # When the choices alone (plus compact chrome) fill the viewport, drop the question
        # entirely — the choices are all the user needs to select; the 1-row floor above would
        # push the tail of the choices off-screen.
        if chrome_rows + mandatory >= available:
            max_question_rows = 0
        question_wrapped = wrap(question, inner_text_width)
        if max_question_rows <= 0:
            question_wrapped = []
        elif len(question_wrapped) > max_question_rows:
            # The marker is itself a row: with a 1-row budget show the marker alone so the
            # rendered question never exceeds max_question_rows.
            question_wrapped = question_wrapped[:max(0, max_question_rows - 1)] + ["… (question truncated)"]

        panel = _Panel('class:clarify-border', box_width, title, 'class:clarify-title')
        if not use_compact_chrome:
            panel.blank()
        for wrapped in question_wrapped:
            panel.row('class:clarify-question', wrapped)
        if not use_compact_chrome:
            panel.blank()
        if freetext and not choices:
            for wrapped in other_wrapped:
                panel.row('class:clarify-choice', wrapped)
            if not use_compact_chrome:
                panel.blank()
        if choices:
            for i, wrapped in choice_wrapped:
                style = 'class:clarify-selected' if i == selected and not freetext else 'class:clarify-choice'
                panel.row(style, wrapped)
            if selected == other_idx and not freetext:
                other_style = 'class:clarify-selected'
            elif freetext:
                other_style = 'class:clarify-active-other'
            else:
                other_style = 'class:clarify-choice'
            for wrapped in other_wrapped:
                panel.row(other_style, wrapped)
        if not use_compact_chrome:
            panel.blank()
        return panel.close()

    def _render_scroll_list_panel(self, state, title, hint, labels, *, min_width, max_width, indent):
        """Titled panel with a hint row and a scrolling selectable list (model picker, palette).

        The panel renders into a Window with no max height, so the visible slice is limited to
        the terminal rows or the bottom border and trailing items get clipped on long lists
        (e.g. Ollama Cloud's 36+ models). ``state["_scroll_offset"]`` is updated in place.
        """
        from cli import HermesCLI, _panel_box_width, _wrap_panel_text
        box_width = _panel_box_width(title, [hint] + labels, min_width=min_width, max_width=max_width)
        inner_text_width = max(8, box_width - 6)
        selected = state.get("selected", 0)
        try:
            from prompt_toolkit.application import get_app
            term_rows = get_app().output.get_size().rows
        except Exception:
            term_rows = _term_rows()
        scroll_offset, visible = HermesCLI._compute_model_picker_viewport(
            selected, state.get("_scroll_offset", 0), len(labels), term_rows)
        state["_scroll_offset"] = scroll_offset

        panel = _Panel('class:clarify-border', box_width, title, 'class:clarify-title')
        panel.blank()
        panel.row('class:clarify-hint', hint)
        panel.blank()
        for idx in range(scroll_offset, min(scroll_offset + visible, len(labels))):
            style = 'class:clarify-selected' if idx == selected else 'class:clarify-choice'
            prefix = '❯ ' if idx == selected else '  '
            for wrapped in _wrap_panel_text(prefix + labels[idx], inner_text_width, subsequent_indent=indent):
                panel.row(style, wrapped)
        panel.blank()
        return panel.close()

    def _get_model_picker_display_fragments(self):
        state = self._model_picker_state
        if not state:
            return []
        if state.get("stage", "provider") == "provider":
            title = "⚙ Model Picker — Select Provider"
            choices = []
            _providers = state.get("providers")
            for p in _providers if isinstance(_providers, list) else []:
                count = p.get("total_models", len(p.get("models", [])))
                label = f"{p['name']} ({count} model{'s' if count != 1 else ''})"
                if p.get("is_current"):
                    label += "  ← current"
                choices.append(label)
            choices.append("Cancel")
            hint = (
                f"Current: {state.get('current_model', 'unknown')} "
                f"on {state.get('current_provider', 'unknown')}")
        else:
            provider_data = state.get("provider_data") or {}
            model_list = state.get("model_list") or []
            title = f"⚙ Model Picker — {provider_data.get('name', provider_data.get('slug', 'Provider'))}"
            # Fuzzy filter narrows the concrete list; selection still resolves to a real entry via
            # the filtered_pairs index mapping, so this never makes model resolution ambiguous.
            _query = state.get("filter", "") or ""
            filtered_pairs = self._filter_model_picker_entries(model_list, _query)
            state["_filtered_pairs"] = filtered_pairs
            model_labels = [e for (_i, e) in filtered_pairs]
            choices = list(model_labels) + ["← Back", "Cancel"]
            if _query:
                hint = (
                    f"Filter: {_query}▏  ({len(model_labels)}/{len(model_list)} match "
                    "— type to narrow, Backspace to clear)")
            elif model_list:
                hint = f"Select a model ({len(model_list)} available) — type to filter"
            else:
                hint = "No models listed for this provider. Use Back or Cancel."
        return self._render_scroll_list_panel(
            state, title, hint, choices, min_width=46, max_width=84, indent='  ')

    def _get_command_palette_display_fragments(self):
        state = self._command_palette_state
        if not state:
            return []
        rows = self._command_palette_visible_entries()
        state["_visible_count"] = len(rows)
        _query = state.get("filter", "") or ""
        total = len(state.get("entries") or [])
        if _query:
            hint = f"Filter: {_query}▏  ({len(rows)}/{total} match — Enter inserts, Esc cancels)"
        else:
            hint = f"Type to filter {total} commands — ↑/↓ then Enter inserts, Esc cancels"
        labels = [f"{c}  —  {d}" if d else c for (c, _cat, d) in rows] or ["(no matching commands)"]
        return self._render_scroll_list_panel(
            state, "⚙ Command Palette", hint, labels, min_width=50, max_width=90, indent='    ')

    def _render_sudo_style_panel(self, title: str, body_lines: list[str]):
        """Bordered ``sudo-*`` panel: blank, each body line, blank, body-final line, blank."""
        from cli import _panel_box_width
        box_width = _panel_box_width(title, body_lines)
        panel = _Panel('class:sudo-border', box_width, title, 'class:sudo-title')
        panel.blank()
        for i, text in enumerate(body_lines):
            if i == len(body_lines) - 1 and i > 0:
                panel.blank()
            panel.row('class:sudo-text', text)
        panel.blank()
        return panel.close()

    def _get_sudo_display_fragments(self):
        if not self._sudo_state:
            return []
        return self._render_sudo_style_panel(
            '🔐 Sudo Password Required', ['Enter password below (hidden), or press Enter to skip'])

    def _get_secret_display_fragments(self):
        state = self._secret_state
        if not state:
            return []
        prompt = state.get("prompt") or f"Enter value for {state.get('var_name', 'secret')}"
        help_text = (state.get("metadata") or {}).get("help")
        content_lines = [prompt, 'Enter secret below (hidden), ESC or Ctrl+C to skip']
        if help_text:
            content_lines.insert(1, str(help_text))
        return self._render_sudo_style_panel('🔑 Skill Setup Required', content_lines)

    # (state attr, deadline attr, hint) for the modal prompts with a countdown hint row.
    _TUI_MODAL_HINTS = (
        ("_sudo_state", "_sudo_deadline", '  password hidden · Enter to skip'),
        ("_secret_state", "_secret_deadline", '  secret hidden · Enter to skip'),
        ("_approval_state", "_approval_deadline", '  ↑/↓ to select, Enter to confirm'),
        ("_slash_confirm_state", "_slash_confirm_deadline", '  type 1/2/3, or ↑/↓ to select, Enter to confirm'),
    )

    def _tui_hint_text(self):
        for state_attr, deadline_attr, hint in self._TUI_MODAL_HINTS:
            if getattr(self, state_attr):
                remaining = max(0, int(getattr(self, deadline_attr) - time.monotonic()))
                return [('class:hint', hint), ('class:clarify-countdown', f'  ({remaining}s)')]
        if self._clarify_state:
            # None deadline = unlimited wait → hide the countdown entirely.
            if self._clarify_deadline is None:
                countdown = ''
            else:
                countdown = f'  ({max(0, int(self._clarify_deadline - time.monotonic()))}s)'
            if self._clarify_freetext:
                hint = '  type your answer and press Enter'
            elif self._clarify_state.get("questions"):
                hint = '  ↑/↓ to select, Enter to lock, Tab next question'
            else:
                hint = '  ↑/↓ to select, Enter to confirm'
            return [('class:hint', hint), ('class:clarify-countdown', countdown)]
        if self._command_running:
            frame = self._command_spinner_frame()
            if self._command_blocks_input:
                detail = "input temporarily disabled"
            else:
                detail = "input stays active; Enter queues"
            return [('class:hint', f'  {frame} command in progress · {detail}')]
        return []

    def _tui_placeholder_text(self):
        if self._voice_recording:
            return f"recording... {self._voice_record_key_label()} to stop, Ctrl+C to cancel"
        if self._voice_processing:
            return "transcribing..."
        if self._sudo_state:
            return "type password (hidden), Enter to submit · ESC to skip"
        if self._secret_state:
            return "type secret (hidden), Enter to submit · ESC to skip"
        if self._approval_state:
            return ""
        if self._slash_confirm_state:
            return "type 1/2/3, or use ↑/↓ then Enter"
        if self._clarify_freetext:
            return "type your answer here and press Enter"
        if self._clarify_state:
            return ""
        if self._command_running:
            return f"{self._command_spinner_frame()} {self._command_status or 'Processing command...'}"
        if self._agent_running:
            return "msg=interrupt · /queue · /bg · /steer · Ctrl+C cancel"
        if self._voice_mode:
            return f"type or {self._voice_record_key_label()} to record"
        # Advertise a parked draft so the stash can never be silently forgotten.
        try:
            _stash_hint = self._prompt_stash.placeholder_hint()
        except Exception:
            _stash_hint = ""
        if _stash_hint:
            return _stash_hint
        # Idle + empty composer: a task-oriented example chosen once per session
        # (self._composer_placeholder) so it stays stable while being read, not flickering.
        return getattr(self, "_composer_placeholder", "") or ""

    def _get_stash_panel_display_fragments(self):
        try:
            _stash = self._prompt_stash
            return self._render_stash_panel(
                _stash.panel_rows(), _stash.panel_cursor, self._get_tui_terminal_width())
        except Exception:
            return []

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

    def _tui_wake_startup(self):
        from cli import logger
        try:
            self._maybe_start_wake_word()
        except Exception as e:
            logger.debug("wake-word startup skipped: %s", e)

    def _tui_suppress_closed_loop_errors(self, loop, context):
        exc = context.get("exception")
        if isinstance(exc, RuntimeError) and "Event loop is closed" in str(exc):
            return
        if isinstance(exc, KeyError) and "is not registered" in str(exc):
            return  # selector registration failures (#6393)
        if isinstance(exc, OSError) and getattr(exc, "errno", None) == errno.EIO:
            return  # broken stdout on interrupt (#13710)
        loop.default_exception_handler(context)

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
            self._submit_secret_response(buf.text)
            buf.reset()
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

    def _tui_image_bar_fragments(self):
        from cli import _format_image_attachment_badges
        if not self._attached_images:
            return []
        badges = _format_image_attachment_badges(self._attached_images, self._image_counter)
        return [("class:image-badge", f" {badges} ")]

    def _tui_voice_status_fragments(self):
        return self._get_voice_status_fragments()

    def _tui_spinner_text(self):
        spinner_line = self._render_spinner_text()
        return [('class:hint', spinner_line)] if spinner_line else []

    def _tui_spinner_height(self):
        return self._spinner_widget_height()

    def _tui_hint_height(self):
        if (
            self._sudo_state or self._secret_state or self._approval_state
            or self._slash_confirm_state or self._clarify_state or self._command_running):
            return 1
        # Keep a spacer while the agent runs on roomy terminals; reclaim the row on narrow screens.
        return self._agent_spacer_height()

    def _tui_init_run_state(self):
        """Reset the per-run REPL state (queues, modal states, voice state, config watcher)."""
        self._agent_running = False
        self._pending_input = queue.Queue()     # normal input (commands + new queries)
        self._interrupt_queue = queue.Queue()   # messages typed while the agent is running
        # Seeded -q handoff: main() can't put directly into _pending_input (this reinit would
        # discard it), so the seeded first message rides in on an attribute and is enqueued here.
        _seed_msg = getattr(self, "_seeded_first_message", None)
        if _seed_msg is not None:
            self._seeded_first_message = None
            self._pending_input.put(_seed_msg)
        # See constructor note; mirrored for the run() path that skips the earlier __init__ branch.
        self._last_turn_interrupted = False
        self._should_exit = False
        self._last_ctrl_c_time = 0  # double Ctrl+C force-exit tracking

        # Plugins get a CLI reference so they can inject messages.
        from hermes_cli.plugins import get_plugin_manager
        get_plugin_manager()._cli_ref = self

        # Config file watcher — detect mcp_servers changes and auto-reload.
        from hermes_cli.config import get_config_path as _get_config_path
        _cfg_path = _get_config_path()
        self._config_mtime: float = _cfg_path.stat().st_mtime if _cfg_path.exists() else 0.0
        self._config_mcp_servers: dict = self.config.get("mcp_servers") or {}
        self._last_config_check: float = 0.0  # monotonic time of last check

        # Modal overlay states: each is a dict (with a response_queue) while active, else None.
        # The prompt_toolkit UI switches to the matching selection/input mode.
        self._clarify_state = None
        self._clarify_freetext = False  # True when the user chose "Other" and is typing
        self._clarify_deadline = 0      # monotonic timeout
        self._sudo_state = None
        self._sudo_deadline = 0
        self._modal_input_snapshot = None
        self._approval_state = None
        self._approval_deadline = 0
        self._approval_lock = threading.Lock()  # serialize concurrent approval prompts (delegation race)
        # Destructive slash-command confirmations (/new, /clear, /undo) are answered through the
        # composer, not raw input(), so the labels stay visible and Enter can't EOF the app.
        self._slash_confirm_state = None
        self._slash_confirm_deadline = 0
        self._command_running = False
        self._command_blocks_input = False
        self._command_status = ""
        self._secret_state = None       # skill-setup secret capture
        self._secret_deadline = 0

        self._attached_images: list[Path] = []  # clipboard image attachments
        self._image_counter = 0

        # Voice mode state (protected by _voice_lock for cross-thread access).
        self._voice_lock = threading.Lock()
        self._voice_mode = False
        self._voice_tts = False
        self._voice_recorder = None     # AudioRecorder (lazy init)
        self._voice_recording = False
        self._voice_processing = False  # STT in progress
        self._voice_continuous = False  # auto-restart after the agent responds
        self._voice_tts_done = threading.Event()  # TTS playback finished
        self._voice_tts_done.set()  # initially "done" (no TTS pending)
        self._voice_tts_stop = None  # active streaming pipeline's stop event
        self._voice_barge_capture = threading.Event()  # barge monitor is capturing the interruption
        self._voice_last_tts_text = ""  # most recently spoken TTS text (echo guard, #75780)
        self._voice_barge_phase = None  # "generation" or "playback" phase of the last barge trip

        if os.environ.get("HERMES_DEFER_AGENT_STARTUP") != "1":
            self._install_tool_callbacks()
            self._ensure_tirith_security()

    def _tui_build_key_bindings(self):
        """Build the prompt_toolkit KeyBindings for the REPL input area.

        Registration ORDER matters: for the same key, prompt_toolkit picks the last matching
        binding, so the generic handlers (Tab, history Up/Down) are registered before or after
        their filtered modal overrides deliberately.
        """
        from cli import (
            CLI_CONFIG,
            _bind_prompt_submit_keys,
            _cli_multiline_shortcuts_enabled,
            _preserve_ctrl_enter_newline)
        from prompt_toolkit.keys import Keys
        kb = KeyBindings()
        _multiline_shortcuts_enabled = _cli_multiline_shortcuts_enabled(self.config or CLI_CONFIG)
        self._tui_multiline_shortcuts = _multiline_shortcuts_enabled

        kb.add(Keys.Ignore, eager=True)(self._tui_handle_ignored_terminal_sequence)
        _bind_prompt_submit_keys(
            kb, self._tui_handle_enter, multiline_shortcuts_enabled=_multiline_shortcuts_enabled)
        kb.add('escape', 'enter')(self._tui_insert_newline)
        # Ctrl+J inserts a newline (Claude Code / Codex / OpenCode). Windows Terminal delivers
        # Ctrl+Enter as the same c-j code. display.cli_multiline_shortcuts: false restores legacy
        # c-j submit on unusual POSIX PTYs where Enter is LF.
        if _multiline_shortcuts_enabled or _preserve_ctrl_enter_newline():
            kb.add('c-j')(self._tui_insert_newline)

        self._tui_bind_editor_and_stash(kb)
        self._tui_bind_overlay_navigation(kb)

        # History: the TextArea is multiline so Up/Down alone only move the cursor;
        # Buffer.auto_up/auto_down browse history when on the first/last line.
        _normal_input = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._slash_confirm_state
            and not self._sudo_state and not self._secret_state and not self._model_picker_state
            and not self._command_palette_state)
        kb.add('up', filter=_normal_input)(self._tui_history_up)
        kb.add('down', filter=_normal_input)(self._tui_history_down)
        kb.add('c-l')(self._tui_handle_ctrl_l)
        kb.add('c-c')(self._tui_handle_ctrl_c)
        # No Ctrl+Shift+C binding: terminal emulators intercept it before stdin, and
        # prompt_toolkit's key parser doesn't recognise 'c-S-c' anyway (#19884/#19895).
        kb.add('c-q')(self._tui_handle_ctrl_q)
        kb.add('c-d')(self._tui_handle_ctrl_d)
        _modal_prompt_active = Condition(
            lambda: bool(self._secret_state or self._sudo_state or self._slash_confirm_state))
        kb.add('escape', filter=_modal_prompt_active, eager=True)(self._tui_handle_escape_modal)
        kb.add('escape', 'escape', filter=~_modal_prompt_active)(self._tui_handle_double_escape)
        kb.add('c-z')(self._tui_handle_ctrl_z)

        kb.add(*self._tui_voice_record_key_sequence())(self._tui_handle_voice_record)
        kb.add(Keys.BracketedPaste, eager=True)(self._tui_handle_paste)
        kb.add('c-v')(self._tui_handle_ctrl_v)
        kb.add('escape', 'v')(self._tui_handle_alt_v)
        return kb

    def _tui_bind_editor_and_stash(self, kb) -> None:
        # VSCode/Cursor bind Ctrl+G to "Find Next" so it never reaches the terminal; Alt+G is
        # unbound there and arrives as ('escape', 'g') — register it as a fallback.
        _editor_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state
            and not self._sudo_state and not self._secret_state)
        kb.add('c-g', filter=_editor_filter)(
            kb.add('escape', 'g', filter=_editor_filter)(self._tui_handle_open_in_editor))
        # Ctrl+S prompt stash: park a draft, send something else, bring it back. Suppressed while
        # a modal prompt owns the composer so Ctrl+S can't stash a password.
        _stash_filter = Condition(
            lambda: not self._clarify_state and not self._approval_state and not self._sudo_state
            and not self._secret_state and not self._slash_confirm_state and not self._model_picker_state
        )
        _stash_panel_filter = Condition(lambda: self._prompt_stash.panel_open and bool(len(self._prompt_stash)))
        kb.add('c-s', filter=_stash_filter)(self._tui_handle_prompt_stash)
        kb.add('up', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_up)
        kb.add('down', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_down)
        kb.add('enter', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_restore)
        kb.add('d', filter=_stash_panel_filter, eager=True)(
            kb.add('D', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_delete)
        )
        kb.add('escape', filter=_stash_panel_filter, eager=True)(self._tui_handle_stash_panel_close)
        kb.add('tab', eager=True)(self._tui_handle_tab)

    def _tui_bind_overlay_navigation(self, kb) -> None:
        """Clarify / approval / slash-confirm / model picker / command palette navigation keys."""
        _clarify_nav = Condition(lambda: bool(self._clarify_state) and not self._clarify_freetext)
        _clarify_batch = Condition(
            lambda: bool(self._clarify_state) and bool(self._clarify_state.get("questions"))
            and not self._clarify_freetext)
        kb.add('up', filter=_clarify_nav)(self._tui_clarify_up)
        kb.add('down', filter=_clarify_nav)(self._tui_clarify_down)
        # Multi-select: Space toggles the checkbox under the cursor.
        kb.add('space', filter=Condition(
            lambda: bool(self._clarify_state) and not self._clarify_freetext
            and self._clarify_state.get("multi_select")
        ))(self._tui_clarify_toggle)
        # Batch clarify: Tab / Shift-Tab cycle the active question (any-order answering; moving
        # onto an answered question lets the user re-answer it). Registered after the generic
        # tab handler so this filtered binding wins while the batch panel is open.
        kb.add('tab', filter=_clarify_batch, eager=True)(self._tui_clarify_batch_tab)
        kb.add('s-tab', filter=_clarify_batch, eager=True)(self._tui_clarify_batch_backtab)
        # Number keys: 1-9 select items 0-8, 0 selects item 9 (10th).
        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_clarify_nav)(self._tui_make_clarify_number_handler(_idx))

        _approval = Condition(lambda: bool(self._approval_state))
        _slash_confirm = Condition(lambda: bool(self._slash_confirm_state))
        _picker = Condition(lambda: bool(self._model_picker_state))
        kb.add('up', filter=_approval)(self._tui_approval_up)
        kb.add('down', filter=_approval)(self._tui_approval_down)
        kb.add('up', filter=_slash_confirm)(self._tui_slash_confirm_up)
        kb.add('down', filter=_slash_confirm)(self._tui_slash_confirm_down)
        kb.add('up', filter=_picker)(self._tui_model_picker_up)
        kb.add('down', filter=_picker)(self._tui_model_picker_down)

        def _model_picker_typing_active() -> bool:
            # Type-to-filter is only live on the model stage (concrete list).
            st = self._model_picker_state
            return bool(st) and st.get("stage") == "model"

        _picker_typing = Condition(_model_picker_typing_active)
        for _ch in _TYPING_CHARS:
            kb.add(_ch, filter=_picker_typing)(self._tui_make_model_filter_char_handler(_ch))
        kb.add('backspace', filter=_picker_typing)(self._tui_model_picker_filter_backspace)
        kb.add('escape', filter=_picker, eager=True)(self._tui_model_picker_escape)

        _palette = Condition(lambda: bool(self._command_palette_state))
        kb.add('c-p', filter=Condition(
            lambda: not self._command_palette_state and not self._model_picker_state and not self._clarify_state
            and not self._approval_state and not self._slash_confirm_state and not self._sudo_state
            and not self._secret_state
        ))(self._tui_open_command_palette)
        kb.add('up', filter=_palette)(self._tui_command_palette_up)
        kb.add('down', filter=_palette)(self._tui_command_palette_down)
        kb.add('enter', filter=_palette)(self._tui_command_palette_enter)
        kb.add('backspace', filter=_palette)(self._tui_command_palette_backspace)
        kb.add('escape', filter=_palette, eager=True)(self._tui_command_palette_escape)
        for _pch in _TYPING_CHARS:
            kb.add(_pch, filter=_palette)(self._tui_make_palette_char_handler(_pch))

        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_approval)(self._tui_make_approval_number_handler(_idx))
        for _num in range(10):
            _idx = 9 if _num == 0 else _num - 1
            kb.add(str(_num), filter=_slash_confirm)(self._tui_make_slash_confirm_number_handler(_idx))

    def _tui_voice_record_key_sequence(self) -> tuple:
        """Resolve the push-to-talk key (voice.record_key, default Ctrl+B) to a prompt_toolkit
        key sequence and cache the UI label.

        Config spellings (ctrl/control/alt/option/opt) are normalized to c-x / a-x so the same
        value binds identically in TUI and CLI. super/win/windows silently fall back to the
        default (prompt_toolkit has no super modifier) — warn so users notice the split. The
        label cache uses the same ``_raw_key`` that drives the binding, so status/placeholder/
        recording-hint renders can never drift from the live key even if the config is edited
        mid-session.
        """
        from cli import logger
        # Voice push-to-talk key: configurable via config.yaml (voice.record_key) Default: Ctrl+B (avoids
        # conflict with Ctrl+R readline reverse-search). Config spellings (ctrl/control/alt/option/opt) are
        # normalized to prompt_toolkit's c-x / a-x format via
        # ``normalize_voice_record_key_for_prompt_toolkit`` so the same config value binds identically in
        # the TUI and CLI (Copilot round-9 review on #19835). ``super``/``win``/``windows`` configs silently
        # fall back to the default here since prompt_toolkit has no super modifier — log a warning so users
        # notice the TUI/CLI split instead of a silent mismatch (round-11).
        _raw_key: object = "ctrl+b"
        try:
            from hermes_cli.config import load_config
            from hermes_cli.voice import (
                normalize_voice_record_key_for_prompt_toolkit,
                pt_key_to_sequence,
                voice_record_key_from_config)
            _raw_key = voice_record_key_from_config(load_config())
            _voice_key = normalize_voice_record_key_for_prompt_toolkit(_raw_key)
            if (
                isinstance(_raw_key, str)
                and _raw_key.strip().lower().split("+", 1)[0].strip() in {"super", "win", "windows"}
                and _voice_key == "c-b"):
                logger.warning(
                    "voice.record_key %r uses a TUI-only modifier (super/win); "
                    "CLI fell back to Ctrl+B. Use ctrl+<key> or alt+<key> for "
                    "cross-runtime parity.",
                    _raw_key)
        except Exception:
            _voice_key = "c-b"
        # Cache the UI label here — same ``_raw_key`` that drives the prompt_toolkit binding below. Every
        # status / placeholder / recording-hint render reads this cached value so display can never drift
        # from the live keybinding even if the user edits voice.record_key mid-session (Copilot round-13 on
        # #19835).
        self.set_voice_record_key_cache(_raw_key)
        return pt_key_to_sequence(_voice_key)

    def _tui_overlay_widget(self, fragments_fn, state_attr: str):
        """Wrapped, auto-sized panel shown while ``self.<state_attr>`` is not None."""
        return ConditionalContainer(
            Window(FormattedTextControl(fragments_fn), wrap_lines=True),
            filter=Condition(lambda: getattr(self, state_attr) is not None))

    def _tui_build_layout(self, kb):
        """Build the TUI widgets, Layout and Style; registers wrapper keybindings on ``kb``."""
        cli_ref = self
        input_area = self._tui_build_input_area()
        spinner_widget = Window(
            content=FormattedTextControl(self._tui_spinner_text),
            height=self._tui_spinner_height,
            wrap_lines=True)
        # Petdex mascot — right-aligned Kitty placeholder or half-block sprite above the prompt;
        # height 0 when no pet is enabled. The animation thread queues virtual Kitty frames;
        # after_render writes them out-of-band while prompt_toolkit owns the placeholder grid.
        self._pet_widget = Window(
            content=FormattedTextControl(self._pet_fragments),
            height=self._pet_widget_height,
            align=WindowAlign.RIGHT)
        # Hint line above the input: only for interactive prompts that need extra instructions
        # (sudo countdown, approval navigation, clarify); the agent-running hint is the placeholder.
        spacer = Window(content=FormattedTextControl(self._tui_hint_text), height=self._tui_hint_height)
        clarify_widget = self._tui_overlay_widget(self._get_clarify_display_fragments, "_clarify_state")
        sudo_widget = self._tui_overlay_widget(self._get_sudo_display_fragments, "_sudo_state")
        secret_widget = self._tui_overlay_widget(self._get_secret_display_fragments, "_secret_state")
        approval_widget = self._tui_overlay_widget(self._get_approval_display_fragments, "_approval_state")
        slash_confirm_widget = self._tui_overlay_widget(
            self._get_slash_confirm_display_fragments, "_slash_confirm_state")
        model_picker_widget = self._tui_overlay_widget(
            self._get_model_picker_display_fragments, "_model_picker_state")
        command_palette_widget = self._tui_overlay_widget(
            self._get_command_palette_display_fragments, "_command_palette_state")
        # Rules above/below the input; narrow terminals hide the bottom one to recover a row.
        input_rule_top = Window(
            char='─', height=lambda: cli_ref._tui_input_rule_height("top"), style='class:input-rule',
        )
        input_rule_bot = Window(
            char='─', height=lambda: cli_ref._tui_input_rule_height("bottom"), style='class:input-rule',
        )
        image_bar = Window(
            content=FormattedTextControl(self._tui_image_bar_fragments),
            height=Condition(lambda: bool(cli_ref._attached_images)))
        voice_status_bar = ConditionalContainer(
            Window(FormattedTextControl(self._tui_voice_status_fragments), height=1),
            filter=Condition(lambda: cli_ref._voice_mode))
        status_bar = ConditionalContainer(
            Window(
                content=FormattedTextControl(lambda: cli_ref._get_status_bar_fragments()),
                height=1,
                # wrap_lines=False: fragments overflowing the width must never wrap onto a second
                # row (looked like a duplicated status bar on long SSH sessions with stale
                # shutil sizes). _get_status_bar_fragments reads prompt_toolkit's own width, so
                # this is the belt-and-suspenders guard.
                wrap_lines=False),
            filter=Condition(
                lambda: cli_ref._status_bar_visible
                and not getattr(cli_ref, "_status_bar_suppressed_after_resize", False)))
        # Stash browse panel — just above the status bar, Ctrl+S on an empty composer with 2+ drafts.
        self._stash_panel_widget = ConditionalContainer(
            Window(FormattedTextControl(self._get_stash_panel_display_fragments), wrap_lines=False),
            filter=Condition(lambda: cli_ref._prompt_stash.panel_open and bool(len(cli_ref._prompt_stash))),
        )
        self._register_extra_tui_keybindings(kb, input_area=input_area)
        layout = Layout(HSplit(self._build_tui_layout_children(
            sudo_widget=sudo_widget,
            secret_widget=secret_widget,
            approval_widget=approval_widget,
            slash_confirm_widget=slash_confirm_widget,
            clarify_widget=clarify_widget,
            model_picker_widget=model_picker_widget,
            command_palette_widget=command_palette_widget,
            spinner_widget=spinner_widget,
            spacer=spacer,
            status_bar=status_bar,
            input_rule_top=input_rule_top,
            image_bar=image_bar,
            input_area=input_area,
            input_rule_bot=input_rule_bot,
            voice_status_bar=voice_status_bar,
            completions_menu=CompletionsMenu(max_height=12, scroll_offset=1))))
        self._tui_set_base_style()
        return layout, PTStyle.from_dict(self._build_tui_style_dict())

    def _tui_build_input_area(self):
        """Multi-line prompt TextArea with slash completion, paste-collapse tracking and
        placeholder/password processors."""
        from cli import _estimate_tui_input_height, get_skill_bundles, get_skill_commands
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import ThreadedCompleter
        cli_ref = self

        def get_prompt():
            return cli_ref._get_tui_prompt_fragments()

        _completer = SlashCommandCompleter(
            skill_commands_provider=lambda: get_skill_commands(),
            command_filter=cli_ref._command_available,
            skill_bundles_provider=lambda: get_skill_bundles())
        input_area = TextArea(
            height=Dimension(min=1, max=8, preferred=1),
            prompt=get_prompt,
            style='class:input-area',
            multiline=True,
            wrap_lines=True,
            read_only=Condition(lambda: bool(cli_ref._command_blocks_input)),
            history=FileHistory(str(self._history_file)),
            # The completer does blocking work (fuzzy @-file indexing shells out to rg/fd with a
            # 2s timeout; path completion hits os.listdir/stat), so complete_while_typing inline
            # would stall the render loop per keystroke (WSL2/slow FS). ThreadedCompleter moves
            # it off the UI event loop.
            completer=ThreadedCompleter(_completer),
            complete_while_typing=True,
            auto_suggest=SlashCommandAutoSuggest(history_suggest=AutoSuggestFromHistory(), completer=_completer),
        )
        # Keep prompt_toolkit on its simple tempfile path: buffer.tempfile = "prompt.md" takes
        # the complex-tempfile branch that re-mkdir()s the mkdtemp() dir and raises EEXIST.
        input_area.buffer.tempfile_suffix = '.md'

        def _input_height():
            # Accounts for explicit newlines AND visual wrapping so the area fits its content.
            try:
                from prompt_toolkit.application import get_app
                doc = input_area.buffer.document
                try:
                    terminal_columns = get_app().output.get_size().columns
                except Exception:
                    terminal_columns = shutil.get_terminal_size((80, 24)).columns
                return _estimate_tui_input_height(doc.lines, self._get_tui_prompt_text(), terminal_columns)
            except Exception:
                return 1

        input_area.window.height = _input_height
        # Paste collapsing state (large pastes are saved to a file and replaced by a placeholder).
        self._tui_paste_counter = 0
        self._tui_prev_text_len = 0
        self._tui_prev_newline_count = 0
        self._tui_paste_just_collapsed = False
        self._skip_paste_collapse = False
        input_area.buffer.on_text_changed += self._tui_on_text_changed
        # Mask input with '*' while a sudo/secret prompt is active.
        input_area.control.input_processors.append(ConditionalProcessor(
            PasswordProcessor(),
            filter=Condition(lambda: bool(cli_ref._sudo_state) or bool(cli_ref._secret_state))))

        class _PlaceholderProcessor(Processor):
            """Render grayed-out placeholder text inside the input when empty."""
            def __init__(self, get_text):
                self._get_text = get_text

            def apply_transformation(self, ti):
                if not ti.document.text and ti.lineno == 0:
                    text = self._get_text()
                    if text:
                        # Append after existing fragments (preserves the ❯ prompt).
                        return Transformation(fragments=ti.fragments + [('class:placeholder', text)])
                return Transformation(fragments=ti.fragments)

        input_area.control.input_processors.append(_PlaceholderProcessor(self._tui_placeholder_text))
        return input_area

    def _tui_set_base_style(self):
        """Populate ``self._tui_style_base`` (skin-aware defaults the style dict is built from)."""
        self._tui_style_base = {
            # Empty input/prompt styles inherit the terminal's own fg/bg so typed text is readable
            # in both light and dark schemes (a hardcoded near-white was invisible on light).
            'input-area': '',
            'placeholder': '#888888 italic',
            'prompt': '',
            'prompt-working': '#888888 italic',
            'hint': '#888888 italic',
            'status-bar': 'bg:#1a1a2e #C0C0C0',
            'status-bar-strong': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-dim': 'bg:#1a1a2e #8B8682',
            'status-bar-good': 'bg:#1a1a2e #8FBC8F bold',
            'status-bar-warn': 'bg:#1a1a2e #FFD700 bold',
            'status-bar-bad': 'bg:#1a1a2e #FF8C00 bold',
            'status-bar-critical': 'bg:#1a1a2e #FF6B6B bold',
            'status-bar-yolo': 'bg:#1a1a2e #FF4444 bold',
            'status-bar-session-title': 'bg:#FFD700 #1a1a2e bold',
            'input-rule': '#CD7F32',
            'image-badge': '#87CEEB bold',
            'completion-menu': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion': 'bg:#1a1a2e #FFF8DC',
            'completion-menu.completion.current': 'bg:#333355 #FFD700',
            'completion-menu.meta.completion': 'bg:#1a1a2e #888888',
            'completion-menu.meta.completion.current': 'bg:#333355 #FFBF00',
            'clarify-border': '#CD7F32',
            'clarify-title': '#FFD700 bold',
            'clarify-question': '#FFF8DC bold',
            'clarify-choice': '#AAAAAA',
            'clarify-selected': '#FFD700 bold',
            'clarify-active-other': '#FFD700 italic',
            'clarify-answer': '#98FB98',
            'clarify-countdown': '#CD7F32',
            'sudo-prompt': '#FF6B6B bold',
            'sudo-border': '#CD7F32',
            'sudo-title': '#FF6B6B bold',
            'sudo-text': '#FFF8DC',
            'approval-border': '#CD7F32',
            'approval-title': '#FF8C00 bold',
            'approval-desc': '#FFF8DC bold',
            'approval-cmd': '#AAAAAA italic',
            'approval-choice': '#AAAAAA',
            'approval-selected': '#FFD700 bold',
            'voice-prompt': '#87CEEB',
            'voice-recording': '#FF4444 bold',
            'voice-processing': '#FFA500 italic',
            'voice-status': 'bg:#1a1a2e #87CEEB',
            'voice-status-recording': 'bg:#1a1a2e #FF4444 bold'}
