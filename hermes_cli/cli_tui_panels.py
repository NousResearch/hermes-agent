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


class CLITuiPanelsMixin:
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
        if code := self._sudo_state.get("vault_code"):
            return self._render_sudo_style_panel(
                f'🔐 Verification code for {code["site"]}',
                [f'{code["site"]} is asking for a one-time code (text message, email or authenticator app).',
                 'Type the code and press Enter; Hermes enters it into the page for you.',
                 'Enter on an empty line skips. The model never sees the code.'])
        if save := self._sudo_state.get("vault_save"):
            if save["step"] == "identifier":
                return self._render_sudo_style_panel(
                    f'🔐 Save login for {save["site"]}',
                    ['The agent reached a sign-in page with no saved login for this site.',
                     'Type the email / username you sign in with (shown), then Enter.',
                     'Enter on an empty line skips. Nothing here is shown to the model.'])
            return self._render_sudo_style_panel(
                f'🔐 Save login for {save["site"]}',
                ['Now the password (hidden). It is encrypted on this machine, bound to',
                 f'{save["origin"]}, and filled into the page without the model ever seeing it.',
                 'Enter on an empty line skips.'])
        if backend := self._sudo_state.get("vault_backend"):
            return self._render_sudo_style_panel(
                f'🔐 Unlock {backend}',
                [f'The agent wants to sign into a site with a login saved in {backend}.',
                 'Type your master password (hidden) to unlock it for this session.',
                 'Enter on an empty line keeps it locked. The model never sees the password.'])
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


