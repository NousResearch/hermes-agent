"""Codex-style backtrack for the classic CLI: Esc Esc on an idle, empty composer (or ``/rewind``)
lists earlier prompts newest first; Enter edits & resends one, ``f`` forks a branch before it.

The transcript above is terminal scrollback that prompt_toolkit cannot scroll, so the picker cannot
jump there; instead it shows the selected prompt in full with the reply that followed it (PgUp/PgDn
scroll that context) — the same exchange the transcript holds, brought to the picker.

Edit & resend cuts NOTHING until the edited text is submitted: the submission travels to the
process loop as a :class:`BacktrackResend`, which re-validates the picked turn and only then runs
the same durable rewind ``/undo`` uses (``_rewind_to_user_ordinal``). Cancelling (Esc Esc, Ctrl+C,
an empty submit, or submitting something else) leaves the session as it was. Fork reuses
``/branch`` and rewinds only the child, so the original session's rows stay active.

Mixin bound FIRST in ``HermesCLI``'s bases so its overrides wrap the TUI handlers via ``super()``.
cli.py-internal symbols are imported lazily (import cycle).
"""

from __future__ import annotations

import shutil

from agent.i18n import t

_PANEL_MAX_INNER = 88
_LIST_MAX_ROWS = 8
# Panel rows besides the hint, list and context: borders (2), four blanks, the "Prompt N of M" label.
_PANEL_FIXED_ROWS = 7


class BacktrackResend(str):
    """An edited prompt to send after rewinding to the turn it was picked from."""

    __slots__ = ("expected", "number", "ordinal", "session_id")

    def __new__(cls, text: str, *, ordinal: int, expected: str, session_id, number: int):
        obj = super().__new__(cls, text)
        obj.ordinal, obj.expected, obj.session_id, obj.number = ordinal, expected, session_id, number
        return obj


def _clip(line: str, limit: int) -> str:
    from hermes_cli.cli_render import _clip_cells
    return _clip_cells(line, limit)


def _one_line(text: str, limit: int) -> str:
    return _clip(" ".join((text or "").split()), limit)


def _term_size() -> tuple[int, int]:
    """(columns, rows) of the terminal the picker draws in: the running app's output, else the tty."""
    from prompt_toolkit.application import get_app_or_none
    app = get_app_or_none()
    if app is not None and app.is_running:
        size = app.output.get_size()
        return size.columns, size.rows
    size = shutil.get_terminal_size((100, 24))
    return size.columns, size.lines


def _picker_budget(free_rows: int, entries: int) -> tuple[int, int]:
    """(list rows, context rows) sharing ``free_rows``: the list takes at most a third
    (shrinking around the selection), the selected exchange the rest; neither goes under one row."""
    listed = min(entries, _LIST_MAX_ROWS, max(1, free_rows // 3))
    return listed, max(1, free_rows - listed)


class CLIBacktrackMixin:
    # picker: {"entries": [(ordinal, text)] newest first, "selected", "_scroll_offset", "context_offset"}
    _backtrack_state = None
    _backtrack_edit = None  # pending edit & resend: {"ordinal", "expected", "session_id", "number"}
    _backtrack_widget = None

    # ---- model ---------------------------------------------------------------------------
    def _backtrack_prompts(self) -> list[str]:
        """Editable text of every real user turn, oldest first (ordinal = index); the same
        predicate /undo counts turns with, so an ordinal here is the ordinal it rewinds."""
        from agent.context_compressor import user_originated_turn_view
        from agent.message_metadata import typed_prompt_content
        from hermes_cli.cli_session_mixin import _user_turn_indices
        history = self.conversation_history or []
        prompts = []
        for index in _user_turn_indices(history):
            view = user_originated_turn_view(history[index]) or {}
            prompts.append(self._undo_content_to_text(typed_prompt_content(view.get("content"), history[index])))
        return prompts

    def _backtrack_reply(self, ordinal: int) -> str:
        """The assistant text that answered user turn ``ordinal`` (everything up to the next turn), as
        the transcript showed it: a compaction handoff summary is model-facing, never the answer."""
        from agent.compaction_display import project_compaction_message_for_display
        from hermes_cli.cli_session_mixin import _user_turn_indices
        history = self.conversation_history or []
        indices = _user_turn_indices(history)
        if not 0 <= ordinal < len(indices):
            return ""
        end = indices[ordinal + 1] if ordinal + 1 < len(indices) else len(history)
        shown = (project_compaction_message_for_display(m) for m in history[indices[ordinal] + 1:end])
        texts = [self._undo_content_to_text(m.get("content")).strip() for m in shown
                 if m is not None and m.get("role") == "assistant"]
        return "\n\n".join(text for text in texts if text)

    def _backtrack_context_rows(self, ordinal: int, text: str, width: int) -> list[str]:
        """The selected exchange as panel rows: the prompt in full, then the reply that followed."""
        from cli import _wrap_panel_text
        rows = [t("cli.backtrack.prompt_label")]
        rows += [f"  {row}" for row in _wrap_panel_text(text or "", width, keep_ws=True)]
        reply = self._backtrack_reply(ordinal)
        rows += ["", t("cli.backtrack.reply_label")]
        rows += ([f"  {row}" for row in _wrap_panel_text(reply, width, keep_ws=True)] if reply
                 else [f"  {t('cli.backtrack.no_reply')}"])
        return rows

    def _backtrack_blocked(self) -> bool:
        return bool(self._agent_running or self._command_running or self._clarify_state or self._approval_state
                    or self._sudo_state or self._secret_state or self._connection_state or self._slash_confirm_state
                    or self._model_picker_state or self._command_palette_state)

    # ---- picker --------------------------------------------------------------------------
    def _open_backtrack_picker(self) -> bool:
        from cli import _cprint
        if self._backtrack_state or self._backtrack_blocked():
            return False
        prompts = self._backtrack_prompts()
        if not prompts:
            _cprint(f"  {t('cli.backtrack.nothing')}")
            return False
        self._backtrack_state = {
            "entries": list(reversed(list(enumerate(prompts)))), "selected": 0, "_scroll_offset": 0,
            "context_offset": 0}
        self._invalidate(min_interval=0.0)
        return True

    def _close_backtrack_picker(self) -> None:
        self._backtrack_state = None
        self._invalidate(min_interval=0.0)

    def _backtrack_selected(self):
        state = self._backtrack_state or {}
        entries = state.get("entries") or []
        selected = state.get("selected", 0)
        return entries[selected] if 0 <= selected < len(entries) else None

    def _get_backtrack_display_fragments(self):
        """The picker panel, every row budgeted: wrapped hint rows, one row per listed prompt and the
        selected exchange together fit the terminal above the CLI's own input/status rows."""
        from cli import _wrap_panel_text
        from hermes_cli.cli_panel import _Panel
        from hermes_cli.cli_tui_mixin import _PANEL_RESERVED_BELOW
        state = self._backtrack_state
        if not state:
            return []
        entries, selected = state["entries"], state.get("selected", 0)
        cols, term_rows = _term_size()
        inner = max(24, min(_PANEL_MAX_INNER, cols - 6))
        hints = _wrap_panel_text(t("cli.backtrack.keys_hint"), inner)
        free = term_rows - _PANEL_RESERVED_BELOW - _PANEL_FIXED_ROWS - len(hints)
        listed, window = _picker_budget(free, len(entries))
        state["_context_window"] = window
        scroll = max(0, min(state.get("_scroll_offset", 0), selected, len(entries) - listed))
        scroll = max(scroll, selected - listed + 1)
        state["_scroll_offset"] = scroll

        panel = _Panel("class:clarify-border", inner + 2, _clip(t("cli.backtrack.title"), inner - 2),
                       "class:clarify-title")
        panel.blank()
        for row in hints:
            panel.row("class:clarify-hint", row)
        panel.blank()
        picked = self._backtrack_selected()
        if picked is not None:
            ordinal, text = picked
            rows = self._backtrack_context_rows(ordinal, text, inner - 2)
            offset = max(0, min(state.get("context_offset", 0), len(rows) - window))
            state["context_offset"] = offset
            label = t("cli.backtrack.preview_label", number=ordinal + 1, total=len(entries))
            if len(rows) > window:
                label += "  " + t("cli.backtrack.scroll_hint", first=offset + 1,
                                  last=min(len(rows), offset + window), total=len(rows))
            panel.row("class:clarify-hint", _clip(label, inner))
            for row in rows[offset:offset + window]:
                panel.row("class:clarify-hint", row)
            panel.blank()
        for idx in range(scroll, min(scroll + listed, len(entries))):
            ordinal, text = entries[idx]
            lead = "❯ " if idx == selected else "  "
            style = "class:clarify-selected" if idx == selected else "class:clarify-choice"
            panel.row(style, _clip(f"{lead}{ordinal + 1:>3}. {_one_line(text, inner)}", inner))
        panel.blank()
        return panel.close()

    def _backtrack_move(self, delta: int) -> None:
        state = self._backtrack_state
        if state:
            state["selected"] = max(0, min(len(state["entries"]) - 1, state.get("selected", 0) + delta))
            state["context_offset"] = 0
            self._invalidate(min_interval=0.0)

    def _backtrack_scroll_context(self, pages: int) -> None:
        """PgUp/PgDn: page the selected exchange; the render clamps the offset to its length."""
        state = self._backtrack_state
        if state:
            step = max(1, state.get("_context_window", 1) - 1)
            state["context_offset"] = max(0, state.get("context_offset", 0) + pages * step)
            self._invalidate(min_interval=0.0)

    def _backtrack_choose_edit(self) -> None:
        picked = self._backtrack_selected()
        self._close_backtrack_picker()
        if picked is not None:
            self._backtrack_begin_edit(*picked)

    def _backtrack_choose_fork(self) -> None:
        picked = self._backtrack_selected()
        self._close_backtrack_picker()
        if picked is not None:
            # Branching writes the DB and switches sessions: run it on the process loop like /branch.
            self._pending_input.put(f"/rewind fork {picked[0] + 1}")

    # ---- edit & resend -------------------------------------------------------------------
    def _backtrack_begin_edit(self, ordinal: int, text: str) -> None:
        self._backtrack_edit = {"ordinal": ordinal, "expected": text, "session_id": self.session_id,
                                "number": ordinal + 1}
        self._prefill_input_buffer(text)

    def _backtrack_cancel_edit(self) -> bool:
        from cli import _cprint
        if self._backtrack_edit is None:
            return False
        self._backtrack_edit = None
        _cprint(f"  {t('cli.backtrack.edit_cancelled')}")
        return True

    def _backtrack_apply_resend(self, item: BacktrackResend):
        """Rewind to the picked turn, then return the edited text to send (None: nothing sent)."""
        from cli import _cprint
        prompts = self._backtrack_prompts()
        if (item.session_id != self.session_id or item.ordinal >= len(prompts)
                or prompts[item.ordinal] != item.expected or self._agent_running):
            _cprint(f"  {t('cli.backtrack.target_changed', number=item.number)}")
            self._prefill_input_buffer(str(item))
            return None
        rewound = self._rewind_to_user_ordinal(item.ordinal, failure_key="cli.backtrack.rewind_failed")
        if rewound is None:
            self._prefill_input_buffer(str(item))
            return None
        _cprint(f"  {t('cli.backtrack.rewound', number=item.number, count=rewound[1])}")
        return str(item)

    # ---- fork ----------------------------------------------------------------------------
    def _backtrack_fork(self, ordinal: int) -> None:
        """/branch, then rewind the CHILD to just before ``ordinal`` and prefill that prompt."""
        from cli import _cprint
        parent_session_id = self.session_id
        self._handle_branch_command("/branch")
        if self.session_id == parent_session_id:
            return  # /branch refused and said why; nothing changed
        rewound = self._rewind_to_user_ordinal(ordinal, failure_key="cli.backtrack.rewind_failed")
        if rewound is None:
            return
        _cprint(f"  {t('cli.backtrack.forked', number=ordinal + 1, parent=parent_session_id)}")
        self._prefill_input_buffer(rewound[0])

    # ---- /rewind -------------------------------------------------------------------------
    def _handle_rewind_command(self, cmd_original: str) -> None:
        """/rewind opens the picker; ``/rewind N`` edits prompt N; ``/rewind fork N`` forks before it."""
        from cli import _cprint
        if self._agent_running:
            return _cprint(f"  {t('cli.commands.shared.agent_busy', command='/rewind')}")
        args = cmd_original.split()[1:]
        fork = bool(args) and args[0].lower() == "fork"
        args = args[1:] if fork else args
        if not args and not fork:
            self._open_backtrack_picker()
            return None
        if len(args) != 1 or not args[0].isdigit():
            return _cprint(f"  {t('cli.backtrack.usage')}")
        prompts = self._backtrack_prompts()
        number = int(args[0])
        if not prompts:
            return _cprint(f"  {t('cli.backtrack.nothing')}")
        if not 1 <= number <= len(prompts):
            return _cprint(f"  {t('cli.backtrack.invalid', value=args[0], total=len(prompts))}")
        if fork:
            return self._backtrack_fork(number - 1)
        self._backtrack_begin_edit(number - 1, prompts[number - 1])
        return None

    # ---- TUI hooks (wrap CLITuiMixin / CLITuiRuntimeMixin via super()) --------------------
    def _tui_handle_double_escape(self, event):
        """Idle + empty composer + no overlay: open the picker. A draft is still discarded (and a
        pending edit cancelled with it); mid-turn behaviour is unchanged."""
        buf = event.app.current_buffer
        if not (buf.text or self._attached_images):
            if self._backtrack_edit is not None:
                self._backtrack_cancel_edit()
            elif not self._backtrack_blocked():
                self._open_backtrack_picker()
            return None
        self._backtrack_cancel_edit()
        return super()._tui_handle_double_escape(event)

    def _tui_submit_buffer(self, event, buf):
        from cli import _looks_like_slash_command
        edit = self._backtrack_edit
        if edit is None:
            return super()._tui_submit_buffer(event, buf)
        text = buf.text.strip()
        if not text and not self._attached_images:
            self._backtrack_cancel_edit()
            return None
        if (self._agent_running or self._attached_images or _looks_like_slash_command(text)
                or text.startswith("!")):
            self._backtrack_cancel_edit()
            return super()._tui_submit_buffer(event, buf)
        self._backtrack_edit = None
        self._inline_pastes(buf)
        self._pending_input.put(BacktrackResend(buf.text.strip(), **edit))
        buf.reset(append_to_history=True)
        event.app.invalidate()
        return None

    def _tui_process_one_input(self, user_input):
        if isinstance(user_input, BacktrackResend):
            user_input = self._backtrack_apply_resend(user_input)
            if user_input is None:
                return None
        return super()._tui_process_one_input(user_input)

    def _tui_clear_or_exit(self, event) -> None:
        if self._backtrack_cancel_edit() and not (event.app.current_buffer.text or self._attached_images):
            return None  # the press cancelled the pending edit; never exit on it
        return super()._tui_clear_or_exit(event)

    def _tui_handle_ctrl_c(self, event):
        if self._backtrack_state:
            self._close_backtrack_picker()
            return None
        return super()._tui_handle_ctrl_c(event)

    def _tui_hint_text(self):
        hint = super()._tui_hint_text()
        if not hint and self._backtrack_edit is not None and not self._backtrack_state:
            return [("class:hint", "  " + t("cli.backtrack.editing_hint", number=self._backtrack_edit["number"]))]
        return hint

    def _tui_hint_height(self):
        if self._backtrack_edit is not None:
            return 1
        return super()._tui_hint_height()

    def _get_extra_tui_widgets(self) -> list:
        extra = super()._get_extra_tui_widgets()
        return [self._backtrack_widget, *extra] if self._backtrack_widget is not None else extra

    def _tui_build_layout(self, kb):
        self._backtrack_widget = self._tui_overlay_widget(self._get_backtrack_display_fragments, "_backtrack_state")
        return super()._tui_build_layout(kb)

    def _tui_build_key_bindings(self):
        """Picker keys, registered after the generic ones so they win while it is open (the
        last matching binding wins); Keys.Any swallows typing so nothing lands in the composer."""
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.keys import Keys
        kb = super()._tui_build_key_bindings()
        picker = Condition(lambda: bool(self._backtrack_state))
        kb.add(Keys.Any, filter=picker)(lambda event: None)
        kb.add("up", filter=picker)(lambda event: self._backtrack_move(-1))
        kb.add("down", filter=picker)(lambda event: self._backtrack_move(1))
        kb.add("pageup", filter=picker)(lambda event: self._backtrack_scroll_context(-1))
        kb.add("pagedown", filter=picker)(lambda event: self._backtrack_scroll_context(1))
        kb.add("enter", filter=picker, eager=True)(lambda event: self._backtrack_choose_edit())
        kb.add("f", filter=picker)(lambda event: self._backtrack_choose_fork())
        kb.add("escape", filter=picker, eager=True)(lambda event: self._close_backtrack_picker())
        return kb
