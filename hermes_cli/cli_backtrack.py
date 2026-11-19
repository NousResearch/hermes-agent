"""Codex-style backtrack for the classic CLI. On an idle, empty composer the first Esc PRIMES
(hint ``esc again to edit previous message``; any other key cancels it), the second opens BROWSE
mode: a transcript pager of the session's prompts and replies with the selected prompt in reverse
video (``cli_backtrack_browse``). ←/→ pick an older/newer prompt, ↑/↓ PgUp/PgDn scroll, Enter puts
the prompt back in the composer, ``f`` forks a branch before it, Esc/q close. ``/rewind`` opens
browse directly; Esc Esc in quick succession goes straight to browse, as before.

Edit & resend cuts NOTHING until the edited text is submitted: the submission travels to the
process loop as a :class:`BacktrackResend`, which re-validates the picked turn and only then runs
the same durable rewind ``/undo`` uses (``_rewind_to_user_ordinal``). Cancelling (Esc Esc, Ctrl+C,
an empty submit, or submitting something else) leaves the session as it was. Fork reuses
``/branch`` and rewinds only the child, so the original session's rows stay active.

Mixin bound FIRST in ``HermesCLI``'s bases so its overrides wrap the TUI handlers via ``super()``.
cli.py-internal symbols are imported lazily (import cycle).
"""

from __future__ import annotations

from agent.i18n import t


_BROWSE_KEYS = ("left", "right", "h", "l", "up", "down", "j", "k", "pageup", "pagedown", "enter", "q", "f")


class BacktrackResend(str):
    """An edited prompt to send after rewinding to the turn it was picked from."""

    __slots__ = ("expected", "number", "ordinal", "session_id")

    def __new__(cls, text: str, *, ordinal: int, expected: str, session_id, number: int):
        obj = super().__new__(cls, text)
        obj.ordinal, obj.expected, obj.session_id, obj.number = ordinal, expected, session_id, number
        return obj


class CLIBacktrackMixin:
    # browse: {"entries": [(ordinal, text)] newest first, "selected", "turns", "_scroll_offset", "_reveal"}
    _backtrack_state = None
    _backtrack_edit = None  # pending edit & resend: {"ordinal", "expected", "session_id", "number"}
    _backtrack_primed = None  # None | "hint" | "silent": the first Esc on an idle, empty composer
    _backtrack_widget = None

    # ---- model ---------------------------------------------------------------------------
    def _backtrack_prompts(self) -> list[str]:
        """Editable text of every real user turn, oldest first (ordinal = index); the same
        predicate /undo counts turns with, so an ordinal here is the ordinal it rewinds. An image
        prompt gives what the user typed: the vision note or attach hint the model got is dropped,
        and the image is not re-attached (an armed edit sends text only)."""
        from session_rules.turns import user_originated_turn_view, typed_prompt_content
        from agent.image_routing import split_image_notes
        from hermes_cli.cli_session_mixin import _user_turn_indices
        history = self.conversation_history or []
        prompts = []
        for index in _user_turn_indices(history):
            view = user_originated_turn_view(history[index]) or {}
            text = self._undo_content_to_text(typed_prompt_content(view.get("content"), history[index]))
            prompts.append(split_image_notes(text)[0])
        return prompts

    def _backtrack_blocked(self) -> bool:
        return bool(self._agent_running or self._command_running or self._clarify_state or self._approval_state
                    or self._sudo_state or self._secret_state or self._connection_state or self._slash_confirm_state
                    or self._model_picker_state or self._command_palette_state)

    # ---- prime (first Esc) -----------------------------------------------------------------
    def _backtrack_composer_empty(self, event_or_app) -> bool:
        app = getattr(event_or_app, "app", event_or_app)
        return not (app.current_buffer.text or self._attached_images)

    def _backtrack_handle_escape(self, event) -> None:
        """A lone Esc (no Alt-sequence followed within the escape timeout). On an idle, empty
        composer the first one primes and the second acts like Esc Esc; with a draft it does
        nothing, as before."""
        if not self._backtrack_composer_empty(event) or self._backtrack_blocked():
            self._backtrack_unprime()
            return
        if self._backtrack_primed:
            self._backtrack_primed = None
            self._backtrack_second_escape()
            return
        self._backtrack_primed = "hint" if self._backtrack_edit is None and self._backtrack_prompts() else "silent"
        self._invalidate(min_interval=0.0)

    def _backtrack_second_escape(self) -> None:
        from cli import _cprint
        if self._backtrack_edit is not None:
            self._backtrack_cancel_edit()
        elif not self._backtrack_blocked():
            if not self._backtrack_prompts():
                _cprint(f"  {t('cli.backtrack.no_previous')}")
            else:
                self._open_backtrack_picker()

    def _backtrack_primed_chord(self, event) -> None:
        """Primed Esc, then Esc + a browse key in one read: the second Esc, then that key."""
        from prompt_toolkit.key_binding.key_processor import KeyPress
        self._backtrack_primed = None
        self._backtrack_second_escape()
        last = event.key_sequence[-1]
        event.app.key_processor.feed_multiple([KeyPress(last.key, last.data)], first=True)

    def _backtrack_unprime(self) -> None:
        if self._backtrack_primed:
            self._backtrack_primed = None
            self._invalidate(min_interval=0.0)

    def _backtrack_after_key_press(self, key_processor) -> None:
        """Any key other than Esc cancels a prime (that key already acted normally). The Esc that
        primed, or one still waiting in the key buffer for a possible Alt-sequence, keeps it."""
        from prompt_toolkit.keys import Keys
        if not self._backtrack_primed:
            return
        pending = key_processor.key_buffer or key_processor._previous_key_sequence
        if not (pending and all(k.key == Keys.Escape for k in pending)):
            self._backtrack_unprime()

    # ---- browse --------------------------------------------------------------------------
    def _open_backtrack_picker(self) -> bool:
        from cli import _cprint
        from hermes_cli.cli_backtrack_browse import transcript_turns
        self._backtrack_primed = None
        if self._backtrack_state or self._backtrack_blocked():
            return False
        prompts = self._backtrack_prompts()
        if not prompts:
            _cprint(f"  {t('cli.backtrack.nothing')}")
            return False
        self._backtrack_state = {
            "entries": list(reversed(list(enumerate(prompts)))), "selected": 0,
            "turns": transcript_turns(self), "_scroll_offset": 0, "_reveal": True}
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
        from hermes_cli.cli_backtrack_browse import render_pager
        from hermes_cli.cli_tui_mixin import _PANEL_RESERVED_BELOW
        state = self._backtrack_state
        if not state:
            return []
        return render_pager(state, state["turns"], t("cli.backtrack.title"), t("cli.backtrack.keys_hint"),
                             _PANEL_RESERVED_BELOW)

    def _backtrack_step(self, older: int) -> None:
        """←/h (older=1) and →/l (older=-1): pick the neighbouring prompt, clamped, and bring
        it into view."""
        state = self._backtrack_state
        if state:
            state["selected"] = max(0, min(len(state["entries"]) - 1, state.get("selected", 0) + older))
            state["_reveal"] = True
            self._invalidate(min_interval=0.0)

    def _backtrack_scroll(self, rows: int = 0, *, pages: int = 0) -> None:
        """↑/↓ scroll a row, PgUp/PgDn a page; the selection stays put (the next ←/→ re-centres)."""
        state = self._backtrack_state
        if state:
            step = rows + pages * max(1, state.get("_page", 10) - 1)
            state["_scroll_offset"] = max(0, state.get("_scroll_offset", 0) + step)
            self._invalidate(min_interval=0.0)

    def _backtrack_close_and_replay(self, event) -> None:
        """Any other key while browsing closes browse and then acts normally (typing lands in the
        composer), as Codex cancels browsing on other keys."""
        self._close_backtrack_picker()
        event.app.key_processor.feed_multiple(list(event.key_sequence), first=True)

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
        """Esc Esc in quick succession. Idle + empty composer + no overlay: open browse (or cancel a
        pending edit). A draft is still discarded (and a pending edit cancelled with it);
        mid-turn behaviour is unchanged."""
        self._backtrack_primed = None
        if self._backtrack_composer_empty(event):
            self._backtrack_second_escape()
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
        self._backtrack_unprime()  # a turn starting ends the prime
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
        if hint or self._backtrack_state:
            return hint
        if self._backtrack_primed == "hint":
            return [("class:hint", "  " + t("cli.backtrack.prime_hint"))]
        if self._backtrack_edit is not None:
            return [("class:hint", "  " + t("cli.backtrack.editing_hint", number=self._backtrack_edit["number"]))]
        return hint

    def _tui_hint_height(self):
        if self._backtrack_edit is not None or self._backtrack_primed == "hint":
            return 1
        return super()._tui_hint_height()

    def _get_extra_tui_widgets(self) -> list:
        extra = super()._get_extra_tui_widgets()
        return [self._backtrack_widget, *extra] if self._backtrack_widget is not None else extra

    def _tui_build_layout(self, kb):
        self._backtrack_widget = self._tui_overlay_widget(self._get_backtrack_display_fragments, "_backtrack_state")
        return super()._tui_build_layout(kb)

    def _tui_build_application(self, layout, kb, style):
        app = super()._tui_build_application(layout, kb, style)
        app.key_processor.after_key_press += self._backtrack_after_key_press
        return app

    def _tui_build_key_bindings(self):
        """Prime + browse keys, registered after the generic ones so they win (the last matching
        binding wins). The lone Esc is eager only on an idle, empty composer with no further key
        already queued: an Alt-sequence arrives from the terminal in one read (Esc+Enter newline,
        Esc+V), so its second key is queued and the Esc stays a prefix. With a draft it is not
        eager, so Esc Esc still discards the draft. Off in vi insert mode (Esc leaves insert mode
        there) and whenever a prompt owns the composer."""
        from prompt_toolkit.application import get_app
        from prompt_toolkit.filters import Condition, vi_insert_mode
        from prompt_toolkit.keys import Keys

        from hermes_cli.cli_subagent_monitor import modal_prompt_active
        kb = super()._tui_build_key_bindings()
        picker = Condition(lambda: bool(self._backtrack_state))
        primable = Condition(lambda: not self._backtrack_state and not modal_prompt_active(self)
                             and not self._prompt_stash.panel_open) & ~vi_insert_mode
        idle_empty = Condition(lambda: self._backtrack_composer_empty(get_app()) and not self._backtrack_blocked())
        lone_esc = idle_empty & Condition(lambda: not get_app().key_processor.input_queue)
        kb.add("escape", filter=primable, eager=lone_esc)(self._backtrack_handle_escape)
        # A browse key typed within the escape timeout of the second Esc reaches us as an Alt chord
        # (the parser holds a lone Esc for 0.5 s). While primed on an empty composer that chord is
        # the gesture: open browse, then apply the key; keys queued behind it run after, in order.
        # The empty-queue test is only for lone-Esc eagerness. Unprimed or with a draft, Alt keys
        # are unchanged.
        primed_chord = primable & Condition(lambda: bool(self._backtrack_primed)) & idle_empty
        for key in _BROWSE_KEYS:
            kb.add("escape", key, filter=primed_chord)(self._backtrack_primed_chord)
        kb.add(Keys.Any, filter=picker)(self._backtrack_close_and_replay)
        # A paste is text, never a browse key: close browse and let the normal (eager) paste
        # handler take it, or the pasted text vanishes with browse still open.
        kb.add(Keys.BracketedPaste, filter=picker, eager=True)(self._backtrack_close_and_replay)
        for keys, handler in (
                (("left",), lambda event: self._backtrack_step(1)), (("h",), lambda event: self._backtrack_step(1)),
                (("right",), lambda event: self._backtrack_step(-1)), (("l",), lambda event: self._backtrack_step(-1)),
                (("up",), lambda event: self._backtrack_scroll(-1)), (("k",), lambda event: self._backtrack_scroll(-1)),
                (("down",), lambda event: self._backtrack_scroll(1)), (("j",), lambda event: self._backtrack_scroll(1)),
                (("pageup",), lambda event: self._backtrack_scroll(pages=-1)),
                (("pagedown",), lambda event: self._backtrack_scroll(pages=1)),
                (("f",), lambda event: self._backtrack_choose_fork()),
                (("q",), lambda event: self._close_backtrack_picker()),
                (("c-d",), lambda event: self._close_backtrack_picker())):  # never exit from browse
            kb.add(*keys, filter=picker)(handler)
        kb.add("enter", filter=picker, eager=True)(lambda event: self._backtrack_choose_edit())
        kb.add("escape", filter=picker, eager=True)(lambda event: self._close_backtrack_picker())
        return kb
