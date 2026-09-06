"""Modal overlays for the interactive CLI: clarify, approval, sudo/secret capture, command palette,
slash-confirm, external editor.

Mixin split out of ``cli.py``; bound onto ``HermesCLI`` via the MRO. cli.py-internal symbols are
imported LAZILY inside each method — the mixin never imports ``cli`` at module load time (cycle).
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time as _time

from hermes_cli.callbacks import prompt_for_secret
from typing import Optional

_TIMED_OUT = object()  # sentinel returned by _poll_modal_queue when the deadline passes

# Typed answers accepted by the slash-confirm modal, mapped onto the canonical choice values.
_CONFIRM_ALIASES = {
    "1": "once", "once": "once", "approve": "once", "yes": "once", "y": "once", "ok": "once",
    "2": "always", "always": "always", "remember": "always",
    "3": "cancel", "cancel": "cancel", "nevermind": "cancel", "no": "cancel", "n": "cancel"}

_APPROVAL_OUTCOME_LABELS = {
    "once": "allowed once",
    "session": "allowed for session",
    "always": "added to allowlist",
    "deny": "denied"}

_CLARIFY_TIMEOUT_REPLY = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed.")


def _approval_gate_on(key: str) -> bool:
    """Read ``approvals.<key>`` (default on); any load failure keeps the prompt enabled."""
    from cli import load_cli_config
    try:
        cfg = load_cli_config()
        approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
        if isinstance(approvals, dict):
            return bool(approvals.get(key, True))
    except Exception:
        pass
    return True


def _gated_confirm(self, command, key, *, title, detail, choices, unchanged, always_msg, once_verb):
    """Shared once/always/cancel confirm behind ``approvals.<key>`` (destructive slash, /reload-mcp).

    Returns ``"once"`` without prompting when the gate is off; ``None`` on cancel / no input /
    unrecognized answer (already reported to the user). Picking "always" persists the opt-out.
    """
    from cli import save_config_value
    if not _approval_gate_on(key):
        return "once"
    raw = self._prompt_text_input_modal(title=title, detail=detail, choices=choices)
    if raw is None:
        print(f"🟡 /{command} cancelled (no input).")
        return None
    choice = self._normalize_slash_confirm_choice(raw, choices)
    if choice is None:
        print(f"🟡 Unrecognized choice '{raw}'. /{command} cancelled.")
        return None
    if choice == "cancel":
        print(f"🟡 /{command} cancelled. {unchanged}")
        return None
    if choice == "always":
        if save_config_value(f"approvals.{key}", False):
            print(always_msg)
            print(f"   Re-enable via `approvals.{key}: true` in config.yaml.")
        else:
            print(f"⚠️  Couldn't persist opt-out — {once_verb} once.")
    return choice


class CLIModalMixin:
    """Modal overlays for the interactive CLI: clarify, approval, sudo/secret capture, command
    palette, slash-confirm, external editor."""

    def _open_external_editor(self, buffer=None) -> bool:
        """Open the active input buffer in an external editor."""
        from cli import _DIM, _RST, _cprint
        app = getattr(self, "_app", None)
        if not app:
            _cprint(f"{_DIM}External editor is only available inside the interactive CLI.{_RST}")
            return False
        if self._command_running:
            _cprint(f"{_DIM}Wait for the current command to finish before opening the editor.{_RST}")
            return False
        if (self._sudo_state or self._secret_state or self._approval_state
                or getattr(self, "_slash_confirm_state", None) or self._clarify_state):
            _cprint(f"{_DIM}Finish the active prompt before opening the editor.{_RST}")
            return False
        target_buffer = buffer or getattr(app, "current_buffer", None)
        if target_buffer is None:
            _cprint(f"{_DIM}No active input buffer is available for the external editor.{_RST}")
            return False
        try:
            # Inline pastes so the editor sees real content; set the skip flag unconditionally so
            # the editor-close text-change doesn't re-collapse it.
            self._inline_pastes(target_buffer)
            self._skip_paste_collapse = True
            # Submission here is driven by the custom `enter` keybinding, NOT the buffer's
            # accept_handler, so validate_and_handle can't route through it; chain a done-callback
            # that re-uses the real submit pipeline (TUI Ctrl+G parity: save == send).
            task = target_buffer.open_in_editor(validate_and_handle=False)
            if task is not None and hasattr(task, "add_done_callback"):
                task.add_done_callback(lambda _t, b=target_buffer: self._submit_editor_buffer(b))
            return True
        except Exception as exc:
            _cprint(f"{_DIM}Failed to open external editor: {exc}{_RST}")
            return False

    def _submit_editor_buffer(self, buffer) -> None:
        """Submit the draft an external editor left in ``buffer`` (Ctrl+G done-callback), mirroring
        the `enter` keybinding: empty save ignored, bang/slash dispatched, else queued. Runs on the
        prompt_toolkit loop, so it must stay cheap/non-blocking."""
        from cli import _DIM, _RST, _cprint, _looks_like_slash_command
        try:
            text = (getattr(buffer, "text", "") or "").strip()
        except Exception:
            return
        if not text:
            return

        app = getattr(self, "_app", None)

        def _done() -> None:
            self._reset_input_buffer(buffer)
            if app is not None:
                app.invalidate()

        # `!<command>` shell mode is checked before slash dispatch, matching the Enter path.
        try:
            if self.handle_bang_shell(text):
                _done()
                return
        except Exception as exc:
            _cprint(f"  {_DIM}Shell command failed: {exc}{_RST}")
            _done()
            return

        if _looks_like_slash_command(text):
            try:
                if not self.process_command(text):
                    self._should_exit = True
                    if app is not None and app.is_running:
                        app.exit()
            except Exception as exc:
                _cprint(f"  {_DIM}Command failed: {exc}{_RST}")
            finally:
                _done()
            return

        if self._agent_running:
            # Agent busy → honour the configured busy-input behaviour (interrupt/steer remain
            # reachable via the normal Enter path).
            if self.busy_input_mode == "interrupt":
                self._interrupt_queue.put(text)
            else:
                self._pending_input.put(text)
            preview = text[:80] + ("..." if len(text) > 80 else "")
            _cprint(f"  Queued for the next turn: {preview}")
        else:
            self._pending_input.put(text)
        _done()

    def _inline_pastes(self, buffer) -> None:
        """Replace collapsed ``[Pasted text #N -> file]`` placeholders in ``buffer`` with real text.

        History recall and the external editor need the content (the file may be gone or on another
        machine); inlining before ``reset(append_to_history=True)`` also lets prompt_toolkit persist
        it. Sets ``_skip_paste_collapse`` so the ensuing text-change doesn't re-collapse it.
        """
        from cli import logger
        try:
            existing = getattr(buffer, "text", "")
            expanded = self._expand_paste_references(existing)
            if expanded != existing and hasattr(buffer, "text"):
                self._skip_paste_collapse = True
                buffer.text = expanded
                if hasattr(buffer, "cursor_position"):
                    buffer.cursor_position = len(expanded)
        except Exception:
            logger.debug("Failed to inline paste placeholders", exc_info=True)

    def _reset_input_buffer(self, buffer) -> None:
        """Clear an input buffer after a programmatic submit (best-effort)."""
        try:
            buffer.reset(append_to_history=True)
        except Exception:
            try:
                buffer.text = ""
            except Exception:
                pass

    def _prefill_input_buffer(self, text: str) -> None:
        """Place ``text`` in the active prompt_toolkit buffer, editable."""
        from cli import logger
        app = getattr(self, "_app", None)
        if app is None:
            return
        try:
            buf = app.current_buffer
            buf.text = text
            if hasattr(buf, "cursor_position"):
                buf.cursor_position = len(text)
            app.invalidate()
        except Exception as e:
            logger.debug("undo: prefill buffer failed: %s", e)

    def _prompt_text_input(self, prompt_text: str) -> str | None:
        """Prompt for free-text input safely inside or outside prompt_toolkit.

        ``run_in_terminal`` only works on the main-thread loop; on the ``process_loop`` daemon
        thread a bare ``input()`` would block forever on loop-owned stdin, so with an app running
        off-main we cancel cleanly (None) — mirroring ``_stdin_fallback`` in the modal prompt.

        Mirrors the thread-aware guard in ``_run_curses_picker``: ``run_in_terminal`` returns a coroutine
        that must be awaited by the prompt_toolkit event loop, which only exists on the main thread. Slash
        commands are dispatched from the ``process_loop`` daemon thread (see issue #23185), so calling
        ``run_in_terminal`` from there orphans the coroutine — ``_ask`` never runs, and user keystrokes leak
        into the composer instead. Fall back to a direct ``input()`` when we're off the main thread.
        """
        result = [None]

        def _ask():
            try:
                result[0] = input(prompt_text).strip() or None
            except (KeyboardInterrupt, EOFError):
                pass

        in_main_thread = threading.current_thread() is threading.main_thread()
        # Slash-worker guard (#23185 / billing auto-reload hang): when a prompt_toolkit app is running but
        # we're on a non-main thread (the process_loop / TUI slash-worker daemon thread), stdin is owned by
        # the event loop / JSON-RPC pipe. A bare input() there blocks forever until the worker's 45s timeout
        # fires. We cannot safely prompt off the main thread, so cancel cleanly (None) instead of hanging —
        # mirrors the _stdin_fallback discipline in _prompt_text_input_modal.
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
                # WSL / Warp / some emulators silently drop the scheduled coroutine — fall back to
                # a direct input() so keystrokes don't leak into the agent buffer.
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

    def _poll_modal_queue(self, response_queue, deadline_attr, *, refresh=1.0, paint=None):
        """Block until a value lands on ``response_queue`` or ``self.<deadline_attr>`` passes
        (``None`` deadline = unlimited). Returns the value or ``_TIMED_OUT``.

        Repaints every ``refresh`` seconds (``0`` = on every idle tick) so countdown hints stay
        live; ``paint`` defaults to ``_paint_now`` — modal prompts must bypass the ``_invalidate``
        throttle/resize guard or the panel can be dropped and time out unseen.
        """
        paint = paint or self._paint_now
        last = _time.monotonic()
        while True:
            try:
                return response_queue.get(timeout=1)
            except queue.Empty:
                deadline = getattr(self, deadline_attr)
                if deadline is not None and deadline - _time.monotonic() <= 0:
                    return _TIMED_OUT
                now = _time.monotonic()
                if now - last >= refresh:
                    last = now
                    paint()

    def _prompt_text_input_modal(
        self, *, title: str, detail: str, choices: list[tuple[str, str, str]], timeout: float = 120
    ) -> str | None:
        """Slash-command confirmation through the prompt_toolkit composer (raw input() fought
        prompt_toolkit's stdin ownership: prompt above the TUI, Enter read as EOF). All platforms
        drive the modal via ``self._app.loop`` + ``call_soon_threadsafe``; raw ``input()`` is kept
        only for the safe cases (no app, no loop, scheduling failure) — on Windows a non-main-thread
        input() deadlocks against prompt_toolkit, so that case cancels instead.

        **Platform note (Windows — issue #33961):** Earlier code bypassed the modal on ``sys.platform ==
        "win32"`` and fell back to a raw ``input()`` prompt. When the confirm was triggered from the
        ``process_loop`` daemon thread (the normal case) that ``input()`` ran off the main thread and
        deadlocked against prompt_toolkit's stdin ownership — the user saw a frozen cursor and Ctrl-C was
        swallowed (bare ``/reset`` froze; ``/reset now`` worked only because it skips the prompt entirely).
        """
        if not choices:
            return None
        if not getattr(self, "_app", None):
            return self._prompt_text_input("Choice [1/2/3]: ")

        try:
            app_loop = self._app.loop
        except Exception:
            app_loop = None
        in_main_thread = threading.current_thread() is threading.main_thread()

        def _stdin_fallback() -> str | None:
            # On native Windows a raw input() from a non-main thread deadlocks against prompt_toolkit's
            # stdin ownership (#33961). With an app running we cannot safely prompt off the main thread, so
            # cancel cleanly (None) rather than hang the terminal.
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
                "response_queue": response_queue}
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
        try:
            result = self._poll_modal_queue(
                response_queue, "_slash_confirm_deadline", refresh=5.0, paint=self._invalidate)
            if result is not _TIMED_OUT:
                _run_on_app_loop(_teardown_modal)
                return result
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
        self, raw: str | None, choices: list[tuple[str, str, str]]) -> str | None:
        if raw is None:
            return None
        choice_raw = raw.strip().lower()
        if not choice_raw:
            return None
        allowed = {choice[0] for choice in choices}
        normalized = _CONFIRM_ALIASES.get(choice_raw)
        if normalized in allowed:
            return normalized
        if choice_raw in allowed:
            return choice_raw
        return None

    def _build_command_palette_entries(self) -> list:
        """Flat (command, category, desc) rows for the Ctrl+P palette: the COMMAND_REGISTRY behind
        /help filtered to this surface, plus installed skill commands. Selecting inserts the exact
        command string — never a fuzzy resolution."""
        from cli import _ensure_skill_commands
        from hermes_cli.commands import COMMANDS_BY_CATEGORY

        entries: list[tuple[str, str, str]] = []
        for category, commands in COMMANDS_BY_CATEGORY.items():
            for cmd, desc in commands.items():
                if self._command_available(cmd):
                    entries.append((cmd, category, desc))
        try:
            for cmd, info in sorted(_ensure_skill_commands().items()):
                entries.append((cmd, "Skill", info.get("description", "")))
        except Exception:
            pass
        return entries

    def _open_command_palette(self) -> None:
        """Open the Ctrl+P fuzzy command palette modal (never stacked over another modal)."""
        if getattr(self, "_command_palette_state", None):
            return
        if (self._model_picker_state or self._clarify_state or self._approval_state
                or self._slash_confirm_state or self._sudo_state or self._secret_state):
            return
        self._capture_modal_input_snapshot()
        self._command_palette_state = {
            "entries": self._build_command_palette_entries(),
            "filter": "",
            "selected": 0,
            "_scroll_offset": 0}
        self._invalidate(min_interval=0.0)

    def _close_command_palette(self) -> None:
        self._command_palette_state = None
        self._restore_modal_input_snapshot()
        self._invalidate(min_interval=0.0)

    def _command_palette_visible_entries(self) -> list:
        """Rows matching the active filter, ranked command-name-first (a bare subsequence over
        "cmd category desc" is uselessly permissive — "steer" would match 130+ rows via text):
        0 exact command, 1 command startswith, 2 substring in command, 3 subsequence in command,
        4 substring in description. Non-matches are dropped; ties keep registry order."""
        state = self._command_palette_state or {}
        entries = state.get("entries") or []
        q = (state.get("filter", "") or "").strip().lower()
        if not q:
            return list(entries)

        def _subseq(needle: str, hay: str) -> bool:
            it = iter(hay)
            return all(ch in it for ch in needle)

        qn = q.lstrip("/")
        ranked = []
        for order, row in enumerate(entries):
            cmd, _cat, desc = row
            name = cmd.lower().lstrip("/")
            desc_l = (desc or "").lower()
            if name == qn:
                rank = 0
            elif name.startswith(qn):
                rank = 1
            elif qn in name:
                rank = 2
            elif _subseq(qn, name):
                rank = 3
            elif q in desc_l:
                rank = 4
            else:
                continue
            ranked.append((rank, order, row))
        ranked.sort(key=lambda t: (t[0], t[1]))
        return [row for (_r, _o, row) in ranked]

    def _handle_command_palette_selection(self) -> None:
        """Prefill the selected command into the composer — never auto-run (many take args)."""
        from cli import logger
        state = self._command_palette_state
        if not state:
            return
        rows = self._command_palette_visible_entries()
        selected = state.get("selected", 0)
        if not (0 <= selected < len(rows)):
            self._close_command_palette()
            return
        cmd = rows[selected][0]
        self._close_command_palette()
        try:
            app = getattr(self, "_app", None)
            if app is not None:
                buf = app.current_buffer
                buf.text = cmd + " "
                buf.cursor_position = len(buf.text)
                self._invalidate(min_interval=0.0)
        except Exception:
            logger.debug("command palette prefill failed", exc_info=True)

    @classmethod
    def _split_destructive_skip(cls, cmd_text: Optional[str]) -> tuple[str, bool]:
        """Split inline-skip tokens out of a destructive slash command → ``(remainder, skip)``.

        ``remainder`` is the text minus the leading "/cmd" word and any skip tokens; ``skip`` is
        True iff one was found: "/reset now" -> ("", True); "/reset --yes My title" ->
        ("My title", True); "/new My title" -> ("My title", False).
        """
        tokens = (cmd_text or "").strip().split()
        if not tokens:
            return "", False
        if tokens[0].startswith("/"):
            tokens = tokens[1:]
        kept = [tok for tok in tokens if tok.lower() not in cls._DESTRUCTIVE_SKIP_TOKENS]
        return " ".join(kept), len(kept) != len(tokens)

    def _confirm_destructive_slash(
        self, command: str, detail: str, cmd_original: Optional[str] = None) -> Optional[str]:
        """Confirm a destructive slash command (``/clear``, ``/new``/``/reset``, ``/undo``): returns
        ``"once"``, ``"always"`` (persists the opt-out) or ``None`` (cancelled). Gate off → "once"
        silently; ``now`` / ``--yes`` / ``-y`` in ``cmd_original`` bypasses the modal (callers strip
        the tokens via :meth:`_split_destructive_skip`).

        Inline-skip: if ``cmd_original`` contains ``now``, ``--yes``, or ``-y`` as an argument (e.g.
        ``/reset now``, ``/new --yes My title``), the modal is bypassed and ``"once"`` is returned
        immediately. This is an escape hatch for non-interactive use and for the degraded path where the
        modal can't be marshaled onto the app loop (native Windows itself now drives the modal normally —
        see #33961). Callers are responsible for stripping the skip tokens from any remaining argument
        parsing (see :meth:`_split_destructive_skip`).
        """
        if cmd_original and self._split_destructive_skip(cmd_original)[1]:
            return "once"
        return _gated_confirm(
            self, command, "destructive_slash_confirm",
            title=f"⚠️  /{command} — destroys conversation state",
            detail=detail,
            choices=[
                ("once", "Approve Once", "proceed this time only"),
                ("always", "Always Approve", "proceed and silence this prompt permanently"),
                ("cancel", "Cancel", "keep current conversation")],
            unchanged="Conversation unchanged.",
            always_msg="🔒 Future /clear, /new, /reset, and /undo will run without confirmation.",
            once_verb="proceeding")

    def _ring_bell(self, prompt: bool = False, context: str = "", detail: str = "") -> None:
        """Terminal bell (\\a) gated by ``display.bell_on_prompt`` (``prompt=True``, blocking modals)
        or ``display.bell_on_complete`` (end of turn); works over SSH. The same flag also emits the
        OSC 9 / Warp OSC 777 desktop notification; ``context`` is the short notification body."""
        flag = "bell_on_prompt" if prompt else "bell_on_complete"
        if not getattr(self, flag, False):
            return
        try:
            sys.stdout.write("\a")
            sys.stdout.flush()
        except Exception:
            pass
        try:
            from hermes_cli.terminal_notify import notify as _terminal_notify
            _terminal_notify(
                context or ("input needed" if prompt else "turn complete"),
                prompt=prompt,
                session_id=getattr(self, "session_id", "") or "",
                detail=detail)
        except Exception:
            pass

    def _clarify_teardown(self) -> None:
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_deadline = None
        self._clarify_multi_base = None
        self._paint_now()

    def _clarify_callback(self, question, choices, multi_select=False, questions=None):
        """Clarify-tool platform callback (agent thread): show the selection UI (or freetext for
        open-ended questions) and block until the key bindings answer or the timeout dismisses it
        (the agent is then told to decide). ``multi_select`` shows checkboxes (Space toggles).
        A non-empty ``questions`` list switches to the batch panel and returns
        ``{"answers": {qid: raw}}`` (plus ``"timed_out": True`` on a partial deadline expiry).

        The single-question path below is unchanged. See #18450.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        from tools.clarify_gateway import resolve_clarify_timeout

        if questions:
            return self._clarify_callback_batch(questions)

        # Canonical clarify timeout, shared with the gateway/TUI path; `<= 0` = unlimited.
        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()
        is_open_ended = not choices
        effective_multi = multi_select and not is_open_ended
        self._clarify_state = {
            "question": question,
            "choices": choices if not is_open_ended else [],
            "selected": 0,
            "multi_select": effective_multi,
            "selected_indices": set() if effective_multi else None,
            "response_queue": response_queue}
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        self._clarify_freetext = is_open_ended  # open-ended → straight to freetext
        self._clarify_multi_base = None
        self._ring_bell(prompt=True, context="clarify")
        self._paint_now()

        result = self._poll_modal_queue(response_queue, "_clarify_deadline")
        if result is not _TIMED_OUT:
            self._clarify_deadline = None
            self._persist_prompt_summary("?", "Clarify", question, str(result))
            return result
        self._clarify_teardown()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — agent will decide){_RST}")
        return _CLARIFY_TIMEOUT_REPLY

    # --- Batch clarify (multi-question, issue #18450) -----------------------
    def _clarify_batch_set_active(self, state, index) -> None:
        """Point the batch clarify panel at question ``index``: mirror it into the flat keys the
        single-question keybindings/renderer read so ↑/↓/Space/number keys work unchanged;
        open-ended drops into freetext; re-visiting restores the earlier cursor/checkboxes."""
        questions_list = state["questions"]
        index = max(0, min(index, len(questions_list) - 1))
        entry = questions_list[index]
        choices = entry["choices"] or []
        state["active"] = index
        state["question"] = entry["question"]
        state["choices"] = choices
        state["selected"] = 0
        state["multi_select"] = bool(entry["multi_select"])
        state["selected_indices"] = set() if entry["multi_select"] else None
        self._clarify_freetext = not entry["choices"]
        self._clarify_multi_base = None
        meta = (state.get("answer_meta") or {}).get(entry["qid"])
        if meta is None:
            return
        kind = meta.get("kind")
        if kind == "choice":
            answer = state["answers"].get(entry["qid"])
            if answer in choices:
                state["selected"] = choices.index(answer)
        elif kind == "other":
            state["selected"] = len(choices)
        elif kind == "multi":
            checked = {choices.index(c) for c in meta.get("choices") or [] if c in choices}
            if meta.get("other_text"):
                checked.add(len(choices))
            state["selected_indices"] = checked

    def _clarify_batch_lock(self, state, answer, meta=None) -> None:
        """Lock ``answer`` for the active batch question (overwriting an earlier one) and advance to
        the next unanswered; ``meta`` ({"kind": "choice"|"other"|"multi", ...}) lets a re-visit
        restore the cursor / prefill an "Other" edit. All answered → resolve the queue, tear down."""
        entry = state["questions"][state["active"]]
        state["answers"][entry["qid"]] = answer
        state.setdefault("answer_meta", {})[entry["qid"]] = meta or {"kind": "choice"}
        self._persist_prompt_summary("?", "Clarify", entry["question"], str(answer))
        total = len(state["questions"])
        for offset in range(1, total + 1):
            candidate = (state["active"] + offset) % total
            if state["questions"][candidate]["qid"] not in state["answers"]:
                self._clarify_batch_set_active(state, candidate)
                return
        try:
            state["response_queue"].put(dict(state["answers"]))
        except Exception:
            pass
        self._clarify_state = None
        self._clarify_freetext = False
        self._clarify_multi_base = None

    def _clarify_batch_enter(self, state) -> None:
        """Enter in batch choice mode: lock the active selection. Multi-select locks a JSON array of
        checked labels (parsed by the tool core); "Other" switches to freetext, prefilled with an
        earlier typed answer so Enter on an answered Other edits instead of retyping."""
        choices = state.get("choices") or []
        selected = state.get("selected", 0)
        entry = state["questions"][state["active"]]
        meta = (state.get("answer_meta") or {}).get(entry["qid"]) or {}
        if state.get("multi_select"):
            sorted_idx = sorted(state.get("selected_indices") or set())
            selected_choices = [choices[i] for i in sorted_idx if i < len(choices)]
            if len(choices) in sorted_idx:
                # Stash the checked real choices so the freetext submit appends the typed answer.
                self._clarify_multi_base = selected_choices
                self._clarify_freetext = True
                self._clarify_prefill = meta.get("other_text") or ""
                return
            self._clarify_batch_lock(
                state,
                json.dumps(selected_choices, ensure_ascii=False),
                meta={"kind": "multi", "choices": selected_choices, "other_text": ""})
            return
        if selected < len(choices):
            self._clarify_batch_lock(state, choices[selected], meta={"kind": "choice"})
            return
        self._clarify_freetext = True
        self._clarify_prefill = meta.get("other_text") or "" if meta.get("kind") == "other" else ""

    def _clarify_callback_batch(self, questions):
        """Batch clarify panel (A-compact): all questions, one active. Returns
        ``{"answers": {qid: raw}}`` when every question is locked, plus ``"timed_out": True`` when
        the deadline expires with partial answers; a cancel string passes through unchanged so the
        tool core resolves the batch empty."""
        from cli import CLI_CONFIG, _DIM, _RST, _cprint
        from tools.clarify_gateway import resolve_clarify_timeout

        timeout = resolve_clarify_timeout(CLI_CONFIG)
        response_queue = queue.Queue()
        state = {
            "questions": list(questions),
            "answers": {},
            "answer_meta": {},
            "active": 0,
            "response_queue": response_queue,
            # Flat keys mirroring the active question — filled by _clarify_batch_set_active.
            "question": "",
            "choices": [],
            "selected": 0,
            "multi_select": False,
            "selected_indices": None}
        self._clarify_state = state
        self._clarify_batch_set_active(state, 0)
        self._clarify_deadline = None if timeout <= 0 else _time.monotonic() + timeout
        self._ring_bell(prompt=True, context="clarify")
        self._paint_now()

        result = self._poll_modal_queue(response_queue, "_clarify_deadline")
        if result is not _TIMED_OUT:
            self._clarify_deadline = None
            return {"answers": result} if isinstance(result, dict) else result
        partial = dict(state["answers"])
        self._clarify_teardown()
        _cprint(f"\n{_DIM}(clarify timed out after {timeout}s — locked answers returned){_RST}")
        return {"answers": partial, "timed_out": True}

    def _sudo_password_callback(self) -> str:
        """Prompt for a sudo password through the prompt_toolkit UI (agent thread); clarify-style
        state + queue answered by the Enter binding."""
        from cli import _DIM, _RST, _cprint

        response_queue = queue.Queue()
        self._capture_modal_input_snapshot()
        self._sudo_state = {"response_queue": response_queue}
        self._sudo_deadline = _time.monotonic() + 45
        self._ring_bell(prompt=True, context="sudo password")
        self._paint_now()

        result = self._poll_modal_queue(response_queue, "_sudo_deadline", refresh=0)
        self._sudo_state = None
        self._sudo_deadline = 0
        self._restore_modal_input_snapshot()
        self._paint_now()
        if result is _TIMED_OUT:
            _cprint(f"\n{_DIM}  ⏱ Timeout — continuing without sudo{_RST}")
            return ""
        if result:
            _cprint(f"\n{_DIM}  ✓ Password received (cached for session){_RST}")
        else:
            _cprint(f"\n{_DIM}  ⏭ Skipped{_RST}")
        return result

    def _approval_callback(self, command: str, description: str,
                           *, allow_permanent: bool = True,
                           allow_session: bool = True,
                           smart_denied: bool = False) -> str:
        """Dangerous-command approval through the prompt_toolkit UI (agent thread).

        Choices: once / session / always / deny (see ``_approval_choices``), plus 'view' for long
        commands. ``_approval_lock`` serializes concurrent requests (parallel delegation subtasks)
        so the shared ``_approval_state`` / ``_approval_deadline`` aren't clobbered.
        """
        from cli import CLI_CONFIG, _DIM, _RST, _cprint

        with self._approval_lock:
            timeout = int(CLI_CONFIG.get("approvals", {}).get("timeout", 300))
            response_queue = queue.Queue()
            self._approval_state = {
                "command": command,
                "description": description,
                "choices": self._approval_choices(
                    command,
                    allow_permanent=allow_permanent,
                    allow_session=allow_session,
                    smart_denied=smart_denied),
                "selected": 0,
                "response_queue": response_queue}
            self._approval_deadline = _time.monotonic() + timeout
            self._ring_bell(prompt=True, context="approval", detail=command)
            self._paint_now()

            result = self._poll_modal_queue(response_queue, "_approval_deadline")
            self._approval_state = None
            self._approval_deadline = 0
            self._paint_now()
            if result is _TIMED_OUT:
                _cprint(f"\n{_DIM}  ⏱ Timeout — denying command{_RST}")
                self._persist_prompt_summary("⚠", "Approval", command, "timed out (no response)")
                return "timeout"
            self._persist_prompt_summary(
                "⚠", "Approval", command, _APPROVAL_OUTCOME_LABELS.get(result, str(result)))
            return result

    def _approval_choices(self, command: str, *, allow_permanent: bool = True,
                          allow_session: bool = True,
                          smart_denied: bool = False) -> list[str]:
        """Smart-DENY overrides and re-ask-every-time gates (allow_session=False) show only
        once/deny; ``allow_permanent=False`` for another reason (e.g. tirith) hides only 'always'."""
        if smart_denied or not allow_session:
            choices = ["once", "deny"]
        elif allow_permanent:
            choices = ["once", "session", "always", "deny"]
        else:
            choices = ["once", "session", "deny"]
        if len(command) > 70:
            choices.append("view")
        return choices

    def _computer_use_approval_callback(self, action: str, args: dict, summary: str) -> str:
        """Adapt the generic approval UI (once/session/always/deny) to the computer_use verdicts
        (approve_once/approve_session/always_approve/deny)."""
        verdict = self._approval_callback(
            command=f"computer_use: {summary}",
            description=f"Allow computer_use to perform `{action}`?")
        return {
            "once": "approve_once",
            "session": "approve_session",
            "always": "always_approve",
            "deny": "deny",
            "timeout": "timeout",
        }.get(verdict, "deny")

    def _handle_approval_selection(self) -> None:
        """Process the currently selected dangerous-command approval choice."""
        state = self._approval_state
        if not state:
            return
        selected = state.get("selected", 0)
        choices = state.get("choices")
        if not isinstance(choices, list):
            choices = []
        if not (0 <= selected < len(choices)):
            return
        chosen = choices[selected]
        if chosen == "view":
            state["show_full"] = True
            state["choices"] = [choice for choice in choices if choice != "view"]
            if state["selected"] >= len(state["choices"]):
                state["selected"] = max(0, len(state["choices"]) - 1)
            self._invalidate()
            return
        state["response_queue"].put(chosen)
        self._approval_state = None
        self._invalidate()

    def _secret_capture_callback(self, var_name: str, prompt: str, metadata=None) -> dict:
        return prompt_for_secret(self, var_name, prompt, metadata)

    def _capture_modal_input_snapshot(self) -> None:
        """Temporarily clear the input buffer and save the user's in-progress draft."""
        if self._modal_input_snapshot is not None or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            self._modal_input_snapshot = {"text": buf.text, "cursor_position": buf.cursor_position}
            buf.reset()
        except Exception:
            self._modal_input_snapshot = None

    def _restore_modal_input_snapshot(self) -> None:
        """Restore any draft text that was present before a modal prompt opened."""
        snapshot = self._modal_input_snapshot
        self._modal_input_snapshot = None
        if not snapshot or not getattr(self, "_app", None):
            return
        try:
            buf = self._app.current_buffer
            buf.text = snapshot.get("text", "")
            buf.cursor_position = min(snapshot.get("cursor_position", 0), len(buf.text))
        except Exception:
            pass

    def _clear_active_overlays_for_interrupt(self) -> None:
        """Drain and clear every input-blocking overlay left by an interrupted agent: the worker
        thread is gone but the state dict still gates input (frozen terminal until its timeout).
        Push a safe value onto each queue (approval -> "deny", others -> cancel), nil the state,
        restore the draft; each step is wrapped so a dead queue can't block the others."""
        def _put(state, value) -> None:
            try:
                state["response_queue"].put(value)
            except Exception:
                pass

        if self._approval_state:
            _put(self._approval_state, "deny")
            self._approval_state = None
        if self._clarify_state:
            _put(self._clarify_state, "The user cancelled. Use your best judgement to proceed.")
            self._clarify_state = None
            self._clarify_freetext = False
            self._clarify_multi_base = None
        if self._sudo_state:
            _put(self._sudo_state, "")
            self._sudo_state = None
            self._sudo_deadline = 0
            self._restore_modal_input_snapshot()
        if self._secret_state:
            try:
                self._cancel_secret_capture()
            except Exception:
                self._secret_state = None

    def _submit_secret_response(self, value: str) -> None:
        if not self._secret_state:
            return
        self._secret_state["response_queue"].put(value)
        self._secret_state = None
        self._secret_deadline = 0
        self._paint_now()  # direct paint so the secret panel clears at once (no throttle)

    def _cancel_secret_capture(self) -> None:
        self._submit_secret_response("")

    def _clear_secret_input_buffer(self) -> None:
        if getattr(self, "_app", None):
            try:
                self._app.current_buffer.reset()
            except Exception:
                pass
