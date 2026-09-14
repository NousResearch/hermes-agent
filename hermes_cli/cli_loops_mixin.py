"""Simple slash-command wrappers plus goal/heartbeat/loop manager hooks for the interactive CLI.
Mixin on ``HermesCLI``; cli.py symbols are imported lazily inside methods (import cycle)."""

from __future__ import annotations

import logging
import os
import shutil
import threading
import time

from rich.markup import escape as _escape

_FRESH_START = "  ✨ (◕‿◕)✨ Fresh start! Screen cleared and conversation reset.\n"


def _preview(payload: str) -> str:
    return f"{payload[:80]}{'...' if len(payload) > 80 else ''}"


def _print_decision_message(decision: dict) -> bool:
    """Print a manager decision's ``message`` (if any) via _cprint; True when one was printed."""
    from cli import _cprint
    msg = decision.get("message") or ""
    if msg:
        _cprint(f"  {msg}")
    return bool(msg)


class CLILoopsMixin:
    """Simple slash-command wrappers plus goal/heartbeat/loop manager hooks for the interactive CLI"""

    def _cmd_exit(self, cmd_original: str):
        # /exit --delete also removes the session's transcripts + SQLite history.
        from cli import _DIM, _RST, _cprint, _slash_args
        _args = _slash_args(cmd_original).lower()
        # Ported from google-gemini/gemini-cli#19332.
        if _args in {"--delete", "-d"}:
            self._delete_session_on_exit = True
        elif _args:
            _cprint(f"  {_DIM}✗ Unknown argument: {_escape(_args)}. Use /exit --delete to also remove session history.{_RST}")
            return True
        return False

    def _cmd_help(self, cmd_original: str):
        from cli import _slash_args
        self.show_help(_slash_args(cmd_original))

    def _cmd_redraw(self, cmd_original: str):
        # Manual recovery for terminal buffer drift (mux tab switches, subshell ``clear``,
        # SSH restores; #8688). Ctrl+L is bound to the same helper.
        from cli import _DIM, _RST, _cprint
        self._force_full_redraw()
        _cprint(f"  {_DIM}✓ UI redrawn{_RST}")

    def _cmd_clear(self, cmd_original: str):
        from cli import ChatConsole, _build_compact_banner, _clear_output_history, _cprint, get_tool_definitions
        from hermes_cli.banner import build_welcome_banner
        if self._confirm_destructive_slash(
            "clear",
            "This clears the screen and starts a new session.\n"
            "The current conversation history will be discarded.",
            cmd_original=cmd_original,
        ) is None:
            return True  # confirmation cancelled — command handled, keep REPL alive
        self.new_session(silent=True)
        _clear_output_history()
        if not self._app:
            self.console.clear()
            self.show_banner()
            print(_FRESH_START)
            self._print_random_tip()
            return
        # Inside the TUI, Rich's console.clear() and self.console both go through
        # patch_stdout's StdoutProxy, which swallows the clear escapes / mangles raw
        # output: clear via prompt_toolkit's output and print through ChatConsole.
        out = self._app.output
        out.erase_screen()
        out.cursor_goto(0, 0)
        out.flush()
        cc = ChatConsole()
        if self.compact or shutil.get_terminal_size().columns < 80:
            cc.print(_build_compact_banner())
        else:
            tools = get_tool_definitions(enabled_toolsets=self.enabled_toolsets,
                                         disabled_toolsets=self.disabled_toolsets, quiet_mode=True)
            agent = getattr(self, "agent", None)
            ctx_len = None
            if agent and hasattr(agent, "context_compressor"):
                ctx_len = agent.context_compressor.context_length
            build_welcome_banner(
                console=cc, model=self.model, cwd=os.getenv("TERMINAL_CWD", os.getcwd()),
                tools=tools, enabled_toolsets=self.enabled_toolsets, session_id=self.session_id,
                context_length=ctx_len, provider=self.provider)
        _cprint(_FRESH_START)
        self._print_random_tip()

    def _cmd_title(self, cmd_original: str):
        from cli import _cprint
        from hermes_state import format_session_db_unavailable
        parts = cmd_original.split(maxsplit=1)
        if len(parts) == 1:
            # No argument: show current title and session ID.
            if not self._session_db:
                _cprint(f"  {format_session_db_unavailable()}")
                return
            _cprint(f"  Session ID: {self.session_id}")
            session = self._session_db.get_session(self.session_id)
            if session and session.get("title"):
                _cprint(f"  Title: {session['title']}")
            elif self._pending_title:
                _cprint(f"  Title (pending): {self._pending_title}")
            else:
                _cprint("  No title set. Usage: /title <your session title>")
            return
        raw_title = parts[1].strip()
        if not raw_title:
            _cprint("  Usage: /title <your session title>")
            return
        if not self._session_db:
            _cprint(f"  {format_session_db_unavailable()}")
            return
        # Sanitize early so feedback matches what gets stored. A rejection (e.g. too
        # long) prints that one reason and stops — never a second, contradictory
        # "empty after cleanup" error (SC-05).
        try:
            from hermes_state import SessionDB
            new_title = SessionDB.sanitize_title(raw_title)
        except ValueError as e:
            _cprint(f"  {e}")
            return True
        if not new_title:
            _cprint("  Title is empty after cleanup. Please use printable characters.")
        elif self._session_db.get_session(self.session_id):
            try:
                if self._session_db.set_session_title(self.session_id, new_title):
                    self._status_bar_title_checked_at = 0.0
                    _cprint(f"  Session title set: {new_title}")
                else:
                    _cprint("  Session not found in database.")
            except ValueError as e:
                _cprint(f"  {e}")
        else:
            # Session not created yet — check uniqueness now, defer the title.
            existing = self._session_db.get_session_by_title(new_title)
            if existing:
                _cprint(f"  Title '{new_title}' is already in use by session {existing['id']}")
            else:
                self._pending_title = new_title
                _cprint(f"  Session title queued: {new_title} (will be saved on first message)")

    def _cmd_new(self, cmd_original: str):
        # Strip inline-skip tokens (now/--yes/-y) before deriving the title so
        # "/new now My Session" yields title="My Session". See _split_destructive_skip.
        _new_args, _ = self._split_destructive_skip(cmd_original)
        title = _new_args.strip() or None
        if self._confirm_destructive_slash(
            "new",
            "This starts a fresh session.\n"
            "The current conversation history will be discarded.",
            cmd_original=cmd_original,
        ) is None:
            return True  # confirmation cancelled — command handled, keep REPL alive
        self.new_session(title=title)

    def _cmd_retry(self, cmd_original: str):
        retry_msg = self.retry_last()
        if retry_msg and hasattr(self, '_pending_input'):
            self._pending_input.put(retry_msg)  # process_loop sends it to the agent

    def _cmd_undo(self, cmd_original: str):
        # "/undo" → 1, "/undo 3" → 3.
        _undo_n = 1
        _undo_parts = cmd_original.split()
        if len(_undo_parts) > 1:
            try:
                _undo_n = max(1, int(_undo_parts[1]))
            except ValueError:
                print(f"(._.) Invalid count {_undo_parts[1]!r} — use /undo or /undo N.")
                return True  # bad arg — command handled, keep the REPL alive
        # Nothing to undo → say so; no destructive confirmation for a no-op (SC-06).
        if not self.conversation_history:
            print("(._.) No messages to undo.")
            return True
        _undo_desc = (
            "This removes the last user/assistant exchange from history."
            if _undo_n == 1
            else f"This removes the last {_undo_n} user turns from history.")
        if self._confirm_destructive_slash("undo", _undo_desc, cmd_original=cmd_original) is None:
            return True  # confirmation cancelled — command handled, keep REPL alive
        self.undo_last(_undo_n)

    def _cmd_skills(self, cmd_original: str):
        with self._busy_command(self._slow_command_status(cmd_original)):
            self._handle_skills_command(cmd_original)

    def _cmd_egress(self, cmd_original: str):
        from hermes_cli.slash_exec import CommandContext, execute_command
        text = execute_command("egress", CommandContext(surface="cli")).text
        self._console_print(text, highlight=False, markup=False)

    def _cmd_statusbar(self, cmd_original: str):
        self._status_bar_visible = not self._status_bar_visible
        self._console_print(f"  Status bar {'visible' if self._status_bar_visible else 'hidden'}")

    def _cmd_update(self, cmd_original: str) -> bool:
        # A truthy result means the process is relaunching — leave the REPL.
        return not self._handle_update_command()

    def _cmd_version(self, cmd_original: str):
        from hermes_cli.main import _print_version_info
        _print_version_info(check_updates=True)

    def _cmd_reload(self, cmd_original: str):
        from hermes_cli.config import reload_env
        count = reload_env()
        print(f"  Reloaded .env ({count} var(s) updated)")

    def _cmd_reload_skills(self, cmd_original: str):
        with self._busy_command(self._slow_command_status(cmd_original)):
            self._reload_skills()

    def _cmd_plugins(self, cmd_original: str):
        from hermes_constants import display_hermes_home
        try:
            # Discover from disk (bundled + user) like `hermes plugins list`, so
            # installed-but-not-enabled plugins show up; the plugin manager only knows
            # *loaded* plugins and made fresh installs look like "nothing installed".
            from hermes_cli.plugins_cmd import (
                _discover_all_plugins, _get_disabled_set, _get_enabled_set, _plugin_status)
            entries = _discover_all_plugins()
            enabled = _get_enabled_set()
            disabled = _get_disabled_set()

            # `/plugins` is a quick glance: user plugins only, bundled ones summarized
            # on one line (full catalog behind `hermes plugins list`).
            user_entries = [e for e in entries if e[3] != "bundled"]
            bundled_count = len(entries) - len(user_entries)
            if not user_entries:
                print("No user plugins installed.")
                print("  Install one: hermes plugins install owner/repo")
                print(f"  Or drop a plugin directory into {display_hermes_home()}/plugins/")
                if bundled_count:
                    print(f"  ({bundled_count} bundled plugins available — see: hermes plugins list)")
                return
            try:  # loaded-plugin details (tools/hooks/commands counts, errors) by name
                from hermes_cli.plugins import get_plugin_manager
                loaded = {p["name"]: p for p in get_plugin_manager().list_plugins()}
            except Exception:
                loaded = {}
            print(f"User plugins ({len(user_entries)}):")
            for name, version, _desc, source, _dir, key in sorted(user_entries):
                state = _plugin_status(name, enabled, disabled, key=key)
                info = loaded.get(name) or {}
                bits = [f"{info[k]} {k}" for k in ("tools", "hooks", "commands") if info.get(k)]
                glyph = {"enabled": "✓", "disabled": "✗"}.get(state, "○")
                ver = f" v{version}" if version else ""
                detail = f" ({', '.join(bits)})" if bits else ""
                label = "" if state == "enabled" else f" [{state}]"
                error = f" — {info['error']}" if info.get("error") else ""
                print(f"  {glyph} {name}{ver}{label}{detail}{error}")
            if bundled_count:
                print(f"  (+{bundled_count} bundled — see: hermes plugins list)")
            print("  Enable/disable: hermes plugins enable/disable <name>")
        except Exception as e:
            print(f"Plugin system error: {e}")

    def _cmd_queue(self, cmd_original: str):
        from cli import _cprint, _slash_args
        payload = self._expand_paste_references(_slash_args(cmd_original))
        if not payload:
            _cprint("  Usage: /queue <prompt>")
        else:
            self._pending_input.put(payload)
            when = " for the next turn" if self._agent_running else ""
            _cprint(f"  Queued{when}: {_preview(payload)}")

    def _cmd_steer(self, cmd_original: str):
        # Inject a message after the next tool call without interrupting: while the
        # agent runs, push into its pending_steer slot (drained by _execute_tool_calls_*
        # into the next tool result); otherwise fall back to /queue semantics.
        from cli import _cprint, _slash_args
        payload = _slash_args(cmd_original)
        if not payload:
            _cprint("  Usage: /steer <prompt>")
        elif self._agent_running and self.agent is not None and hasattr(self.agent, "steer"):
            try:
                accepted = self.agent.steer(payload)
            except Exception as exc:
                _cprint(f"  Steer failed: {exc}")
            else:
                if accepted:
                    _cprint(f"  ⏩ Steer queued — arrives after the next tool call: {_preview(payload)}")
                else:
                    _cprint("  Steer rejected (empty payload).")
        else:
            self._pending_input.put(payload)
            _cprint(f"  No agent running; queued as next turn: {_preview(payload)}")

    # ────────────────────────────────────────────────────────────────
    # Session-bound managers: /goal (Ralph-style loop), /heartbeat, /loop
    # ────────────────────────────────────────────────────────────────
    def _session_bound_manager(self, attr: str, label: str, load):
        """Return the manager cached on ``self.<attr>``, rebuilt when ``session_id`` changed
        (after /new or a compression-driven session split).

        ``load()`` does the imports and returns a ``sid -> manager`` factory; an import
        failure is logged and yields None, as does an empty session_id.
        """
        try:
            make = load()
        except Exception as exc:
            logging.debug("%s unavailable: %s", label, exc)
            return None
        sid = getattr(self, "session_id", None) or ""
        if not sid:
            return None
        existing = getattr(self, attr, None)
        if existing is not None and getattr(existing, "session_id", None) == sid:
            return existing
        mgr = make(sid)
        setattr(self, attr, mgr)
        return mgr

    def _get_goal_manager(self):
        """GoalManager bound to the current session_id (see ``_session_bound_manager``)."""
        def load():
            from hermes_cli.goals import GoalManager
            from hermes_cli.config import load_config

            def make(sid):
                try:
                    goals_cfg = (load_config() or {}).get("goals") or {}
                    max_turns = int(goals_cfg.get("max_turns", 20) or 20)
                except Exception:
                    max_turns = 20
                return GoalManager(session_id=sid, default_max_turns=max_turns)
            return make
        return self._session_bound_manager("_goal_manager", "goal manager", load)

    def _get_heartbeat_manager(self):
        """HeartbeatManager bound to the current session_id (see ``_session_bound_manager``)."""
        def load():
            from hermes_cli.heartbeat import HeartbeatManager
            return lambda sid: HeartbeatManager(session_id=sid)
        return self._session_bound_manager("_heartbeat_manager", "heartbeat manager", load)

    def _get_loop_manager(self):
        """LoopManager bound to the current session_id (see ``_session_bound_manager``)."""
        def load():
            from hermes_cli.loops import LoopManager
            return lambda sid: LoopManager(session_id=sid)
        return self._session_bound_manager("_loop_manager", "loop manager", load)

    def _start_heartbeat_watchdog(self):
        """Start the idle-poll daemon that injects a due heartbeat prompt into
        ``_pending_input`` as a normal user turn when the session is idle. Missed ticks
        coalesce (the anchor resets on fire, so a busy hour yields ONE heartbeat turn).
        Idempotent; safe to call on every /heartbeat set."""
        if getattr(self, "_heartbeat_watchdog_started", False):
            return
        self._heartbeat_watchdog_started = True
        from hermes_cli.heartbeat import POLL_SECONDS

        def _loop():
            try:
                while not getattr(self, "_should_exit", False):
                    time.sleep(POLL_SECONDS)
                    try:
                        mgr = self._get_heartbeat_manager()
                        if mgr is None or not mgr.is_active():
                            continue
                        busy = (
                            self._agent_running
                            or getattr(self, "_voice_recording", False)
                            or getattr(self, "_voice_processing", False)
                            or not self._pending_input.empty())
                        if busy:
                            continue
                        prompt = mgr.due_prompt()
                        if prompt:
                            self._pending_input.put(prompt)
                    except Exception as exc:
                        logging.debug("heartbeat watchdog tick failed: %s", exc)
            finally:
                self._heartbeat_watchdog_started = False
        threading.Thread(target=_loop, daemon=True, name="heartbeat-watchdog").start()

    def _maybe_resume_parked_goal(self) -> None:
        """Idle hook run from process_loop: when a parked /goal's barrier has lifted (the process
        exited, the timer elapsed, or the wait aged past its cap), queue the continuation so the
        loop resumes WITHOUT waiting for an unrelated turn to re-evaluate it. The barrier used to be
        checked only lazily, on the next turn; a session with nothing else arriving stayed parked
        indefinitely (one run: 3 h 22 min on a grandchild's poller)."""
        now = time.time()
        if now - getattr(self, "_last_goal_barrier_check", 0.0) < 5.0:
            return
        self._last_goal_barrier_check = now
        try:
            if not self._pending_input.empty():
                return
            mgr = self._get_goal_manager()
            state = getattr(mgr, "state", None) if mgr is not None else None
            if state is None or state.status != "active":
                return
            if not (state.waiting_on_pid is not None or state.waiting_on_session is not None or state.waiting_until):
                return  # not parked
            if mgr.is_waiting():
                return  # barrier still holds (is_waiting() also applies the age cap)
            prompt = mgr.next_continuation_prompt()
            if prompt:
                from cli import _DIM, _RST, _cprint
                _cprint(f"  {_DIM}▶ Goal barrier lifted — resuming.{_RST}")
                self._pending_input.put(prompt)
        except Exception as exc:
            logging.debug("parked-goal resume check failed: %s", exc)

    def _maybe_fire_loop_tick(self) -> None:
        """Idle hook run from process_loop: fire a due /loop wakeup.

        Only while the agent is idle and nothing is queued — a real user message always
        wins the idle boundary, and so does an active (non-parked) /goal, whose
        judge-driven continuations own it; the loop defers to the next poll.
        """
        from cli import _DIM, _RST, _cprint
        mgr = self._get_loop_manager()
        if mgr is None or not mgr.is_due():
            return
        # The idle poll runs at ~10 Hz; a due-but-deferred tick would otherwise hit the
        # DB (goal_blocks_loop_tick) on every poll. Throttle the re-check.
        now = time.time()
        if now - getattr(self, "_last_loop_tick_check", 0.0) < 2.0:
            return
        self._last_loop_tick_check = now
        try:
            if not self._pending_input.empty():
                return
        except Exception:
            return
        try:
            from hermes_cli.loops import goal_blocks_loop_tick
            if goal_blocks_loop_tick(mgr.session_id):
                return
        except Exception:
            pass
        wakeup = mgr.fire_tick()
        if not wakeup:
            return
        try:
            state = mgr.state
            tick_no = state.ticks_fired if state else "?"
            _cprint(f"  {_DIM}↻ /loop wakeup #{tick_no} firing…{_RST}")
            self._pending_input.put(wakeup)
        except Exception as exc:
            logging.debug("loop tick injection failed: %s", exc)
            try:
                mgr.abandon_tick()
            except Exception:
                pass
            return
        # A slash-command loop (`/loop 10m /recap`) is dispatched via process_command and
        # never reaches chat()'s post-turn finally, so its tick would never complete and
        # the loop would wedge on awaiting_response. Slash ticks have no model reply to
        # judge; complete them immediately (caps and scheduling still apply).
        if wakeup.lstrip().startswith("/"):
            try:
                _print_decision_message(mgr.complete_tick(""))
            except Exception:
                pass

    def _last_assistant_response_text(self) -> str:
        """Text of the most recent assistant message ("" when none); multimodal parts are flattened."""
        try:
            for msg in reversed(self.conversation_history or []):
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        parts = [
                            p.get("text", "")
                            for p in content
                            if isinstance(p, dict) and p.get("type") in {"text", "output_text"}]
                        return "\n".join(t for t in parts if t)
                    return str(content or "")
        except Exception:
            pass
        return ""

    def _maybe_complete_loop_tick_after_turn(self) -> None:
        """Post-turn hook: evaluate a finished /loop wakeup turn.

        No-op unless the turn was a loop wakeup (``awaiting_response`` set by
        ``fire_tick``). Detects the LOOP_COMPLETE marker, judges --until, applies caps,
        and schedules the next tick. Mirrors _maybe_continue_goal_after_turn's shape.
        """
        from cli import _DIM, _RST, _cprint
        mgr = self._get_loop_manager()
        if mgr is None:
            return
        state = mgr.state
        if state is None or not state.awaiting_response:
            return

        # A user-interrupted wakeup turn pauses the loop (recoverable via /loop resume)
        # — same contract as the goal loop's Ctrl+C handling.
        if getattr(self, "_last_turn_interrupted", False):
            try:
                mgr.pause(reason="user-interrupted (Ctrl+C)")
            except Exception:
                pass
            _cprint(
                f"  {_DIM}⏸ Loop paused — wakeup turn was interrupted. "
                f"Use /loop resume to continue, or /loop stop to end it.{_RST}")
            return
        decision = mgr.complete_tick(self._last_assistant_response_text())
        if (not _print_decision_message(decision) and decision.get("status") == "active"
                and mgr.state is not None):
            _cprint(f"  {_DIM}↻ Loop: {mgr.state.remaining_label()}.{_RST}")

    def _maybe_continue_goal_after_turn(self) -> None:
        """Post-turn hook: judge the goal and maybe re-queue a continuation. A real user
        message already queued preempts judging (re-judged after their turn). Ctrl+C
        AUTO-PAUSES instead of judging — the judge on partial output nearly always says
        "continue" and would re-queue exactly what was cancelled; pausing is recoverable
        via ``/goal resume``. Empty-response skip mirrors ``gateway/run.py``."""
        from cli import _DIM, _RST, _cprint, _looks_like_slash_command
        mgr = self._get_goal_manager()
        if mgr is None or not mgr.is_active():
            return

        # Slash commands don't count as "real user messages": they're dispatched via
        # process_command, not chat(), so a queued /subgoal would consume its slot without
        # ever re-firing this hook and the goal loop would silently stall. Peek at every
        # queued entry (Queue.queue is the deque; FIFO undisturbed) and defer only on a
        # non-slash payload. Bundled payloads are (text, images) tuples.
        try:
            pending = getattr(self, "_pending_input", None)
            if pending is not None and not pending.empty():
                try:
                    entries = [e[0] if isinstance(e, tuple) and e else e for e in list(pending.queue)]
                    has_real_message = any(
                        not (isinstance(e, str) and _looks_like_slash_command(e)) for e in entries)
                except Exception:
                    has_real_message = True  # can't introspect — defer to be safe
                if has_real_message:
                    return
        except Exception:
            pass
        if getattr(self, "_last_turn_interrupted", False):
            try:
                mgr.pause(reason="user-interrupted (Ctrl+C)")
            except Exception as exc:
                logging.debug("goal pause-on-interrupt failed: %s", exc)
            _cprint(
                f"  {_DIM}⏸ Goal paused — turn was interrupted. "
                f"Use /goal resume to continue, or /goal clear to stop.{_RST}")
            return

        # Empty/whitespace responses are almost always transient failures (API error,
        # empty stream): judging would say "continue" and trip the parse-failure backstop.
        last_response = self._last_assistant_response_text()
        if not last_response.strip():
            return
        _active_deleg = 0
        try:
            from hermes_cli.goals import count_active_delegations, gather_background_processes as _gather_bg
            # Only THIS session's processes: subagents' pollers must not park the parent's goal.
            _bg_procs = _gather_bg(owner_task_id=getattr(self, "session_id", None) or None)
            _active_deleg = count_active_delegations(getattr(self.agent, "session_id", None))
        except Exception:
            _bg_procs = None
        decision = mgr.evaluate_after_turn(
            last_response, user_initiated=True, background_processes=_bg_procs, active_delegations=_active_deleg)
        _print_decision_message(decision)
        if decision.get("should_continue"):
            prompt = decision.get("continuation_prompt")
            if prompt:
                try:
                    self._pending_input.put(prompt)
                except Exception as exc:
                    logging.debug("goal continuation enqueue failed: %s", exc)
