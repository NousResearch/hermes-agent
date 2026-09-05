"""Autonomy-loop gateway commands: /goal, /subgoal, /heartbeat, /loop, /refine, /review.
Bound onto ``GatewayRunner`` through ``GatewaySlashCommandsMixin``."""

from __future__ import annotations

import logging

from agent.i18n import t
from gateway.platforms.base import MessageEvent, MessageType

# Log-record parity with gateway/run.py and the origin module.
logger = logging.getLogger("gateway.run")


def _plural(n: int, noun: str) -> str:
    return f"{n} {noun}{'s' if n != 1 else ''}"


def _quiet_bool(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False


def _mgr_call(prefix: str, fn, *args, errors=(RuntimeError, ValueError)):
    """``(result, None)`` from ``fn(*args)``, or ``(None, "<prefix>: <exc>")`` on a manager error."""
    try:
        return fn(*args), None
    except errors as exc:
        return None, f"{prefix}: {exc}"


class GatewayGoalCommandsMixin:
    """Autonomy-loop gateway commands: /goal, /subgoal, /heartbeat, /loop, /refine, /review."""

    async def _handle_goal_command(self, event: MessageEvent) -> str:
        """Handle /goal: status / show / unwait / clear / pause / resume / wait / gate / <new goal>.

        Setting a new goal queues the goal text as the next turn so the agent starts immediately;
        the post-turn continuation hook takes over after.
        """
        args = (event.get_command_args() or "").strip()
        lower = args.lower()
        mgr, _session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")
        if not args or lower == "status":
            return mgr.status_line()
        if lower == "show":
            return f"{mgr.status_line()}\n{mgr.render_contract()}"
        if lower == "unwait":
            return "▶ Wait barrier cleared — goal loop resumes." if mgr.stop_waiting() else "No wait barrier set."
        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            self._clear_goal_continuations(event, "clear")
            return t("gateway.goal_cleared") if had else t("gateway.no_active_goal")
        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                return t("gateway.goal.no_goal_set")
            self._clear_goal_continuations(event, "pause")
            return t("gateway.goal.paused", goal=state.goal)
        if lower == "resume":
            return self._goal_resume(mgr, event)
        # Verb-prefixed forms take the remainder as their argument.
        for verb, handler in (("wait", self._goal_wait), ("gate", self._goal_gate)):
            if lower == verb or lower.startswith(verb + " "):
                return handler(mgr, args[len(verb):].strip(), event)
        return await self._goal_set(mgr, args, lower, event)

    def _clear_goal_continuations(self, event: MessageEvent, verb: str) -> None:
        try:
            adapter, quick_key = self._adapter_and_key_for(event)
            if adapter and quick_key:
                self._clear_goal_pending_continuations(quick_key, adapter)
        except Exception as exc:
            logger.debug("goal %s: pending continuation cleanup failed: %s", verb, exc)

    def _enqueue_goal_turn(
        self, event: MessageEvent, text: str, *, label: str, kickoff: bool, route=None
    ) -> None:
        """Enqueue *text* as the next turn through the adapter FIFO (the post-turn judge's path).

        A kickoff is a distinct internal lifecycle; neither kickoff nor resume reuses the
        triggering message id/channel prompt. *route* is pre-resolved when supplied.
        """
        try:
            adapter, quick_key = route or self._adapter_and_key_for(event)
            if text and adapter and quick_key:
                turn = MessageEvent(
                    text=text,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=None,
                    channel_prompt=None,
                    internal=kickoff,
                )
                self._enqueue_fifo(quick_key, turn, adapter)
        except Exception as exc:
            logger.debug("goal %s failed: %s", label, exc)

    def _goal_resume(self, mgr, event: MessageEvent) -> str:
        state = mgr.resume()
        if state is None:
            return t("gateway.goal.no_resume")
        # Resume must restart work, not just flip persisted state: enqueue the canonical
        # continuation so the next turn fires as soon as this reply is delivered.
        self._enqueue_goal_turn(
            event, mgr.next_continuation_prompt(), label="resume: continuation enqueue", kickoff=False
        )
        return t("gateway.goal.resumed", goal=state.goal)

    @staticmethod
    def _goal_wait(mgr, wait_arg: str, event: MessageEvent) -> str:
        """/goal wait <pid> [reason] — park the loop on a background process."""
        if not wait_arg:
            return "Usage: /goal wait <pid> [reason]"
        wtokens = wait_arg.split(None, 1)
        try:
            pid = int(wtokens[0])
        except ValueError:
            return "/goal wait: <pid> must be an integer process id."
        reason = wtokens[1].strip() if len(wtokens) > 1 else ""
        _, err = _mgr_call("/goal wait", lambda: mgr.wait_on(pid, reason=reason))
        if err:
            return err
        rtxt = f" ({reason})" if reason else ""
        return f"⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits."

    def _goal_gate(self, mgr, gate_arg: str, event: MessageEvent) -> str:
        """/goal gate [list | add <command> | remove <N> | clear] — deterministic quality gates."""
        gate_lower = gate_arg.lower()
        if not gate_arg or gate_lower == "list":
            return mgr.render_gates()
        if gate_lower.startswith("add "):
            # SECURITY: a gate is persisted and later executed with shell=True at every goal turn
            # boundary (run_gate), with no approval prompt. Letting an allowed but non-admin sender
            # choose that string is authenticated RCE under the Hermes process account — and with
            # no admin list configured (the default) every allowed sender is unrestricted. Gate ONLY
            # this shell-creating operation behind a real, explicitly-configured admin (the same
            # fail-closed check that guards cross-origin /resume); list/remove/clear stay open so
            # a non-admin can still recover.
            if not self._resume_caller_is_admin(event.source):
                return (
                    "⛔ /goal gate add requires an explicitly configured "
                    "gateway admin (allow_admin_from for DMs, "
                    "group_allow_admin_from for groups)."
                )
            gate, err = _mgr_call("/goal gate add", mgr.add_gate, gate_arg[len("add"):].strip())
            if err:
                return err
            return (
                f"⚿ Gate added: $ {gate.command} "
                f"({gate.max_retries} retries, {gate.timeout_seconds}s timeout). "
                f"It must pass before the goal can complete."
            )
        if gate_lower.startswith(("remove ", "rm ")):
            removed, err = _mgr_call(
                "/goal gate remove", lambda: mgr.remove_gate(int(gate_arg.split(None, 1)[1].strip())),
                errors=(RuntimeError, ValueError, IndexError),
            )
            return err or f"✓ Gate removed: $ {removed}"
        if gate_lower == "clear":
            prev, err = _mgr_call("/goal gate clear", mgr.clear_gates, errors=(RuntimeError,))
            return err or f"✓ Cleared {_plural(prev, 'gate')}."
        return "Usage: /goal gate [list | add <command> | remove <N> | clear]"

    async def _goal_set(self, mgr, args: str, lower: str, event: MessageEvent) -> str:
        """Set a new goal from free text, inline ``field: value`` contract lines, or ``draft <objective>``."""
        drafting = lower.startswith("draft")
        if drafting:
            objective = args[len("draft"):].strip()
            if not objective:
                return "Usage: /goal draft <objective in plain language>"
            try:
                from hermes_cli.goals import draft_contract

                # _run_in_executor_with_context, not a bare hop: drafting calls the auxiliary LLM,
                # whose provider/credential resolution reads the profile secret scope — a
                # contextvar a default-executor hop drops.
                contract = await self._run_in_executor_with_context(draft_contract, objective)
            except Exception as exc:
                logger.debug("goal draft failed: %s", exc)
                contract = None
            args = objective  # the goal text is the objective
        else:
            # Inline `field: value` lines parse into a completion contract; the remaining prose is
            # the goal headline. Plain free-form goals (no such lines) behave exactly as before.
            from hermes_cli.goals import parse_contract
            headline, parsed = parse_contract(args)
            args = headline or args
            contract = parsed if not parsed.is_empty() else None
        try:
            state = mgr.set(args, contract=contract)
        except ValueError as exc:
            return t("gateway.goal.invalid", error=str(exc))

        # Queue the goal text as an immediate first turn; the post-turn hook takes over after.
        self._enqueue_goal_turn(
            event,
            mgr.next_continuation_prompt() or state.goal,
            label="kickoff enqueue",
            kickoff=True,
            route=self._adapter_and_key_for(event),
        )

        base = t("gateway.goal.set", budget=state.max_turns, goal=state.goal)
        if state.has_contract():
            return f"{base}\nCompletion contract:\n{state.contract.render_block()}"
        if drafting:
            return f"{base}\n(Couldn't draft a contract — running as a free-form goal.)"
        return base

    async def _handle_heartbeat_command(self, event: MessageEvent) -> str:
        """Handle /heartbeat (mirror of the CLI handler): the session's one recurring re-entry
        prompt. The gateway-wide poller injects due heartbeats through the adapter FIFO as
        ordinary user turns, so alternation and caching hold."""
        from hermes_cli.heartbeat import parse_interval, format_interval, MIN_INTERVAL_SECONDS
        args = (event.get_command_args() or "").strip()
        lower = args.lower()
        mgr, _session_entry = await self._get_heartbeat_manager_for_event(event)
        if mgr is None:
            return "Heartbeats unavailable (no session)."
        quick_key = self._session_key_for_source(event.source) if event.source else None

        def _watch():
            if quick_key and event.source is not None:
                self._register_heartbeat_watch(quick_key, event.source, mgr.session_id)

        if not args or lower == "status":
            return mgr.status_line()
        if lower == "pause":
            state = mgr.pause()
            return f"⏸ Heartbeat paused: {state.prompt}" if state else "No heartbeat set."
        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return "No heartbeat to resume."
            _watch()
            return f"▶ Heartbeat resumed (every {format_interval(state.interval_seconds)}): {state.prompt}"
        if lower in {"clear", "stop", "off"}:
            had = mgr.clear()
            if quick_key:
                self._unregister_heartbeat_watch(quick_key)
            return "✓ Heartbeat cleared." if had else "No heartbeat set."

        # Set: `/heartbeat every 10m <prompt>` (also accepts `10m <prompt>`).
        tokens = args.split(None, 2)
        interval, prompt = None, ""
        if tokens[0].lower() == "every" and len(tokens) >= 2:
            interval = parse_interval(f"every {tokens[1]}")
            prompt = tokens[2] if len(tokens) > 2 else ""
        else:
            interval = parse_interval(tokens[0])
            prompt = args[len(tokens[0]):].strip() if interval and interval > 0 else ""
        if interval is None:
            return (
                "Usage: /heartbeat every <interval> <prompt>  (e.g. /heartbeat every 10m Check CI)\n"
                "Also: /heartbeat status | pause | resume | clear"
            )
        if interval < 0:
            return f"Interval too small — minimum is {MIN_INTERVAL_SECONDS}s."
        if not prompt.strip():
            return "Usage: /heartbeat every <interval> <prompt> — the prompt is required."
        state, err = _mgr_call("Invalid heartbeat", mgr.set, prompt, interval, errors=(ValueError,))
        if err:
            return err
        _watch()
        return (
            f"♥ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}\n"
            "Fires as a normal turn whenever this session is idle and the interval has "
            "elapsed. Lives while the gateway runs — use `hermes cron` for durable schedules."
        )

    def _idle_cached_agent_or_error(self, event: MessageEvent, verb: str):
        """``(session_key, cached_agent, None)`` for /refine and /review, or ``(_, _, error_text)``:
        both need a cached agent from a completed turn and refuse while a run is in flight."""
        quick_key = self._session_key_for_source(event.source) if event.source else None
        if not quick_key:
            return None, None, f"{verb.capitalize()} unavailable (no session)."
        if quick_key in self._running_agents:
            return quick_key, None, f"Agent is running — wait for the turn to finish, then /{verb}."
        agent = self._cached_agent_for(quick_key)
        if agent is None:
            return quick_key, None, f"Nothing to {verb} yet — send a message first."
        return quick_key, agent, None

    async def _handle_refine_command(self, event: MessageEvent) -> str:
        """Handle /refine — run the memory/skill review fork on demand, in a daemon thread against a
        snapshot of the cached AIAgent's conversation (live session and prompt cache untouched)."""
        args = (event.get_command_args() or "").strip()
        _quick_key, agent, error = self._idle_cached_agent_or_error(event, "refine")
        if error:
            return error
        snapshot = list(getattr(agent, "_session_messages", None) or [])
        if not snapshot:
            return "Nothing to refine yet — the conversation is empty."
        try:
            agent._spawn_background_review(
                messages_snapshot=snapshot, review_memory=True,
                review_skills="skill_manage" in getattr(agent, "valid_tool_names", set()), focus=args or None,
            )
        except Exception as exc:
            return f"/refine failed to start: {exc}"
        tail = f" (focus: {args})" if args else ""
        return (
            f"⚗ Reviewing this conversation in the background{tail} — "
            f"any memory/skill updates will be reported when done."
        )

    async def _handle_review_command(self, event: MessageEvent) -> str:
        """Handle /review — spawn an independent reviewer subagent. The approval session-key
        contextvar is only bound during agent turns, so bind it explicitly here or the completion
        event carries no gateway route and never re-enters this chat."""
        args = (event.get_command_args() or "").strip()
        quick_key, agent, error = self._idle_cached_agent_or_error(event, "review")
        if error:
            return error
        snapshot = list(getattr(agent, "_session_messages", None) or [])
        from tools.approval_context import reset_current_session_key, set_current_session_key

        def _dispatch():
            token = set_current_session_key(quick_key)
            try:
                from agent.review_engine import start_review
                return start_review(agent, snapshot, args)
            finally:
                reset_current_session_key(token)

        try:
            # _run_in_executor_with_context, not a bare hop: the reviewer subagent is spawned from
            # the worker and inherits its context; a bare hop would run it under the launch home.
            result = await self._run_in_executor_with_context(_dispatch)
        except ValueError as exc:
            return str(exc)
        except Exception as exc:
            return f"/review failed to start: {exc}"
        from agent.review_engine import format_dispatch_note
        return format_dispatch_note(result, args)

    async def _handle_subgoal_command(self, event: MessageEvent) -> str:
        """Handle /subgoal (mirror of the CLI handler): extra criteria appended to the active goal
        mid-loop. They modify state read at the next turn boundary, so this is safe while the
        agent is running."""
        args = (event.get_command_args() or "").strip()
        mgr, _session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")
        if not mgr.has_goal():
            return "No active goal. Set one with /goal <text>."
        if not args:
            return f"{mgr.status_line()}\n{mgr.render_subgoals()}"
        tokens = args.split(None, 1)
        verb = tokens[0].lower()
        rest = tokens[1].strip() if len(tokens) > 1 else ""
        if verb == "remove":
            if not rest:
                return "Usage: /subgoal remove <n>"
            try:
                idx = int(rest.split()[0])
            except ValueError:
                return "/subgoal remove: <n> must be an integer (1-based index)."
            removed, err = _mgr_call(
                "/subgoal remove", mgr.remove_subgoal, idx, errors=(IndexError, RuntimeError)
            )
            return err or f"✓ Removed subgoal {idx}: {removed}"
        if verb == "clear":
            prev, err = _mgr_call("/subgoal clear", mgr.clear_subgoals, errors=(RuntimeError,))
            if err:
                return err
            return f"✓ Cleared {_plural(prev, 'subgoal')}." if prev else "No subgoals to clear."
        text, err = _mgr_call("/subgoal", mgr.add_subgoal, args)
        if err:
            return err
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return f"✓ Added subgoal {idx}: {text}"

    async def _handle_loop_command(self, event: MessageEvent) -> str:
        """Handle /loop — recurring in-session wakeups, via ``dispatch_loop_command`` (CLI mirror)."""
        try:
            from hermes_cli.loops import LoopManager, dispatch_loop_command, goal_blocks_loop_tick
        except Exception as exc:
            logger.debug("loops module unavailable: %s", exc)
            return "Loops unavailable."

        # Warm the SessionDB cache off-loop: a cold cache drops the first /loop write while the
        # reply claims the loop was set (same class as the /goal false-ack fix).
        await self._warm_goals_session_db("loop manager")
        try:
            session_entry = await self.async_session_store.get_or_create_session(event.source)
        except Exception:
            session_entry = None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return "Loops unavailable (no active session)."
        mgr = LoopManager(session_id=sid)

        # New loops capture the event's routing so the idle loop-wakeup watcher can inject ticks
        # here after a restart; best-effort, empty fields dropped.
        route: dict = {}
        try:
            src = event.source
            if src is not None:
                platform = getattr(src, "platform", "")
                route = {"platform": platform.value if hasattr(platform, "value") else str(platform or "")}
                for key in ("chat_id", "chat_type", "thread_id", "user_id", "user_name"):
                    route[key] = str(getattr(src, key, "") or "")
                route = {k: v for k, v in route.items() if v}
        except Exception:
            route = {}
        result = dispatch_loop_command(mgr, (event.get_command_args() or "").strip(), route=route)
        output = result.get("output") or ""
        if result.get("created") and _quiet_bool(lambda: goal_blocks_loop_tick(mgr.session_id)):
            output += (
                "\nNote: an active /goal is driving this session — loop "
                "wakeups defer until the goal finishes, pauses, or parks."
            )
        return output
