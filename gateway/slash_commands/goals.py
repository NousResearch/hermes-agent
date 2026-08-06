"""/goal, /subgoal, /heartbeat, /refine slash-command handlers for GatewayRunner.

Moved verbatim from ``gateway/slash_commands.py``. Method bodies are
byte-identical; ``self`` remains the ``GatewayRunner`` through the MRO.
"""

from __future__ import annotations

from agent.i18n import t
from gateway.platforms.base import MessageEvent
from gateway.platforms.base import MessageType

from gateway.slash_commands._shared import logger

class GoalsCommandsMixin:
    """/goal, /subgoal, /heartbeat, /refine handlers."""

    async def _handle_goal_command(self, event: "MessageEvent") -> str:
        """Handle /goal for gateway platforms.

        Subcommands: ``/goal`` / ``/goal status`` / ``/goal pause`` /
        ``/goal resume`` / ``/goal clear``. Any other text becomes the
        new goal.

        Setting a new goal queues the goal text as the next turn so the
        agent starts working on it immediately — the post-turn
        continuation hook then takes over from there.
        """
        args = (event.get_command_args() or "").strip()
        lower = args.lower()

        mgr, session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")

        if not args or lower == "status":
            return mgr.status_line()

        # /goal show → print the active goal's completion contract
        if lower == "show":
            return f"{mgr.status_line()}\n{mgr.render_contract()}"

        if lower == "pause":
            state = mgr.pause(reason="user-paused")
            if state is None:
                return t("gateway.goal.no_goal_set")
            try:
                adapter = self.adapters.get(event.source.platform) if event.source else None
                _quick_key = self._session_key_for_source(event.source) if event.source else None
                if adapter and _quick_key:
                    self._clear_goal_pending_continuations(_quick_key, adapter)
            except Exception as exc:
                logger.debug("goal pause: pending continuation cleanup failed: %s", exc)
            return t("gateway.goal.paused", goal=state.goal)

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return t("gateway.goal.no_resume")
            return t("gateway.goal.resumed", goal=state.goal)

        if lower in {"clear", "stop", "done"}:
            had = mgr.has_goal()
            mgr.clear()
            try:
                adapter = self.adapters.get(event.source.platform) if event.source else None
                _quick_key = self._session_key_for_source(event.source) if event.source else None
                if adapter and _quick_key:
                    self._clear_goal_pending_continuations(_quick_key, adapter)
            except Exception as exc:
                logger.debug("goal clear: pending continuation cleanup failed: %s", exc)
            return t("gateway.goal_cleared") if had else t("gateway.no_active_goal")

        # /goal wait <pid> [reason] — park the loop on a background process.
        if lower == "wait" or lower.startswith("wait "):
            wait_arg = args[len("wait"):].strip()
            if not wait_arg:
                return "Usage: /goal wait <pid> [reason]"
            wtokens = wait_arg.split(None, 1)
            try:
                pid = int(wtokens[0])
            except ValueError:
                return "/goal wait: <pid> must be an integer process id."
            reason = wtokens[1].strip() if len(wtokens) > 1 else ""
            try:
                mgr.wait_on(pid, reason=reason)
            except (RuntimeError, ValueError) as exc:
                return f"/goal wait: {exc}"
            rtxt = f" ({reason})" if reason else ""
            return f"⏳ Goal parked on pid {pid}{rtxt}. Loop pauses until it exits."

        # /goal unwait — clear the wait barrier.
        if lower == "unwait":
            if mgr.stop_waiting():
                return "▶ Wait barrier cleared — goal loop resumes."
            return "No wait barrier set."

        # /goal gate ... — manage deterministic quality gates.
        if lower == "gate" or lower.startswith("gate "):
            gate_arg = args[len("gate"):].strip()
            gate_lower = gate_arg.lower()
            if not gate_arg or gate_lower == "list":
                return mgr.render_gates()
            if gate_lower.startswith("add "):
                command = gate_arg[len("add"):].strip()
                try:
                    gate = mgr.add_gate(command)
                except (RuntimeError, ValueError) as exc:
                    return f"/goal gate add: {exc}"
                return (
                    f"⚿ Gate added: $ {gate.command} "
                    f"({gate.max_retries} retries, {gate.timeout_seconds}s timeout). "
                    f"It must pass before the goal can complete."
                )
            if gate_lower.startswith("remove ") or gate_lower.startswith("rm "):
                idx_text = gate_arg.split(None, 1)[1].strip()
                try:
                    removed = mgr.remove_gate(int(idx_text))
                except (RuntimeError, ValueError, IndexError) as exc:
                    return f"/goal gate remove: {exc}"
                return f"✓ Gate removed: $ {removed}"
            if gate_lower == "clear":
                try:
                    prev = mgr.clear_gates()
                except RuntimeError as exc:
                    return f"/goal gate clear: {exc}"
                return f"✓ Cleared {prev} gate{'s' if prev != 1 else ''}."
            return "Usage: /goal gate [list | add <command> | remove <N> | clear]"

        # /goal draft <objective> → draft a structured completion contract,
        # then set it. The aux LLM call is sync; run it off the event loop.
        draft_contract_obj = None
        if lower.startswith("draft"):
            objective = args[len("draft"):].strip()
            if not objective:
                return "Usage: /goal draft <objective in plain language>"
            try:
                import asyncio
                from hermes_cli.goals import draft_contract

                draft_contract_obj = await asyncio.get_running_loop().run_in_executor(
                    None, draft_contract, objective
                )
            except Exception as exc:
                logger.debug("goal draft failed: %s", exc)
                draft_contract_obj = None
            args = objective  # the goal text is the objective
            contract = draft_contract_obj
        else:
            # Inline `field: value` lines parse into a completion contract;
            # the remaining prose is the goal headline. Plain free-form goals
            # (no such lines) behave exactly as before.
            from hermes_cli.goals import parse_contract

            headline, parsed = parse_contract(args)
            args = headline or args
            contract = parsed if not parsed.is_empty() else None

        # Otherwise — treat the remaining text as the new goal.
        try:
            state = mgr.set(args, contract=contract)
        except ValueError as exc:
            return t("gateway.goal.invalid", error=str(exc))

        # Queue the goal text as an immediate first turn so the agent
        # starts making progress. The post-turn hook takes over after.
        adapter = self.adapters.get(event.source.platform) if event.source else None
        _quick_key = self._session_key_for_source(event.source) if event.source else None
        if adapter and _quick_key:
            try:
                kickoff_event = MessageEvent(
                    text=state.goal,
                    message_type=MessageType.TEXT,
                    source=event.source,
                    message_id=event.message_id,
                    channel_prompt=event.channel_prompt,
                )
                self._enqueue_fifo(_quick_key, kickoff_event, adapter)
            except Exception as exc:
                logger.debug("goal kickoff enqueue failed: %s", exc)

        base = t("gateway.goal.set", budget=state.max_turns, goal=state.goal)
        if state.has_contract():
            return f"{base}\nCompletion contract:\n{state.contract.render_block()}"
        if lower.startswith("draft"):
            # Drafting was requested but the aux model couldn't produce one.
            return f"{base}\n(Couldn't draft a contract — running as a free-form goal.)"
        return base

    async def _handle_heartbeat_command(self, event: "MessageEvent") -> str:
        """Handle /heartbeat for gateway platforms (mirror of CLI handler).

        Sets/manages the session's one recurring re-entry prompt. The
        gateway-wide poller injects due heartbeats through the adapter FIFO
        as ordinary user turns, so alternation and caching are untouched.
        """
        from hermes_cli.heartbeat import parse_interval, format_interval, MIN_INTERVAL_SECONDS

        args = (event.get_command_args() or "").strip()
        lower = args.lower()

        mgr, session_entry = await self._get_heartbeat_manager_for_event(event)
        if mgr is None:
            return "Heartbeats unavailable (no session)."

        quick_key = self._session_key_for_source(event.source) if event.source else None

        if not args or lower == "status":
            return mgr.status_line()

        if lower == "pause":
            state = mgr.pause()
            return f"⏸ Heartbeat paused: {state.prompt}" if state else "No heartbeat set."

        if lower == "resume":
            state = mgr.resume()
            if state is None:
                return "No heartbeat to resume."
            if quick_key and event.source is not None:
                self._register_heartbeat_watch(quick_key, event.source, mgr.session_id)
            return f"▶ Heartbeat resumed (every {format_interval(state.interval_seconds)}): {state.prompt}"

        if lower in {"clear", "stop", "off"}:
            had = mgr.clear()
            if quick_key:
                self._unregister_heartbeat_watch(quick_key)
            return "✓ Heartbeat cleared." if had else "No heartbeat set."

        # Set: `/heartbeat every 10m <prompt>` (also accepts `10m <prompt>`).
        tokens = args.split(None, 2)
        interval = None
        prompt = ""
        if tokens and tokens[0].lower() == "every" and len(tokens) >= 2:
            interval = parse_interval(f"every {tokens[1]}")
            prompt = tokens[2] if len(tokens) > 2 else ""
        elif tokens:
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

        try:
            state = mgr.set(prompt, interval)
        except ValueError as exc:
            return f"Invalid heartbeat: {exc}"
        if quick_key and event.source is not None:
            self._register_heartbeat_watch(quick_key, event.source, mgr.session_id)
        return (
            f"♥ Heartbeat set (every {format_interval(state.interval_seconds)}): {state.prompt}\n"
            "Fires as a normal turn whenever this session is idle and the interval has "
            "elapsed. Lives while the gateway runs — use `hermes cron` for durable schedules."
        )

    async def _handle_refine_command(self, event: "MessageEvent") -> str:
        """Handle /refine — run the memory/skill review fork on demand.

        Uses the session's cached AIAgent (idle agents live in
        ``_agent_cache``). The review runs in a daemon thread against a
        snapshot of the conversation; the live session and prompt cache are
        untouched. Requires the session to have at least one completed turn.
        """
        args = (event.get_command_args() or "").strip()
        quick_key = self._session_key_for_source(event.source) if event.source else None
        if not quick_key:
            return "Refine unavailable (no session)."
        if quick_key in self._running_agents:
            return "Agent is running — wait for the turn to finish, then /refine."

        agent = None
        cache_lock = getattr(self, "_agent_cache_lock", None)
        if cache_lock is not None:
            with cache_lock:
                cached = self._agent_cache.get(quick_key)
                agent = cached[0] if isinstance(cached, tuple) else cached if cached else None
        if agent is None:
            return "Nothing to refine yet — send a message first."

        snapshot = list(getattr(agent, "_session_messages", None) or [])
        if not snapshot:
            return "Nothing to refine yet — the conversation is empty."

        review_skills = "skill_manage" in getattr(agent, "valid_tool_names", set())
        try:
            agent._spawn_background_review(
                messages_snapshot=snapshot,
                review_memory=True,
                review_skills=review_skills,
                focus=args or None,
            )
        except Exception as exc:
            return f"/refine failed to start: {exc}"
        tail = f" (focus: {args})" if args else ""
        return (
            f"⚗ Reviewing this conversation in the background{tail} — "
            f"any memory/skill updates will be reported when done."
        )

    async def _handle_subgoal_command(self, event: "MessageEvent") -> str:
        """Handle /subgoal for gateway platforms (mirror of CLI handler).

        Subgoals are extra criteria appended to the active goal mid-loop.
        They modify state read at the next turn boundary, so this is safe
        to invoke while the agent is running.
        """
        args = (event.get_command_args() or "").strip()
        mgr, _session_entry = await self._get_goal_manager_for_event(event)
        if mgr is None:
            return t("gateway.goal.unavailable")
        if not mgr.has_goal():
            return "No active goal. Set one with /goal <text>."

        # No args → list current subgoals.
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
            try:
                removed = mgr.remove_subgoal(idx)
            except (IndexError, RuntimeError) as exc:
                return f"/subgoal remove: {exc}"
            return f"✓ Removed subgoal {idx}: {removed}"

        if verb == "clear":
            try:
                prev = mgr.clear_subgoals()
            except RuntimeError as exc:
                return f"/subgoal clear: {exc}"
            if prev:
                return f"✓ Cleared {prev} subgoal{'s' if prev != 1 else ''}."
            return "No subgoals to clear."

        try:
            text = mgr.add_subgoal(args)
        except (ValueError, RuntimeError) as exc:
            return f"/subgoal: {exc}"
        idx = len(mgr.state.subgoals) if mgr.state else 0
        return f"✓ Added subgoal {idx}: {text}"
