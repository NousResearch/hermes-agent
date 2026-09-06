"""Goal/heartbeat continuation, post-turn hooks and loop-wakeup watcher methods for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from gateway.platforms.base import MessageEvent, MessageType

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner  # noqa: F401
    from gateway.run_turn_runner import TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayGoalsMixin:
    """Goal/heartbeat continuation, post-turn hooks and loop-wakeup watcher methods for GatewayRunner."""

    # ── /goal — persistent cross-turn goals (Ralph-style loop) ──────────
    def _goal_max_turns_from_config(self) -> int:
        """Configured /goal turn budget. GatewayRunner.config is a GatewayConfig dataclass, so the
        top-level ``goals`` block is only reachable via hermes_cli.config.load_config()."""
        try:
            goals_cfg = (
                (self.config or {}).get("goals", {})
                if isinstance(self.config, dict)
                else getattr(self.config, "goals", {}) or {}
            )
            if not goals_cfg:
                from hermes_cli.config import load_config

                goals_cfg = (load_config() or {}).get("goals") or {}
            return int(goals_cfg.get("max_turns", 20) or 20)
        except Exception:
            return 20

    async def _warm_goals_session_db(self, label: str) -> None:
        """Warm the goals SessionDB cache off-loop (best-effort): a cold cache runs the state.db
        init on the loop thread and freezes the loop. The executor hop keeps the profile home
        override alive under multiplex; a failed warm-up is a bounded stall, never a crash."""
        try:
            from hermes_cli.goals import _get_session_db as _warm_goals_db

            await self._run_in_executor_with_context(_warm_goals_db)
        except Exception as exc:
            logger.warning("%s: session DB warm-up failed: %s", label, exc)

    async def _session_entry_for_manager(self, event: "MessageEvent", label: str):
        """Session entry for a /goal or /heartbeat manager, or None when lookup fails. Warms the
        SessionDB cache first (a cold cache drops the first write while the reply claims it was
        set). Internal events never touch activity (idle/daily reset clock)."""
        await self._warm_goals_session_db(label)
        try:
            session_entry = await self.async_session_store.get_or_create_session(
                event.source, touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception as exc:
            logger.debug("%s: session lookup failed: %s", label, exc)
            return None
        return session_entry if getattr(session_entry, "session_id", None) else None

    async def _manager_for_event(self, event: "MessageEvent", kind: str, load):
        """``(manager, session_entry)`` for *kind* ("goal"/"heartbeat"), or ``(None, None)``.
        ``load()`` imports the manager class and returns a ``session_id -> manager`` factory."""
        try:
            factory = load()
        except Exception as exc:
            logger.debug("%s manager unavailable: %s", kind, exc)
            return None, None
        session_entry = await self._session_entry_for_manager(event, f"{kind} manager")
        if session_entry is None:
            return None, None
        return factory(session_entry.session_id), session_entry

    async def _get_goal_manager_for_event(self, event: "MessageEvent"):
        """Return ``(GoalManager, session_entry)`` for this event, or ``(None, None)``."""
        def _load():
            from hermes_cli.goals import GoalManager
            max_turns = self._goal_max_turns_from_config()
            return lambda sid: GoalManager(session_id=sid, default_max_turns=max_turns)
        return await self._manager_for_event(event, "goal", _load)

    async def _get_heartbeat_manager_for_event(self, event: "MessageEvent"):
        """Return ``(HeartbeatManager, session_entry)`` for this event, or ``(None, None)``."""
        def _load():
            from hermes_cli.heartbeat import HeartbeatManager
            return lambda sid: HeartbeatManager(session_id=sid)
        return await self._manager_for_event(event, "heartbeat", _load)

    @staticmethod
    def _synthetic_prompt_event(source: Any, text: str, *, internal: bool = False) -> MessageEvent:
        """Build the TEXT event used to inject a goal/heartbeat/loop prompt into a session."""
        return MessageEvent(text=text, message_type=MessageType.TEXT, source=source, internal=internal)

    def _register_heartbeat_watch(self, quick_key: str, source: Any, session_id: str) -> None:
        """Track a session with an active heartbeat (``quick_key`` → ``(source, session_id)``) and
        start the poller. In-memory by design: heartbeat STATE survives restarts in SessionDB, but
        firing resumes only when the user touches /heartbeat again."""
        watch = getattr(self, "_heartbeat_watch", None)
        if watch is None:
            watch = self._heartbeat_watch = {}
        watch[quick_key] = (source, session_id)
        self._start_heartbeat_poller()

    def _unregister_heartbeat_watch(self, quick_key: str) -> None:
        (getattr(self, "_heartbeat_watch", None) or {}).pop(quick_key, None)

    async def _heartbeat_poll_once(self, watch: dict) -> None:
        """One heartbeat poll pass: enqueue every due prompt of a non-busy watched session."""
        # Off-loop warm-up covers the degraded path where /heartbeat's own warm-up failed.
        await self._warm_goals_session_db("heartbeat poll")
        for quick_key, (source, session_id) in list(watch.items()):
            try:
                if quick_key in self._running_agents:
                    continue  # busy sessions coalesce their tick to the next idle poll
                from hermes_cli.heartbeat import HeartbeatManager

                mgr = HeartbeatManager(session_id=session_id)
                if not mgr.has_heartbeat():
                    watch.pop(quick_key, None)
                    continue
                prompt = mgr.due_prompt()
                adapter = self._adapter_for_source(source) if prompt else None
                if adapter is not None:
                    self._enqueue_fifo(quick_key, self._synthetic_prompt_event(source, prompt), adapter)
            except Exception as exc:
                logger.debug("heartbeat poll for %s failed: %s", quick_key, exc)

    def _start_heartbeat_poller(self) -> None:
        """Start the single gateway-wide heartbeat poll task (idempotent)."""
        existing = getattr(self, "_heartbeat_poll_task", None)
        if existing is not None and not existing.done():
            return

        from hermes_cli.heartbeat import POLL_SECONDS

        async def _poll_loop():
            while True:
                await asyncio.sleep(POLL_SECONDS)
                watch = getattr(self, "_heartbeat_watch", None)
                if watch:
                    await self._heartbeat_poll_once(watch)

        try:
            task = self._heartbeat_poll_task = asyncio.create_task(_poll_loop())
            # PERMANENT once started (infinite loop) — tag it like a _spawn_supervised watcher so
            # _scale_to_zero_has_live_background_work() doesn't treat the gateway as busy forever.
            task._hermes_supervised_watcher = True  # type: ignore[attr-defined]
            _bg = getattr(self, "_background_tasks", None)
            if _bg is not None:
                _bg.add(task)
                task.add_done_callback(_bg.discard)
        except Exception:
            logger.debug("Failed to start heartbeat poller", exc_info=True)

    def _goal_notice_adapter(self, source: Any):
        adapter = self._adapter_for_source(source)
        if not adapter:
            logger.debug("goal continuation: no adapter for %s", getattr(source, "platform", None))
        return adapter

    async def _send_goal_status_notice(self, source: Any, message: str) -> None:
        """Send a /goal judge status line back to the originating chat/thread."""
        adapter = self._goal_notice_adapter(source)
        if not adapter:
            return
        metadata = None
        with suppress(Exception):
            metadata = self._thread_metadata_for_source(source)
        result = await adapter.send(source.chat_id, message, metadata=metadata)
        if result is not None and not getattr(result, "success", True):
            logger.warning(
                "goal continuation: status send failed: %s", getattr(result, "error", "unknown error"),
            )

    async def _defer_goal_status_notice_after_delivery(self, source: Any, message: str) -> None:
        """Send a /goal status line after the main response is delivered.

        The adapter sends the agent response after this caller returns, so for reading order use
        its one-shot post-delivery callback when available, else deliver directly (never drop).
        """
        adapter = self._goal_notice_adapter(source)
        if not adapter:
            return

        async def _deliver() -> None:
            try:
                await self._send_goal_status_notice(source, message)
            except Exception as exc:
                logger.warning("goal continuation: status send failed: %s", exc, exc_info=True)

        session_key = None
        with suppress(Exception):
            session_key = self._session_key_for_source(source)
        if session_key and hasattr(adapter, "register_post_delivery_callback"):
            try:
                active = getattr(adapter, "_active_sessions", {}).get(session_key)
                generation = getattr(active, "_hermes_run_generation", None) if active is not None else None
                adapter.register_post_delivery_callback(session_key, _deliver, generation=generation)
                return
            except Exception as exc:
                logger.debug("goal continuation: post-delivery callback registration failed: %s", exc)
        await _deliver()

    async def _post_turn_manager(self, session_entry: Any, label: str, module: str, load):
        """Shared head of the post-turn hooks: ``load()`` imports the manager module and returns a
        ``session_id -> manager`` factory; None when unavailable / no session id. Warms the
        SessionDB cache first — a cold cache at the turn boundary drops the read/write."""
        try:
            factory = load()
        except Exception as exc:
            logger.debug("%s: %s module unavailable: %s", label, module, exc)
            return None
        sid = getattr(session_entry, "session_id", None) or ""
        if not sid:
            return None
        await self._warm_goals_session_db(label)
        return factory(sid)

    async def _post_turn_goal_continuation(
        self, *, session_entry: Any, source: Any, final_response: str,
    ) -> None:
        """Run the goal judge after a gateway turn (AFTER delivery) and, if still active, enqueue a
        continuation through the adapter FIFO so a simultaneous real user message takes priority."""
        def _load():
            from hermes_cli.goals import GoalManager
            max_turns = self._goal_max_turns_from_config()
            return lambda sid: GoalManager(session_id=sid, default_max_turns=max_turns)

        mgr = await self._post_turn_manager(session_entry, "goal continuation", "goals", _load)
        if mgr is None or not mgr.is_active():
            return

        _bg_procs, _active_deleg = None, 0
        with suppress(Exception):
            from hermes_cli.goals import count_active_delegations, gather_background_processes as _gather_bg
            # Only THIS session's processes (gateway turns register under turn_ctx.session_id):
            # subagents' pollers must not park the parent's goal.
            _bg_procs = _gather_bg(owner_task_id=getattr(session_entry, "session_id", None) or None)
            _active_deleg = count_active_delegations(getattr(session_entry, "session_id", None))

        # judge_goal() is a synchronous aux-LLM HTTP call (10-40 s; would block Discord heartbeats).
        # _run_in_executor_with_context carries the profile secret scope / aux runtime contextvars
        # without which aux credential resolution fails under multiplexing.
        decision = await self._run_in_executor_with_context(
            lambda: mgr.evaluate_after_turn(
                final_response or "", user_initiated=True, background_processes=_bg_procs,
                active_delegations=_active_deleg,
            ),
        )
        msg = decision.get("message") or ""
        # Deferred until the visible final response is delivered, else "✓ Goal achieved" precedes it.
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)
        prompt = decision.get("continuation_prompt") or ""
        if not decision.get("should_continue") or not prompt or source is None:
            return
        # Enqueue via the adapter's FIFO so a user message already in flight preempts naturally.
        try:
            adapter = self._adapter_for_source(source)
            _quick_key = self._session_key_for_source(source)
            if adapter and _quick_key:
                self._enqueue_fifo(_quick_key, self._synthetic_prompt_event(source, prompt), adapter)
        except Exception as exc:
            logger.debug("goal continuation: enqueue failed: %s", exc)

    async def _run_post_turn_hooks(
        self, *, agent_result: Any, source: Any, is_internal: bool, event: Any = None,
    ) -> None:
        """Run goal and loop bookkeeping after an agent turn returns."""
        final_text = self._final_text_for_post_turn_hooks(agent_result, event)
        try:
            session_entry = await self.async_session_store.get_or_create_session(
                source, touch_activity=not is_internal,
            )
        except Exception as exc:
            logger.debug("post-turn session resolution failed: %s", exc)
            return
        # Empty interrupted/errored responses must not drive /goal, but an in-flight /loop tick
        # still needs to be released and rescheduled.
        hooks = [("loop completion", self._post_turn_loop_completion)]
        if final_text.strip():
            hooks.insert(0, ("goal continuation", self._post_turn_goal_continuation))
        for label, hook in hooks:
            try:
                await hook(session_entry=session_entry, source=source, final_response=final_text)
            except Exception as exc:
                logger.debug("%s hook failed: %s", label, exc)

    @staticmethod
    def _final_text_for_post_turn_hooks(agent_result, event=None) -> str:
        """Text for /goal and /loop after a gateway turn. Streamed turns return None from
        _handle_message_with_agent (already_sent); the delivered reply is stashed on the event."""
        text = ""
        if isinstance(agent_result, dict):
            text = str(agent_result.get("final_response") or "")
        elif isinstance(agent_result, str):
            text = agent_result
        if text.strip():
            return text
        streamed = getattr(event, "_streamed_final_response", None)
        return streamed if isinstance(streamed, str) and streamed.strip() else text

    async def _post_turn_loop_completion(
        self, *, session_entry: Any, source: Any, final_response: str,
    ) -> None:
        """Complete a /loop wakeup tick after a gateway turn. No-op unless a tick is in flight
        (``awaiting_response``, set when the wakeup was injected); applies the LOOP_COMPLETE marker
        / --until judge / caps and schedules the next tick for the idle wakeup watcher."""
        def _load():
            from hermes_cli.loops import LoopManager
            return lambda sid: LoopManager(session_id=sid)

        mgr = await self._post_turn_manager(session_entry, "loop completion", "loops", _load)
        state = mgr.state if mgr is not None else None
        if state is None or not state.awaiting_response:
            return
        # The --until judge is a sync aux-LLM call — keep it off the event loop.
        decision = await asyncio.get_running_loop().run_in_executor(
            None, mgr.complete_tick, final_response or ""
        )
        msg = decision.get("message") or ""
        if msg and source is not None:
            await self._defer_goal_status_notice_after_delivery(source, msg)

    async def _loop_wakeup_fire_one(self, sid: str, state: Any, now: float, warned_no_route: set) -> None:
        """Inject one due /loop wakeup into its session, applying every deferral rule."""
        from hermes_cli.loops import LoopManager, goal_blocks_loop_tick

        if state.awaiting_response or now < state.next_due_at:
            return
        route = state.route or {}
        platform_name = route.get("platform", "")
        chat_id = route.get("chat_id", "")
        if not platform_name or not chat_id:
            return  # CLI / TUI-owned loop — their own schedulers drive it.
        adapter = next((a for p, a in self.adapters.items() if p.value == platform_name), None)
        if adapter is None:
            if sid not in warned_no_route:
                warned_no_route.add(sid)
                logger.debug(
                    "loop wakeup: no adapter for platform %r (session %s)", platform_name, sid,
                )
            return

        source = self._build_process_event_source({
            "session_key": "",
            "platform": platform_name,
            "chat_id": chat_id,
            **{k: route.get(k, "") for k in ("chat_type", "thread_id", "user_id", "user_name")},
        })
        if source is None:
            return
        session_key = None
        with suppress(Exception):
            session_key = self._session_key_for_source(source)
        if session_key and session_key in self._running_agents:
            return  # busy — stays due, next scan retries
        if goal_blocks_loop_tick(sid):
            return

        mgr = LoopManager(session_id=sid)
        if not mgr.is_due(now):
            return
        # fire_tick()/complete_tick() are writes (BEGIN IMMEDIATE) taking the SessionDB writer lock; a slow
        # writer elsewhere holding it while the loop thread blocked froze the gateway until the watchdog
        # fired. The context-preserving executor keeps the profile HERMES_HOME override under multiplex.
        wakeup = await self._run_in_executor_with_context(mgr.fire_tick)
        if not wakeup:
            return
        # #85957: after the parent turn's event.complete the CLIENT owns the next turn on this stateless
        # surface. Persist the completion as a durable delivery row — never self-post it as a new role=user
        # prompt.
        # #85957: same client-owns-the-turn rule as the raw-key branch above — persist the completion as a
        # delivery row, never self-post it as a new role=user prompt.
        try:
            logger.info(
                "loop wakeup #%s — injecting for %s chat=%s thread=%s",
                mgr.state.ticks_fired if mgr.state else "?",
                platform_name, source.chat_id, source.thread_id,
            )
            await adapter.handle_message(self._synthetic_prompt_event(source, wakeup, internal=True))
            # Slash-command loops dispatch through the command path and never hit the post-turn
            # completion hook — complete the tick immediately (caps + scheduling).
            if wakeup.lstrip().startswith("/"):
                await self._run_in_executor_with_context(mgr.complete_tick, "")
        except Exception as exc:
            logger.warning("loop wakeup injection failed for %s: %s", sid, exc)
            with suppress(Exception):
                mgr.abandon_tick()

    async def _loop_wakeup_watcher(self, interval: float = 15.0) -> None:
        """Fire due /loop wakeups for idle gateway sessions: a coarse ticker scans persisted loops
        (SessionDB ``loop:*`` rows) and injects each due prompt via the synthetic-message path.
        Deferrals: session running a turn (FIFO would race the live turn); active non-parked /goal
        (goal owns the idle boundary); no routing metadata (one-time warning)."""
        await asyncio.sleep(5)  # let platforms finish connecting
        warned_no_route: set = set()
        while self._running:
            try:
                from hermes_cli.loops import list_active_loops

                # Warm once per scan: the scan reads every persisted loop and a cold cache would
                # run the state.db init on the loop thread before the first read.
                await self._warm_goals_session_db("loop wakeup")
                # Off-loop too: the read is lock-free under WAL but convoys on the writer lock without it.
                active_loops = await self._run_in_executor_with_context(list_active_loops)
                now = time.time()
                for sid, state in active_loops:
                    await self._loop_wakeup_fire_one(sid, state, now, warned_no_route)
            except Exception as exc:
                logger.debug("loop wakeup watcher error: %s", exc)
            await asyncio.sleep(interval)
