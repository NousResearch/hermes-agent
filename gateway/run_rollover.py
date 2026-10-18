"""Gateway-owned handoff/cutover/bootstrap under the existing lane claim.

The model authors continuity; only the store transaction may move ownership.
Native busy queues remain process-local, not a crash-durable inbox.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from gateway.rollover_packet import (
    RolloverBlocked, RolloverPolicy, RolloverPreflightBlocked,
    handoff_instruction, validate_packet, verify_packet_binding,
)

logger = logging.getLogger("gateway.run")


@dataclass
class LaneRollover:
    policy: RolloverPolicy | None
    session_id: str
    guard: object = None
    phase: str = "watching"
    hygiene: object = None
    fallback_used: bool = False


def lane_rollover(runner, key):
    state = runner._peek_session_state(key)
    return getattr(state.persistent, "rollover", None) if state is not None else None


def queue_during_rollover(runner, event, key) -> bool:
    lane = lane_rollover(runner, key)
    if lane is None or lane.phase not in {"authoring", "committing", "bootstrapping"}:
        return False
    from hermes_cli.commands import resolve_command
    # Unknown slash-prefixed text is ordinary input, not control authority.
    command = event.get_command()
    if command and resolve_command(command):
        return False
    adapter = runner._delivery_adapter_for(event.source)
    if adapter is None:
        raise RolloverBlocked("Rollover cannot retain input without its owning adapter")
    runner._enqueue_fifo(key, event, adapter)
    return True


def bind_rollover_guard(runner, ctx, agent) -> None:
    from agent.session_rollover import RolloverBudgetUnavailable, RolloverGuard, resolve_rollover_budget

    lane = lane_rollover(runner, ctx.session_key)
    if lane is not None and lane.phase == "refresh":
        lane.session_id = agent.session_id
    if (lane is None or lane.policy is None or lane.session_id != agent.session_id
            or lane.phase in {"bootstrapping", "admitting"}):
        agent._session_rollover = None
        return
    if ctx.moa_config or getattr(agent, "api_mode", "") == "codex_app_server":
        raise RolloverPreflightBlocked("This runtime does not expose every rollover safe point")
    if lane.guard is None:
        compressor = agent.context_compressor
        lane.guard = RolloverGuard(
            runway_tokens=lane.policy.runway_tokens,
            safety_margin_tokens=lane.policy.safety_margin_tokens,
            hygiene_threshold_tokens=int(compressor.context_length * lane.hygiene.threshold_pct),
            hygiene_message_limit=lane.hygiene.hard_msg_limit,
        )
    if lane.phase == "authoring":
        lane.guard.phase = "authoring"
    elif lane.phase == "refresh":
        lane.guard.phase = "requested"
    try:
        resolve_rollover_budget(agent, lane.guard)
    except RolloverBudgetUnavailable as exc:
        raise RolloverPreflightBlocked("The bound runtime cannot satisfy the rollover context budget") from exc
    agent._session_rollover = lane.guard


def _queue_is_empty(runner, key, source) -> bool:
    adapter = runner._delivery_adapter_for(source)
    state = runner._session_state(key)
    if runner._queue_depth(key, adapter=adapter) or state.persistent.pending_command_text:
        return False
    if state.conversation.sidecar_notes:
        return False
    # The debounce bucket is accepted input even before it reaches the FIFO head.
    if adapter is not None and key in adapter._text_debounce_store():
        return False
    return True


def _blocking_obligation(runner, key, session_id) -> str | None:
    """Runs in the owning profile's executor scope; never on the gateway event loop."""
    from tools.approval import has_blocking_approval, get_pending_gateway_approval
    from tools.async_delegation import has_live_for_session
    from tools.clarify_gateway import get_pending_for_session
    from tools.process_registry import process_registry
    from tools.slash_confirm import get_pending

    state = runner._session_state(key)
    if (state.persistent.approvals or state.persistent.update_prompt_pending
            or has_blocking_approval(key) or get_pending_gateway_approval(key)
            or get_pending_for_session(key, include_choice_prompts=True) or get_pending(key)):
        return "an approval or unanswered question is pending"
    if state.conversation.one_turn_restore:
        return "a one-turn model override has not settled"
    if has_live_for_session(session_key=key, parent_session_id=session_id):
        return "a callback-dependent worker is still active"
    db = runner.session_store._db_for_key(key)
    with db._read_ctx() as conn:
        pending = conn.execute(
            "SELECT 1 FROM async_delegations WHERE (origin_session=? OR parent_session_id=?) "
            "AND (state IN ('running','stalling','finalizing') OR delivery_state='pending') LIMIT 1",
            (key, session_id),
        ).fetchone()
    if pending:
        return "a worker completion has not been reconciled"
    for process in process_registry.list_sessions(task_id=session_id, session_key=key):
        callback = process.get("notify_on_complete") or process.get("watch_patterns")
        if callback and not _process_callback_settled(runner, process):
            return "a callback-dependent process has not settled"
    return None


def _process_callback_settled(runner, process) -> bool:
    from tools.process_registry import process_registry

    if process["status"] != "exited":
        return False
    session_id = process["session_id"]
    if process_registry.is_completion_consumed(session_id):
        return True
    record = process_registry.get(session_id)
    lock = getattr(runner, "_completion_delivery_lock", None)
    if record is None or lock is None:
        return False
    identity = ("completion", session_id, record.started_at)
    with lock:
        return identity in runner._completion_deliveries_delivered


class GatewayRolloverMixin:
    async def _rollover_before_hygiene(self, event, source, entry, key, history) -> bool:
        """Admit the opt-in before hygiene, deferring its decision to the actual agent engine.

        A cold lane has no resolved compressor yet. Reconstructing one from profile YAML
        would miss provider overrides; its turn-start guard makes the decision instead.
        """
        from gateway.run import _load_gateway_config
        from gateway.config import Platform

        lane = lane_rollover(self, key)
        if lane is not None and lane.phase == "admitting" and lane.session_id == entry.session_id:
            return True
        policy = await asyncio.to_thread(lambda: RolloverPolicy.for_lane(_load_gateway_config(), key))
        if policy is None:
            self._session_state(key).persistent.rollover = None
            return False
        if source.platform == Platform.LOCAL:
            raise RolloverBlocked("This rollover path owns gateway messaging lanes, not local-session admissions")
        if self._get_proxy_url():
            raise RolloverBlocked("Rollover requires the local gateway agent and session store")
        if getattr(event, "_moa_config", None):
            raise RolloverBlocked("Mixture-of-agents turns do not expose every rollover safe point")
        if lane is None or lane.policy != policy or lane.session_id != entry.session_id:
            probe_path = await asyncio.to_thread(policy.new_packet_path, entry.session_id)
            await asyncio.to_thread(handoff_instruction, policy, probe_path, entry.session_id)
            lane = LaneRollover(policy, entry.session_id)
            self._session_state(key).persistent.rollover = lane
        lane.hygiene = await self._hmwa_hygiene_settings(source, key)
        if lane.guard is not None and lane.guard.phase == "invalidated":
            # The emergency path explicitly releases hygiene only after invalidation.
            lane.guard = None
            lane.phase = "refresh"
            return False
        return True

    async def _rollover_pending_receipt(self, entry, key):
        receipt = await self.async_session_store.get_rollover(key)
        lane = lane_rollover(self, key)
        if not receipt or receipt["phase"] in {"complete", "invalidated"}:
            if lane is not None and (lane.phase == "admitting" or lane.session_id != entry.session_id):
                self._session_state(key).persistent.rollover = None
            return None
        owner = receipt["new_session_id"] if receipt["owner_committed"] else receipt["old_session_id"]
        if owner != entry.session_id:
            await self.async_session_store.set_rollover_phase(
                key, receipt["new_session_id"], receipt["phase"], "invalidated",
                "The native route was replaced; the old handoff no longer owns this lane.",
            )
            self._session_state(key).persistent.rollover = None
            return None
        return receipt

    async def _rollover_recover_before_turn(self, event, source, entry, key, generation):
        receipt = await self._rollover_pending_receipt(entry, key)
        admission_id = getattr(event, "_rollover_admission_session_id", None)
        if admission_id and (receipt is None or receipt["new_session_id"] != admission_id
                             or receipt["phase"] != "admitting"):
            return True, None
        if receipt is None:
            return bool(getattr(event, "_rollover_recovery_wake", False)), None
        adapter = self._delivery_adapter_for(source)
        if receipt["phase"] != "committed" and adapter is not None and key not in adapter._pending_messages:
            self._promote_queued_event(key, adapter, event)
        lane = lane_rollover(self, key)
        if receipt["phase"] == "admitting" and lane is not None and lane.phase == "admitting":
            if admission_id or getattr(event, "_rollover_recovery_wake", False):
                await self._hmwa_acquire_turn_lease(key, generation, entry, ())
                await self._mark_durable_active_turn(event, key)
                try:
                    return True, await self._rollover_admit(source, entry, key, generation, receipt, event)
                finally:
                    await self._clear_durable_active_turn(event)
            return False, None
        if receipt["phase"] != "committed":
            return True, (
                "Automatic continuity is blocked for reconciliation: " +
                (receipt.get("phase_reason") or receipt.get("reason") or
                 "a previous rollover was interrupted; input or action outcomes may be uncertain") +
                ". No operational action or new successor was started."
            )
        # Retain this arrival; the bound bootstrap is compulsory first-turn context.
        if not getattr(event, "_rollover_recovery_wake", False):
            if adapter is None:
                raise RolloverBlocked("No owning adapter can retain the pending input")
            previous = adapter._pending_messages.get(key)
            if previous is not None:
                self._session_state(key).conversation.queued_events.insert(0, previous)
            adapter._pending_messages[key] = event
            event._rollover_fifo_input = True
        return True, await self._rollover_bootstrap(source, entry, key, generation, receipt, event)

    async def _rollover_after_retained_turn(self, event, source, entry, key, generation, result):
        receipt = await self._rollover_pending_receipt(entry, key)
        if receipt is None:
            return False, None
        if result.get("failed") or result.get("interrupted") or result.get("error"):
            return True, await self._rollover_block_successor(
                key, entry, receipt, "Retained input did not complete; reconcile before continuation.",
            )
        if _queue_is_empty(self, key, source):
            from gateway.platforms.event import MessageEvent
            adapter = self._delivery_adapter_for(source)
            admission = MessageEvent(
                text="[Internal continuity admission pending]", source=source, internal=True,
                allow_gateway_control=False, metadata={"gateway_session_key": key},
            )
            admission._rollover_admission_session_id = entry.session_id
            self._enqueue_fifo(key, admission, adapter)
        # Let the native caller deliver this retained input's own shaped reply
        # before the adapter drains the identity-bound continuation event.
        return False, None

    async def _rollover_after_turn(self, event, source, entry, key, generation, result):
        lane = lane_rollover(self, key)
        if lane is None:
            return False, None
        if result.get("rollover_fallback"):
            lane.fallback_used = True
            lane.session_id = entry.session_id
        if lane.phase == "admitting":
            return await self._rollover_after_retained_turn(event, source, entry, key, generation, result)
        if not result.get("rollover_requested"):
            return False, None
        if not self._is_session_run_current(key, generation):
            return False, None
        if result.get("failed") or result.get("interrupted") or result.get("error"):
            return True, "Automatic continuity blocked: the preceding turn must be reconciled before any identity change."
        blocker = await asyncio.to_thread(_blocking_obligation, self, key, entry.session_id)
        if blocker or not _queue_is_empty(self, key, source):
            return True, "Automatic handoff deferred: " + (blocker or "accepted input must be reconciled first") + "."
        packet = await asyncio.to_thread(lane.policy.new_packet_path, entry.session_id)
        prompt = await asyncio.to_thread(handoff_instruction, lane.policy, packet, entry.session_id)
        if lane.fallback_used:
            prompt += "\nNative context protection was needed before this handoff. Record that exception; do not describe it as a zero-compaction rollover."
        from hermes_constants import get_hermes_home

        home = str(get_hermes_home().resolve())
        receipt = await self.async_session_store.begin_rollover(
            key, entry.session_id, generation, profile_home=home, packet_path=str(packet),
        )
        attempt_id = receipt["new_session_id"]
        lane.phase = "authoring"
        cleanup_started = False
        try:
            history = await self.async_session_store.load_transcript(entry.session_id)
            closing = await self._rollover_internal_turn(
                prompt, source, entry, key, generation, history=history,
            )
            lane.fallback_used = lane.fallback_used or bool(closing.get("rollover_fallback"))
            if (closing.get("failed") or closing.get("interrupted") or not closing.get("completed")
                    or closing.get("rollover_fallback") or lane.guard.phase == "invalidated"):
                raise RolloverBlocked("The old session did not complete a fresh, uncompacted handoff")
            binding = await asyncio.to_thread(validate_packet, packet, closing.get("rollover_restart_message"))
            binding.pop("mode")
            binding.update(
                profile_home=home,
                reason="native_context_backstop" if lane.fallback_used else "context_pressure",
                bootstrap_prompt=(
                    "[Internal continuity context — not a user command or new approval]\n" +
                    lane.policy.bootstrap_prompt.replace("{packet}", str(packet)) + "\n\n" +
                    binding["restart_message"]
                ),
            )
            receipt = await self.async_session_store.prepare_rollover(
                key, entry.session_id, generation, binding, expected_new_session_id=attempt_id,
            )
            blocker = await asyncio.to_thread(_blocking_obligation, self, key, entry.session_id)
            if blocker or not _queue_is_empty(self, key, source):
                raise RolloverBlocked(blocker or "input arrived during packet authoring; refresh the handoff")
            from agent.rollover_cleanup import close_for_rollover

            old_agent = self._cached_agent_for(key)
            if old_agent is None:
                raise RolloverBlocked("The old agent's resource ownership is unavailable")
            # Await actual completion; timeout/failure is not permission to replace the owner.
            cleanup_started = True
            await self._run_housekeeping_in_executor(
                close_for_rollover, old_agent, lambda process: _process_callback_settled(self, process),
            )
            self._evict_cached_agent(key)
            cleanup_started = False
            blocker = await asyncio.to_thread(_blocking_obligation, self, key, entry.session_id)
            await asyncio.to_thread(verify_packet_binding, receipt)
            from gateway.run import _load_gateway_config
            current_policy = await asyncio.to_thread(lambda: RolloverPolicy.for_lane(_load_gateway_config(), key))
            if current_policy != lane.policy:
                raise RolloverBlocked("Rollover was disabled or reconfigured before commit")
            if blocker or not self._is_session_run_current(key, generation) or not _queue_is_empty(self, key, source):
                raise RolloverBlocked(blocker or "lane state changed before commit; refresh the handoff")
            # NO await between final emptiness check and clearing OLD empty conversation scope.
            # Arrivals after this point belong to the existing FIFO and are never cleared again.
            overrides = self._session_state(key).conversation
            retained = (overrides.model_override, overrides.reasoning_override, overrides.service_tier_override)
            self._clear_conversation_scope(key, reason="automatic_rollover", persist_yolo=False)
            overrides.model_override, overrides.reasoning_override, overrides.service_tier_override = retained
            lane.phase = "committing"
            successor = await self.async_session_store.commit_rollover(
                key, entry.session_id, generation, expected_new_session_id=attempt_id,
            )
            if not self._is_session_run_current(key, generation):
                raise RolloverBlocked("A newer control command superseded this committed transition")
            if not self._rebind_turn_lease(key, generation, successor.session_id):
                raise RolloverBlocked("The committed successor could not inherit the held turn lease")
            self._evict_cached_agent(key)
            lane.session_id = successor.session_id
            receipt = await self.async_session_store.get_rollover(key)
            if (receipt["new_session_id"] != attempt_id
                    or not self._is_session_run_current(key, generation)):
                raise RolloverBlocked("A newer transition superseded this handoff")
            return True, await self._rollover_bootstrap(source, successor, key, generation, receipt, event)
        except asyncio.CancelledError:
            await self._rollover_fail_attempt(
                key, lane, attempt_id, True, "Handoff turn was cancelled; reconcile uncertain outcomes",
            )
            raise
        except Exception as exc:
            logger.exception("Automatic rollover could not complete for %s", key)
            return True, await self._rollover_fail_attempt(key, lane, attempt_id, cleanup_started, exc)

    async def _rollover_fail_attempt(self, key, lane, attempt_id, uncertain, error):
        current = await self.async_session_store.get_rollover(key)
        same_attempt = current and current["new_session_id"] == attempt_id
        committed = same_attempt and current.get("owner_committed")
        blocked = committed or uncertain
        if same_attempt and current["phase"] not in {"complete", "blocked", "invalidated"}:
            await self.async_session_store.set_rollover_phase(
                key, current["new_session_id"], current["phase"],
                "blocked" if blocked else "invalidated", str(error),
            )
        if lane_rollover(self, key) is lane and same_attempt:
            lane.phase = "blocked" if blocked else "watching"
            if not blocked and lane.guard is not None and lane.guard.phase != "invalidated":
                lane.guard.phase = "requested"
        return f"Automatic continuity {'blocked' if blocked else 'deferred'}: {error}."

    async def _rollover_internal_turn(self, prompt, source, entry, key, generation, *, history):
        from gateway.session import build_session_context
        from gateway.run import _load_gateway_config

        context = build_session_context(source, self.config, entry)
        tokens = self._set_session_env(context)
        try:
            config = await asyncio.to_thread(_load_gateway_config)
            redact_pii = bool((config.get("privacy") or {}).get("redact_pii", False))
            context_prompt = self._pinned_session_context_prompt(context, redact_pii, key, internal=True)
            channel_prompt, turn_source = self._pinned_channel_inputs(key, None, source, internal=True)
            result = await self._run_agent(
                message=prompt, context_prompt=context_prompt, history=history, source=turn_source,
                session_id=entry.session_id, session_key=key, run_generation=generation,
                channel_prompt=channel_prompt, persist_user_message=prompt,
                persist_user_display_kind="internal", title_user_message="",
            )
            await self.async_session_store.update_session(
                key, last_prompt_tokens=result.get("last_prompt_tokens", 0), touch_activity=False,
            )
            await self._refresh_agent_cache_message_count(key, entry.session_id)
            return result
        finally:
            self._clear_session_env(tokens)

    async def _rollover_deliver_result(self, result, event, source, entry, key, generation):
        response = result.get("final_response") or ""
        return await self._hmwa_deliver_turn_response(
            event, source, entry, key, generation, result, result.get("messages", []),
            response, "", self._is_intentional_silence(result, response),
        )

    async def _rollover_block_successor(self, key, entry, receipt, reason):
        current = await self.async_session_store.get_rollover(key)
        if (current and current["new_session_id"] == receipt["new_session_id"]
                and current["phase"] in {"bootstrapping", "admitting"}):
            await self.async_session_store.set_rollover_phase(
                key, entry.session_id, current["phase"], "blocked", reason,
            )
        lane = lane_rollover(self, key)
        if lane is not None and lane.session_id == entry.session_id:
            lane.phase = "blocked"
        return f"Automatic continuity blocked in successor {entry.session_id}: {reason}."

    async def _rollover_complete(self, key, entry, phase):
        await self.async_session_store.set_rollover_phase(key, entry.session_id, phase, "complete")
        lane = lane_rollover(self, key)
        lane.phase = "watching"
        lane.guard = None
        lane.fallback_used = False

    async def _rollover_admit(self, source, entry, key, generation, receipt, event):
        try:
            history = await self.async_session_store.load_transcript(entry.session_id)
            await asyncio.to_thread(verify_packet_binding, receipt)
            if not self._is_session_run_current(key, generation):
                raise RolloverBlocked("A newer stop or reset superseded continuation")
            if not _queue_is_empty(self, key, source):
                return None
            result = await self._rollover_internal_turn(
                "[Internal continuity admission — no new authority]\n"
                "Recentered continuity is admitted. Account for every newer input and stop before "
                "following the reconciled Continuation Directive: take only its still-authorized "
                "next action or existing monitoring/reconciliation duty. Never revive cancelled "
                "work. If awaiting the user or nothing remains authorized, remain idle.",
                source, entry, key, generation, history=history,
            )
            if (result.get("failed") or result.get("interrupted") or not result.get("completed")
                    or not self._is_session_run_current(key, generation)):
                raise RolloverBlocked("Continuation did not complete; reconcile uncertain outcomes")
            await self._rollover_complete(key, entry, "admitting")
            return await self._rollover_deliver_result(result, event, source, entry, key, generation)
        except asyncio.CancelledError:
            await self._rollover_block_successor(
                key, entry, receipt, "Continuation was cancelled; reconcile uncertain outcomes",
            )
            raise
        except Exception as exc:
            logger.exception("Automatic rollover continuation failed for %s", key)
            return await self._rollover_block_successor(key, entry, receipt, str(exc))

    async def _rollover_bootstrap(self, source, entry, key, generation, receipt, delivery_event):
        from gateway.platforms.event import MessageEvent

        await asyncio.to_thread(verify_packet_binding, receipt)
        lane = lane_rollover(self, key)
        if lane is None:
            lane = LaneRollover(None, entry.session_id)
            self._session_state(key).persistent.rollover = lane
        lane.phase = "bootstrapping"
        lane.session_id = entry.session_id
        # The normal cutover still holds its rebound lease. Recovery acquires it once.
        if self._held_turn_lease(key, generation) is None:
            await self._hmwa_acquire_turn_lease(key, generation, entry, ())
        await self.async_session_store.set_rollover_phase(key, entry.session_id, "committed", "bootstrapping")
        event = MessageEvent(text=receipt["bootstrap_prompt"], source=source, internal=True,
                             allow_gateway_control=False)
        try:
            await self._rehydrate_prompt_pins(key, entry.session_id)
            await self._hmwa_open_session(entry, key, source)
            await self._mark_durable_active_turn(event, key)
            # Recenter before acting: later queued input can supersede the old directive.
            prompt = (
                receipt["bootstrap_prompt"] + "\n\n[Gateway first-turn admission]\n"
                "This first turn is recentering only. Read the bound packet and current authorities, "
                "reconcile state, and identify the directive's mode and next authorized action. Do not "
                "start operational work yet. For existing authorized duties the gateway will admit "
                "continuation after bootstrap and newer queued input. For await-user, ask only the "
                "directive's unresolved question in this response, then wait; do not postpone the "
                "question to another turn. Idle means no work. A later genuine user instruction "
                "can supply new authority normally."
            )
            result = await self._rollover_internal_turn(prompt, source, entry, key, generation, history=[])
            if result.get("failed") or result.get("interrupted") or not result.get("completed"):
                raise RolloverBlocked("Successor bootstrap did not complete; reconcile this same successor")
            if not self._is_session_run_current(key, generation):
                raise RolloverBlocked("A newer stop or reset superseded bootstrap")
            bound = await asyncio.to_thread(verify_packet_binding, receipt)
            if bound["mode"] in {"await-user", "idle"}:
                await self._rollover_complete(key, entry, "bootstrapping")
            else:
                receipt = await self.async_session_store.set_rollover_phase(
                    key, entry.session_id, "bootstrapping", "admitting",
                )
                lane.phase = "admitting"
                if _queue_is_empty(self, key, source):
                    return await self._rollover_admit(source, entry, key, generation, receipt, delivery_event)
            return await self._rollover_deliver_result(result, delivery_event, source, entry, key, generation)
        except asyncio.CancelledError:
            await self._rollover_block_successor(
                key, entry, receipt, "Bootstrap was cancelled; reconcile uncertain outcomes",
            )
            raise
        except Exception as exc:
            logger.exception("Automatic rollover bootstrap failed for %s", key)
            return await self._rollover_block_successor(key, entry, receipt, str(exc))
        finally:
            await self._clear_durable_active_turn(event)

    def _schedule_rollover_recoveries(self, platform=None):
        first_recovery = not getattr(self, "_rollover_recovery_started", False)
        self._rollover_recovery_started = True
        task = self._retain_background_task(asyncio.create_task(
            self._restore_rollover_sessions(platform, recover_interrupted=first_recovery),
        ))
        if getattr(self, "_startup_restore_in_progress", False):
            tasks = getattr(self, "_startup_restore_tasks", None)
            if tasks is None:
                tasks = self._startup_restore_tasks = []
            tasks.append(task)

    async def _restore_rollover_sessions(self, platform=None, *, recover_interrupted=False):
        from gateway.platforms.event import MessageEvent

        from gateway.run import _multiplex_profile_homes
        from hermes_constants import get_routing_process_hermes_home
        homes = (
            [home for _name, home in _multiplex_profile_homes(self.config)]
            if getattr(self.config, "multiplex_profiles", False) else [get_routing_process_hermes_home()]
        )
        receipts = await self.async_session_store.list_pending_rollovers(
            profile_homes=homes, recover_interrupted=recover_interrupted,
        )
        for receipt in receipts:
            key = receipt["session_key"]
            try:
                entry = await self.async_session_store.lookup_by_session_key(key)
                if entry is None or entry.origin is None or self._is_session_running(key):
                    continue
                source = self._restored_source(entry)
                if platform is not None and source.platform != platform:
                    continue
                adapter = self._delivery_adapter_for(source)
                if adapter is None or not self._resume_owner_authorized(key, source):
                    continue
                # The adapter owns ingress; a preclaimed runner sentinel queues its own wake.
                # A concurrent ordinary arrival sees the receipt and must bootstrap first.
                event = MessageEvent(
                    text="", source=source, internal=True, allow_gateway_control=False,
                    metadata={"gateway_session_key": key},
                )
                event._rollover_recovery_wake = True
                async with self._async_profile_scope_for_source(source):
                    await self._run_startup_resume_event(adapter, event, key)
            except Exception:
                logger.exception("Rollover recovery failed for %s; continuing other lanes", key)
