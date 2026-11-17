"""Settle only a successfully completed, still-owned turn; late input uses the normal queue."""
import asyncio
import logging
from contextlib import nullcontext
import uuid

logger = logging.getLogger(__name__)


def is_group_topic(source):
    return (source is not None and getattr(source.platform, "value", source.platform) == "telegram"
            and source.chat_type in ("group", "supergroup") and bool(source.thread_id))


def _settle(runner, source, entry, key, generation, result, terminal=None, owners=()):
    from gateway.fresh_session.publication import publish
    from tools.async_delegation import has_live_for_session
    from agent.conversation_compression import _compression_lock_holder

    store = runner.session_store
    db = runner._sync_session_db()
    intent = result.get("fresh_topic_handoff")
    if db is None or intent is None:
        return False
    with store._lock:
        current = store._entry_locked(key)
        if (current is None or current.session_id != intent["session_id"]
                or not runner._is_session_run_current(key, generation)):
            return False
        state = runner._peek_session_state(key)
        adapter = runner._delivery_adapter_for(source)
        if (adapter is None or key in adapter._pending_messages
                or (state and state.conversation.queued_events)
                or (state and runner._agent_has_active_subagents(state.turn.agent))
                or any(_has_work(runner, owner) for owner in owners)
                or has_live_for_session(session_key=key, parent_session_id=intent["session_id"])
                or store._has_active_processes_safe(key, context="fresh-session handoff")):
            return False
        holder = _compression_lock_holder(None)
        if not db.try_acquire_session_turn_lease(intent["session_id"], holder):
            return False
        try:
            child_id = str(uuid.uuid4())
            if not publish(db, parent_id=intent["session_id"], child_id=child_id, summary=intent["summary"],
                           final_response=result.get("final_response") or "", source=source,
                           session_key=key, holder=holder, terminal=terminal):
                return False
            # The binding committed with the child is authoritative if sessions.json save crashes.
        finally:
            db.release_session_turn_lease(intent["session_id"], holder)
        # Keep publication and mirror adoption under the same route lock. An inbound
        # resolver must not recover the child in the gap and make our CAS retire it.
        switched = store.switch_session(key, child_id, expected_session_id=intent["session_id"], require_live_target=True)
        if switched is None:
            db.end_session(child_id, "superseded_by_new")
        else:
            runner._clear_session_boundary_security_state(key)
        return switched is not None


async def settle(runner, source, entry, key, generation, result, *, terminal=None):
    from gateway.fresh_session.serialization import topic_boundary

    async with topic_boundary(runner, source):
        return await _settle_owned(runner, source, entry, key, generation, result, terminal=terminal)


async def _settle_owned(runner, source, entry, key, generation, result, *, terminal=None):
    if not is_group_topic(source) or not result.get("fresh_topic_handoff"):
        return False
    if result.get("completed") is not True or any(result.get(k) for k in ("failed", "error", "interrupted", "partial")):
        return False
    if terminal is None and _defer_to_terminal(runner, source, entry, key, generation, result):
        return False
    try:
        settled = await asyncio.to_thread(_settle, runner, source, entry, key, generation, result, terminal, _work_owners(runner, key, result))
    except Exception:
        logger.exception("Fresh-session handoff settlement failed; durable topic binding governs recovery")
        return False
    if settled and not runner._is_session_run_current(key, generation):
        return False
    if settled:
        runner._evict_cached_agent(key, commit_memory=True, release_after_turn=True)
    if settled:
        from gateway.fresh_session.delivery import defer_note, send_note
        if terminal is None and result.get("already_sent"):
            await send_note(runner, source)
        else:
            defer_note(runner, source, key, generation, terminal)
    return settled


def _defer_to_terminal(runner, source, entry, key, generation, result):
    from gateway.session_authorities import active_authority
    from gateway.session_ingress import executing_admission
    from gateway.session_results import execution_result

    if not executing_admission.get():
        return False
    authority = active_authority(runner)
    row = (execution_result.get() or {}).get("admission")
    if authority is not None and row is not None and key not in authority.deferred_resets:
        terminal = (row["admission_id"], row["generation"], row["owner_epoch"])
        async def after_terminal():
            from gateway.session_ingress import _execution_scope
            with await _execution_scope(authority, None):
                return await settle(runner, source, entry, key, generation, result, terminal=terminal)
        authority.deferred_resets[key] = after_terminal
    # An admitted turn never attempts early settlement, even if its owner is unavailable.
    return True


def _work_owners(runner, key, result):
    from gateway.run_agent_cache import _first_agent
    from gateway.session_authorities import active_authority

    # Snapshot cache separately: never acquire its lock while holding the route lock.
    with getattr(runner, "_agent_cache_lock", None) or nullcontext():
        cached = _first_agent(getattr(runner, "_agent_cache", {}).get(key))
    authority = active_authority(runner)
    parent_id = result["fresh_topic_handoff"]["session_id"]
    live = authority.sessions.get(parent_id) if authority else None
    return (cached, live.review if live else None, live)


def _has_work(runner, owner):
    review = getattr(owner, "_background_review_run", None)
    controls = getattr(owner, "controls", None)
    return (bool(controls and (controls.pending or controls.remote_responders))
            or runner._agent_has_active_subagents(owner)
            or (review is not None and not review.request_done.is_set()))
