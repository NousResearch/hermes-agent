"""Consume a checkpoint-bound worker model handoff using the existing gateway model lifecycle."""
from __future__ import annotations

import asyncio
import logging
from copy import deepcopy
from types import SimpleNamespace

from hermes_cli import goal_model_recovery as recovery

logger = logging.getLogger(__name__)


def model_context(runner, source):
    from gateway.slash_commands_model import _ModelSwitchContext
    from hermes_constants import get_hermes_home
    key = runner._session_key_for_source(source)
    ctx = _ModelSwitchContext(key, source, get_hermes_home() / "config.yaml", False)
    ctx.read_config()
    runner._rehydrate_session_model_override(key)
    ctx.apply_override(runner._session_model_override(key) or {})
    agent = runner._cached_agent_for(key)
    if agent is not None:
        ctx.current_model = getattr(agent, "model", ctx.current_model)
        ctx.current_provider = getattr(agent, "provider", ctx.current_provider)
        ctx.route_provider = ctx.current_provider
    return ctx


def worker_route(runner, source):
    ctx = model_context(runner, source)
    return recovery._route({"provider": ctx.current_provider, "model": ctx.current_model})


def _scope_enabled(session_id, target):
    cfg = recovery.settings()
    routes = cfg.get("routes")
    return (recovery.enabled_for(session_id, cfg) and isinstance(routes, list)
            and target in [recovery._route(value) for value in routes])


async def probe_candidate(runner, result):
    """One benign generation/tool-call probe; no tool is executed and a fallback is rejected."""
    def probe():
        from agent.auxiliary_client import call_llm
        info = {}
        response = call_llm(
            provider=result.target_provider, model=result.new_model, api_key=result.api_key,
            base_url=result.base_url, api_mode=result.api_mode, timeout=12, max_tokens=128,
            messages=[{"role":"user", "content":"Call hermes_route_probe with an empty object. This is a harmless capability probe."}],
            tools=[{"type":"function","function":{"name":"hermes_route_probe",
                    "description":"A no-effect model capability probe; never executed.",
                    "parameters":{"type":"object","properties":{},"additionalProperties":False}}}],
            route_info=info,
        )
        if info.get("provider") != result.target_provider or info.get("model") != result.new_model:
            return False
        calls = response.choices[0].message.tool_calls or []
        return any(call.function.name == "hermes_route_probe" for call in calls)
    return await asyncio.wait_for(runner._run_in_executor_with_context(probe), timeout=20)


def _same_worker(runner, key, session_id, source):
    return (runner.session_store.peek_session_id(key) == session_id
            and runner._is_user_authorized_for_source(source))


async def apply(runner, entry, source, request):
    """Returns the committed reservation, or None; only then may its continuation be queued."""
    with runner._profile_scope_for_source(source):
        key = runner._session_key_for_source(source)
        sid = entry.session_id
        if request.get("session_id") != sid:
            return None
        if not _same_worker(runner, key, sid, source):
            manager = recovery.current_pending(sid, request["id"])
            if manager is not None:
                recovery.settle(manager, request["id"], "cancelled", error_type="OwnerOrRouteChanged")
            return None
        async with runner._model_switch_lock():
            manager = recovery.current_pending(sid, request["id"])
            if manager is None:
                return None
            if not _scope_enabled(sid, request["target"]):
                recovery.settle(manager, request["id"], "cancelled", error_type="ScopeDisabled")
                return None
            ctx = model_context(runner, source)
            current = {"provider":ctx.current_provider, "model":ctx.current_model}
            if current == request["target"]:
                # Adopt only an override whose generation qualification was durably recorded.
                proof = manager.state.model_recovery["pending"].get("qualification") or {}
                if (proof.get("provider") == current["provider"] and proof.get("model") == current["model"]
                        and proof.get("probe") == "tool_generation"):
                    return recovery.settle(manager, request["id"], "committed")
                recovery.settle(manager, request["id"], "failed", error_type="UnqualifiedPersistedRoute")
                return None
            if current != request["prior"]:
                recovery.settle(manager, request["id"], "superseded")
                return None
            snapshot = runner._snapshot_session_model_override(key)
            try:
                result, error = await runner._perform_model_switch(
                    ctx, request["target"]["model"], request["target"]["provider"], source)
                if error or result is None:
                    raise RuntimeError("Model resolution failed")
                if {"provider":result.target_provider, "model":result.new_model} != request["target"]:
                    raise RuntimeError("Resolved route differs from approved target")
                if not await probe_candidate(runner, result):
                    raise RuntimeError("Generation/provenance/tool capability probe failed")
                if (recovery.current_pending(sid, request["id"]) is None
                        or not _same_worker(runner, key, sid, source)
                        or not _scope_enabled(sid, request["target"])
                        or runner._snapshot_session_model_override(key) != snapshot):
                    recovery.settle(manager, request["id"], "cancelled", error_type="UserOrScopeChanged")
                    return None
                if recovery.qualify(manager, request["id"]) is None:
                    return None
                durable = {"model":result.new_model, "provider":result.target_provider, "base_url":result.base_url}
                # Existing /model persistence is best-effort. Recovery must not claim a durable
                # switch until this strict non-secret write succeeds.
                await runner.async_session_store.set_model_override(key, durable)
                switch_error = runner._switch_cached_agent_model(result, ctx, False)
                if switch_error:
                    old = snapshot.get("override") if snapshot.get("had_override") else None
                    await runner.async_session_store.set_model_override(key, old)
                    raise RuntimeError("Cached model switch rolled back")
                await runner._record_model_switch(result, ctx, source=source, one_turn=False, picker=False)
                return recovery.settle(manager, request["id"], "committed")
            except Exception as exc:
                logger.exception("Worker model handoff failed (%s)", type(exc).__name__)
                recovery.settle(manager, request["id"], "failed", error_type=type(exc).__name__)
                return None


async def prepare_worker_turn(runner, event, source, key, internal):
    """Retain the original post-admission consumer before preparing a model handoff."""
    if not internal:
        from gateway.run_inbound_consumer import run_post_admission_hook
        consumed, reply = await run_post_admission_hook(runner, event, source, key)
        if consumed:
            return False, reply
    return await before_turn(runner, event, source, key), None


async def before_turn(runner, event, source, key):
    """Recover pending commits before running the original worker; duplicates cannot start a turn."""
    with runner._profile_scope_for_source(source):
        sid = runner.session_store.peek_session_id(key)
        if not sid:
            return True
        cfg = recovery.settings()
        token = getattr(event, "_goal_model_handoff_id", None)
        if not recovery.enabled_for(sid, cfg):
            return token is None
        await runner._warm_goals_session_db("model handoff")
        manager = recovery.current_pending(sid)
        if manager is None:
            return token is None
        pending = deepcopy(manager.state.model_recovery["pending"])
        if token is None:
            # A real user/FIFO event wins; never overwrite it with our pending goal prompt.
            recovery.settle(manager, pending["id"], "user_turn")
            return True
        if token != pending["id"]:
            return False
        committed = await apply(runner, SimpleNamespace(session_id=sid), source, pending)
        if committed is None:
            return False
        # Claim before tool/model execution, not after the reply. The existing active-turn marker
        # recovers an interrupted turn; a replayed handoff event cannot create a second one.
        if recovery.settle(manager, committed["id"], "continued") is None:
            return False
        event.text = committed["prompt"]
        return True


async def restore_markers(runner):
    """Mark only authorized pending handoffs for the existing startup/reconnect resume mechanism."""
    with runner.session_store._lock:
        runner.session_store._ensure_loaded_locked()
        entries = list(runner.session_store._entries.values())
    for entry in entries:
        if entry.suspended or entry.origin is None or runner._is_session_running(entry.session_key):
            continue
        source = runner._restored_source(entry)
        with runner._profile_scope_for_source(source):
            cfg = recovery.settings()
            if not recovery.enabled_for(entry.session_id, cfg):
                continue
            await runner._warm_goals_session_db("restore model handoff")
            if recovery.current_pending(entry.session_id) is None:
                continue
            if not runner._resume_owner_authorized(entry.session_key, source):
                continue
            await runner.async_session_store.mark_resume_pending(entry.session_key, "goal_model_handoff")
