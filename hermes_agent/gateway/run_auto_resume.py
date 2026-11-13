"""The gateway's ``delegation.auto_resume`` scheduler: one watcher per gateway process, a tick per served profile.

The policy (what is due, supersession, the attempt cap) lives in ``hermes_agent/tools/delegate_tool_auto_resume.py``; this module
binds each profile's scope (``_profile_runtime_scope``, never ``os.environ``) and performs the continue the way a
parent turn would: a stand-in parent agent built on the parent conversation's own next-turn route with its session id,
the parent conversation's routing bound in a fresh context, then ``delegate_task(action='continue')``'s own entry point. The
continued child's completion goes through the normal completion queue to the parent's conversation.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

TICK_SECONDS = 15


def _session_route(unit: dict[str, Any]) -> dict[str, str]:
    """``set_session_vars`` kwargs that route a completion back to the parent conversation (its ledger row)."""
    from hermes_agent.gateway.run import _parse_session_key
    task = json.loads(unit.get("task_json") or "{}")
    key = str(unit.get("origin_session") or "")
    parsed = _parse_session_key(key) or {}
    # An api_server parent is addressed by its raw session id (what ``_current_origin_session_id`` reads back).
    chat_id = str(unit.get("origin_session_id") or "") or parsed.get("chat_id", "")
    return {
        "platform": "api_server" if unit.get("origin_session_id") else parsed.get("platform", ""), "chat_id": chat_id,
        "chat_type": parsed.get("chat_type", ""), "thread_id": parsed.get("thread_id", ""),
        "user_id": str(task.get("user_id") or ""), "user_name": str(task.get("user_name") or ""),
        "scope_id": str(task.get("scope_id") or ""), "session_key": key,
        "session_id": str(unit.get("parent_session_id") or ""),
        "ui_session_id": str(unit.get("origin_ui_session_id") or ""),
        # An api_server parent that resumes server history is addressed by its session id (``origin_session_id``).
        **({"session_history_delivery": "1"} if unit.get("origin_session_id") else {}),
    }


def _session_source(runner: Any, unit: dict[str, Any]) -> Any:
    """The parent conversation's ``SessionSource``, rebuilt from its ledger row the way a completion is routed."""
    if unit.get("origin_session_id") and not unit.get("origin_session"):
        from hermes_agent.gateway.config import Platform
        from hermes_agent.gateway.session import SessionSource
        return SessionSource(platform=Platform.API_SERVER, chat_id=str(unit["origin_session_id"]))
    return runner._build_process_event_source({"session_key": str(unit.get("origin_session") or "")})


def _stand_in_parent(runner: Any, db: Any, record: dict[str, Any], source: Any, route: dict[str, str],
                     policy: Any = None) -> Any:
    """A parent agent for the continue only, on the route the parent conversation's next turn would run on
    (``_resolve_session_agent_runtime``: its ``/model``, channel override or launch policy; its own ``/fast`` tier,
    never the runner-wide one the last turn of another conversation left), with the parent's session id and the
    profile's own store. It runs no turn. Its toolsets are the child's own, so the rebuild keeps every tool it had."""
    from hermes_agent.run_agent import AIAgent
    key = route["session_key"] or None
    model, runtime = runner._resolve_session_agent_runtime(source=source, session_key=key)
    runner._pre_agent_fallback_notice = None  # a turn's notice; this agent runs none
    tier = (policy.current_service_tier(runner, key) if policy else
            runner._resolve_session_service_tier(source=source, session_key=key))
    turn = runner._resolve_turn_agent_config("", model, runtime, tier=tier)
    parent = AIAgent(model=turn["model"], **turn["runtime"], request_overrides=turn.get("request_overrides"),
                     service_tier=tier,
                     quiet_mode=True, verbose_logging=False, session_id=route["session_id"], platform="gateway",
                     gateway_session_key=route["session_key"] or None, skip_memory=True, skip_context_files=True,
                     session_db=db, enabled_toolsets=record.get("toolsets") or None)
    parent._end_session_on_close = False
    parent._oneshot_session = False  # a background continue, never the joined one-shot variant
    return parent


def _continue_as_parent(runner: Any, db: Any, record: dict[str, Any], unit: dict[str, Any], route: dict[str, str],
                        message: str) -> dict[str, Any]:
    from hermes_agent.gateway.session_authorities import active_authority
    from hermes_agent.gateway.session_local_recovery import restore_completion_route
    from hermes_agent.gateway.session_policy import policy_for_source, policy_scope
    from hermes_agent.tools.delegate_tool_auto_resume import STAND_IN_FAILED
    from hermes_agent.tools.delegate_tool_continue import continue_subagent
    try:
        # As a completion's delivery does: boot leaves an automatically ended local parent unrestored, and without
        # its frozen policy the stand-in would run on the launch route.
        restore_completion_route(runner, str(unit.get("origin_session") or ""))
        source = _session_source(runner, unit)
        policy = policy_for_source(runner, source)
    except Exception as exc:  # health: allow BLE001 -- OUR stand-in failed, not the child's route: a later tick retries
        return {"error": f"{STAND_IN_FAILED}: {type(exc).__name__}: {exc}"}
    with policy_scope(policy, authority=active_authority(runner) if policy else None):
        try:
            parent = _stand_in_parent(runner, db, record, source, route, policy)
        except Exception as exc:  # health: allow BLE001 -- as above
            logger.warning("automatic resume of %s: no stand-in parent", record["subagent_id"], exc_info=True)
            return {"error": f"{STAND_IN_FAILED}: {type(exc).__name__}: {exc}"}
        try:
            return json.loads(continue_subagent(record["subagent_id"], message, parent, background=True, auto=True))
        finally:
            parent.release_clients()


def _start_continue(runner: Any, record: dict[str, Any], unit: dict[str, Any], message: str) -> Optional[str]:
    """Continue one claimed child; None once it is dispatched, else the refusal text. Runs on a worker thread in the
    profile's scope, inside a fresh context so the parent's routing binding never leaks into another task."""
    from hermes_agent.gateway.session_context import reset_session_vars, set_session_vars
    from hermes_constants import get_hermes_home
    from hermes_agent.hermes_state_registry import acquire, release_or_close
    reset_session_vars()
    route = _session_route(unit)
    # An api_server parent has no push channel: its result is persisted for its next request, as at dispatch.
    set_session_vars(**route, async_delivery=not unit.get("origin_session_id"), cron_session="")
    db = acquire(get_hermes_home() / "state.db")  # the bound profile's store, never the launch profile's
    try:
        out = _continue_as_parent(runner, db, record, unit, route, message)
    finally:
        release_or_close(db)
    if out.get("status") == "dispatched":
        return None
    return str(out.get("error") or out.get("note") or "the continue returned no dispatch")


def _profile_homes(runner: Any) -> list[tuple[str, Path]]:
    from hermes_constants import get_hermes_home
    homes = dict(getattr(runner, "_served_profile_homes", None) or {})
    primary = getattr(runner, "_primary_profile_name", None) or "default"
    homes.setdefault(primary, get_hermes_home())
    return [(name, Path(home)) for name, home in homes.items()]


def _capacity() -> int:
    """The bound profile's ``delegation.max_concurrent_children`` minus the units THIS profile has running."""
    from hermes_constants import get_hermes_home
    from hermes_agent.tools.async_delegation import active_count
    from hermes_agent.tools.delegate_tool_config import _get_max_concurrent_children
    return _get_max_concurrent_children() - active_count(home=get_hermes_home())


def tick_profiles(runner: Any) -> list[str]:
    """One scheduler pass over every served profile, each under its own scope (blocking: run off the loop)."""
    from hermes_agent.gateway.run import _profile_runtime_scope
    from hermes_agent.tools.delegate_tool_auto_resume import tick
    resumed: list[str] = []
    for name, home in _profile_homes(runner):
        if not (home / "state.db").is_file():
            continue
        try:
            with _profile_runtime_scope(home, None):
                start = (lambda record, unit, message:
                         contextvars.copy_context().run(_start_continue, runner, record, unit, message))
                resumed += tick(start, capacity=_capacity())
        except Exception:  # health: allow BLE001 -- one profile's failure must not stop the others
            logger.warning("delegation auto-resume pass failed for profile %r", name, exc_info=True)
    return resumed


class GatewayAutoResumeMixin:
    """``_delegation_auto_resume_watcher``: resumes stopped subagents when ``delegation.auto_resume`` is on."""

    async def _delegation_auto_resume_watcher(self, interval: float = TICK_SECONDS) -> None:
        await asyncio.sleep(5)  # platforms connect and boot recovery runs first
        while self._running:
            try:
                await asyncio.to_thread(tick_profiles, self)
            except Exception as exc:  # health: allow BLE001 -- the watcher keeps its cadence
                logger.debug("delegation auto-resume watcher error: %s", exc)
            await asyncio.sleep(interval)
