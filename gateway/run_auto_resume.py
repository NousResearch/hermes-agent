"""The gateway's ``delegation.auto_resume`` scheduler: one watcher per gateway process, a tick per served profile.

The policy (what is due, supersession, the attempt cap) lives in ``tools/delegate_tool_auto_resume.py``; this module
binds each profile's scope (``_profile_runtime_scope``, never ``os.environ``) and performs the continue the way a
parent turn would: a stand-in parent agent built on the profile's own route with the parent's session id, the parent
conversation's routing bound in a fresh context, then ``delegate_task(action='continue')``'s own entry point. The
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
    from gateway.run import _parse_session_key
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


def _stand_in_parent(db: Any, record: dict[str, Any], parent_sid: str, session_key: str) -> Any:
    """A parent agent for the continue only, on the bound profile's route with the parent's session id and the
    profile's own store. It runs no turn. Its toolsets are the child's own, so the rebuild keeps every tool it had."""
    from gateway.run import _load_gateway_config, _resolve_gateway_model, _resolve_runtime_agent_kwargs
    from run_agent import AIAgent
    runtime = _resolve_runtime_agent_kwargs()
    runtime.pop("_fallback_notice", None)
    model = runtime.pop("model", None) or _resolve_gateway_model(_load_gateway_config())
    parent = AIAgent(model=model, **runtime, quiet_mode=True, verbose_logging=False, session_id=parent_sid,
                     platform="gateway", gateway_session_key=session_key or None, skip_memory=True,
                     skip_context_files=True, session_db=db, enabled_toolsets=record.get("toolsets") or None)
    parent._end_session_on_close = False
    parent._oneshot_session = False  # a background continue, never the joined one-shot variant
    return parent


def _start_continue(record: dict[str, Any], unit: dict[str, Any], message: str) -> Optional[str]:
    """Continue one child; None once it is dispatched, else the refusal text. Runs on a worker thread in the
    profile's scope, inside a fresh context so the parent's routing binding never leaks into another task."""
    from gateway.session_context import reset_session_vars, set_session_vars
    from hermes_constants import get_hermes_home
    from hermes_state_registry import acquire, release_or_close
    from tools.delegate_tool_continue import continue_subagent
    reset_session_vars()
    route = _session_route(unit)
    # An api_server parent has no push channel: its result is persisted for its next request, as at dispatch.
    set_session_vars(**route, async_delivery=not unit.get("origin_session_id"), cron_session="")
    db = acquire(get_hermes_home() / "state.db")  # the bound profile's store, never the launch profile's
    parent = None
    try:
        parent = _stand_in_parent(db, record, route["session_id"], route["session_key"])
        out = json.loads(continue_subagent(record["subagent_id"], message, parent, background=True, auto=True))
    finally:
        if parent is not None:
            parent.release_clients()
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
    from tools.async_delegation import active_count
    from tools.delegate_tool_config import _get_max_concurrent_children
    return _get_max_concurrent_children() - active_count()


def tick_profiles(runner: Any) -> list[str]:
    """One scheduler pass over every served profile, each under its own scope (blocking: run off the loop)."""
    from gateway.run import _profile_runtime_scope
    from tools.delegate_tool_auto_resume import tick
    resumed: list[str] = []
    for name, home in _profile_homes(runner):
        if not (home / "state.db").is_file():
            continue
        try:
            with _profile_runtime_scope(home, None):
                start = (lambda record, unit, message:
                         contextvars.copy_context().run(_start_continue, record, unit, message))
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
