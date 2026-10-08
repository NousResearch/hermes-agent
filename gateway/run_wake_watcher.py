"""Gateway driver for one-shot ``schedule_wake`` deadlines armed inside a messaging chat (#122444).

Mirror of ``GatewayGoalsMixin._loop_wakeup_watcher`` for the ``wake:*`` rows: a coarse ticker scans
every served profile's SessionDB for ARMED wakes whose ``route`` names a platform + chat, and
injects each due prompt through that profile's adapter as a synthetic internal event. Wakes with no
route (CLI/TUI-owned sessions) are left to their own drivers. Deferrals mirror the loop watcher:
session running a turn, ``hermes pause`` engaged, no adapter for the route.

Inspired by ChatGPT Work's dots, which decide for themselves when to pause and wake back up and
reach the user through the chat they were working in (ChatGPT, Slack, Teams).
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import Any, Optional

logger = logging.getLogger("gateway.run")

WAKE_WATCH_INTERVAL_SECONDS = 15.0


async def wake_fire_one(
    runner: Any, sid: str, state: Any, now: float, warned_no_route: set, profile: Optional[str] = None,
) -> bool:
    """Inject one due wake into its routed chat, applying every deferral rule. Returns True when
    the prompt was handed to the adapter. ``profile`` is the store being scanned (None = default);
    a ``profile`` persisted in the route wins."""
    from hermes_cli.wake import abandon_wake_fire, due_wake_prompt, route_is_gateway_chat

    if not state.is_due(now) or not route_is_gateway_chat(state.route):
        return False
    route = state.route
    platform_name, chat_id = route.get("platform", ""), route.get("chat_id", "")
    profile = route.get("profile") or profile
    # The wake's OWN profile's adapter map, fail closed (same rule as the loop watcher): a secondary
    # session's wake must never inject through the default bot on a bare chat_id.
    adapters = runner._adapters_for_profile(profile)
    adapter = next((a for p, a in adapters.items() if p.value == platform_name), None)
    if adapter is None:
        if sid not in warned_no_route:
            warned_no_route.add(sid)
            logger.debug("wake: no adapter for platform %r (session %s, profile %s)", platform_name, sid, profile)
        return False
    source = runner._build_process_event_source({
        "session_key": "", "platform": platform_name, "chat_id": chat_id,
        **{k: route.get(k, "") for k in ("chat_type", "thread_id", "user_id", "user_name")},
    })
    if source is None:
        return False
    if profile and not getattr(source, "profile", None):
        source.profile = profile
    session_key = None
    with suppress(Exception):
        session_key = runner._session_key_for_source(source)
    if session_key and session_key in runner._running_agents:
        return False  # busy — stays armed, next scan retries
    from agent.estop import check_paused

    if check_paused("wake", logger):
        return False  # `hermes pause`: leave the wake armed so it fires after `hermes resume`
    prompt = await runner._run_in_executor_with_context(due_wake_prompt, sid, now)
    if not prompt:
        return False
    try:
        logger.info("wake firing for %s chat=%s thread=%s", platform_name, source.chat_id, source.thread_id)
        await adapter.handle_message(runner._synthetic_prompt_event(source, prompt, internal=True))
        return True
    except Exception:  # health: allow BLE001 -- adapter boundary (any platform SDK error); refund the fire so it stays armed
        logger.warning("wake injection failed for %s", sid, exc_info=True)
        with suppress(Exception):
            await runner._run_in_executor_with_context(abandon_wake_fire, sid)
        return False


async def wake_watcher(runner: Any, interval: float = WAKE_WATCH_INTERVAL_SECONDS) -> None:
    """Fire due chat-routed wakes for idle gateway sessions across every served profile's store
    (same multiplex shape as ``_loop_wakeup_watcher``: enter each profile's runtime scope, skip
    scopes whose store holds no armed wake)."""
    from gateway.run import _async_profile_runtime_scope, _resolve_handoff_watch_scopes
    from gateway.run_idle_gates import profile_has_armed_wake

    await asyncio.sleep(5)  # let platforms finish connecting
    warned_no_route: set = set()

    def _scope(profile_home):
        if profile_home is not None:
            return _async_profile_runtime_scope(profile_home)
        from tui_gateway.launch_profile_policy import async_launch_profile_scope_if_multiplexed
        return async_launch_profile_scope_if_multiplexed()

    async def _scan_one_store(profile_name: Optional[str]) -> None:
        from hermes_cli.wake import list_armed_wakes

        await runner._warm_goals_session_db("wake watcher")
        armed = await runner._run_in_executor_with_context(list_armed_wakes)
        now = time.time()
        for sid, state in armed:
            await wake_fire_one(runner, sid, state, now, warned_no_route, profile_name)

    while runner._running:
        try:
            for profile_name, profile_home in await _resolve_handoff_watch_scopes(runner):
                if profile_home is not None and not await runner._run_in_executor_with_context(
                        profile_has_armed_wake, profile_home):
                    continue
                async with _scope(profile_home):
                    await _scan_one_store(profile_name)
        except Exception:  # health: allow BLE001 -- supervised watcher loop must survive any scan error; same shape as _loop_wakeup_watcher
            logger.debug("wake watcher error", exc_info=True)
        await asyncio.sleep(interval)
