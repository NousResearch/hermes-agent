"""Wisdom's consumer of the messaging gateway's idle-session scheduling rail."""

from __future__ import annotations

import asyncio
import logging
import time

from hermes_wisdom.consent import ConsentActor
from hermes_wisdom.mediation import WisdomMediation, delivery_mode, session_runtime
from hermes_wisdom.mediation_view import advice_view, delivery_groups

logger = logging.getLogger(__name__)
ACTIVE_SECONDS = 10 * 60


async def schedule(
    gateway, adapter, source, session_id: str, *, observe_only: bool = False
) -> bool:
    """Return whether agent mode owns notification delivery for this profile."""
    platform = str(getattr(source.platform, "value", source.platform))
    profile = getattr(source, "profile", None) or getattr(
        adapter, "_owner_profile", None
    )
    if not profile and delivery_mode() == "fixed":
        return False
    if not callable(getattr(adapter, "_run_wisdom_profile_operation", None)):
        return False

    async def scoped(fn):
        if platform == "slack":
            return await adapter._run_wisdom_profile_operation(fn, profile=profile)
        return await adapter._run_wisdom_profile_operation(fn)

    if await scoped(delivery_mode) != "agent":
        return False
    if platform not in {"telegram", "slack"}:
        return True
    if str(getattr(source, "chat_type", "")) not in {"dm", "private"}:
        return True
    if not getattr(source, "user_id", None) or not gateway._is_user_authorized(source):
        return True
    key = gateway._session_key_for_source(source)
    actor = ConsentActor(
        key,
        platform,
        str(source.user_id),
        str(source.chat_id),
        str(getattr(source, "thread_id", None) or ""),
        str(getattr(source, "scope_id", None) or ""),
    )
    from hermes_wisdom.service import WisdomService

    def register():
        service = WisdomService()
        service.require_setup()
        org = service.store.active_org_id()
        mediation = WisdomMediation(service)
        mediation.queue.register_session(
            org,
            session_key=key,
            session_id=session_id,
            platform=platform,
            actor_id=actor.actor_id,
            private=True,
            available=False,
            user_activity=observe_only,
            address=actor.address,
        )

    await scoped(register)
    if observe_only:
        return True
    tasks = getattr(gateway, "_wisdom_mediation_tasks", None)
    if tasks is None:
        tasks = gateway._wisdom_mediation_tasks = {}
    deadlines = getattr(gateway, "_wisdom_mediation_active_until", None)
    if deadlines is None:
        deadlines = gateway._wisdom_mediation_active_until = {}
    deadlines[key] = time.monotonic() + ACTIVE_SECONDS
    if key in tasks and not tasks[key].done():
        return True

    async def tick():
        from tools.approval import get_pending_gateway_approval
        from tools.clarify_gateway import has_pending

        if get_pending_gateway_approval(key) or has_pending(key):
            return
        with gateway._agent_cache_lock:
            cached = gateway._agent_cache.get(key)
            agent = cached[0] if isinstance(cached, tuple) else cached
            if agent is None:
                return
            runtime = session_runtime(agent)
            history = list(getattr(agent, "_session_messages", None) or [])

        def prepare():
            service = WisdomService()
            mediation = WisdomMediation(service)
            org = service.store.active_org_id()
            mediation.queue.register_session(
                org,
                session_key=key,
                session_id=session_id,
                platform=platform,
                actor_id=actor.actor_id,
                private=True,
                available=True,
                address=actor.address,
            )
            if mediation.queue.claim_refresh(org):
                try:
                    service.check(apply_automatic=False)
                    mediation.ingest()
                except Exception as exc:
                    logger.debug(
                        "Wisdom feed refresh deferred (%s)", type(exc).__name__
                    )
            return org, mediation.prepare(org, actor, runtime=runtime, history=history)

        org, items = await scoped(prepare)
        if not items:
            return
        guard = adapter._active_sessions.get(key)
        if guard is None or guard.is_set():
            return

        def begin(group):
            mediation = WisdomMediation(WisdomService())
            introduction = not mediation.queue.introduced(org)
            selected = [
                item
                for item in group
                if mediation.queue.begin_delivery(
                    org, item["assessment"]["id"], item["assessment"]["lease_token"]
                )
            ]
            return selected, introduction

        for group in delivery_groups(items):
            selected, introduction = await scoped(lambda: begin(group))
            if not selected:
                continue
            view = advice_view(selected, introduction=introduction)
            await adapter.send_wisdom_mediation(view, source=source)

            def finish():
                mediation = WisdomMediation(WisdomService())
                for item in selected:
                    job = item["assessment"]
                    mediation.queue.complete_delivery(
                        org, job["id"], job["lease_token"], introduced=introduction
                    )

            await scoped(finish)

    async def watch():
        try:
            # The post-delivery hook still owns its guard until it returns.
            await asyncio.sleep(1)
            while time.monotonic() < deadlines.get(key, 0):
                if await scoped(delivery_mode) != "agent":
                    return
                if not gateway._is_user_authorized(source):
                    return
                try:
                    await adapter.run_idle_activity(key, tick)
                except Exception as exc:
                    logger.warning(
                        "Wisdom session mediation deferred (%s)", type(exc).__name__
                    )
                await asyncio.sleep(60)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Wisdom session mediation paused (%s)", type(exc).__name__)
        finally:
            if tasks.get(key) is asyncio.current_task():
                tasks.pop(key, None)
                deadlines.pop(key, None)

    task = asyncio.create_task(watch())
    tasks[key] = task
    adapter._background_tasks.add(task)
    task.add_done_callback(adapter._background_tasks.discard)
    return True
