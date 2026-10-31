"""Gateway housekeeping chores that keep the data disk healthy: stale media-cache cleanup and the
home-channel prompt for the low-disk auto-prune decision (see ``hermes_cli/prune_advisor.py``).

A headless gateway on a small VPS is where a disk fills up, and its owner may never open the CLI,
so the question goes to each served profile's home channel; Desktop and Ink TUI viewers of local
sessions get it as a notice. It is posted at most once per
snooze window per profile: posting records the snooze, and ``/autoprune on|off`` ends it for good.
Runs from the housekeeping thread; the send is scheduled onto the gateway loop.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def housekeeping_media_caches() -> None:
    """Hourly media-cache prune (24h max age); spillover (None) keeps the session retention window."""
    from gateway.platforms.base import (
        cleanup_audio_cache, cleanup_document_cache, cleanup_image_cache, cleanup_screenshot_cache,
        cleanup_video_cache)
    from tools.tool_result_storage import cleanup_spillover_cache
    from tools.environments.local import cleanup_terminal_temp_cache
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_relay import cleanup_bot_relay_artifacts
    from tools.bot_delivery_queue import cleanup_bot_delivery_queue
    from agent.provider_media import MEDIA_CACHE_MAX_AGE_HOURS
    from gateway.run import _housekeeping_chore

    for cache_name, cleanup_fn in (
        ("Image", cleanup_image_cache), ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache), ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache), ("Spillover", cleanup_spillover_cache),
        ("Terminal temp", cleanup_terminal_temp_cache), ("Bot DM", cleanup_bot_dm_cache),
        ("Bot relay", cleanup_bot_relay_artifacts),
        # Trigger 4 for peer deliveries (VERIFICATION D1): recover orphaned claims and
        # expire over-age records hourly, even on a gateway that sees no peer send in
        # that hour.
        ("Bot delivery", cleanup_bot_delivery_queue)):
        def _one(name=cache_name, fn=cleanup_fn):
            removed = fn(max_age_hours=None if fn is cleanup_spillover_cache else MEDIA_CACHE_MAX_AGE_HOURS)
            if removed:
                logger.info("%s cache cleanup: removed %d stale file(s)", name, removed)
        _housekeeping_chore(f"{cache_name} cache cleanup", _one)


def maybe_post_auto_prune_prompt(runner, loop) -> None:
    """One housekeeping tick: post the prompt to every served profile that has a home channel, a
    low disk and no decision yet. Cheap when nothing is due (two syscalls + a YAML read per
    profile); the database estimate runs only for a profile that can actually receive it."""
    if runner is None or loop is None:
        return
    from agent.async_utils import safe_schedule_threadsafe
    from hermes_cli import prune_advisor
    from hermes_constants import get_routing_process_hermes_home

    reachable = {profile for profile, *_rest in runner._served_home_channel_transports()}
    homes: dict = {None: get_routing_process_hermes_home()}
    homes.update(getattr(runner, "_served_profile_homes", None) or {})
    for profile, home in homes.items():
        if profile not in reachable or prune_advisor.should_ask(Path(home)) is None:
            continue
        advice = prune_advisor.advise(Path(home))
        if advice is None:
            continue
        fut = safe_schedule_threadsafe(_post(runner, profile, home, advice), loop, logger=logger,
                                       log_message="Auto-prune prompt scheduling error")
        # Snooze only on a delivered post: a send that failed must not swallow the question.
        if fut is not None and fut.result(timeout=60):
            prune_advisor.snooze(Path(home))


_LOCAL_NOTIFIED: dict[str, float] = {}


async def publish_local_auto_prune_notice(authority, ref) -> None:
    """Desktop / Ink TUI sessions (``AuthorityConnection.resume``): one sticky ``notification.show``
    per profile per snooze window while the disk is low and auto-prune is undecided. The classic
    CLI asks at its own startup, so its sessions are skipped here."""
    import asyncio
    import time
    from gateway.config import Platform
    from gateway.session_policy import policy_for_source
    from hermes_cli import prune_advisor
    from hermes_state_runtime import RuntimeStoreError

    live = authority.sessions.get(ref.session_id)
    source = getattr(live, "source", None)
    if source is None or source.platform != Platform.LOCAL:
        return
    try:
        policy = policy_for_source(authority.runner, source)
    except RuntimeStoreError:  # policy not restored yet: this viewer gets no notice, the next one may
        return
    if policy is None or policy.source not in {"tui", "gui"}:
        return
    home = Path(authority.db.db_path).parent
    if not prune_advisor.notice_due(home, _LOCAL_NOTIFIED):
        return
    _LOCAL_NOTIFIED[str(home)] = time.time()  # in-flight claim: one estimate per home at a time
    advice = await asyncio.to_thread(prune_advisor.advise, home)
    # Answered (here or in another window) during the estimate: publishing would re-ask it.
    delivered = (advice is not None and prune_advisor.should_ask(home) is not None
                 and live.event_stream.publish(ref.session_id, prune_advisor.notice_payload(advice),
                                               event_type="notification.show"))
    if not delivered:
        _LOCAL_NOTIFIED.pop(str(home), None)  # nobody saw it: the next window may ask


def clear_local_auto_prune_notice(runner, source) -> None:
    """After /autoprune is answered on a local session, drop its sticky notice for that viewer."""
    from gateway.config import Platform
    from gateway.session_authorities import active_authority
    from hermes_cli.prune_advisor import NOTICE_KEY
    if getattr(source, "platform", None) != Platform.LOCAL:
        return
    authority = active_authority(runner)
    live = authority.sessions.get(source.chat_id) if authority is not None else None
    if live is not None:
        live.event_stream.publish(source.chat_id, {"key": NOTICE_KEY}, event_type="notification.clear")


async def _post(runner, profile, home, advice) -> bool:
    """Send to the profile's home channels; True when at least one send was delivered."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.warning_notifications import present_notification
    from hermes_cli.prune_advisor import advice_lines

    delivered = False
    scope = _async_profile_runtime_scope(Path(home)) if profile is not None else contextlib.nullcontext()
    async with scope:
        for target_profile, platform, _cfg, home_ch, transport in list(runner._served_home_channel_transports()):
            if target_profile != profile:
                continue
            prefix = getattr(transport.adapter, "typed_command_prefix", "/") if transport.adapter else "/"
            message = "⚠ " + "\n".join(advice_lines(advice, command_prefix=prefix))
            sent: list[bool] = []

            async def _send(platform=platform, home_ch=home_ch, transport=transport, message=message):
                sent.append(await runner._send_home_channel_message(
                    platform, home_ch, transport, message, "Auto-prune prompt failed for %s:%s: %s"))

            await present_notification(_send, platform=platform)
            delivered = delivered or any(sent)
    return delivered
