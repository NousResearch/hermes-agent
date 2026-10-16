"""Gateway housekeeping chores that keep the data disk healthy: stale media-cache cleanup and the
home-channel prompt for the low-disk auto-prune decision (see ``hermes_cli/prune_advisor.py``).

A headless gateway on a small VPS is where a disk fills up, and its owner may never open the CLI,
so the question goes to each served profile's home channel. It is posted at most once per
snooze window per profile: posting records the snooze, and ``/autoprune on|off`` ends it for good.
Runs from the housekeeping thread; the send is scheduled onto the gateway loop.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def housekeeping_media_caches() -> None:
    """Every platform media cache prunes on the same hourly cadence (24h max age)."""
    from gateway.platforms.base import (
        cleanup_audio_cache, cleanup_document_cache, cleanup_image_cache, cleanup_screenshot_cache,
        cleanup_video_cache)
    from tools.tool_result_storage import cleanup_spillover_cache
    from tools.environments.local import cleanup_terminal_temp_cache
    from tools.bot_mode_dm import cleanup_bot_dm_cache
    from tools.bot_relay import cleanup_bot_relay_artifacts
    from agent.provider_media import MEDIA_CACHE_MAX_AGE_HOURS
    from gateway.run import _housekeeping_chore

    for cache_name, cleanup_fn in (
        ("Image", cleanup_image_cache), ("Document", cleanup_document_cache),
        ("Audio", cleanup_audio_cache), ("Video", cleanup_video_cache),
        ("Screenshot", cleanup_screenshot_cache), ("Spillover", cleanup_spillover_cache),
        ("Terminal temp", cleanup_terminal_temp_cache), ("Bot DM", cleanup_bot_dm_cache),
        ("Bot relay", cleanup_bot_relay_artifacts)):
        def _one(name=cache_name, fn=cleanup_fn):
            removed = fn(max_age_hours=MEDIA_CACHE_MAX_AGE_HOURS)
            if removed:
                logger.info("%s cache cleanup: removed %d stale file(s)", name, removed)
        _housekeeping_chore(f"{cache_name} cache cleanup", _one)


def maybe_post_auto_prune_prompt(runner, loop) -> None:
    """One housekeeping tick: post the prompt to every served profile whose disk is low and who
    has not decided. Cheap when nothing is due (two syscalls + a YAML read per profile)."""
    if runner is None or loop is None:
        return
    from agent.async_utils import safe_schedule_threadsafe
    from hermes_cli import prune_advisor
    from hermes_constants import get_routing_process_hermes_home

    homes: dict = {None: get_routing_process_hermes_home()}
    homes.update(getattr(runner, "_served_profile_homes", None) or {})
    for profile, home in homes.items():
        if prune_advisor.should_ask(Path(home)) is None:
            continue
        advice = prune_advisor.advise(Path(home))
        if advice is None:
            continue
        fut = safe_schedule_threadsafe(_post(runner, profile, home, advice), loop, logger=logger,
                                       log_message="Auto-prune prompt scheduling error")
        # Snooze only on a delivered post: a gateway with no home channel (the one a CLI
        # auto-starts) must not swallow the question the CLI/TUI is about to ask.
        if fut is not None and fut.result(timeout=60):
            prune_advisor.snooze(Path(home))


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
