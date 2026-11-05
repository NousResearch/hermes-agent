"""The shutdown notice's home-channel pass (``GatewayShutdownMixin._notify_active_sessions_of_shutdown``)."""
from __future__ import annotations

from contextlib import nullcontext
import logging

logger = logging.getLogger("gateway.run")


async def broadcast_home_shutdown_notice(runner, msg: str, notified: set) -> None:
    """EVERY served profile's home channel, through that profile's OWN bot: ``runner.adapters`` and
    ``runner.config`` are the launch profile's alone, so iterating them left the secondaries' channels
    silent (#118233). ``list(...)`` snapshots the adapter maps: adapter.send() can hit a fatal path
    (_handle_fatal) that pops the adapter -> "dictionary changed size during iteration"."""
    from gateway.run import _async_profile_runtime_scope
    from gateway.run_shutdown import _delivery_target_key
    from gateway.warning_notifications import present_notification

    profile_adapters = getattr(runner, "_profile_adapters", None) or {}
    for profile, platform, platform_cfg in list(runner._served_home_channel_configs()):
        home = platform_cfg.home_channel
        if not home or not home.chat_id:
            continue
        adapter = (runner.adapters if profile is None else profile_adapters.get(profile) or {}).get(platform)
        if adapter is None:
            continue
        if not runner._notice_allowed(platform, "home channel", platform_cfg):
            continue
        dedup_key = _delivery_target_key(platform.value, home.chat_id, home.thread_id, profile=profile)
        if dedup_key in notified:
            continue
        try:
            metadata = runner._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=adapter)
        except Exception as e:
            logger.debug(
                "Failed to send shutdown notification to home channel %s:%s: %s", platform.value, home.chat_id, e,
                exc_info=True,
            )
            continue

        async def _send_home(adapter=adapter, home=home, platform=platform, metadata=metadata, dedup_key=dedup_key):
            if await runner._send_shutdown_notice(
                adapter, str(home.chat_id), msg, "home channel", platform.value, metadata=metadata,
            ):
                notified.add(dedup_key)
        # present_notification reads the ACTIVE profile's display settings: bind the served one's.
        profile_home = (getattr(runner, "_served_profile_homes", None) or {}).get(profile) if profile else None
        async with _async_profile_runtime_scope(profile_home) if profile_home else nullcontext():
            await present_notification(_send_home, platform=platform)
