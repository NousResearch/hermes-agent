"""Shutdown notices for active sessions, home channels and interrupted cron runs."""

from __future__ import annotations

import logging
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol

from agent.i18n import t
from gateway.config import Platform

if TYPE_CHECKING:
    from cron.scheduler_preflight import SharedRouteAdapters
    from gateway.platforms.base import BasePlatformAdapter


class _ThreadMetadataForTarget(Protocol):
    def __call__(
        self, platform: Optional[Platform], chat_id: Optional[str], thread_id: Optional[str], *,
        chat_type: Optional[str] = None, reply_to_message_id: Optional[str] = None,
        adapter: Optional[Any] = None,
    ) -> Optional[dict[str, Any]]: ...


logger = logging.getLogger("gateway.run")


class GatewayShutdownNoticesMixin:
    _restart_requested: bool
    _cron_delivery_adapters: Callable[
        [Optional[str]], dict[Platform, BasePlatformAdapter] | SharedRouteAdapters
    ]
    _thread_metadata_for_target: _ThreadMetadataForTarget

    def _restart_notification_allowed(self, platform: Platform) -> bool:
        """False when the platform config sets ``gateway_restart_notification=false``."""
        platform_cfg = self.config.platforms.get(platform)
        return platform_cfg is None or bool(platform_cfg.gateway_restart_notification)

    def _notice_allowed(self, platform: Platform, what: str, platform_cfg=None) -> bool:
        """``_restart_notification_allowed`` with the INFO suppression line for shutdown notices.
        ``platform_cfg`` is a SERVED profile's own platform entry; ``self.config`` is the launch profile's."""
        allowed = (self._restart_notification_allowed(platform) if platform_cfg is None
                   else bool(platform_cfg.gateway_restart_notification))
        if allowed:
            return True
        logger.info(
            "Shutdown notification suppressed for %s: %s has gateway_restart_notification=false", what, platform.value,
        )
        return False

    async def _notify_interrupted_cron_jobs(self, interrupted_runs) -> int:
        """Tell the owner of each just-interrupted cron run that it died; returns notices sent.

        The cron worker can't (its thread reaches ``_deliver_result`` after teardown closed the
        transport), so this runs post-interrupt while adapters are still connected. Best-effort.

        Its thread reaches ``_deliver_result`` asynchronously, and by then ``_bounded_adapter_teardown`` has
        closed the transport — so the notice never leaves the process, and ``_consume_interrupted_flag``
        discards the resulting ``delivery_error`` along with it. The run's only trace is a line in jobs.json
        nobody reads (#82232).
        Must therefore be called from the post-interrupt phase, while adapters are still connected — the
        same window ``_notify_active_sessions_of_shutdown`` relies on for chat sessions, which is blind to
        cron work because cron runs on the scheduler's own thread pool rather than ``self._running_agents``
        (#60432).

        One process ticks every profile's store, and ``stop()`` runs in the launch profile's scope. Each
        run is therefore handled inside its own profile's scope: the job is read from that profile's
        store, the text is in that profile's language, and the notice leaves through that profile's bot.
        """
        from gateway.run_shutdown import _log_suppressed
        if not interrupted_runs:
            return 0
        from gateway.run import _async_profile_runtime_scope
        notified: set = set()
        for run in interrupted_runs:
            with _log_suppressed(logging.DEBUG, "Cron interrupt notice for %s under %s failed: %s", run.job_id, run.home):
                async with _async_profile_runtime_scope(run.home):
                    await self._notify_interrupted_cron_run(run.job_id, run.home, notified)
        if notified:
            logger.info("Shutdown: delivered %d interrupted-cron-job notice(s)", len(notified))
        return len(notified)

    async def _shutdown_notification_target(self, session_key: str):
        """``(source, platform_str, chat_id, thread_id, profile)``: persisted origin > cached source >
        parsed key. ``profile`` is the owning profile from the source or the ``agent:<profile>:`` key
        namespace (``None`` = default) so the notice leaves through that profile's bot."""
        from gateway.run import _parse_session_key
        source = None
        try:
            if getattr(self, "session_store", None) is not None:
                await self.async_session_store._ensure_loaded()
                entry = self.session_store._entries.get(session_key)
                source = getattr(entry, "origin", None) if entry else None
        except Exception as e:
            logger.debug("Failed to load session origin for shutdown notification %s: %s", session_key, e)
        if source is None:
            source = self._get_cached_session_source(session_key)
        if source is not None:
            return source, source.platform.value, str(source.chat_id), source.thread_id, getattr(source, "profile", None)
        _parsed = _parse_session_key(session_key)
        if not _parsed:
            return None
        return None, _parsed["platform"], _parsed["chat_id"], _parsed.get("thread_id"), _parsed.get("profile")

    async def _send_shutdown_notice(
        self, adapter, chat_id: str, msg: str, kind: str, platform_str: str, **send_kwargs
    ) -> bool:
        """Send one shutdown notice; True when delivered. Failures are debug-logged, never raised."""
        where = "home channel " if kind == "home channel" else ""
        fail_fmt = f"Failed to send shutdown notification to {where}%s:%s: %s"
        if not await self._send_notice_logged(adapter, chat_id, msg, platform_str, fail_fmt, **send_kwargs):
            return False
        logger.info("Sent shutdown notification to %s %s:%s", kind, platform_str, chat_id)
        return True

    @staticmethod
    async def _send_notice_logged(
        adapter, chat_id: str, msg: str, platform_str: str, fail_fmt: str, raise_fmt: Optional[str] = None, **kw
    ) -> bool:
        """``adapter.send`` whose failure is debug-logged as ``fmt % (platform, chat, error)`` — ``fail_fmt``
        for success=False, ``raise_fmt`` (default ``fail_fmt``) for a raise; True only on a delivered send.
        Every shutdown notice races live turns, so it always carries the interim marker (#98432)."""
        from gateway.run_shutdown import _send_error, _send_failed
        from gateway.run import _interim_metadata
        kw["metadata"] = _interim_metadata(kw.get("metadata"))
        try:
            result = await adapter.send(chat_id, msg, **kw)
        except Exception as e:
            logger.debug(raise_fmt or fail_fmt, platform_str, chat_id, e)
            return False
        if _send_failed(result):
            logger.debug(fail_fmt, platform_str, chat_id, _send_error(result))
            return False
        return True

    async def _notify_active_sessions_of_shutdown(self) -> None:
        """Send shutdown/restart notifications to active chats and home channels.

        Called at the start of stop() while adapters are connected; send failures never block shutdown.
        """
        from gateway.run_shutdown import _delivery_target_key, _log_suppressed, _notice_target_key
        restart_source = self._restart_command_source if self._restart_requested else None
        # Translate per target, inside that profile's scope. ``stop()`` runs in the launch profile's
        # scope, so text resolved here would be in the launch profile's language for every profile.
        notice_key = "gateway.shutdown.notice_restart" if self._restart_requested else "gateway.shutdown.notice_shutdown"
        served_homes = getattr(self, "_served_profile_homes", None) or {}
        restart_key = None
        if restart_source is not None:
            with suppress(Exception):
                restart_key = _notice_target_key(
                    restart_source.platform.value, restart_source.chat_id, restart_source.thread_id
                )
        notified: set[tuple[str, str, Optional[str]]] = set()
        for session_key in self._snapshot_running_agents():
            target = await self._shutdown_notification_target(session_key)
            if target is None:
                continue
            source, platform_str, chat_id, thread_id, profile = target
            dedup_key = _delivery_target_key(platform_str, chat_id, thread_id, profile=profile)
            if dedup_key in notified:
                continue
            try:
                platform = Platform(platform_str)
                # The session's OWN profile's bot (transport ref → profile map), never a bare
                # self.adapters hit: under multiplex that is the default bot, so a secondary session's
                # "Gateway shutting down" would land in the user's chat with the wrong bot.
                adapter = self._delivery_adapter_for(source) if source is not None else None
                if adapter is None:
                    adapter = self._authorization_adapter(platform, profile)
                if not adapter:
                    continue
                if not self._notice_allowed(platform, "active session", self._served_platform_config(profile, platform)):
                    continue
                reply_to_message_id = getattr(source, "message_id", None)
                if reply_to_message_id is None and restart_key == dedup_key:
                    reply_to_message_id = getattr(restart_source, "message_id", None)
                metadata = self._thread_metadata_for_target(
                    platform, chat_id, thread_id, chat_type=getattr(source, "chat_type", None),
                    reply_to_message_id=reply_to_message_id, adapter=adapter,
                )
            except Exception as e:
                logger.debug("Failed to send shutdown notification to %s:%s: %s", platform_str, chat_id, e)
                continue
            # Automatic interrupt diagnostic, resolved under the session's own profile scope (same
            # shape as the stall watcher). The requester's own chat on an in-chat /restart is the
            # requested outcome of that command and is never suppressed.
            async def _send_active(adapter=adapter, chat_id=chat_id, platform_str=platform_str,
                                   metadata=metadata, dedup_key=dedup_key):
                if await self._send_shutdown_notice(
                    adapter, chat_id, t(notice_key), "active chat", platform_str, metadata=metadata
                ):
                    notified.add(dedup_key)
            from gateway.warning_notifications import present_notification
            from gateway.run import _async_profile_runtime_scope
            profile_home = (self._resolve_profile_home_for_source(source) if source is not None
                            else served_homes.get(profile or "default"))
            scope = _async_profile_runtime_scope(profile_home) if profile_home else nullcontext()
            async with scope:
                presented = await present_notification(_send_active, platform=platform, diagnostic=restart_key != dedup_key)
            if not presented:
                notified.add(dedup_key)  # suppressed: latch so the home-channel pass does not re-target it
        if self._restart_requested and restart_source is not None:
            logger.debug("Skipping home-channel shutdown notifications for in-chat restart")
            return
        # A quiet drain (routine fleet auto-update) suppresses ONLY the home-channel broadcast; per-session
        # pings above stay. Current-epoch marker only; a failing check fails toward the louder behaviour.
        with _log_suppressed(logging.DEBUG, "drain_notification_suppressed check failed: %s"):
            from gateway.drain_control import drain_notification_suppressed
            if drain_notification_suppressed():
                logger.info(
                    "Home-channel shutdown broadcast suppressed by drain marker (suppress_notification=true)"
                )
                return
        # EVERY served profile's home channel, through that profile's OWN bot: ``self.adapters`` and
        # ``self.config`` are the launch profile's alone, so iterating them left the secondaries'
        # channels silent (#118233). ``list(...)`` snapshots the adapter maps: adapter.send() can hit
        # a fatal path (_handle_fatal) that pops the adapter -> "dictionary changed size during iteration".
        profile_adapters = getattr(self, "_profile_adapters", None) or {}
        for profile, platform, platform_cfg in list(self._served_home_channel_configs()):
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue
            adapter = (self.adapters if profile is None else profile_adapters.get(profile) or {}).get(platform)
            if adapter is None:
                continue
            if not self._notice_allowed(platform, "home channel", platform_cfg):
                continue
            dedup_key = _delivery_target_key(platform.value, home.chat_id, home.thread_id, profile=profile)
            if dedup_key in notified:
                continue
            try:
                metadata = self._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=adapter)
            except Exception as e:
                logger.debug(
                    "Failed to send shutdown notification to home channel %s:%s: %s", platform.value, home.chat_id, e,
                )
                continue
            async def _send_home(adapter=adapter, home=home, platform=platform, metadata=metadata):
                if await self._send_shutdown_notice(
                    adapter, str(home.chat_id), t(notice_key), "home channel", platform.value, metadata=metadata,
                ):
                    notified.add(dedup_key)
            from gateway.warning_notifications import present_notification
            from gateway.run import _async_profile_runtime_scope
            # present_notification reads the ACTIVE profile's display settings: bind the served one's.
            profile_home = served_homes.get(profile or "default")
            async with _async_profile_runtime_scope(profile_home) if profile_home else nullcontext():
                await present_notification(_send_home, platform=platform)

    async def _notify_interrupted_cron_run(self, job_id: str, home: Path, notified: set) -> None:
        """Send one interrupted run's notices; the caller has bound the scope of ``home``'s profile."""
        from gateway.run_shutdown import _log_suppressed, _notice_target_key
        try:
            from cron.jobs import get_job
            from cron.scheduler import _resolve_delivery_targets
            from cron.scheduler_preflight import SharedRouteAdapters
            from hermes_constants import profile_name_for_home
        except Exception as e:
            logger.debug("Cron interrupt notification unavailable: %s", e)
            return
        try:
            job = get_job(job_id)
            if not job:
                return
            # deliver=local / unresolvable-origin jobs resolve to zero targets and stay silent (no home-
            # channel fallback). Interrupted notices are failure-category status: honor failure_deliver.
            # See #43014.
            targets = _resolve_delivery_targets(job, for_failure=True)
        except Exception as e:
            logger.debug("Cron interrupt targets unresolved for %s: %s", job_id, e)
            return
        profile = profile_name_for_home(home)
        adapters = self._cron_delivery_adapters(profile)
        action = t("gateway.shutdown.action_restarting" if self._restart_requested else "gateway.shutdown.action_shutting_down")
        msg = t("gateway.shutdown.cron_interrupted", job=job.get("name") or job_id, action=action)
        for target in targets or ():
            try:
                platform = Platform(str(target.get("platform", "")).lower())
            except Exception:
                continue
            adapter = (adapters.get(platform, target) if isinstance(adapters, SharedRouteAdapters)
                       else adapters.get(platform))
            if adapter is None or not self._notice_allowed(
                platform, "cron job", self._served_platform_config(profile, platform),
            ):
                continue
            chat_id = str(target.get("chat_id"))
            thread_id = target.get("thread_id")
            platform_str = str(platform.value)
            dedup_key = (profile, job_id, *_notice_target_key(platform_str, chat_id, thread_id))
            if dedup_key in notified:
                continue
            with _log_suppressed(logging.DEBUG, "Cron interrupt notice to %s:%s raised: %s", platform.value, chat_id):
                metadata = self._thread_metadata_for_target(platform, chat_id, thread_id, adapter=adapter)
                async def send_notice():
                    if await self._send_notice_logged(
                        adapter, chat_id, msg, str(platform.value), "Cron interrupt notice to %s:%s failed: %s",
                        "Cron interrupt notice to %s:%s raised: %s", metadata=metadata,
                    ):
                        notified.add(dedup_key)
                from gateway.warning_notifications import present_notification
                await present_notification(send_notice, platform=platform)

    def _served_platform_config(self, profile: Optional[str], platform: Platform):
        """A served secondary profile's own entry for *platform*, for ``_notice_allowed``. ``None`` for the
        primary profile and for a profile whose config has no entry for *platform* (a shared-bot
        satellite), so the decision falls back to ``self.config``."""
        profile_config = (getattr(self, "_profile_configs", None) or {}).get(profile) if profile else None
        return profile_config.platforms.get(platform) if profile_config is not None else None
