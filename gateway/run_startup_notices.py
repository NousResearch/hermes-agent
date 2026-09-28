"""Gateway startup notices and configured home-channel delivery."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Callable, Optional

from agent.i18n import t
from gateway.config import GatewayConfig, Platform
from gateway.run_shutdown import _delivery_target_key, _notice_target_key, _send_error, _send_failed

logger = logging.getLogger("gateway.run")


def _served_notice_target_key(profile: Optional[str], platform_value: str, chat_id, thread_id) -> tuple:
    """Notice-dedupe key for one SERVED profile's home channel.

    A secondary uses the ``<profile>:<platform>`` key convention the runtime status already
    stamps in ``gateway_state.json``; the launch profile keeps the bare platform value so a
    marker written before this change still matches its delivered targets.
    """
    return _notice_target_key(
        platform_value if profile is None else f"{profile}:{platform_value}", chat_id, thread_id)


def _safe_delivery_transport(platform, config, adapters, *, profile: Optional[str] = None):
    """``resolve_delivery_transport`` isolated to one target: ``None`` (logged) on failure.

    The fan-out spans every served profile, so one profile's broken adapter must not abort the
    pass and starve every profile after it in dict order.
    """
    from gateway.delivery import resolve_delivery_transport
    try:
        return resolve_delivery_transport(platform, config, adapters)
    except Exception as exc:
        logger.debug(
            "Home-channel transport unavailable for %s%s: %s",
            f"{profile}:" if profile else "", getattr(platform, "value", platform), exc)
        return None


class GatewayStartupNoticesMixin:
    config: GatewayConfig
    _marker_profile: Callable[[dict], Optional[str]]
    _pending_marker_metadata: Callable[..., Optional[dict]]

    async def _send_restart_notification(self) -> Optional[tuple[str, str, Optional[str]]]:
        """Notify the chat that initiated /restart that the gateway is back."""
        from gateway.delivery import resolve_delivery_transport
        from gateway.run import _hermes_home, _non_conversational_metadata
        notify_path = _hermes_home / ".restart_notify.json"
        if not notify_path.exists():
            return None
        try:
            data = json.loads(notify_path.read_text(encoding="utf-8-sig"))
            platform_str = data.get("platform")
            chat_id = data.get("chat_id")
            thread_id = data.get("thread_id")
            if not platform_str or not chat_id:
                return None
            platform = Platform(platform_str)
            # The receiving bot can differ from the runtime profile. Legacy markers have only
            # the runtime hint; an explicit "default" transport must still select the primary bot.
            transport_profile = data.get("transport_profile")
            if transport_profile is None:
                transport_profile = self._marker_profile(data)
            transport = resolve_delivery_transport(
                platform, self.config, self._adapters_for_profile(transport_profile))
            if transport is None:
                logger.debug("Restart notification skipped: no live transport for %s", platform_str)
                return None
            platform_cfg = self.config.platforms.get(platform)
            if platform_cfg is not None and not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Restart notification suppressed: %s has gateway_restart_notification=false", platform_str
                )
                return None
            metadata = self._pending_marker_metadata(platform, chat_id, data, transport.adapter)
            if data.get("delivered_via_upstream_relay") is True:
                metadata = dict(metadata or {})
                for field in ("user_id", "scope_id"):
                    if data.get(field):
                        metadata[field] = str(data[field])
            result = await transport.send(
                platform, str(chat_id), t("gateway.startup.restarted"),
                metadata=_non_conversational_metadata(metadata, platform=platform),
            )
            # adapter.send() catches provider errors (e.g. "Chat not found") and returns
            # SendResult(success=False) rather than raising, so inspect the result before claiming success.
            if _send_failed(result):
                logger.warning(
                    "Restart notification to %s:%s was not delivered: %s", platform_str, chat_id, _send_error(result),
                )
                return None
            logger.info("Sent restart notification to %s:%s", platform_str, chat_id)
            return str(platform_str), str(chat_id), str(thread_id) if thread_id else None
        except Exception as e:
            logger.warning("Restart notification failed: %s", e)
            return None
        finally:
            notify_path.unlink(missing_ok=True)

    def _home_channel_transports(self):
        """Yield ``(platform, platform_cfg, home, transport)`` for every home channel with a live transport."""
        for platform, platform_cfg in self.config.platforms.items():
            home = platform_cfg.home_channel
            if not home or not home.chat_id:
                continue
            transport = _safe_delivery_transport(platform, self.config, self.adapters)
            if transport is None:
                continue
            yield platform, platform_cfg, home, transport

    def _served_home_channel_configs(self):
        """``(profile, platform, platform_cfg)`` for every SERVED profile's configured home channel.

        ``self.config`` is the launch profile's alone, but one host process multiplexes every
        profile, so a host-wide notice built from it silently skips the others' channels. The
        secondary configs are the ones ``_load_secondary_profile_config`` already cached at
        adapter start; ``profile`` is ``None`` for the launch profile.
        """
        for platform, platform_cfg in self.config.platforms.items():
            yield None, platform, platform_cfg
        for profile, profile_cfg in (getattr(self, "_profile_configs", None) or {}).items():
            for platform, platform_cfg in profile_cfg.platforms.items():
                yield profile, platform, platform_cfg

    def _served_home_channel_transports(self):
        """``(profile, platform, platform_cfg, home, transport)`` for every served profile's home
        channel with a live transport — the launch profile's (``profile`` ``None``) first."""
        for platform, platform_cfg, home, transport in self._home_channel_transports():
            yield None, platform, platform_cfg, home, transport
        for profile, profile_cfg in (getattr(self, "_profile_configs", None) or {}).items():
            adapters = (getattr(self, "_profile_adapters", None) or {}).get(profile) or {}
            for platform, platform_cfg in profile_cfg.platforms.items():
                home = platform_cfg.home_channel
                if not home or not home.chat_id:
                    continue
                transport = _safe_delivery_transport(platform, profile_cfg, adapters, profile=profile)
                if transport is None:
                    continue
                yield profile, platform, platform_cfg, home, transport

    async def _send_home_channel_message(self, platform, home, transport, message: str, failure_fmt: str) -> bool:
        """Best-effort send to one home channel; True on success, failures logged with ``failure_fmt``."""
        from gateway.run import _non_conversational_metadata
        try:
            metadata = self._thread_metadata_for_target(platform, home.chat_id, home.thread_id, adapter=transport.adapter)
            if transport.is_relay:
                metadata = dict(metadata or {})
                if home.user_id:
                    metadata["user_id"] = home.user_id
                if home.scope_id:
                    metadata["scope_id"] = home.scope_id
            send_metadata = _non_conversational_metadata(metadata, platform=platform)
            if send_metadata is not None or transport.is_relay:
                result = await transport.send(platform, str(home.chat_id), message, metadata=send_metadata)
            else:
                result = await transport.adapter.send(str(home.chat_id), message)
            if _send_failed(result):
                logger.warning(failure_fmt, platform.value, home.chat_id, _send_error(result))
                return False
            return True
        except Exception as exc:
            logger.warning(failure_fmt, platform.value, home.chat_id, exc)
            return False

    def _free_tier_startup_line(self) -> Optional[str]:
        """Extra startup line when the gateway's inference is carried by the Nous free tier; None otherwise.

        Best-effort: a resolution failure (no provider, auth error) must not block the online notice."""
        try:
            # Persisted state only. The free-tier check reads auth.json; it runs FIRST so the resolver
            # is only consulted when a free-tier identity already exists and its own free-tier rung
            # (which may mint on a fresh install, NS-829) answers from that identity without a network
            # call. No token refresh at boot either way.
            from hermes_cli.anon_auth import free_tier_route
            if not free_tier_route():
                return None
        except Exception as exc:
            logger.debug("Free tier startup line skipped: %s", exc)
            return None
        return t("gateway.startup.free_tier_line")

    _planned_restart_notice_lock: Optional[asyncio.Lock] = None

    async def _replay_pending_planned_restart_notification(self) -> None:
        """Send the planned-restart online notice to every home channel still owed one; clear
        ``.restart_pending.json`` only once all of them were reached.

        Runs from the boot pass and again from ``_install_reconnected_adapter``, so a home whose
        platform was down at boot gets its notice when the platform comes back (#112109). Delivered
        targets are recorded in the marker so neither a later replay nor the next process (if this
        one restarts first) notifies a home twice. The lock serializes a boot pass that outlived the
        restore gate against a concurrent reconnect replay.
        """
        from gateway.run import _planned_restart_notification_path
        from utils import atomic_json_write

        if self._planned_restart_notice_lock is None:
            self._planned_restart_notice_lock = asyncio.Lock()
        async with self._planned_restart_notice_lock:
            path = _planned_restart_notification_path()
            if not path.exists():
                return
            try:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                delivered = {tuple(target) for target in data.get("delivered_targets", [])}
                # Owed targets come from config, not live transports: a removed home or an opt-out
                # (gateway_restart_notification=false) must not keep the marker alive forever.
                owed = {
                    _served_notice_target_key(
                        profile, platform.value, cfg.home_channel.chat_id, cfg.home_channel.thread_id)
                    for profile, platform, cfg in self._served_home_channel_configs()
                    if cfg.home_channel and cfg.home_channel.chat_id and cfg.gateway_restart_notification
                }
                delivered |= await self._send_home_channel_startup_notifications(skip_targets=delivered)
                if owed <= delivered:
                    path.unlink(missing_ok=True)
                    return
                data["delivered_targets"] = [list(target) for target in delivered]
                atomic_json_write(path, data, indent=None)
            except Exception:
                logger.warning("Planned-restart notification remains pending", exc_info=True)

    async def _send_home_channel_startup_notifications(
        self, *, skip_targets: Optional[set[tuple[str, str, Optional[str]]]] = None
    ) -> set[tuple[str, str, Optional[str]]]:
        """Notify EVERY served profile's configured home channels that the gateway is back online.

        Best-effort, once per home CHAT — several served profiles can share one chat (a single
        Telegram group for the whole host), and one host process restarting once owes that chat
        one notice. Accounting stays per profile so the marker's owed set still discharges.
        ``skip_targets`` lets startup avoid duplicate messages when a more specific restart
        notification is queued for the same chat.
        """
        delivered: set[tuple[str, str, Optional[str]]] = set()
        skipped = skip_targets or set()
        message = t("gateway.startup.online")
        free_tier_line = self._free_tier_startup_line()
        if free_tier_line:
            message = f"{message}\n{free_tier_line}"
        targets = list(self._served_home_channel_transports())
        # A chat already notified for ANOTHER profile is not notified again.
        notified_chats = {
            _delivery_target_key(platform.value, home.chat_id, home.thread_id, profile=profile)
            for profile, platform, _cfg, home, _transport in targets
            if _served_notice_target_key(profile, platform.value, home.chat_id, home.thread_id) in skipped
        }
        for profile, platform, platform_cfg, home, transport in targets:
            if not platform_cfg.gateway_restart_notification:
                logger.info(
                    "Home-channel startup notification suppressed: %s has gateway_restart_notification=false",
                    platform.value,
                )
                continue
            target = _served_notice_target_key(profile, platform.value, home.chat_id, home.thread_id)
            if target in skipped or target in delivered:
                continue
            chat = _delivery_target_key(platform.value, home.chat_id, home.thread_id, profile=profile)
            if chat in notified_chats:
                delivered.add(target)
                continue
            if await self._send_home_channel_message(
                platform, home, transport, message, "Home-channel startup notification failed for %s:%s: %s",
            ):
                notified_chats.add(chat)
                delivered.add(target)
                logger.info("Sent home-channel startup notification to %s:%s", platform.value, home.chat_id)
        return delivered

