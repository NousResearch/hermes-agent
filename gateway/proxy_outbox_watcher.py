"""Thin-gateway consumer for proactive proxy-outbox deliveries."""

import asyncio
import logging
import os
from typing import Optional
from urllib.parse import quote

from gateway.config import Platform


logger = logging.getLogger("gateway.run")


class GatewayProxyOutboxMixin:
    """Consume each configured profile's outbox through native adapters."""

    @staticmethod
    def _get_proxy_key() -> str:
        """Read the per-profile proxy credential through the active secret scope."""
        try:
            from agent.secret_scope import UnscopedSecretError, get_secret

            try:
                return (get_secret("GATEWAY_PROXY_KEY") or "").strip()
            except UnscopedSecretError:
                pass
        except Exception:
            pass
        return os.getenv("GATEWAY_PROXY_KEY", "").strip()

    def _should_run_proxy_outbox_watcher(self) -> bool:
        """Start for root proxy mode or to discover scoped multiplex proxies."""
        return bool(self._get_proxy_url() and self._get_proxy_key()) or bool(
            getattr(self.config, "multiplex_profiles", False)
        )

    async def _proxy_outbox_watcher(self, interval: float = 1.0) -> None:
        """Deliver proactive host output through each profile's native adapters."""
        from aiohttp import ClientSession, ClientTimeout
        from gateway.proxy_outbox import deliver_once
        from gateway.run import _async_profile_runtime_scope, _handoff_watch_scopes

        excluded = {Platform.LOCAL, Platform.API_SERVER, Platform.RELAY}

        async def _poll(profile_name: Optional[str]) -> None:
            proxy_url = self._get_proxy_url()
            proxy_key = self._get_proxy_key()
            if not proxy_url or not proxy_key:
                return
            adapter_map = (
                self.adapters
                if profile_name is None
                else (self._profile_adapters or {}).get(profile_name, {})
            )
            adapters = {
                platform: adapter
                for platform, adapter in adapter_map.items()
                if platform not in excluded
            }
            if not adapters:
                return
            if profile_name is not None:
                proxy_url = (
                    f"{proxy_url.rstrip('/')}/p/{quote(profile_name, safe='')}"
                )
            await deliver_once(
                proxy_url,
                proxy_key,
                adapters,
                session=session,
            )

        async with ClientSession(timeout=ClientTimeout(total=30)) as session:
            while self._running:
                for profile_name, profile_home in _handoff_watch_scopes(self):
                    try:
                        if profile_home is None:
                            await _poll(profile_name)
                        else:
                            async with _async_profile_runtime_scope(profile_home):
                                await _poll(profile_name)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.warning(
                            "Proxy outbox poll failed for profile %s (%s)",
                            profile_name or "default",
                            type(exc).__name__,
                        )
                await asyncio.sleep(interval)
