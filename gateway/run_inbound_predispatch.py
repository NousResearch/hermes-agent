"""The once-only gate on the ``pre_gateway_dispatch`` plugin hook, shared by the idle path
(``_hm_admit_event``) and busy arrivals the adapter diverts before ``_handle_message``
(``_hm_admit_busy_ingress``, wired in ``run_adapters`` through the handlers below). The hook call itself is
``GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook``."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

from gateway.session import SessionSource

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent


class GatewayPreDispatchMixin:
    async def _hm_pre_gateway_dispatch_once(
        self, event: MessageEvent, source: SessionSource
    ) -> Optional[MessageEvent]:
        """Apply the pre-dispatch hook once to one process-local inbound event."""
        if getattr(event, "_pre_gateway_dispatch_applied", False):
            return event
        event = await self._hm_pre_gateway_dispatch_hook(event, source)
        if event is not None:
            event._pre_gateway_dispatch_applied = True
        return event

    async def _hm_admit_busy_ingress(
        self, event: MessageEvent
    ) -> Optional[MessageEvent]:
        """Run the normal pre-dispatch contract before an active-session diversion."""
        if getattr(event, "internal", False):
            return event
        return await self._hm_pre_gateway_dispatch_once(event, event.source)

    def _make_profile_pre_gateway_dispatch_handler(self, profile_name: str):
        """Apply pre-dispatch under the owning secondary profile before a busy diversion
        (canonicalize FIRST, like the busy-path twin, so the hook sees the routed identity)."""
        from gateway.run import _async_profile_runtime_scope
        profile_home = self._routed_profile_home(profile_name)

        async def _handler(event):
            self._canonicalize(getattr(event, "source", None), transport_profile=profile_name)
            async with self._async_scope_or_null(_async_profile_runtime_scope, profile_home):
                return await self._hm_admit_busy_ingress(event)

        return _handler

    def _make_default_profile_pre_gateway_dispatch_handler(self):
        """Apply pre-dispatch under a primary adapter event's routed profile."""
        from gateway.run import _async_profile_runtime_scope, get_hermes_home
        default_home = Path(get_hermes_home())

        async def _handler(event):
            source = event.source
            profile_home = self._admit_primary_source(source, default_home)
            if profile_home is None:
                return None
            async with _async_profile_runtime_scope(profile_home):
                return await self._hm_admit_busy_ingress(event)

        return _handler

    def _primary_pre_gateway_dispatch_handler(self):
        """Return the correctly scoped pre-dispatch handler for busy primary ingress."""
        if self._multiplex_on():
            return self._make_default_profile_pre_gateway_dispatch_handler()
        return self._hm_admit_busy_ingress
