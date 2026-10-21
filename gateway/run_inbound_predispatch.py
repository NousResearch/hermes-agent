"""The once-only gate on the ``pre_gateway_dispatch`` plugin hook, shared by the idle path
(``_hm_admit_event``) and busy arrivals the adapter diverts before ``_handle_message``
(``_hm_admit_busy_ingress``, wired in ``run_adapters``). The hook call itself is
``GatewayInboundHooksMixin._hm_pre_gateway_dispatch_hook``."""
from __future__ import annotations

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
