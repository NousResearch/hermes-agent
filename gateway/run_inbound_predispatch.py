"""The ``pre_gateway_dispatch`` plugin hook for gateway inbound: the hook call and its once-only
gate, shared by the idle path (``_hm_admit_event``) and busy arrivals the adapter diverts before
``_handle_message`` (``_hm_admit_busy_ingress``, wired in ``run_adapters``)."""
from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

from gateway.session import SessionSource

if TYPE_CHECKING:
    from gateway.platforms.event import MessageEvent

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayPreDispatchMixin:
    async def _hm_pre_gateway_dispatch_hook(
        self, event: MessageEvent, source: SessionSource
    ) -> Optional[MessageEvent]:
        """Run the ``pre_gateway_dispatch`` plugin hook; None = drop, else the (maybe rewritten) event.
        Results: ``{"action": "skip"}`` → drop; ``{"action": "rewrite", "text"}`` → replace ``event.text``;
        ``allow``/None → normal dispatch. Runs BEFORE auth so plugins can handle unauthorized senders."""
        try:
            from hermes_cli.lifecycle import ainvoke_hook as _ainvoke_hook
            _hook_results = await _ainvoke_hook(
                "pre_gateway_dispatch", event=event, gateway=self,
                # getattr: bare-runner tests build GatewayRunner via object.__new__ without __init__.
                session_store=getattr(self, "session_store", None),
            )
        except Exception as _hook_exc:
            logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc, exc_info=True)
            _hook_results = []

        for _result in _hook_results:
            if not isinstance(_result, dict):
                continue
            _action = _result.get("action")
            if _action == "skip":
                logger.info(
                    "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                    _result.get("reason"), source.platform.value if source.platform else "unknown",
                    source.chat_id or "unknown",
                )
                return None
            if _action == "rewrite":
                _new_text = _result.get("text")
                if isinstance(_new_text, str):
                    event = dataclasses.replace(event, text=_new_text)
                break
            if _action == "allow":
                break
        return event

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
