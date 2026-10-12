"""Plugin dispatch handling for inbound gateway events."""

from __future__ import annotations

import dataclasses
import logging
from typing import Optional

from gateway.platforms.event import MessageEvent
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


async def pre_gateway_dispatch(
    gateway, event: MessageEvent, source: SessionSource
) -> Optional[MessageEvent]:
    """Run the hook; None drops, while rewrite replaces text before authorization."""
    try:
        from hermes_cli.lifecycle import ainvoke_hook

        hook_results = await ainvoke_hook(
            "pre_gateway_dispatch",
            event=event,
            gateway=gateway,
            # Bare-runner tests may construct GatewayRunner without __init__.
            session_store=getattr(gateway, "session_store", None),
        )
    except Exception as hook_exc:
        logger.warning(
            "pre_gateway_dispatch invocation failed: %s", hook_exc, exc_info=True
        )
        hook_results = []

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        if action == "skip":
            logger.info(
                "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                result.get("reason"),
                source.platform.value if source.platform else "unknown",
                source.chat_id or "unknown",
            )
            return None
        if action == "rewrite":
            new_text = result.get("text")
            if isinstance(new_text, str):
                event = dataclasses.replace(event, text=new_text)
            break
        if action == "allow":
            break
    return event
