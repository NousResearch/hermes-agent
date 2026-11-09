"""Minimal startup for the gateway's lifetime control-only mode."""

from __future__ import annotations

import logging
from dataclasses import replace

from gateway.config import Platform

logger = logging.getLogger(__name__)


async def start_control_only(runner) -> bool:
    """Open only the authenticated API control surface, with no serving recovery."""
    from gateway.platforms.api_server import APIServerAdapter

    platform_config = runner.config.platforms.get(Platform.API_SERVER)
    if platform_config is None or not platform_config.enabled:
        logger.error("Control-only mode requires platforms.api_server to be enabled")
        return False
    adapter_config = replace(
        platform_config, extra={**(platform_config.extra or {}), "_hermes_control_only": True})
    adapter = APIServerAdapter(adapter_config)
    adapter.gateway_runner = runner
    if not await adapter.connect():
        return False
    runner.adapters = {Platform.API_SERVER: adapter}
    runner._running = True
    runner._update_runtime_status("control_only")
    return True
