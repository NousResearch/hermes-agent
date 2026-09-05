"""Runtime config hooks registration for serve and dashboard startup (#102504).

Registers config-defined shell hooks (pre_tool_call, post_tool_call, etc.)
and outbound webhooks for the dashboard/serve in-process runtime.
Consolidates prior defect family implementations (#61844, #70461, #69832, #102513, #81409).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def register_runtime_config_hooks(args: Optional[Any] = None) -> None:
    """Register config shell hooks and outbound webhooks for the dashboard/serve runtime.

    Must be invoked after synchronous ``discover_plugins()`` so plugin callbacks
    retain deterministic priority over config shell hooks in the hook registry.
    """
    try:
        from hermes_cli.config import load_config
        config = load_config()
    except Exception:
        logger.debug("Failed to load config for runtime hook registration", exc_info=True)
        return

    if not isinstance(config, dict):
        return

    # 1. Register config shell hooks (pre_tool_call, post_tool_call, user_prompt, etc.)
    try:
        from agent import shell_hooks
        accept_hooks = getattr(args, "accept_hooks", False) if args is not None else False
        shell_hooks.register_from_config(config, accept_hooks=accept_hooks)
    except Exception:
        logger.debug("Failed to register shell hooks from config", exc_info=True)

    # 2. Register outbound webhooks (session start/end, tool events, etc.)
    try:
        from agent import outbound_webhooks
        outbound_webhooks.register_from_config(config)
    except Exception:
        logger.debug("Failed to register outbound webhooks from config", exc_info=True)
