"""Warm the existing process-local toolchain probe before the first gateway prompt."""

import asyncio
import logging
import time

from gateway.run_shutdown import _log_suppressed

logger = logging.getLogger("gateway.run")


def _warm_environment_probe_sync() -> bool:
    from hermes_cli.config import load_config_readonly
    from tools.env_probe import get_environment_probe_line

    config = load_config_readonly() or {}
    agent_config = config.get("agent")
    if not isinstance(agent_config, dict):
        agent_config = {}
    if not agent_config.get("environment_probe", True):
        return False
    # This resolver owns remote-backend omission, the single worker, cache and
    # bounded wait. Reuse it rather than taking a new prompt or tool snapshot.
    get_environment_probe_line()
    return True


async def warm_environment_probe() -> None:
    with _log_suppressed(
        logging.WARNING, "Environment-probe warm-up failed; first inbound turn will initialize lazily",
        exc_info=True,
    ):
        started = time.monotonic()
        # Profile config and terminal-policy ContextVars must reach the resolver.
        if await asyncio.to_thread(_warm_environment_probe_sync):
            logger.info("Environment probe prepared in %.1fs", time.monotonic() - started)
