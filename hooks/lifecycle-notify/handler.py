"""Send lifecycle notifications to the configured home channel.

This hook fires on gateway:startup and logs a notification that the agent
is online. Full adapter-based message sending (actually delivering the
notification to the home channel via Signal/Telegram/etc.) will be wired
in Phase 2 when the HSM wizard deploys this hook into agent containers.

Environment variables read:
  HERMES_HOME_CHANNEL    — target channel/group ID for lifecycle messages
  HERMES_AGENT_NAME      — display name of this agent (default: "hermes")
  HERMES_DEFAULT_PLATFORM — platform adapter name (e.g. "signal", "telegram")
"""

import logging
import os

logger = logging.getLogger("hooks.lifecycle-notify")


async def handle(event_type: str, context: dict) -> None:
    """On gateway:startup, log an 'online' notification for the home channel."""
    if event_type != "gateway:startup":
        return

    home_channel = os.environ.get("HERMES_HOME_CHANNEL")
    if not home_channel:
        return

    agent_name = os.environ.get("HERMES_AGENT_NAME", "hermes")
    platform = os.environ.get("HERMES_DEFAULT_PLATFORM", "")

    if not platform:
        return

    logger.info(
        "%s is online (home_channel=%s, platform=%s)",
        agent_name,
        home_channel,
        platform,
    )
