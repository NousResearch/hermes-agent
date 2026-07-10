"""Gateway lifecycle command guard for cron jobs.

Cron normally runs inside the gateway process. A cron job that restarts/stops
that same gateway can SIGTERM the scheduler mid-turn and create an auto-resume
restart loop under launchd/systemd. Keep this guard in the cron package (not the
CLI) so every creation path and runtime script execution path shares it.
"""

from __future__ import annotations

import re

# Patterns that indicate a cron job targets the Hermes gateway lifecycle.
# Deliberately specific: do not block normal prompts that merely mention an
# unrelated gateway and restart events (e.g. API gateway incident summaries).
GATEWAY_LIFECYCLE_PATTERNS = re.compile(
    r"(?i)"
    r"(hermes\s+gateway\s+(restart|stop|start))"
    r"|(launchctl\s+(kickstart|bootout|bootstrap|unload|load|stop|restart)\s+.*hermes)"
    r"|(systemctl\s+(-\S+\s+)*(restart|stop|start)\s+.*hermes)"
    r"|(p?kill\s+.*hermes.*gateway)"
)


BLOCK_MESSAGE = (
    "Blocked: cron job contains a gateway lifecycle command "
    "(restart/stop/kill). This is blocked to prevent restart loops (#30719). "
    "Use `hermes gateway restart` from a shell outside the gateway."
)


def contains_gateway_lifecycle_command(text: str | None) -> bool:
    """Return True if *text* contains a Hermes gateway lifecycle command."""
    if not text:
        return False
    return bool(GATEWAY_LIFECYCLE_PATTERNS.search(str(text)))


def assert_no_gateway_lifecycle_command(text: str | None) -> None:
    """Raise ValueError when *text* contains a blocked gateway lifecycle command."""
    if contains_gateway_lifecycle_command(text):
        raise ValueError(BLOCK_MESSAGE)
