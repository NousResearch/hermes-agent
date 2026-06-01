"""Runtime handles shared by gateway-hosted tools.

The gateway owns long-lived platform clients. Tool modules can reuse those
clients through this tiny indirection instead of opening duplicate connections
or requiring credentials to be copied into every execution context.
"""

from __future__ import annotations

from typing import Any, Optional

_slack_adapter: Optional[Any] = None


def set_slack_adapter(adapter: Optional[Any]) -> None:
    """Expose the active Slack adapter to gateway-hosted Slack tools."""
    global _slack_adapter
    _slack_adapter = adapter


def get_slack_adapter() -> Optional[Any]:
    """Return the active Slack adapter, if the gateway registered one."""
    return _slack_adapter
