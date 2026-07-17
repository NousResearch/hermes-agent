"""Per-agent plugin hook policy for reduced-authority turns."""

from typing import Any


def invoke_agent_hook(agent: Any, event: str, **kwargs):
    """Invoke a plugin hook unless this agent forbids plugin egress."""
    if getattr(agent, "_skip_plugin_hooks", False):
        return []

    from hermes_cli.plugins import invoke_hook

    return invoke_hook(event, **kwargs)
