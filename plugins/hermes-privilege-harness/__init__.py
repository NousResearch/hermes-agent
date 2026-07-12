"""
Hermes Privilege Harness — Passive VIP Plugin.

Architecture: Hermes handles approval, we handle execution.
"""

import logging
from . import guard

logger = logging.getLogger("hermes-vip.plugin")


def register(ctx):
    # pre_tool_call — only intercept vip_sudo for native approval card
    ctx.register_hook("pre_tool_call", _hook)

    # vip_sudo — the ONLY privileged tool
    ctx.register_tool(
        name="vip_sudo",
        toolset="terminal",
        description=(
            "Execute commands as root via a secure privilege daemon. "
            "Hermes will prompt for approval before execution."
        ),
        schema={
            "name": "vip_sudo",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute as root",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why root is needed",
                    },
                },
                "required": ["command"],
            },
        },
        handler=lambda args, **kw: guard.vip_sudo(
            args.get("command", "") if isinstance(args, dict) else str(args),
            args.get("reason", "") if isinstance(args, dict) else "",
        ),
    )

    logger.info("hermes-privilege-harness plugin ready")


def _hook(tool_name, args, **kwargs):
    return guard.check(tool_name, args if isinstance(args, dict) else {})
