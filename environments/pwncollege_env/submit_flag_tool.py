"""submit_flag tool for pwn.college RL environments.

Registers a `submit_flag` tool in the hermes-agent tool registry under the
"pwncollege" toolset. The handler checks flags against the dojo RL API using
per-task context (SDK client + slot) stored in a module-level dict.

Usage in an environment:
    from environments.pwncollege_env.submit_flag_tool import (
        register_flag_context, clear_flag_context,
    )

    # Before agent loop
    register_flag_context(task_id, sync_client, slot)

    # After agent loop
    clear_flag_context(task_id)
"""

import json
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

# Per-task context: task_id → {"client": DojoRLSyncClient, "slot": int}
_task_flag_context: Dict[str, Dict[str, Any]] = {}


def register_flag_context(task_id: str, sync_client: Any, slot: int) -> None:
    """Register dojo client + slot for a rollout so submit_flag can verify flags."""
    _task_flag_context[task_id] = {"client": sync_client, "slot": slot}


def clear_flag_context(task_id: str) -> None:
    """Remove flag context after rollout completes."""
    _task_flag_context.pop(task_id, None)


def _submit_flag_handler(args: dict, **kw) -> str:
    """Handle submit_flag tool calls by checking the flag against the dojo API."""
    task_id = kw.get("task_id", "default")
    flag = args.get("flag", "")

    if not flag:
        return json.dumps({"success": False, "message": "No flag provided."})

    ctx = _task_flag_context.get(task_id)
    if not ctx:
        return json.dumps({
            "success": False,
            "message": "No active challenge instance for this task.",
        })

    try:
        correct = ctx["client"].check_flag(ctx["slot"], flag)
    except Exception as e:
        logger.error("Flag check failed for task %s: %s", task_id, e, exc_info=True)
        return json.dumps({"success": False, "message": f"Flag check error: {type(e).__name__}"})

    if correct:
        return json.dumps({"success": True, "message": "Flag accepted! Challenge solved."})
    return json.dumps({"success": False, "message": "Incorrect flag."})


# ---------------------------------------------------------------------------
# Register in hermes-agent tool registry
# ---------------------------------------------------------------------------

SUBMIT_FLAG_SCHEMA = {
    "name": "submit_flag",
    "description": (
        "Submit a flag for verification. Use this when you find a flag "
        "(format: pwn.college{...}) to check if it is correct."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "flag": {
                "type": "string",
                "description": "The flag string (format: pwn.college{...}).",
            }
        },
        "required": ["flag"],
    },
}

from tools.registry import registry
from toolsets import create_custom_toolset

registry.register(
    name="submit_flag",
    toolset="pwncollege",
    schema=SUBMIT_FLAG_SCHEMA,
    handler=_submit_flag_handler,
    emoji="🚩",
)

create_custom_toolset(
    name="pwncollege",
    description="PwnCollege CTF tools",
    tools=["submit_flag"],
)
