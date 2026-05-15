"""RTK Plugin for Hermes — rewrites terminal commands through rtk to save tokens.

Install: place this directory at ~/.hermes/plugins/rtk/
Disable: set HERMES_RTK_DISABLE=1
"""

import os
import shutil
import subprocess
import logging
from typing import Any, Dict, Optional

from utils import env_var_enabled

logger = logging.getLogger(__name__)


def _rewrite_terminal_command(
    tool_name: str, args: dict, **kwargs
) -> Optional[Dict[str, Any]]:
    """pre_tool_call hook that rewrites terminal commands through rtk.

    Returns a rewrite directive when the command can be optimized,
    None otherwise (pass-through).
    """
    if tool_name != "terminal":
        return None

    if env_var_enabled("HERMES_RTK_DISABLE"):
        return None

    command = args.get("command")
    if not command or not isinstance(command, str):
        return None

    if not shutil.which("rtk"):
        return None

    try:
        result = subprocess.run(
            ["rtk", "rewrite", command],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except subprocess.TimeoutExpired:
        logger.debug("rtk rewrite timed out for: %s", command[:80])
        return None
    except Exception as exc:
        logger.debug("rtk rewrite error: %s", exc)
        return None

    if result.returncode != 0:
        return None

    rewritten = result.stdout.strip()
    if not rewritten or rewritten == command:
        return None

    logger.debug("rtk rewrite: %s -> %s", command[:60], rewritten[:60])
    return {"action": "rewrite", "args": {"command": rewritten}}


def register(ctx):
    """Plugin entry point — register the RTK pre_tool_call hook."""
    ctx.register_hook("pre_tool_call", _rewrite_terminal_command)
