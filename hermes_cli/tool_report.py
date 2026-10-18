"""Content-free, effective runtime capabilities for supervised invocations."""
from __future__ import annotations

import contextlib
import json
import sys


def report_tools(toolsets: str, required: list[str]) -> int:
    """Use the same resolver and suppression settings as the agent CLI."""
    from hermes_cli.config import load_config
    from model_tools import get_tool_definitions

    requested = list(dict.fromkeys(t.strip() for t in toolsets.split(",") if t.strip()))
    if not requested:
        raise ValueError("at least one toolset is required")
    config = load_config()
    with contextlib.redirect_stdout(sys.stderr):
        definitions = get_tool_definitions(
            enabled_toolsets=requested,
            disabled_toolsets=config.get("agent", {}).get("disabled_toolsets") or [],
            quiet_mode=True,
        )
    tools = sorted({d.get("function", d)["name"] for d in definitions})
    missing = sorted(set(required) - set(tools))
    print(json.dumps({
        "schema": "hermes.tool-report.v1", "ok": not missing,
        "toolsets": requested, "tools": tools, "missing_required": missing,
    }, sort_keys=True))
    return 1 if missing else 0
