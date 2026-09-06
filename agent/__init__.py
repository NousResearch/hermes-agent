"""Agent internals extracted from run_agent.py so it stays focused on AIAgent."""

# Python imports the parent package before executing a -m entry point. Gate
# this exact MCP launch before jiter_preload can map mutable venv extensions.
import sys as _sys

if "agent.transports.hermes_tools_mcp_server" in getattr(_sys, "orig_argv", ()):
    try:
        from hermes_mcp_update_gate import should_quiesce_mcp_bridge as _update_gate
        _paused_for_update = _update_gate()
    except Exception:
        _paused_for_update = False
    if _paused_for_update:
        _sys.stderr.write("Hermes tools are paused while the installation is updating.\n")
        raise SystemExit(0)

from . import jiter_preload as _jiter_preload  # noqa: F401
