"""Probe-only serve bootstrap: observe real ownership and inject config-read failure.

Not imported by production. The failure seam raises once during model-target
preparation; agent construction, rebuild predicates, WebSocket dispatch, and
SessionDB teardown remain real.
"""
import inspect
import os
from pathlib import Path

from tui_gateway import server
import hermes_state_registry as registry


original_target = server._config_model_target
marker = Path(os.environ["HERMES_HOME"]) / "fail-rebuild"


def config_target():
    if marker.exists() and any(frame.function in {"_reset_session_agent", "_sync_bot_capabilities"}
                               for frame in inspect.stack()):
        marker.unlink()
        raise OSError("probe config read failure")
    return original_target()


server._config_model_target = config_target


def ownership(rid, params):
    session = server._sessions.get(params["session_id"])
    agent = (session or {}).get("agent")
    with registry._lock:
        refs = {str(path): generation.refcount for path, generation in registry._generations.items()}
    return server._ok(rid, {"agent": id(agent) if agent else None,
                            "owns_db": bool(getattr(agent, "_owns_session_db", False)),
                            "refs": refs})


server._methods["probe.ownership"] = ownership

if __name__ == "__main__":
    from hermes_cli.main import main
    main()
