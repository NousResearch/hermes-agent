"""llm-switch plugin — auto-manage local LLM servers when switching models.

Uses the pre_llm_call hook to detect when the active model matches a locally
configured model, and starts/swaps the server automatically.  Also registers
a switch_local_llm tool so the agent can switch models autonomously.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_PLUGIN_DIR = Path(__file__).parent
_config: Optional[Dict[str, Any]] = None


def _load_config() -> Optional[Dict[str, Any]]:
    """Load models.yaml from plugin dir or LLM_SWITCH_MODELS env var."""
    global _config
    if _config is not None:
        return _config

    config_path = os.getenv(
        "LLM_SWITCH_MODELS", str(_PLUGIN_DIR / "models.yaml")
    )
    try:
        import yaml
    except ImportError:
        logger.warning("llm-switch: PyYAML not installed, plugin disabled")
        return None

    path = Path(config_path)
    if not path.exists():
        logger.info(
            "llm-switch: No models.yaml found at %s — copy models.yaml.example to get started",
            path,
        )
        return None

    with open(path, encoding="utf-8") as f:
        _config = yaml.safe_load(f)
    return _config


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

def _on_pre_llm_call(messages: list, model: str, **kwargs: Any) -> None:
    """Auto-start the right local server before each LLM call.

    If the current model name matches a key in models.yaml and the correct
    server isn't already running, kill any existing server and start the
    right one.  This makes ``/model custom:write`` seamless — the server
    spins up on the first message after switching.
    """
    from . import server

    config = _load_config()
    if not config or model not in config.get("models", {}):
        return  # not a local model — nothing to do

    status = server.get_status(config)
    if status.get("running") and status.get("model") == model:
        return  # correct model already running

    desc = config["models"][model].get("description", model)
    print(f"  Starting local model: {model} ({desc})")

    if server.start_server(model, config):
        port = config.get("server", {}).get("port", 8080)
        print(f"  Ready on http://localhost:{port}/v1")
    else:
        print(f"  WARNING: Server timed out — check /tmp/llama-server.log")


def _on_session_end(session_id: str, platform: str, **kwargs: Any) -> None:
    """Kill the local server when the session ends."""
    config = _load_config()
    if not config:
        return
    try:
        from . import server

        status = server.get_status(config)
        if status.get("running"):
            binary = config.get("server", {}).get("binary", "llama-server")
            server.kill_server(binary=binary)
            logger.info("llm-switch: Stopped server on session end")
    except Exception as exc:
        logger.debug("llm-switch: Error stopping server: %s", exc)


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def _handle_switch(args: dict, **kwargs: Any) -> str:
    """Handle the switch_local_llm tool call."""
    from . import server

    model = args.get("model", "").strip()

    config = _load_config()
    if not config:
        return json.dumps({"error": "No models.yaml configured"})

    if model == "status":
        return json.dumps(server.get_status(config))

    if model == "stop":
        binary = config.get("server", {}).get("binary", "llama-server")
        server.kill_server(binary=binary)
        return json.dumps({"status": "stopped"})

    if model == "list":
        models = {
            k: v.get("description", "")
            for k, v in config.get("models", {}).items()
        }
        return json.dumps({"models": models})

    if model not in config.get("models", {}):
        available = list(config.get("models", {}).keys())
        return json.dumps(
            {"error": f"Unknown model '{model}'", "available": available}
        )

    desc = config["models"][model].get("description", model)
    print(f"  Switching to: {model} ({desc})")
    ok = server.start_server(model, config)
    port = config.get("server", {}).get("port", 8080)

    if ok:
        return json.dumps({
            "status": "ready",
            "model": model,
            "description": desc,
            "endpoint": f"http://localhost:{port}/v1",
        })
    return json.dumps({
        "status": "starting",
        "model": model,
        "note": "Server still loading — check /tmp/llama-server.log",
    })


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register(ctx: Any) -> None:
    """Wire schemas to handlers and register lifecycle hooks."""
    from . import schemas

    ctx.register_tool(
        name="switch_local_llm",
        toolset="llm-switch",
        schema=schemas.SWITCH_LOCAL_LLM,
        handler=_handle_switch,
        description="Switch local LLM model",
    )
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
