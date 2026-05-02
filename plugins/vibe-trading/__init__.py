"""Hermes plugin that bridges to a running Vibe-Trading API service."""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

TOOLSET = "vibe-trading"
DEFAULT_BASE_URL = "http://192.168.1.58:8899"
DEFAULT_TIMEOUT_SECONDS = 20.0


def _base_url() -> str:
    return os.getenv("VIBE_TRADING_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _timeout_seconds() -> float:
    raw = os.getenv("VIBE_TRADING_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        return max(1.0, float(raw))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False)


def _request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> str:
    """Call Vibe-Trading and return a JSON string for Hermes tool output."""
    url = f"{_base_url()}/{path.lstrip('/')}"
    if query:
        clean_query = {k: v for k, v in query.items() if v is not None}
        if clean_query:
            url = f"{url}?{urllib.parse.urlencode(clean_query)}"

    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=body, headers=headers, method=method.upper())

    try:
        with urllib.request.urlopen(request, timeout=_timeout_seconds()) as response:
            raw = response.read().decode("utf-8", errors="replace")
            if not raw.strip():
                return _json_dumps({"success": True, "status": getattr(response, "status", 200)})
            try:
                return _json_dumps(json.loads(raw))
            except json.JSONDecodeError:
                return _json_dumps({"success": True, "text": raw})
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(exc)
        return _json_dumps({
            "success": False,
            "error_type": "HTTPError",
            "status": exc.code,
            "error": detail,
            "url": url,
        })
    except Exception as exc:
        return _json_dumps({
            "success": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "url": url,
        })


def _vibe_health(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/health")


def _vibe_list_skills(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/skills")


def _vibe_list_swarm_presets(args: dict[str, Any], **kwargs) -> str:
    return _request_json("GET", "/swarm/presets")


def _vibe_run_swarm(args: dict[str, Any], **kwargs) -> str:
    payload = {
        "preset_name": args.get("preset_name"),
        "variables": args.get("variables") or {},
    }
    return _request_json("POST", "/swarm/runs", payload)


def _vibe_get_swarm_run(args: dict[str, Any], **kwargs) -> str:
    run_id = str(args.get("run_id", "")).strip()
    if not run_id:
        return _json_dumps({"success": False, "error": "run_id is required"})
    return _request_json("GET", f"/swarm/runs/{urllib.parse.quote(run_id, safe='')}")


def _vibe_create_session(args: dict[str, Any], **kwargs) -> str:
    payload = {
        "title": args.get("title") or "",
        "config": args.get("config"),
    }
    return _request_json("POST", "/sessions", payload)


def _vibe_send_message(args: dict[str, Any], **kwargs) -> str:
    session_id = str(args.get("session_id", "")).strip()
    content = str(args.get("content", "")).strip()
    if not session_id:
        return _json_dumps({"success": False, "error": "session_id is required"})
    if not content:
        return _json_dumps({"success": False, "error": "content is required"})
    return _request_json(
        "POST",
        f"/sessions/{urllib.parse.quote(session_id, safe='')}/messages",
        {"content": content},
    )


def _vibe_get_run_result(args: dict[str, Any], **kwargs) -> str:
    run_id = str(args.get("run_id", "")).strip()
    if not run_id:
        return _json_dumps({"success": False, "error": "run_id is required"})
    return _request_json("GET", f"/runs/{urllib.parse.quote(run_id, safe='')}")


def _vibe_list_runs(args: dict[str, Any], **kwargs) -> str:
    limit = args.get("limit", 20)
    return _request_json("GET", "/runs", query={"limit": limit})


VIBE_HEALTH_SCHEMA = {
    "name": "vibe_health",
    "description": "Check whether the Vibe-Trading API service is reachable and healthy.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_LIST_SKILLS_SCHEMA = {
    "name": "vibe_list_skills",
    "description": "List Vibe-Trading finance skills, including A-share analysis skills.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_LIST_SWARM_PRESETS_SCHEMA = {
    "name": "vibe_list_swarm_presets",
    "description": "List Vibe-Trading multi-agent team presets such as investment_committee and risk_committee.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

VIBE_RUN_SWARM_SCHEMA = {
    "name": "vibe_run_swarm",
    "description": (
        "Run a Vibe-Trading multi-agent team. Use this for investment committee, "
        "risk committee, quant strategy, and A-share research desk analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "preset_name": {
                "type": "string",
                "description": "Swarm preset name, e.g. investment_committee, risk_committee, quant_strategy_desk.",
            },
            "variables": {
                "type": "object",
                "description": "Preset variables, e.g. {'target': '600519.SH', 'market': 'A股'}.",
                "additionalProperties": {"type": "string"},
            },
        },
        "required": ["preset_name", "variables"],
    },
}

VIBE_GET_SWARM_RUN_SCHEMA = {
    "name": "vibe_get_swarm_run",
    "description": "Fetch a Vibe-Trading swarm run result by run_id.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "description": "Swarm run identifier."}},
        "required": ["run_id"],
    },
}

VIBE_CREATE_SESSION_SCHEMA = {
    "name": "vibe_create_session",
    "description": "Create a Vibe-Trading session for natural-language finance analysis.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Session title."},
            "config": {"type": "object", "description": "Optional session config."},
        },
        "required": [],
    },
}

VIBE_SEND_MESSAGE_SCHEMA = {
    "name": "vibe_send_message",
    "description": (
        "Send a natural-language request to a Vibe-Trading session. Use this for "
        "A-share buy/sell research, backtest requests, ST risk checks, and sector analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {"type": "string", "description": "Vibe-Trading session identifier."},
            "content": {"type": "string", "description": "Natural-language analysis request."},
        },
        "required": ["session_id", "content"],
    },
}

VIBE_GET_RUN_RESULT_SCHEMA = {
    "name": "vibe_get_run_result",
    "description": "Fetch a Vibe-Trading run result by run_id, including backtest and analysis outputs.",
    "parameters": {
        "type": "object",
        "properties": {"run_id": {"type": "string", "description": "Run identifier."}},
        "required": ["run_id"],
    },
}

VIBE_LIST_RUNS_SCHEMA = {
    "name": "vibe_list_runs",
    "description": "List recent Vibe-Trading runs with summary fields.",
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of runs to return.",
                "default": 20,
                "minimum": 1,
                "maximum": 100,
            }
        },
        "required": [],
    },
}


def register(ctx):
    """Register Vibe-Trading bridge tools."""
    tools = [
        ("vibe_health", VIBE_HEALTH_SCHEMA, _vibe_health),
        ("vibe_list_skills", VIBE_LIST_SKILLS_SCHEMA, _vibe_list_skills),
        ("vibe_list_swarm_presets", VIBE_LIST_SWARM_PRESETS_SCHEMA, _vibe_list_swarm_presets),
        ("vibe_run_swarm", VIBE_RUN_SWARM_SCHEMA, _vibe_run_swarm),
        ("vibe_get_swarm_run", VIBE_GET_SWARM_RUN_SCHEMA, _vibe_get_swarm_run),
        ("vibe_create_session", VIBE_CREATE_SESSION_SCHEMA, _vibe_create_session),
        ("vibe_send_message", VIBE_SEND_MESSAGE_SCHEMA, _vibe_send_message),
        ("vibe_get_run_result", VIBE_GET_RUN_RESULT_SCHEMA, _vibe_get_run_result),
        ("vibe_list_runs", VIBE_LIST_RUNS_SCHEMA, _vibe_list_runs),
    ]
    for name, schema, handler in tools:
        ctx.register_tool(
            name=name,
            toolset=TOOLSET,
            schema=schema,
            handler=handler,
            description=schema["description"],
        )
    logger.info("vibe-trading plugin: registered %s tools", len(tools))
