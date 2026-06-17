"""Native WHOOP tools for Hermes (registered via plugins/whoop)."""

from __future__ import annotations

from typing import Any, Dict

from hermes_cli.auth import get_auth_status
from plugins.whoop.client import (
    WHOOPAPIError,
    WHOOPAuthRequiredError,
    WHOOPClient,
    WHOOPError,
)
from tools.registry import tool_error, tool_result


def _check_whoop_available() -> bool:
    try:
        return bool(get_auth_status("whoop").get("logged_in"))
    except Exception:
        return False


def _whoop_client() -> WHOOPClient:
    return WHOOPClient()


def _whoop_tool_error(exc: Exception) -> str:
    if isinstance(exc, WHOOPAPIError):
        return tool_error(str(exc), status_code=exc.status_code)
    if isinstance(exc, (WHOOPError, WHOOPAuthRequiredError)):
        return tool_error(str(exc))
    return tool_error(f"WHOOP tool failed: {type(exc).__name__}: {exc}")


def _coerce_limit(raw: Any, *, default: int = 25, minimum: int = 1, maximum: int = 25) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _coerce_max_pages(raw: Any, *, default: int = 1, minimum: int = 1, maximum: int = 10) -> int:
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _list_kwargs(args: dict) -> Dict[str, Any]:
    return {
        "start": args.get("start"),
        "end": args.get("end"),
        "limit": _coerce_limit(args.get("limit")) if args.get("limit") is not None else None,
        "next_token": args.get("next_token") or args.get("nextToken"),
        "max_pages": _coerce_max_pages(args.get("max_pages")),
    }


def _first_record(payload: Any, *, empty_message: str) -> Any:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list) and payload["records"]:
        return payload["records"][0]
    return {"empty": True, "message": empty_message, "source": payload}


def _require_id(args: dict, *, field: str = "id") -> str | None:
    raw = args.get(field) or args.get("whoop_id")
    value = str(raw or "").strip()
    return value or None


def _handle_whoop_profile(args: dict, **kw) -> str:
    try:
        return tool_result(_whoop_client().get_profile())
    except Exception as exc:
        return _whoop_tool_error(exc)


def _handle_whoop_cycles(args: dict, **kw) -> str:
    action = str(args.get("action") or "latest").strip().lower()
    try:
        client = _whoop_client()
        if action == "list":
            return tool_result(client.list_cycles(**_list_kwargs(args)))
        if action == "latest":
            return tool_result(_first_record(
                client.list_cycles(**_list_kwargs(args)),
                empty_message="No WHOOP cycles found for the requested window.",
            ))
        if action == "get":
            cycle_id = _require_id(args)
            if not cycle_id:
                return tool_error("id is required for action='get'")
            return tool_result(client.get_cycle(cycle_id))
        return tool_error(f"Unknown whoop_cycles action: {action}")
    except Exception as exc:
        return _whoop_tool_error(exc)


def _handle_whoop_recovery(args: dict, **kw) -> str:
    action = str(args.get("action") or "latest").strip().lower()
    try:
        client = _whoop_client()
        if action == "list":
            return tool_result(client.list_recovery(**_list_kwargs(args)))
        if action == "latest":
            return tool_result(_first_record(
                client.list_recovery(**_list_kwargs(args)),
                empty_message="No WHOOP recovery records found for the requested window.",
            ))
        if action == "get":
            cycle_id = _require_id(args, field="cycle_id")
            if not cycle_id:
                return tool_error("cycle_id is required for action='get'")
            return tool_result(client.get_recovery(cycle_id))
        return tool_error(f"Unknown whoop_recovery action: {action}")
    except Exception as exc:
        return _whoop_tool_error(exc)


def _handle_whoop_sleep(args: dict, **kw) -> str:
    action = str(args.get("action") or "latest").strip().lower()
    try:
        client = _whoop_client()
        if action == "list":
            return tool_result(client.list_sleep(**_list_kwargs(args)))
        if action == "latest":
            return tool_result(_first_record(
                client.list_sleep(**_list_kwargs(args)),
                empty_message="No WHOOP sleep records found for the requested window.",
            ))
        if action == "get":
            sleep_id = _require_id(args)
            if not sleep_id:
                return tool_error("id is required for action='get'")
            return tool_result(client.get_sleep(sleep_id))
        return tool_error(f"Unknown whoop_sleep action: {action}")
    except Exception as exc:
        return _whoop_tool_error(exc)


def _handle_whoop_workouts(args: dict, **kw) -> str:
    action = str(args.get("action") or "list").strip().lower()
    try:
        client = _whoop_client()
        if action == "list":
            return tool_result(client.list_workouts(**_list_kwargs(args)))
        if action == "latest":
            return tool_result(_first_record(
                client.list_workouts(**_list_kwargs(args)),
                empty_message="No WHOOP workouts found for the requested window.",
            ))
        if action == "get":
            workout_id = _require_id(args)
            if not workout_id:
                return tool_error("id is required for action='get'")
            return tool_result(client.get_workout(workout_id))
        return tool_error(f"Unknown whoop_workouts action: {action}")
    except Exception as exc:
        return _whoop_tool_error(exc)


COMMON_STRING = {"type": "string"}
_COMMON_RANGE_PROPS = {
    "start": {"type": "string", "description": "ISO-8601 start timestamp"},
    "end": {"type": "string", "description": "ISO-8601 end timestamp"},
    "limit": {"type": "integer", "minimum": 1, "maximum": 25},
    "next_token": COMMON_STRING,
    "max_pages": {"type": "integer", "minimum": 1, "maximum": 10},
}

WHOOP_PROFILE_SCHEMA = {
    "name": "whoop_profile",
    "description": "Fetch the authenticated user's basic WHOOP profile. Read-only; not medical advice.",
    "parameters": {"type": "object", "properties": {}},
}

WHOOP_CYCLES_SCHEMA = {
    "name": "whoop_cycles",
    "description": "Fetch WHOOP physiological cycles. Use latest for the most recent cycle or list for a paginated range.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["latest", "list", "get"]},
            "id": COMMON_STRING,
            **_COMMON_RANGE_PROPS,
        },
    },
}

WHOOP_RECOVERY_SCHEMA = {
    "name": "whoop_recovery",
    "description": "Fetch WHOOP recovery records. Returns raw metrics only; no medical advice.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["latest", "list", "get"]},
            "cycle_id": {"type": "string", "description": "WHOOP cycle ID for action='get'"},
            "id": {"type": "string", "description": "Alias for cycle_id"},
            **_COMMON_RANGE_PROPS,
        },
    },
}

WHOOP_SLEEP_SCHEMA = {
    "name": "whoop_sleep",
    "description": "Fetch WHOOP sleep records. Returns raw metrics only; no medical advice.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["latest", "list", "get"]},
            "id": COMMON_STRING,
            **_COMMON_RANGE_PROPS,
        },
    },
}

WHOOP_WORKOUTS_SCHEMA = {
    "name": "whoop_workouts",
    "description": "Fetch WHOOP workout/activity records. Returns raw metrics only; no medical advice.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["latest", "list", "get"]},
            "id": COMMON_STRING,
            **_COMMON_RANGE_PROPS,
        },
    },
}
