"""Webhook dispatch support for the jcode bridge plugin."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from gateway.config import Platform
from gateway.platforms.base import SendResult
from plugins.jcode_bridge.tools import handle_jcode_contract_check, handle_jcode_run

logger = logging.getLogger(__name__)


_JCODE_ARG_KEYS = {
    "cwd",
    "session",
    "provider",
    "model",
    "provider_profile",
    "output_mode",
    "execution_mode",
    "ensure_server",
    "timeout_seconds",
    "socket",
    "debug_socket",
    "jcode_bin",
    "confirm_outbound_human_contact",
    "confirm_sensitive_person_data",
    "safety_override_reason",
}

_JCODE_PREFLIGHT_KEYS = {
    "jcode_bin",
    "cwd",
    "provider",
    "model",
    "provider_profile",
    "timeout_seconds",
}


def _route_name_from_chat_id(chat_id: str | None) -> str | None:
    if not chat_id or not chat_id.startswith("webhook:"):
        return None
    parts = chat_id.split(":", 2)
    if len(parts) < 3:
        return None
    return parts[1] or None


def _get_route_config(event: Any, gateway: Any) -> tuple[str | None, dict[str, Any] | None, Any]:
    source = getattr(event, "source", None)
    if getattr(source, "platform", None) != Platform.WEBHOOK:
        return None, None, None

    route_name = _route_name_from_chat_id(getattr(source, "chat_id", None))
    if not route_name:
        return None, None, None

    webhook_adapter = getattr(gateway, "adapters", {}).get(Platform.WEBHOOK)
    routes = getattr(webhook_adapter, "_routes", {}) if webhook_adapter else {}
    route_config = routes.get(route_name)
    if not isinstance(route_config, dict):
        return route_name, None, webhook_adapter
    return route_name, route_config, webhook_adapter


def _dispatches_to_jcode(route_config: dict[str, Any]) -> bool:
    dispatch = route_config.get("dispatch")
    if isinstance(dispatch, str):
        return dispatch.strip().lower() == "jcode"
    if isinstance(dispatch, dict):
        target = dispatch.get("target") or dispatch.get("type")
        return isinstance(target, str) and target.strip().lower() == "jcode"
    return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _route_jcode_config(route_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    jcode_config = route_config.get("jcode", {})
    dispatch = route_config.get("dispatch")
    return (
        jcode_config if isinstance(jcode_config, dict) else {},
        dispatch if isinstance(dispatch, dict) else {},
    )


def _route_value(route_config: dict[str, Any], key: str, default: Any = None) -> Any:
    jcode_config, dispatch = _route_jcode_config(route_config)
    if key in dispatch:
        return dispatch[key]
    if key in jcode_config:
        return jcode_config[key]
    return default


def _jcode_args_from_route(route_config: dict[str, Any], message: str) -> dict[str, Any]:
    jcode_config, dispatch = _route_jcode_config(route_config)
    args: dict[str, Any] = {}
    args.update({
        key: value
        for key, value in jcode_config.items()
        if key in _JCODE_ARG_KEYS
    })
    # Support the compact form:
    #   dispatch:
    #     target: jcode
    #     cwd: /repo
    args.update({
        key: value
        for key, value in dispatch.items()
        if key in _JCODE_ARG_KEYS
    })
    args.setdefault("output_mode", "json")
    args["message"] = message
    return args


def _contract_preflight_args_from_route(route_config: dict[str, Any]) -> dict[str, Any] | None:
    if not _truthy(_route_value(route_config, "preflight_contract")):
        return None

    jcode_config, dispatch = _route_jcode_config(route_config)
    args: dict[str, Any] = {}
    args.update({
        key: value
        for key, value in jcode_config.items()
        if key in _JCODE_PREFLIGHT_KEYS
    })
    args.update({
        key: value
        for key, value in dispatch.items()
        if key in _JCODE_PREFLIGHT_KEYS
    })
    args["live"] = _truthy(_route_value(route_config, "preflight_live"))
    args["live_run"] = _truthy(_route_value(route_config, "preflight_live_run"))

    live_run_message = _route_value(route_config, "preflight_live_run_message")
    if isinstance(live_run_message, str) and live_run_message.strip():
        args["live_run_message"] = live_run_message.strip()

    return args


def _content_from_jcode_payload(payload: dict[str, Any]) -> str:
    if not payload.get("success"):
        error = payload.get("error") or "unknown error"
        stderr = str(payload.get("stderr") or "").strip()
        if stderr:
            return f"jcode dispatch failed: {error}\n\n{stderr[:4000]}"
        return f"jcode dispatch failed: {error}"

    parsed = payload.get("parsed")
    if isinstance(parsed, dict):
        for key in ("text", "final_response", "response", "content"):
            value = parsed.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(parsed, indent=2, ensure_ascii=True)

    if isinstance(parsed, list):
        for event in reversed(parsed):
            if not isinstance(event, dict):
                continue
            for key in ("text", "final_response", "response", "content"):
                value = event.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return json.dumps(parsed[-10:], indent=2, ensure_ascii=True)

    stdout = str(payload.get("stdout") or "").strip()
    if stdout:
        return stdout
    return json.dumps(payload, indent=2, ensure_ascii=True)


async def _run_jcode_and_send(event: Any, webhook_adapter: Any, route_config: dict[str, Any]) -> None:
    args = _jcode_args_from_route(route_config, getattr(event, "text", ""))
    chat_id = getattr(getattr(event, "source", None), "chat_id", "")
    try:
        payload: dict[str, Any] | None = None
        preflight_args = _contract_preflight_args_from_route(route_config)
        if preflight_args is not None:
            raw_preflight = await asyncio.to_thread(handle_jcode_contract_check, preflight_args)
            preflight = json.loads(raw_preflight)
            if not preflight.get("success"):
                payload = {
                    "success": False,
                    "error": "jcode bridge contract preflight failed",
                    "preflight": preflight,
                }

        if payload is None:
            raw_result = await asyncio.to_thread(handle_jcode_run, args)
            payload = json.loads(raw_result)
    except Exception as exc:
        logger.exception("[jcode_bridge] webhook dispatch failed before delivery")
        payload = {
            "success": False,
            "error": f"jcode bridge dispatch error: {exc}",
        }

    content = _content_from_jcode_payload(payload)
    try:
        result = await webhook_adapter.send(chat_id, content)
    except Exception:
        logger.exception("[jcode_bridge] failed to deliver jcode webhook result")
        return

    if isinstance(result, SendResult) and not result.success:
        logger.warning(
            "[jcode_bridge] webhook result delivery failed chat_id=%s error=%s",
            chat_id,
            result.error,
        )


def on_pre_gateway_dispatch(event: Any = None, gateway: Any = None, **_: Any) -> dict[str, str] | None:
    """Intercept webhook routes configured with `dispatch: jcode`.

    The webhook adapter has already authenticated, rate-limited, rendered the
    prompt, and stored delivery metadata by the time this hook runs. Returning
    `skip` prevents the normal Hermes agent run; the background task sends
    jcode's result through the webhook adapter's existing delivery mechanism.
    """
    if event is None or gateway is None:
        return None

    route_name, route_config, webhook_adapter = _get_route_config(event, gateway)
    if not route_config or webhook_adapter is None:
        return None
    if not _dispatches_to_jcode(route_config):
        return None

    try:
        asyncio.get_running_loop().create_task(
            _run_jcode_and_send(event, webhook_adapter, route_config)
        )
    except RuntimeError:
        logger.warning(
            "[jcode_bridge] no running event loop for webhook dispatch route=%s",
            route_name,
        )
        return {"action": "allow"}

    return {
        "action": "skip",
        "reason": f"webhook route '{route_name}' dispatched to jcode",
    }
