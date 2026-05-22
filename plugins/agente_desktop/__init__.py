"""agente_desktop plugin — bridge from the Hermes Gateway to the Agente
desktop Electron main process.

Registers 8 tools under the ``agente-desktop`` toolset. Each tool's handler
proxies the call to an HTTP server inside the Electron main, which dispatches
the request to the matching TypeScript tool in ``electron/main/hermes-tools/``.

Activation contract: the Electron sidecar manager spawns Hermes with the env
vars ``AGENTE_TOOL_PORT`` (ephemeral loopback port) and ``AGENTE_TOOL_SECRET``
(32-byte hex bearer token). When both are present the plugin is "available";
when either is missing the tools are still registered but the check_fn
returns False, so the AIAgent simply omits them from the LLM tool schema.

IPC contract:

    POST http://127.0.0.1:${AGENTE_TOOL_PORT}/dispatch/${tool_name}
    Authorization: Bearer ${AGENTE_TOOL_SECRET}
    Content-Type: application/json
    body: {...tool_args}

    200 → {"ok": true, "result": <tool-specific-json>}
    200 → {"ok": false, "error": "<string>", "details": <optional>}
    401 → {"ok": false, "error": "unauthorized"}

The Python wrapper unwraps `result` (or surfaces `error`) so the model sees
the tool's native return shape.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from plugins.agente_desktop.schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)

TOOLSET = "agente-desktop"
_REQUEST_TIMEOUT_SECONDS = 25

_TOOL_EMOJIS: dict[str, str] = {
    "list_whatsapp_accounts": "📱",
    "list_recent_messages": "💬",
    "create_ticket": "🎫",
    "move_ticket": "🔄",
    "list_tickets": "📋",
    "save_triage_instructions": "📝",
    "request_approval": "✋",
    "get_office_context": "🏢",
    # desktop-202605-161 additions
    "upsert_client": "👤",
    "query_client": "🔍",
    "link_ticket_to_client": "🔗",
    "link_document_to_client": "📎",
    "update_office_context": "🏢",
    "list_tools": "🛠️",
    "list_workflows": "📂",
    "inspect_workflow": "🔬",
    "start_workflow_run": "▶️",
    "get_run_status": "📊",
    "resume_paused_run": "⏯️",
}


def _ipc_endpoint(tool_name: str) -> str | None:
    port = os.environ.get("AGENTE_TOOL_PORT")
    if not port:
        return None
    return f"http://127.0.0.1:{port}/dispatch/{tool_name}"


def _proxy_call(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Forward a tool call to the Electron main process over loopback HTTP.

    Returns the unwrapped ``result`` payload on success or an ``{"error":
    ...}`` dict on any failure. Never raises — every path returns a dict so
    the AIAgent can fold the response into the next conversation turn.
    """
    endpoint = _ipc_endpoint(tool_name)
    secret = os.environ.get("AGENTE_TOOL_SECRET")
    if endpoint is None or not secret:
        return {"error": "agente_tool_ipc_not_configured"}

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(args).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {secret}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return {
            "error": "agente_tool_http_error",
            "status": exc.code,
            "body": exc.read().decode("utf-8", "replace"),
        }
    except Exception as exc:  # noqa: BLE001 — defensive; never raise from a tool handler
        return {"error": "agente_tool_exception", "message": str(exc)}

    try:
        envelope = json.loads(body)
    except json.JSONDecodeError:
        return {"error": "agente_tool_invalid_json", "raw": body[:512]}

    if not isinstance(envelope, dict):
        return {"error": "agente_tool_invalid_envelope", "raw": body[:512]}

    if envelope.get("ok") is True:
        result = envelope.get("result")
        return result if isinstance(result, dict) else {"result": result}

    return {
        "error": envelope.get("error", "agente_tool_unknown_error"),
        "details": envelope.get("details"),
    }


def _check_available() -> bool:
    return bool(os.environ.get("AGENTE_TOOL_PORT") and os.environ.get("AGENTE_TOOL_SECRET"))


def _make_handler(tool_name: str):
    """Bind the tool name into a closure so each registered handler has its
    own dispatch target. Avoids the late-binding-loop pitfall.
    """

    def handler(args: dict[str, Any]) -> dict[str, Any]:
        return _proxy_call(tool_name, args)

    return handler


def register(ctx) -> None:
    """Register the 18 Agente desktop tools.

    Called once by the plugin loader when the plugin is enabled via
    ``plugins.enabled`` in config.yaml AND the ``agente-desktop`` toolset is
    listed under ``toolsets:``. The toolset is auto-activated at runtime when
    AGENTE_TOOL_PORT + AGENTE_TOOL_SECRET env vars are present (see
    gateway/platforms/api_server.py ``_create_agent``).
    """
    for tool_name, schema in TOOL_SCHEMAS.items():
        ctx.register_tool(
            name=tool_name,
            toolset=TOOLSET,
            schema=schema,
            handler=_make_handler(tool_name),
            check_fn=_check_available,
            description=schema.get("description", ""),
            emoji=_TOOL_EMOJIS.get(tool_name, "🔌"),
        )
    logger.info(
        "agente_desktop plugin registered %d tools under toolset %r",
        len(TOOL_SCHEMAS),
        TOOLSET,
    )
