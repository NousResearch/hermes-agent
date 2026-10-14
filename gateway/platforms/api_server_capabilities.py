"""Capability projection for the authenticated native API."""
from aiohttp import web

from gateway.platforms import api_server_workspace as _workspace


def response(adapter):
    from gateway.platforms import api_server as api

    return web.json_response({
        "object": "hermes.api_server.capabilities", "platform": "hermes-agent",
        "model": adapter._model_name,
        "auth": {"type": "bearer", "required": bool(adapter._api_key)},
        "runtime": {
            "mode": "server_agent", "tool_execution": "server", "split_runtime": False,
            "description": (
                "The API server creates a server-side Hermes AIAgent; "
                "tools execute on the API-server host unless a future "
                "explicit split-runtime mode is enabled.")},
        "features": {
            "chat_completions": True, "chat_completions_streaming": True,
            "responses_api": True, "responses_streaming": True, "run_submission": True,
            "runs_idempotency": api._api_runs._idempotency_capabilities(adapter, store_type=api.RunIdempotencyStore),
            **api._STATIC_FEATURE_FLAGS,
            "workspace_io": _workspace.capability(adapter),
            "cors": bool(adapter._cors_origins),
            # Always advertised for feature-detection; enabled follows config.
            "browser_extension_control": {
                "enabled": adapter._browser_control_enabled(),
                "protocol_version": api._BROWSER_CONTROL_PROTOCOL_VERSION,
                "capabilities": sorted(api.BROWSER_CONTROL_CAPABILITIES),
                "artifact_capabilities": sorted(api.BROWSER_CONTROL_ARTIFACT_CAPABILITIES),
                "developer_capabilities": sorted(api.BROWSER_CONTROL_DEVELOPER_CAPABILITIES),
                "developer_mode": adapter._browser_control_developer_mode(),
                "artifact_transport": {
                    "upload": {"method": "POST", "path": "/v1/artifacts/upload"},
                    "download": {
                        "method": "GET", "path": "/v1/artifacts/download/{artifact_id}"},
                    "max_bytes": api.DEFAULT_MAX_ARTIFACT_BYTES,
                    "ttl_seconds": api.DEFAULT_ARTIFACT_TTL_SECONDS,
                    "allowed_mime_types": sorted(api.DEFAULT_ALLOWED_MIME_TYPES)},
                "real_browser_actions": True,
                "transports": {
                    "local_vps": "websocket-subprotocol-ticket",
                    "cloud": "authenticated-gateway-rpc"}}},
        "endpoints": {name: {"method": m, "path": p} for name, (m, p) in api._CAPABILITY_ENDPOINTS},
    })

