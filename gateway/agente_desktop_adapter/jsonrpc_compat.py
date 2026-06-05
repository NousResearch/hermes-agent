"""
jsonrpc_compat.py

JSON-RPC alias handling + BYOK seam for POST /rpc.

Extracted/moved desktop-specific param aliases (sessionKey, gateway_session_key,
request_id -> session_id etc) and oauth subscription polling compat that
were previously inline in api_server._handle_rpc and _dispatch_tui_rpc.

The /rpc route registration itself is performed here (moved from the ad-hoc
block in api_server.py) so that upstream-only builds simply lack the route.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _apply_jsonrpc_desktop_aliases(body: Dict[str, Any]) -> Dict[str, Any]:
    """Apply BYOK/desktop param name aliases (extracted from api_server)."""
    body = dict(body or {})
    params = body.get("params")
    if not isinstance(params, dict):
        params = {}
    method = body.get("method")

    # desktop / BYOK alias: request_id for session in oauth code/poll
    if (
        method in {"auth.submit_oauth_code", "auth.poll_subscription_oauth"}
        and not params.get("session_id")
        and params.get("request_id")
    ):
        params = {**params, "session_id": params.get("request_id")}
        body["params"] = params

    # common session key aliases used by desktop UI for approval + auth
    if isinstance(body.get("params"), dict):
        p = body["params"]
        if not p.get("session_key"):
            p["session_key"] = (
                p.get("sessionKey")
                or p.get("gateway_session_key")
                or p.get("session_id")
            )
    return body


def register(app: Any, adapter: Any = None) -> None:
    """Wire JSON-RPC desktop/BYOK aliases.

    If adapter present, we also ensure /rpc route is registered using the
    (generic) _handle_rpc on the adapter (desktop-specific aliasing is
    applied inside _dispatch_tui_rpc which the core handler calls).
    """
    if adapter is None:
        adapter = getattr(app, "get", lambda k: None)("api_server_adapter") if hasattr(app, "get") else None
        if adapter is None and hasattr(app, "__getitem__"):
            try:
                adapter = app["api_server_adapter"]
            except Exception:
                adapter = None

    if adapter is None:
        return

    # Patch the dispatch helper to always apply aliases (defensive; core already has some).
    try:
        orig_dispatch = getattr(adapter, "_dispatch_tui_rpc", None)
        if orig_dispatch and not getattr(orig_dispatch, "_agente_alias_wrapped", False):
            def _wrapped(body: Dict[str, Any]) -> Dict[str, Any]:
                body = _apply_jsonrpc_desktop_aliases(body)
                return orig_dispatch(body)
            _wrapped._agente_alias_wrapped = True  # type: ignore[attr-defined]
            adapter._dispatch_tui_rpc = _wrapped  # type: ignore[attr-defined]
            logger.debug("agente_desktop_adapter.jsonrpc_compat: aliases patched into _dispatch_tui_rpc")
    except Exception as exc:
        logger.debug("jsonrpc_compat alias patch skipped: %s", exc)

    # Add the /rpc route here (was ad-hoc desktop block in api_server).
    try:
        from aiohttp import web
        # Only add if the core didn't (we removed the add in api_server cleanup).
        paths = {r.path for r in app.router.routes()}
        if "/rpc" not in paths:
            async def rpc_handler(req: "web.Request") -> "web.Response":
                # delegate to the (now generic) handler on adapter
                return await adapter._handle_rpc(req)
            app.router.add_post("/rpc", rpc_handler)
            logger.debug("agente_desktop_adapter.jsonrpc_compat: /rpc route added via adapter")
    except Exception as exc:
        logger.debug("jsonrpc_compat /rpc route add skipped: %s", exc)
