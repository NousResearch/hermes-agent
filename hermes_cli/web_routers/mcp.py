"""MCP dashboard routes.

The OAuth flow registry (``_mcp_oauth_flows``) and the worker/helpers stay in
web_server — reached via the late-binding seam so tests that mutate
``web_server._mcp_oauth_flows`` or monkeypatch its helpers keep working.
"""

import asyncio
from contextlib import contextmanager
import secrets
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from hermes_cli.web_deps import late
from hermes_cli.web_server_mcp import _mcp_oauth_flows, _mcp_server_summary, _normalize_mcp_server_create
from hermes_cli.web_models import MCPEnabledToggle, MCPServerCreate, MCPServersReplace
from hermes_cli.web_routers._common import _profile_scope, config_write_scope, log as _log

router = APIRouter()

_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_require_token = late("_require_token")
_run_dashboard_mcp_oauth = late("_run_dashboard_mcp_oauth", "hermes_cli.web_server_mcp")
load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")

_mcp_oauth_flows_lock = threading.Lock()
_MCP_DASHBOARD_OAUTH_TTL = 15 * 60
_MAX_PENDING_MCP_OAUTH_FLOWS = 8


@contextmanager
def _profile_secret_scope(profile: Optional[str]):
    """Home + secret scope for a probe-class request (#109901). ``_config_profile_scope`` now binds
    the secret scope itself; this stays the probe/OAuth callers' name. Home-only, NOT
    ``_profile_scope``: the body can block for seconds and the latter holds the process-global
    skills lock."""
    with _config_profile_scope(profile):
        yield


def _secret_scoped(profile: Optional[str], fn):
    def _run():
        with _profile_secret_scope(profile):
            return fn()
    return _run


def _gc_mcp_oauth_flows() -> None:
    cutoff = time.time() - _MCP_DASHBOARD_OAUTH_TTL
    with _mcp_oauth_flows_lock:
        stale = [fid for fid, flow in _mcp_oauth_flows.items() if getattr(flow, "created_at", 0) < cutoff]
        for flow_id in stale:
            _mcp_oauth_flows.pop(flow_id, None)


def _mcp_oauth_callback_url(request: Request, server_name: str) -> str:
    """Externally reachable callback URL for a dashboard flow."""
    from urllib.parse import quote, urlparse, urlunparse

    from hermes_cli.dashboard_auth.prefix import prefix_from_request, resolve_public_url

    suffix = f"/api/mcp/oauth/callback/{quote(server_name, safe='')}"
    public_url = resolve_public_url(request.app.state)
    if public_url:
        return f"{public_url}{suffix}"
    base = urlparse(str(request.base_url))
    prefix = prefix_from_request(request)
    return urlunparse(base._replace(path=f"{prefix}{suffix}", params="", query="", fragment=""))


@router.get("/api/mcp/servers")
async def list_mcp_servers(profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers

    def _read():
        config_servers = _get_mcp_servers()
        from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

        return server_configs_with_sources(config_servers)

    servers, plugins = await asyncio.to_thread(_secret_scoped(profile, _read))
    return {"servers": [
        _mcp_server_summary(name, cfg, plugins[name]) for name, cfg in sorted(servers.items())
    ]}


@router.post("/api/mcp/servers")
async def add_mcp_server(body: MCPServerCreate, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers, _save_bearer_auth_token, _save_mcp_server, record_mcp_install

    try:
        name, server_config, bearer_token = _normalize_mcp_server_create(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _save() -> bool:
        # _save_mcp_server does its own load→mutate→save; the duplicate-name
        # check sits under the same lock span so a concurrent add can't slip
        # between check and save.
        with config_write_scope(body.profile or profile):
            config_servers = _get_mcp_servers()
            from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

            servers, plugins = server_configs_with_sources(config_servers)
            if plugin := plugins.get(name):
                raise HTTPException(
                    status_code=409, detail=f"Server '{name}' is provided by plugin '{plugin}' and cannot be modified",
                )
            if name in servers:
                raise HTTPException(status_code=409, detail=f"Server '{name}' already exists")
            if bearer_token is not None:
                server_config["headers"] = _save_bearer_auth_token(name, bearer_token)
            return _save_mcp_server(name, server_config)

    def _run():
        saved = _save()
        # Outside the config mutation lock: a cold first metric call costs imports.
        with _profile_scope(body.profile or profile):
            record_mcp_install("url" if server_config.get("url") else "local", None,
                               "success" if saved else "failed", failure_class=None if saved else "config_rejected")
        if not saved:
            raise HTTPException(
                status_code=400, detail=f"Server '{name}' rejected: suspicious command/args configuration",
            )

    try:
        await asyncio.to_thread(_run)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("POST /api/mcp/servers failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _mcp_server_summary(name, server_config)


@router.put("/api/mcp/servers")
async def replace_mcp_servers(body: MCPServersReplace, profile: Optional[str] = None):
    """Replace the entire ``mcp_servers`` map (the mcp.json editor's save) —
    the deep-merging ``/api/config`` can never delete a key or drop an
    ``enabled: false``, so removals wouldn't persist through it."""
    from hermes_cli.mcp_config import _replace_mcp_servers

    def _run():
        with config_write_scope(body.profile or profile):
            from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

            _servers, plugins = server_configs_with_sources({})
            for name in body.servers:
                if plugin := plugins.get(name):
                    return False, [f"Server '{name}' is provided by plugin '{plugin}' and cannot be modified"]
            return _replace_mcp_servers(body.servers)

    ok, issues = await asyncio.to_thread(_run)
    if not ok:
        raise HTTPException(status_code=400, detail="; ".join(issues))
    return {"ok": True}


@router.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers, _remove_mcp_server

    def _run():
        with config_write_scope(profile):
            config_servers = _get_mcp_servers()
            from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

            _servers, plugins = server_configs_with_sources(config_servers)
            if plugin := plugins.get(name):
                raise HTTPException(
                    status_code=409, detail=f"Server '{name}' is provided by plugin '{plugin}' and cannot be modified",
                )
            return _remove_mcp_server(name)

    if not await asyncio.to_thread(_run):
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    return {"ok": True}


@router.post("/api/mcp/servers/{name}/test")
async def test_mcp_server(name: str, profile: Optional[str] = None):
    """Connect to the server, list its tools, disconnect."""
    from hermes_cli.mcp_config import _get_mcp_servers, _oauth_tokens_present, _probe_single_server

    def _read():
        config_servers = _get_mcp_servers()
        from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

        return server_configs_with_sources(config_servers)[0]

    servers = await asyncio.to_thread(_secret_scoped(profile, _read))
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    details: dict[str, Any] = {}
    # An `auth: oauth` server that serves tools/list anonymously would probe OK
    # with no token — a false green. Require a token on disk, matching /auth.
    needs_oauth_token = servers[name].get("auth") == "oauth"

    def _probe():
        tools = _probe_single_server(name, servers[name], details=details)
        return tools, (_oauth_tokens_present(name) if needs_oauth_token else True)

    try:  # probe blocks on a dedicated MCP event loop — keep it off the FastAPI loop
        tools, token_present = await asyncio.to_thread(_secret_scoped(profile, _probe))
    except Exception as exc:
        from hermes_cli.mcp_config import redact_mcp_probe_text

        return {"ok": False, "error": redact_mcp_probe_text(exc), "tools": []}
    if not token_present:
        return {"ok": False, "error": "OAuth authentication required — no token found.", "tools": []}
    # Optional per-tool schema size (chars) for the desktop's cost overlay;
    # failed probes simply omit it.
    schema_chars = details.get("schema_chars") or {}
    return {
        "ok": True,
        "tools": [
            {
                "name": t,
                "description": d,
                **({"schema_chars": schema_chars[t]} if isinstance(schema_chars.get(t), int) else {}),
            }
            for t, d in tools
        ],
        "prompts": details.get("prompts", 0),
        "resources": details.get("resources", 0),
    }


@router.post("/api/mcp/servers/{name}/auth")
async def auth_mcp_server(name: str, request: Request, profile: Optional[str] = None):
    """Start MCP OAuth and hand the authorization URL to the dashboard browser."""
    from hermes_cli.mcp_config import _get_mcp_servers
    from hermes_constants import get_hermes_home
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow, exception_message

    _require_token(request)
    _gc_mcp_oauth_flows()

    def _home() -> str:
        return str(get_hermes_home().expanduser().resolve(strict=False))

    process_home = _home()

    def _read():
        with _profile_secret_scope(profile):
            config_servers = _get_mcp_servers()
            from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

            servers, plugins = server_configs_with_sources(config_servers)
            return servers, plugins, _home()

    servers, plugins, flow_home = await asyncio.to_thread(_read)
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    if plugin := plugins.get(name):
        raise HTTPException(status_code=409, detail=f"Server '{name}' is provided by plugin '{plugin}' and cannot be modified")
    cfg = dict(servers[name])
    if not cfg.get("url"):
        raise HTTPException(status_code=400, detail="stdio servers authenticate via env keys, not OAuth")
    if cfg.get("headers") and cfg.get("auth") != "oauth":
        raise HTTPException(status_code=400, detail="This server uses header/API-key auth, not OAuth")
    cfg["auth"] = "oauth"

    flow_id = secrets.token_urlsafe(24)
    flow = DashboardOAuthFlow(
        flow_id=flow_id,
        server_name=name,
        profile=profile,
        hermes_home=flow_home,
        redirect_uri=(cfg.get("oauth") or {}).get("redirect_uri") or _mcp_oauth_callback_url(request, name),
        reconnect_live=flow_home == process_home,
    )
    with _mcp_oauth_flows_lock:
        live = [f for f in _mcp_oauth_flows.values() if not f.worker_done]
        if len(live) >= _MAX_PENDING_MCP_OAUTH_FLOWS:
            raise HTTPException(status_code=429, detail="Too many MCP OAuth flows are already in progress")
        if any(f.server_name == name and f.hermes_home == flow_home for f in live):
            raise HTTPException(status_code=409, detail=f"MCP OAuth for '{name}' is already in progress")
        _mcp_oauth_flows[flow_id] = flow
    threading.Thread(target=_run_dashboard_mcp_oauth, args=(flow, cfg), daemon=True, name=f"mcp-oauth-{name}").start()
    try:
        await flow.wait_for_authorization_url(timeout=30)
    except Exception as exc:
        flow.mark_error(exception_message(exc))
    return flow.snapshot()


@router.get("/api/mcp/oauth/flows/{flow_id}")
async def mcp_oauth_flow_status(flow_id: str, request: Request):
    _require_token(request)
    _gc_mcp_oauth_flows()
    flow = _mcp_oauth_flows.get(flow_id)
    if flow is None:
        raise HTTPException(status_code=404, detail="OAuth flow not found or expired")
    snapshot = flow.snapshot()
    snapshot["tools"] = flow.tools
    return snapshot


@router.delete("/api/mcp/oauth/flows/{flow_id}")
async def cancel_mcp_oauth_flow(flow_id: str, request: Request):
    """Cancel an in-flight flow. mark_error unblocks the worker so it frees the
    per-server "already in progress" slot — otherwise a renderer that stops
    polling leaves the flow squatting until the 300s callback timeout and every
    retry 409s. Idempotent: a settled flow is left as-is."""
    _require_token(request)
    flow = _mcp_oauth_flows.get(flow_id)
    if flow is None:  # expired/GC'd is the goal state of a cancel — not an error
        return {"ok": True, "status": "expired"}
    flow.mark_error("Cancelled by user", cancelled=True)
    return {"ok": True, "status": flow.snapshot()["status"]}


@router.get("/api/mcp/oauth/callback/{server_name:path}")
async def mcp_oauth_callback(
    server_name: str,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    iss: Optional[str] = None,
):
    _gc_mcp_oauth_flows()
    with _mcp_oauth_flows_lock:
        candidates = [
            flow for flow in _mcp_oauth_flows.values()
            if flow.server_name == server_name and flow.status == "authorization_required"
        ]
    flow = next(
        (c for c in candidates
         if c.expected_state is not None and state is not None and secrets.compare_digest(c.expected_state, state)),
        None,
    )
    if flow is None:
        return HTMLResponse("<h1>OAuth flow expired</h1><p>Return to Hermes and try again.</p>", status_code=404)
    try:
        flow.deliver_callback(code=code, state=state, error=error, iss=iss)
    except ValueError as exc:
        return HTMLResponse(
            "<h1>OAuth callback rejected</h1><p>The callback was invalid or already used.</p>",
            status_code=409 if "already received" in str(exc) else 400,
        )
    if error:
        return HTMLResponse("<h1>Authorization failed</h1><p>Return to Hermes for details.</p>", status_code=400)
    return HTMLResponse("<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>")


@router.put("/api/mcp/servers/{name}/enabled")
async def set_mcp_server_enabled(name: str, body: MCPEnabledToggle, profile: Optional[str] = None):
    """Toggle ``enabled`` (takes effect on next session/gateway); disabled
    servers stay in config so they can be re-enabled without re-entry."""
    def _run():
        with config_write_scope(body.profile or profile):
            cfg = load_config()
            from tui_gateway.mcp_rpc_helpers import server_configs_with_sources

            _servers, plugins = server_configs_with_sources(cfg.get("mcp_servers") or {})
            if plugin := plugins.get(name):
                raise HTTPException(
                    status_code=409, detail=f"Server '{name}' is provided by plugin '{plugin}' and cannot be modified",
                )
            servers = cfg.get("mcp_servers")
            if not isinstance(servers, dict) or name not in servers:
                raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
            if not isinstance(servers[name], dict):
                raise HTTPException(status_code=400, detail="Malformed server config")
            servers[name]["enabled"] = bool(body.enabled)
            save_config(cfg)
        return {"ok": True, "name": name, "enabled": bool(body.enabled)}

    return await asyncio.to_thread(_run)


@router.get("/api/mcp/catalog")
async def list_removed_catalog():
    """The in-tree MCP catalog is gone; released desktops and dashboards still poll this route."""
    return {"entries": [], "diagnostics": []}


@router.post("/api/mcp/catalog/install")
async def install_removed_catalog_entry():
    raise HTTPException(
        status_code=410,
        detail="The MCP catalog was removed. Add the server with "
        "`hermes mcp add <name> --url <url> [--auth oauth]` or POST /api/mcp/servers.",
    )
