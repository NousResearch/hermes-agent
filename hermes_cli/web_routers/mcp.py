"""MCP dashboard routes.

The OAuth flow registry (``_mcp_oauth_flows``) and the worker/helpers stay in
web_server — reached via the late-binding seam so tests that mutate
``web_server._mcp_oauth_flows`` or monkeypatch its helpers keep working.
"""

import asyncio
import hashlib
import re
import secrets
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from hermes_cli.web_deps import LateState, late
from hermes_cli.web_server_mcp import _mcp_oauth_flows, _mcp_server_summary, _normalize_mcp_server_create
from hermes_cli.web_models import MCPCatalogInstall, MCPEnabledToggle, MCPServerCreate, MCPServersReplace
from hermes_cli.web_routers._common import (
    _profile_cli_args, _profile_scope, _spawn_hermes_action, config_write_scope, http_failure,
    log as _log, scoped_to_thread,
)

router = APIRouter()

_config_profile_scope = late("_config_profile_scope", "hermes_cli.web_server_profiles")
_require_token = late("_require_token")
_run_dashboard_mcp_oauth = late("_run_dashboard_mcp_oauth", "hermes_cli.web_server_mcp")
load_config = late("load_config", "hermes_cli.config")
save_config = late("save_config", "hermes_cli.config")
save_env_value = late("save_env_value", "hermes_cli.config")

_mcp_oauth_flows_lock = threading.Lock()
_MCP_DASHBOARD_OAUTH_TTL = 15 * 60
_MAX_PENDING_MCP_OAUTH_FLOWS = 8


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
    public_url = resolve_public_url()
    if public_url:
        return f"{public_url}{suffix}"
    base = urlparse(str(request.base_url))
    prefix = prefix_from_request(request)
    return urlunparse(base._replace(path=f"{prefix}{suffix}", params="", query="", fragment=""))


def _mcp_install_action_name(name: str) -> str:
    """Unique per-entry mcp-install action name (+ registered log file), so a
    re-click or a second catalog install doesn't overwrite the first's tracked
    process/log while its git clone is still running."""
    from hermes_cli.web_server_gateway import _ACTION_LOG_FILES
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:48] or "server"
    digest = hashlib.sha1(name.encode()).hexdigest()[:8]
    action = f"mcp-install-{slug}-{digest}"
    _ACTION_LOG_FILES.setdefault(action, f"action-{action}.log")
    return action


@router.get("/api/mcp/servers")
async def list_mcp_servers(profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers

    servers = await scoped_to_thread(profile, _get_mcp_servers)
    return {"servers": [_mcp_server_summary(name, cfg) for name, cfg in sorted(servers.items())]}


@router.post("/api/mcp/servers")
async def add_mcp_server(body: MCPServerCreate, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers, _save_bearer_auth_token, _save_mcp_server

    try:
        name, server_config, bearer_token = _normalize_mcp_server_create(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _run():
        # _save_mcp_server does its own load→mutate→save; the duplicate-name
        # check sits under the same lock span so a concurrent add can't slip
        # between check and save.
        with config_write_scope(body.profile or profile):
            if name in _get_mcp_servers():
                raise HTTPException(status_code=409, detail=f"Server '{name}' already exists")
            if bearer_token is not None:
                server_config["headers"] = _save_bearer_auth_token(name, bearer_token)
            if not _save_mcp_server(name, server_config):
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
            return _replace_mcp_servers(body.servers)

    ok, issues = await asyncio.to_thread(_run)
    if not ok:
        raise HTTPException(status_code=400, detail="; ".join(issues))
    return {"ok": True}


@router.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _remove_mcp_server

    def _run():
        with config_write_scope(profile):
            return _remove_mcp_server(name)

    if not await asyncio.to_thread(_run):
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    return {"ok": True}


@router.post("/api/mcp/servers/{name}/test")
async def test_mcp_server(name: str, profile: Optional[str] = None):
    """Connect to the server, list its tools, disconnect."""
    from hermes_cli.mcp_config import _get_mcp_servers, _oauth_tokens_present, _probe_single_server

    servers = await scoped_to_thread(profile, _get_mcp_servers)
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    details: Dict[str, Any] = {}
    # An `auth: oauth` server that serves tools/list anonymously would probe OK
    # with no token — a false green. Require a token on disk, matching /auth.
    needs_oauth_token = servers[name].get("auth") == "oauth"

    def _probe_scoped():
        # Home-only scope (contextvar), NOT _profile_scope: a probe can block for
        # seconds (stdio `npx` cold start) and _profile_scope holds the
        # process-global skills lock for its whole body, serializing every other
        # endpoint. The probe only needs HERMES_HOME for .env + token resolution.
        with _config_profile_scope(profile):
            tools = _probe_single_server(name, servers[name], details=details)
            return tools, (_oauth_tokens_present(name) if needs_oauth_token else True)

    try:  # probe blocks on a dedicated MCP event loop — keep it off the FastAPI loop
        tools, token_present = await asyncio.to_thread(_probe_scoped)
    except Exception as exc:
        return {"ok": False, "error": str(exc), "tools": []}
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
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    _require_token(request)
    _gc_mcp_oauth_flows()

    def _home() -> str:
        return str(get_hermes_home().expanduser().resolve(strict=False))

    process_home = _home()

    def _read():
        with _profile_scope(profile):
            return _get_mcp_servers(), _home()

    servers, flow_home = await asyncio.to_thread(_read)
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
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
        flow.mark_error(str(exc))
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
    flow.mark_error("Cancelled by user")
    return {"ok": True, "status": flow.snapshot()["status"]}


@router.get("/api/mcp/oauth/callback/{server_name:path}")
async def mcp_oauth_callback(
    server_name: str,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
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
        flow.deliver_callback(code=code, state=state, error=error)
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
            servers = cfg.get("mcp_servers")
            if not isinstance(servers, dict) or name not in servers:
                raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
            if not isinstance(servers[name], dict):
                raise HTTPException(status_code=400, detail="Malformed server config")
            servers[name]["enabled"] = bool(body.enabled)
            save_config(cfg)
        return {"ok": True, "name": name, "enabled": bool(body.enabled)}

    return await asyncio.to_thread(_run)


def _catalog_entry_json(entry: Any, installed: bool, enabled: bool) -> Dict[str, Any]:
    auth = entry.auth
    transport = entry.transport
    install = entry.install
    return {
        "name": entry.name,
        "description": entry.description,
        "source": entry.source,
        "homepage": entry.homepage,
        "transport": transport.type,
        "auth_type": getattr(auth, "type", "none"),
        # Env vars the user must supply (names + prompts only, never values).
        "required_env": [
            {"name": e.name, "prompt": e.prompt, "required": e.required}
            for e in getattr(auth, "env", []) or []
        ],
        # Transport details surfaced on purpose: the trust model asks users to
        # inspect command/args/url + bootstrap before installing.
        "command": transport.command,
        "args": list(transport.args or []),
        "url": transport.url,
        # Git bootstrap (present only for entries that clone + build).
        "install_url": install.url if install else None,
        "install_ref": install.ref if install else None,
        "bootstrap": list(install.bootstrap) if install else [],
        "default_enabled": list(entry.tools.default_enabled) if entry.tools.default_enabled is not None else None,
        "post_install": entry.post_install or "",
        # What the user has to go do themselves before this can work.
        "setup": [step.text for step in entry.setup],
        # Composer-suggestion triggers (desktop brand pills), only when the
        # manifest declares a `suggest` block.
        "suggest": {"keywords": list(entry.suggest.keywords), "hosts": list(entry.suggest.hosts)} if entry.suggest else None,
        "needs_install": install is not None,
        "installed": installed,
        "enabled": enabled,
    }


@router.get("/api/mcp/registry/search")
async def search_mcp_registry(q: str = "", limit: int = 12):
    """Search the official MCP Registry for hosted connectors — the tier below the reviewed catalog. Results are
    remote endpoints only (never a package launcher; hermes_cli.mcp_registry owns that filter), each labeled
    ``verified`` when the publisher's registry namespace owns the endpoint's domain, else ``community``.
    Never raises for a registry failure: degrading to the reviewed catalog is correct, a 500 would break the
    consent card and the composer for a purely optional lookup."""
    try:
        from hermes_cli import mcp_registry
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Registry unavailable: {exc}")
    entries = await asyncio.to_thread(mcp_registry.search, q, max(1, min(50, limit)))
    return {"entries": [entry.as_dict() for entry in entries]}


@router.get("/api/mcp/catalog")
async def list_mcp_catalog(profile: Optional[str] = None):
    """Browse the Nous-approved MCP catalog (optional-mcps/ manifests), each
    entry annotated with installed/enabled state for ``profile``."""
    with http_failure("mcp_catalog import failed", 500, "Catalog unavailable"):
        from hermes_cli import mcp_catalog

    entries = []
    try:
        def _read():
            with _profile_scope(profile):
                catalog = list(mcp_catalog.list_catalog())
                state = {e.name: (mcp_catalog.is_installed(e.name), mcp_catalog.is_enabled(e.name)) for e in catalog}
            return catalog, state

        catalog_entries, installed_state = await asyncio.to_thread(_read)
        for entry in catalog_entries:
            installed, enabled = installed_state.get(entry.name, (False, False))
            entries.append(_catalog_entry_json(entry, installed, enabled))
    except HTTPException:  # unknown/invalid profile → 404, not a silently-empty catalog
        raise
    except Exception:
        _log.exception("list_mcp_catalog failed")

    diagnostics = []
    try:
        diagnostics = [{"name": n, "kind": k, "message": m} for (n, k, m) in mcp_catalog.catalog_diagnostics()]
    except Exception:
        pass
    return {"entries": entries, "diagnostics": diagnostics}


@router.post("/api/mcp/catalog/install")
async def install_mcp_catalog_entry(body: MCPCatalogInstall, profile: Optional[str] = None):
    """Install a catalog MCP into config.yaml (declared env vars go to .env
    first; git-bootstrap entries run via the background CLI action path)."""
    from hermes_cli import mcp_catalog
    from hermes_cli.config import validate_env_var_name_for_write

    name = (body.name or "").strip()
    entry = mcp_catalog.get_entry(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No catalog entry '{name}'")

    # Catalog credentials are a closed schema: configuring one MCP must not
    # become a generic write primitive for unrelated process environment.
    declared_env = {spec.name for spec in (entry.auth.env or [])}
    undeclared_env = sorted(set(body.env) - declared_env)
    if undeclared_env:
        raise HTTPException(
            status_code=400,
            detail=f"Catalog entry '{name}' does not declare environment variable(s): {', '.join(undeclared_env)}",
        )
    # Validate the complete map before the first write so a mixed
    # valid+invalid request cannot partially persist credentials.
    try:
        for key in body.env:
            validate_env_var_name_for_write(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    effective_profile = body.profile or profile
    if body.env:
        def _write_env():
            with _profile_scope(effective_profile):
                for k, v in body.env.items():
                    if v:
                        save_env_value(k, v)

        await asyncio.to_thread(_write_env)

    # Git-bootstrap entries can take a while to clone — background action so
    # the request returns immediately (per-entry action name, see helper).
    if entry.install is not None:
        action = _mcp_install_action_name(name)
        try:
            _spawn_hermes_action(_profile_cli_args(effective_profile) + ["mcp", "install", name], action)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Install failed: {exc}")
        return {"ok": True, "name": name, "background": True, "action": action}

    # No git step — install synchronously; install_entry goes through the
    # call-time config/env resolvers so the profile scope covers it.
    try:
        await scoped_to_thread(effective_profile, lambda: mcp_catalog.install_entry(entry, enable=body.enable))
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("install_mcp_catalog_entry failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "name": name, "background": False}


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
import logging  # noqa: F401,E402
# ---- END PLUGIN-COMPAT ----
