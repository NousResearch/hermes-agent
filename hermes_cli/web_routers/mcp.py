"""MCP dashboard routes (extracted verbatim from web_server.py).

Handler bodies are byte-identical.  The OAuth flow registry
(``_mcp_oauth_flows`` + lock + pending cap) and the worker/helpers stay in
web_server - reached via the late-binding seam in :mod:`hermes_cli.web_deps`
(``late`` for callables, ``LateState`` for the mutable registry/lock/limit) so
tests that mutate ``web_server._mcp_oauth_flows`` or
``monkeypatch.setattr(web_server, "_run_dashboard_mcp_oauth", ...)`` keep
working unchanged.
"""

import asyncio  # noqa: F401 — used by handlers
import logging
import secrets  # noqa: F401
import threading  # noqa: F401
from typing import Any, Dict, Optional  # noqa: F401

from fastapi import APIRouter, HTTPException, Request  # noqa: F401
from fastapi.responses import HTMLResponse  # noqa: F401

from hermes_cli.web_deps import late, LateState
from hermes_cli.web_models import (
    MCPCatalogInstall,
    MCPEnabledToggle,
    MCPServerCreate,
    MCPServersReplace,
)

# Same logger the handlers used before extraction (identical logger object).
_log = logging.getLogger("hermes_cli.web_server")

router = APIRouter()

# Late-bound web_server helpers (resolved at call time; cycle-safe,
# monkeypatch-transparent).
_config_profile_scope = late("_config_profile_scope")
_gc_mcp_oauth_flows = late("_gc_mcp_oauth_flows")
_mcp_install_action_name = late("_mcp_install_action_name")
_mcp_oauth_callback_url = late("_mcp_oauth_callback_url")
_mcp_server_summary = late("_mcp_server_summary")
_normalize_mcp_server_create = late("_normalize_mcp_server_create")
_profile_cli_args = late("_profile_cli_args")
_profile_scope = late("_profile_scope")
_require_token = late("_require_token")
_run_dashboard_mcp_oauth = late("_run_dashboard_mcp_oauth")
_spawn_hermes_action = late("_spawn_hermes_action")
load_config = late("load_config")
save_config = late("save_config")
save_env_value = late("save_env_value")

# Live proxies for web_server-owned module state (mutations/monkeypatches
# on web_server remain authoritative; resolved at operation time).
_mcp_oauth_flows = LateState("_mcp_oauth_flows")
_mcp_oauth_flows_lock = LateState("_mcp_oauth_flows_lock")
_MAX_PENDING_MCP_OAUTH_FLOWS = LateState("_MAX_PENDING_MCP_OAUTH_FLOWS")


@router.get("/api/mcp/servers")
async def list_mcp_servers(profile: Optional[str] = None):
    from hermes_cli.mcp_config import _get_mcp_servers

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    return {
        "servers": [
            _mcp_server_summary(name, cfg) for name, cfg in sorted(servers.items())
        ]
    }


@router.post("/api/mcp/servers")
async def add_mcp_server(body: MCPServerCreate, profile: Optional[str] = None):
    from hermes_cli.mcp_config import (
        _get_mcp_servers,
        _save_bearer_auth_token,
        _save_mcp_server,
    )

    try:
        name, server_config, bearer_token = _normalize_mcp_server_create(body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _profile_scope(body.profile or profile):
        existing = _get_mcp_servers()
    if name in existing:
        raise HTTPException(status_code=409, detail=f"Server '{name}' already exists")

    try:
        with _profile_scope(body.profile or profile):
            if bearer_token is not None:
                server_config["headers"] = _save_bearer_auth_token(name, bearer_token)
            if not _save_mcp_server(name, server_config):
                raise HTTPException(
                    status_code=400,
                    detail=f"Server '{name}' rejected: suspicious command/args configuration",
                )
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("POST /api/mcp/servers failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return _mcp_server_summary(name, server_config)


@router.put("/api/mcp/servers")
async def replace_mcp_servers(body: MCPServersReplace, profile: Optional[str] = None):
    """Replace the entire ``mcp_servers`` map (the GUI mcp.json editor's save).

    The generic ``/api/config`` endpoint deep-merges maps, so it can never
    delete a server key, drop an ``enabled: false`` flag, or remove a nested
    field — edits looked saved but the stale entry survived on disk.  This
    endpoint sets the whole map so removals actually persist.  Storage stays
    the config.yaml ``mcp_servers`` key the CLI/TUI already read.
    """
    from hermes_cli.mcp_config import _replace_mcp_servers

    with _profile_scope(body.profile or profile):
        ok, issues = _replace_mcp_servers(body.servers)
    if not ok:
        raise HTTPException(status_code=400, detail="; ".join(issues))
    return {"ok": True}


@router.delete("/api/mcp/servers/{name}")
async def remove_mcp_server(name: str, profile: Optional[str] = None):
    from hermes_cli.mcp_config import _remove_mcp_server

    with _profile_scope(profile):
        removed = _remove_mcp_server(name)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
    return {"ok": True}


@router.post("/api/mcp/servers/{name}/test")
async def test_mcp_server(name: str, profile: Optional[str] = None):
    """Connect to the server, list its tools, disconnect.  Returns tool list."""
    from hermes_cli.mcp_config import (
        _get_mcp_servers,
        _oauth_tokens_present,
        _probe_single_server,
    )

    with _profile_scope(profile):
        servers = _get_mcp_servers()
    if name not in servers:
        raise HTTPException(status_code=404, detail=f"Server '{name}' not found")

    details: Dict[str, Any] = {}
    # An `auth: oauth` server that serves tools/list anonymously would probe OK
    # with no token — a false green. Require a token on disk for it, matching the
    # /auth verification (some providers don't enforce auth on tools/list).
    needs_oauth_token = servers[name].get("auth") == "oauth"

    def _probe_scoped():
        # Home-only scope (contextvar), NOT _profile_scope. A probe blocks for
        # as long as the server takes to spawn/connect — a stdio `npx` cold
        # start is many seconds — and _profile_scope holds a process-global
        # skills lock for its ENTIRE body. Holding that across the probe
        # serialized every other endpoint (config/skills/toolsets all take the
        # same lock), so a slow server made unrelated requests time out at 15s.
        # The probe touches no skills globals; it only needs the HERMES_HOME
        # override for .env interpolation + OAuth token resolution, which the
        # contextvar provides (copied into this to_thread worker; and
        # _run_on_mcp_loop re-wraps it onto the MCP event-loop thread).
        #
        # This endpoint is also called automatically when Desktop opens its MCP
        # page. A status probe must never start an interactive OAuth flow or
        # open the system browser; only the explicit /auth endpoint may do so.
        from tools.mcp_oauth import suppress_interactive_oauth

        with _config_profile_scope(profile), suppress_interactive_oauth():
            tools = _probe_single_server(name, servers[name], details=details)
            token_present = _oauth_tokens_present(name) if needs_oauth_token else True
            return tools, token_present

    try:
        # Probe blocks on a dedicated MCP event loop — run in a thread so the
        # FastAPI event loop is never blocked.
        tools, token_present = await asyncio.to_thread(_probe_scoped)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
            "tools": [],
        }
    if not token_present:
        return {
            "ok": False,
            "error": "OAuth authentication required — no token found.",
            "tools": [],
        }
    return {
        "ok": True,
        "tools": [{"name": t, "description": d} for t, d in tools],
        "prompts": details.get("prompts", 0),
        "resources": details.get("resources", 0),
    }


@router.post("/api/mcp/servers/{name}/auth")
async def auth_mcp_server(name: str, request: Request, profile: Optional[str] = None):
    """Start MCP OAuth and hand the authorization URL to the dashboard browser."""
    from hermes_cli.mcp_config import _get_mcp_servers
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow

    _require_token(request)
    _gc_mcp_oauth_flows()
    from hermes_constants import get_hermes_home

    process_home = str(get_hermes_home().expanduser().resolve(strict=False))
    with _profile_scope(profile):
        servers = _get_mcp_servers()
        flow_home = str(get_hermes_home().expanduser().resolve(strict=False))
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
        redirect_uri=(cfg.get("oauth") or {}).get("redirect_uri")
        or _mcp_oauth_callback_url(request, name),
        reconnect_live=flow_home == process_home,
    )
    with _mcp_oauth_flows_lock:
        pending = sum(
            not flow.worker_done
            for flow in _mcp_oauth_flows.values()
        )
        if pending >= _MAX_PENDING_MCP_OAUTH_FLOWS:
            raise HTTPException(
                status_code=429,
                detail="Too many MCP OAuth flows are already in progress",
            )
        if any(
            flow.server_name == name
            and flow.hermes_home == flow_home
            and not flow.worker_done
            for flow in _mcp_oauth_flows.values()
        ):
            raise HTTPException(
                status_code=409,
                detail=f"MCP OAuth for '{name}' is already in progress",
            )
        _mcp_oauth_flows[flow_id] = flow
    threading.Thread(
        target=_run_dashboard_mcp_oauth,
        args=(flow, cfg),
        daemon=True,
        name=f"mcp-oauth-{name}",
    ).start()
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
            flow
            for flow in _mcp_oauth_flows.values()
            if flow.server_name == server_name
            and flow.status == "authorization_required"
        ]
    flow = next(
        (
            candidate
            for candidate in candidates
            if candidate.expected_state is not None
            and state is not None
            and secrets.compare_digest(candidate.expected_state, state)
        ),
        None,
    )
    if flow is None:
        return HTMLResponse("<h1>OAuth flow expired</h1><p>Return to Hermes and try again.</p>", status_code=404)
    try:
        flow.deliver_callback(code=code, state=state, error=error)
    except ValueError as exc:
        reason = str(exc)
        status_code = 409 if "already received" in reason else 400
        return HTMLResponse(
            "<h1>OAuth callback rejected</h1>"
            "<p>The callback was invalid or already used.</p>",
            status_code=status_code,
        )
    if error:
        return HTMLResponse("<h1>Authorization failed</h1><p>Return to Hermes for details.</p>", status_code=400)
    return HTMLResponse("<h1>Authorization received</h1><p>You can close this tab and return to Hermes.</p>")


@router.put("/api/mcp/servers/{name}/enabled")
async def set_mcp_server_enabled(
    name: str, body: MCPEnabledToggle, profile: Optional[str] = None
):
    """Enable or disable an MCP server (takes effect on next session/gateway).

    Toggles the ``enabled`` key on the server's config.yaml entry — the same
    flag the agent reads at startup.  Disabled servers stay in config so they
    can be re-enabled without re-entering their settings.
    """
    with _profile_scope(body.profile or profile):
        cfg = load_config()
        servers = cfg.get("mcp_servers")
        if not isinstance(servers, dict) or name not in servers:
            raise HTTPException(status_code=404, detail=f"Server '{name}' not found")
        if not isinstance(servers[name], dict):
            raise HTTPException(status_code=400, detail="Malformed server config")
        servers[name]["enabled"] = bool(body.enabled)
        save_config(cfg)
    return {"ok": True, "name": name, "enabled": bool(body.enabled)}


@router.get("/api/mcp/catalog")
async def list_mcp_catalog(profile: Optional[str] = None):
    """Browse the Nous-approved MCP catalog (the optional-mcps/ manifests).

    Each entry reports whether it's already installed and enabled so the UI
    can show install / enabled state inline.  This is the same catalog
    `hermes mcp catalog` / `hermes mcp install` read.  ``profile`` scopes
    the installed/enabled annotations (the catalog itself is repo-shipped
    and identical for every profile).
    """
    try:
        from hermes_cli import mcp_catalog
    except Exception as exc:
        _log.exception("mcp_catalog import failed")
        raise HTTPException(status_code=500, detail=f"Catalog unavailable: {exc}")

    entries = []
    try:
        with _profile_scope(profile):
            catalog_entries = list(mcp_catalog.list_catalog())
            installed_state = {
                e.name: (mcp_catalog.is_installed(e.name), mcp_catalog.is_enabled(e.name))
                for e in catalog_entries
            }
        for entry in catalog_entries:
            auth = entry.auth
            transport = entry.transport
            install = entry.install
            entries.append({
                "name": entry.name,
                "description": entry.description,
                "source": entry.source,
                "transport": transport.type,
                "auth_type": getattr(auth, "type", "none"),
                # Env vars the user must supply (names + prompts only, never values).
                "required_env": [
                    {"name": e.name, "prompt": e.prompt, "required": e.required}
                    for e in getattr(auth, "env", []) or []
                ],
                # Transport details so the UI can show exactly what connects/runs.
                # The trust model (docs: user-guide/features/mcp) tells users to
                # inspect command/args/url and the install bootstrap before
                # installing — surface them rather than hiding them in the repo.
                "command": transport.command,
                "args": list(transport.args or []),
                "url": transport.url,
                # Git bootstrap (present only for entries that clone + build).
                "install_url": install.url if install else None,
                "install_ref": install.ref if install else None,
                "bootstrap": list(install.bootstrap) if install else [],
                # Default tool pre-selection hint and post-install guidance.
                "default_enabled": list(entry.tools.default_enabled)
                if entry.tools.default_enabled is not None
                else None,
                "post_install": entry.post_install or "",
                "needs_install": entry.install is not None,
                "installed": installed_state.get(entry.name, (False, False))[0],
                "enabled": installed_state.get(entry.name, (False, False))[1],
            })
    except HTTPException:
        # Unknown/invalid profile → 404, not a silently-empty catalog.
        raise
    except Exception:
        _log.exception("list_mcp_catalog failed")

    diagnostics = []
    try:
        diagnostics = [
            {"name": n, "kind": k, "message": m}
            for (n, k, m) in mcp_catalog.catalog_diagnostics()
        ]
    except Exception:
        pass

    return {"entries": entries, "diagnostics": diagnostics}


@router.post("/api/mcp/catalog/install")
async def install_mcp_catalog_entry(body: MCPCatalogInstall, profile: Optional[str] = None):
    """Install a catalog MCP into config.yaml.

    For HTTP/stdio entries with required env vars, those are written to .env
    via the standard env path so the agent can read them at session start.
    Entries that need a git bootstrap (``needs_install``) are installed via
    the CLI action path because the clone can take time.
    """
    from hermes_cli import mcp_catalog

    name = (body.name or "").strip()
    entry = mcp_catalog.get_entry(name)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No catalog entry '{name}'")

    # Persist any supplied env vars first (catalog entries declare which names
    # they need; we only write the ones the user provided).
    effective_profile = body.profile or profile
    if body.env:
        with _profile_scope(effective_profile):
            for k, v in body.env.items():
                if v:
                    save_env_value(k, v)

    # Git-bootstrap entries can take a while to clone — run via the background
    # action path so the request returns immediately and the UI can tail logs.
    # The -p subprocess rebinds HERMES_HOME-derived paths in the child.
    if entry.install is not None:
        # Unique per-entry action name: a shared "mcp-install" would let a
        # re-click (or a second entry) overwrite the tracked process/log while
        # the first clone is still running.
        action = _mcp_install_action_name(name)
        try:
            _spawn_hermes_action(
                _profile_cli_args(effective_profile) + ["mcp", "install", name],
                action,
            )
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Install failed: {exc}")
        return {"ok": True, "name": name, "background": True, "action": action}

    # No git step — install synchronously via the catalog API. install_entry
    # routes through load_config/save_config + save_env_value, all call-time
    # resolvers, so the context override scopes it. Wrap the to_thread body
    # in the scope INSIDE the thread (contextvars don't propagate into
    # to_thread the other way around — asyncio.to_thread copies context, so
    # setting it here works; keep it explicit for clarity).
    def _install_scoped():
        with _profile_scope(effective_profile):
            mcp_catalog.install_entry(entry, enable=body.enable)

    try:
        await asyncio.to_thread(_install_scoped)
    except HTTPException:
        raise
    except Exception as exc:
        _log.exception("install_mcp_catalog_entry failed")
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True, "name": name, "background": False}
