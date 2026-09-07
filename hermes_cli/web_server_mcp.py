"""MCP dashboard helpers: create-payload normalisation, env redaction/summary, dashboard-driven MCP OAuth worker.

Wraps the same config layer the CLI uses (hermes_cli.mcp_config); stdio ``env``
secrets are redacted on read.
"""

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from tools.mcp_dashboard_oauth import DashboardOAuthFlow
from hermes_cli.config import redact_key
from hermes_cli.web_models import MCPServerCreate


def _normalize_mcp_server_create(body: MCPServerCreate) -> tuple[str, Dict[str, Any], Optional[str]]:
    """Validate a Dashboard MCP create request and build its safe config.

    The returned config never contains the Bearer token; callers persist it via
    the shared Bearer helper once inside the intended profile scope. Shared by
    the MCP page and the Profile Builder so both enforce one transport/auth contract.
    """
    from hermes_cli.mcp_config import _bearer_auth_headers, _strip_bearer_prefix
    from hermes_cli.mcp_security import validate_mcp_server_entry

    name = (body.name or "").strip()
    if not name:
        raise ValueError("Server name is required")

    url = (body.url or "").strip()
    command = (body.command or "").strip()
    auth = (body.auth or "none").strip().lower()
    bearer_token = body.bearer_token.get_secret_value() if body.bearer_token is not None else None

    if bool(url) == bool(command):
        raise ValueError("Provide exactly one of URL (HTTP/SSE) or command (stdio)")
    if auth not in {"none", "header", "oauth"}:
        raise ValueError(f"Unsupported auth mode: {auth}")

    server_config: Dict[str, Any] = {}
    if url:
        if body.args:
            raise ValueError("Arguments are only supported for stdio MCP servers")
        if body.env:
            raise ValueError("Environment variables are only supported for stdio MCP servers")
        if auth == "header":
            normalized = _strip_bearer_prefix(bearer_token) if bearer_token else ""
            if not normalized or normalized.lower() == "bearer":
                raise ValueError("Bearer token is required")
            server_config["headers"] = _bearer_auth_headers(name)
        elif body.bearer_token is not None:
            raise ValueError("Bearer token requires header authentication")

        server_config["url"] = url
        if auth == "oauth":
            server_config["auth"] = "oauth"
    else:
        if auth != "none" or body.bearer_token is not None:
            raise ValueError("HTTP authentication is not supported for stdio MCP servers")
        server_config["command"] = command
        if body.args:
            server_config["args"] = list(body.args)
        if body.env:
            server_config["env"] = dict(body.env)

    issues = validate_mcp_server_entry(name, server_config)
    if issues:
        raise ValueError(f"Server '{name}' rejected: {'; '.join(issues)}")
    return name, server_config, bearer_token


def _redact_mcp_env(env: Dict[str, Any]) -> Dict[str, str]:
    """Mask secret-shaped MCP env values for read responses."""
    out: Dict[str, str] = {}
    for k, v in (env or {}).items():
        try:
            out[str(k)] = redact_key(str(v)) if v else ""
        except Exception:
            out[str(k)] = "***"
    return out


def _mcp_server_summary(name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    transport = "http" if cfg.get("url") else ("stdio" if cfg.get("command") else "unknown")
    auth = cfg.get("auth")
    headers = cfg.get("headers") or {}
    if not auth and isinstance(headers, dict) and any(str(key).lower() == "authorization" for key in headers):
        auth = "header"
    return {
        "name": name,
        "transport": transport,
        "url": cfg.get("url"),
        "command": cfg.get("command"),
        "args": list(cfg.get("args") or []),
        "env": _redact_mcp_env(cfg.get("env") or {}),
        "auth": auth,
        "enabled": cfg.get("enabled", True) is not False,
        # Tool selection: list of enabled tool names, or None = all.
        "tools": cfg.get("tools"),
    }


_mcp_oauth_flows: dict[str, "DashboardOAuthFlow"] = {}
_mcp_oauth_transactions: dict[tuple[str, str], threading.Lock] = {}
_mcp_oauth_transactions_lock = threading.Lock()


def _mcp_oauth_transaction(flow) -> threading.Lock:
    key = (flow.hermes_home, flow.server_name)
    with _mcp_oauth_transactions_lock:
        return _mcp_oauth_transactions.setdefault(key, threading.Lock())


def _run_dashboard_mcp_oauth(flow, cfg: dict) -> None:
    """Run the normal MCP probe with dashboard redirect/callback handlers."""
    from hermes_cli.mcp_config import _oauth_tokens_present, _probe_single_server, _save_mcp_server
    try:
        from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from tools.mcp_dashboard_oauth import dashboard_oauth_flow
        from tools.mcp_oauth import HermesTokenStorage, force_interactive_oauth
        from tools.mcp_oauth_manager import get_manager

        home_token = set_hermes_home_override(flow.hermes_home)
        secret_token = set_secret_scope(build_profile_secret_scope(Path(flow.hermes_home)))
        try:
            transaction = _mcp_oauth_transaction(flow)
            with transaction, force_interactive_oauth(), dashboard_oauth_flow(flow):
                manager = get_manager()
                storage = HermesTokenStorage(flow.server_name)
                backup = storage.snapshot()
                previous_entry = None
                try:
                    previous_entry = manager.remove(flow.server_name, hermes_home=flow.hermes_home)
                    tools = _probe_single_server(
                        flow.server_name,
                        cfg,
                        connect_timeout=max(float(cfg.get("connect_timeout", 0) or 0), 315),
                    )
                    if not _oauth_tokens_present(flow.server_name):
                        raise RuntimeError(
                            "The server responded, but no OAuth token was obtained — "
                            "this provider may require a manually-registered OAuth client."
                        )
                    _save_mcp_server(flow.server_name, cfg)
                    flow.tools = [{"name": t, "description": d} for t, d in tools]
                    flow.mark_approved()
                    if flow.reconnect_live:
                        from tools.mcp_tool_loop import reconnect_mcp_server

                        reconnect_mcp_server(flow.server_name)
                except Exception:
                    storage.restore(backup, only_if_absent=True)
                    manager.restore_entry(flow.server_name, previous_entry, hermes_home=flow.hermes_home)
                    raise
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)
    except Exception as exc:
        msg = str(exc)
        # Providers gating RFC 7591 registration to pre-approved clients 403 the
        # register call before any auth URL exists; say so, not "403 Forbidden".
        try:
            from tools.mcp_oauth import humanize_oauth_registration_error

            humanized = humanize_oauth_registration_error(
                flow.server_name, exc, server_url=cfg.get("url") if isinstance(cfg, dict) else None
            )
            if humanized:
                msg = humanized
        except Exception:
            pass
        flow.mark_error(msg)
    finally:
        flow.mark_worker_done()
