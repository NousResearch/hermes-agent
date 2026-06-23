"""World Monitor authentication setup for Hermes (API key, OAuth MCP, local sidecar)."""

from __future__ import annotations

import re
import socket
import urllib.error
import urllib.request
from typing import Any

from . import api

WM_KEY_RE = re.compile(r"^wm_[0-9a-f]{40}$")
WM_MCP_URL = "https://worldmonitor.app/mcp"
WM_MCP_NAME = "worldmonitor"
REGISTRATION_URL = "https://www.worldmonitor.app/"
PRO_DOCS_URL = "https://www.worldmonitor.app/docs/usage-auth"
MCP_DOCS_URL = "https://www.worldmonitor.app/docs/mcp-quickstart"

# Hermes LLM / Codex / xAI credentials are a different trust domain — never auto-mapped.
NON_TRANSFERABLE_ENV_HINTS = (
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GROQ_API_KEY",
    "CODEX",
)


def validate_wm_key(key: str) -> tuple[bool, str]:
    """Return (ok, message) for a World Monitor user API key."""
    text = (key or "").strip()
    if not text:
        return False, "empty key"
    if text.startswith("Bearer "):
        return False, "paste the wm_ key only, not a Bearer prefix"
    if WM_KEY_RE.match(text):
        return True, "valid wm_ user key format"
    if len(text) >= 16 and text.startswith("wm_"):
        return True, "wm_ prefix with acceptable length (enterprise/opaque key)"
    return False, "expected wm_ + 40 hex chars (PRO/API user key) — see World Monitor docs"


def probe_sidecar(port: int = api.DEFAULT_LOCAL_PORT, *, timeout: float = 20.0) -> dict[str, Any]:
    """Check whether the World Monitor desktop sidecar responds on localhost."""
    base = f"http://127.0.0.1:{port}"
    sock_ok = False
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            sock_ok = True
    except OSError:
        sock_ok = False

    http_ok = False
    http_status: int | None = None
    probe_url = f"{base}/api/news/v1/list-feed-digest?variant=full&lang=en"
    try:
        req = urllib.request.Request(
            probe_url,
            headers={"Accept": "application/json", "User-Agent": "hermes-worldmonitor-osint/0.1"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            http_ok = 200 <= resp.status < 300
            http_status = resp.status
    except urllib.error.HTTPError as exc:
        http_status = exc.code
        # Sidecar up but endpoint gated is still "reachable".
        http_ok = exc.code in {200, 401, 403}
    except Exception:
        http_ok = False

    return {
        "port": port,
        "base_url": base,
        "socket_open": sock_ok,
        "http_reachable": http_ok,
        "http_status": http_status,
        "running": sock_ok and http_ok,
    }


def test_cloud_key(key: str) -> dict[str, Any]:
    """Probe cloud API with X-WorldMonitor-Key (never Bearer wm_)."""
    import os

    prev = os.environ.get(api.ENV_API_KEY)
    os.environ[api.ENV_API_KEY] = key.strip()
    prev_base = os.environ.get(api.ENV_API_BASE)
    os.environ[api.ENV_API_BASE] = api.DEFAULT_CLOUD_BASE
    try:
        data = api.get_country_risk("JP")
        return {"success": True, "sample": "get-country-risk/JP", "keys": list(data.keys())[:8]}
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if prev is None:
            os.environ.pop(api.ENV_API_KEY, None)
        else:
            os.environ[api.ENV_API_KEY] = prev
        if prev_base is None:
            os.environ.pop(api.ENV_API_BASE, None)
        else:
            os.environ[api.ENV_API_BASE] = prev_base


def _mcp_oauth_configured() -> dict[str, Any]:
    try:
        from hermes_cli.mcp_config import _get_mcp_servers

        servers = _get_mcp_servers()
    except Exception as exc:
        return {"configured": False, "error": str(exc)}

    cfg = servers.get(WM_MCP_NAME) or {}
    if not cfg:
        return {"configured": False}
    return {
        "configured": True,
        "url": cfg.get("url") or WM_MCP_URL,
        "auth": cfg.get("auth"),
        "enabled": cfg.get("enabled", True),
    }


def _ensure_mcp_oauth(*, dry_run: bool = False) -> dict[str, Any]:
    existing = _mcp_oauth_configured()
    if existing.get("configured"):
        return {"status": "already_configured", **existing}
    server_cfg = {"url": WM_MCP_URL, "auth": "oauth", "enabled": True}
    if dry_run:
        return {"status": "would_install", "transport": server_cfg}
    from hermes_cli.mcp_config import _save_mcp_server

    saved = _save_mcp_server(WM_MCP_NAME, server_cfg)
    return {"status": "installed" if saved else "save_failed", "transport": server_cfg}


def _save_api_key(key: str) -> None:
    from hermes_cli.config import save_env_value

    save_env_value(api.ENV_API_KEY, key.strip())


def _save_sidecar_base(port: int) -> None:
    from hermes_cli.config import save_env_value

    base = f"http://127.0.0.1:{port}"
    save_env_value(api.ENV_API_BASE, base)
    save_env_value(api.ENV_LOCAL_PORT, str(port))


def auth_guidance() -> dict[str, Any]:
    """Explain why Hermes LLM keys cannot substitute for World Monitor auth."""
    try:
        from hermes_cli.config import get_env_value
    except Exception:
        get_env_value = lambda _n: ""  # type: ignore

    present_llm_keys = [
        name
        for name in NON_TRANSFERABLE_ENV_HINTS
        if (get_env_value(name) or "").strip()
    ]
    return {
        "worldmonitor_auth_is_separate": True,
        "llm_keys_cannot_be_reused": True,
        "codex_oauth_cannot_be_reused": True,
        "xai_oauth_cannot_be_reused": True,
        "reason": (
            "World Monitor validates X-WorldMonitor-Key (wm_…) or World Monitor OAuth JWT — "
            "not OpenRouter, OpenAI, xAI, Anthropic, or Codex credentials."
        ),
        "hermes_llm_env_vars_present": present_llm_keys,
        "supported_modes": [
            {
                "mode": "oauth_mcp",
                "summary": "World Monitor Pro OAuth via Hermes MCP (recommended for interactive use)",
                "command": "hermes worldmonitor-osint setup-auth --mode oauth",
            },
            {
                "mode": "api_key",
                "summary": "wm_ REST/MCP key (PRO subscription or API tier)",
                "command": "hermes worldmonitor-osint setup-auth --mode key --api-key wm_...",
            },
            {
                "mode": "sidecar",
                "summary": "Local desktop sidecar (no cloud key for local-first reads)",
                "command": "hermes worldmonitor-osint setup-auth --mode sidecar",
            },
            {
                "mode": "dev",
                "summary": "Local Vite dev server (npm run dev on port 3000)",
                "command": "hermes worldmonitor-osint dev setup",
            },
        ],
        "docs": {
            "auth": PRO_DOCS_URL,
            "mcp_quickstart": MCP_DOCS_URL,
            "registration": REGISTRATION_URL,
        },
    }


def setup_auth(
    *,
    mode: str = "auto",
    api_key: str = "",
    port: int = api.DEFAULT_LOCAL_PORT,
    install_mcp: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Configure World Monitor access for Hermes."""
    mode = (mode or "auto").strip().lower()
    result: dict[str, Any] = {
        "success": False,
        "mode": mode,
        "dry_run": dry_run,
        "guidance": auth_guidance(),
        "actions": [],
        "next_steps": [],
    }

    if mode in {"auto", "sidecar", "dev"}:
        if mode in {"auto", "dev"}:
            from . import dev_server

            dev_probe = dev_server.probe_dev_server()
            result["dev_probe"] = dev_probe
            if dev_probe.get("running"):
                result["actions"].append("dev_server_detected")
                if not dry_run:
                    dev_server._save_dev_base(dev_probe["port"])
                result["success"] = True
                result["auth_method"] = "vite_dev"
                result["next_steps"].append(
                    f"Dev server active at {dev_probe['base_url']}. REST plugin uses local Vite API."
                )
                if mode == "dev":
                    return result

        if mode == "sidecar" or (mode == "auto" and not result.get("success")):
            sidecar = probe_sidecar(port)
            result["sidecar_probe"] = sidecar
            if sidecar.get("running"):
                result["actions"].append("sidecar_detected")
                if not dry_run:
                    _save_sidecar_base(port)
                result["success"] = True
                result["auth_method"] = "local_sidecar"
                result["next_steps"].append(
                    "Sidecar active. REST plugin uses local API; restart Hermes if you changed .env."
                )
                if mode == "sidecar":
                    return result

    if mode in {"auto", "key"} and (api_key or "").strip():
        ok, msg = validate_wm_key(api_key)
        result["key_validation"] = {"ok": ok, "message": msg}
        if not ok:
            result["error"] = msg
            return result
        if dry_run:
            result["actions"].append("would_save_api_key")
        else:
            _save_api_key(api_key)
            test = test_cloud_key(api_key)
            result["cloud_test"] = test
            result["success"] = bool(test.get("success"))
            result["auth_method"] = "api_key"
            result["actions"].append("api_key_saved")
        if mode == "key":
            if result.get("success") or dry_run:
                result["next_steps"].append("Restart Hermes session to pick up WORLDMONITOR_API_KEY.")
            return result

    if mode in {"auto", "oauth"} and install_mcp:
        mcp_result = _ensure_mcp_oauth(dry_run=dry_run)
        result["mcp_oauth"] = mcp_result
        if mcp_result.get("status") in {"installed", "already_configured", "would_install"}:
            result["actions"].append("mcp_oauth_configured")
            result["success"] = True
            result.setdefault("auth_method", "oauth_mcp")
            result["next_steps"].extend(
                [
                    "Start a new Hermes session, then run `hermes mcp login worldmonitor` (interactive).",
                    "After login, verify with `hermes mcp test worldmonitor`.",
                    "On first connect, complete 'Sign in with World Monitor Pro' in the browser.",
                    "OAuth tokens are stored by Hermes MCP — not your LLM provider keys.",
                ]
            )
        if mode == "oauth":
            return result

    if result.get("success"):
        return result

    result["next_steps"] = [
        "Option A (OAuth, no wm_ key): hermes worldmonitor-osint setup-auth --mode oauth",
        "Option B (API key): Subscribe to World Monitor Pro/API → copy wm_ key → "
        "hermes worldmonitor-osint setup-auth --mode key --api-key wm_...",
        "Option C (local): Install World Monitor desktop app → sidecar on port 46123 → "
        "hermes worldmonitor-osint setup-auth --mode sidecar",
        "Option D (dev): git clone + npm install + npm run dev → "
        "hermes worldmonitor-osint dev setup",
        f"Docs: {MCP_DOCS_URL}",
    ]
    result["error"] = (
        "No working World Monitor auth path detected. "
        "Hermes LLM / Codex / xAI credentials cannot be substituted."
    )
    return result
