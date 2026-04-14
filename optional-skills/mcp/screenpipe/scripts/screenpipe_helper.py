#!/usr/bin/env python3
"""Helper utilities for wiring Screenpipe into Hermes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

DEFAULT_API_BASE = "http://127.0.0.1:3030"
DEFAULT_MCP_SERVER_NAME = "screenpipe"
DEFAULT_MCP_CONFIG = {
    "command": "npx",
    "args": ["-y", "screenpipe-mcp"],
}


class ScreenpipeError(RuntimeError):
    """Domain-specific Screenpipe integration failure."""


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME", "~/.hermes")).expanduser()


def _config_path(path: str | Path | None = None) -> Path:
    return Path(path).expanduser() if path else (_hermes_home() / "config.yaml")


def _load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = _config_path(path)
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _save_config(config: dict[str, Any], path: str | Path | None = None) -> Path:
    config_path = _config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def _json_request(url: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params, doseq=True)}"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.URLError as exc:
        raise ScreenpipeError(str(exc)) from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScreenpipeError(f"Invalid JSON from {url}") from exc


def doctor(*, base_url: str = DEFAULT_API_BASE, config_path: str | Path | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    api_reachable = False
    health_payload: dict[str, Any] | None = None
    error = ""

    try:
        health_payload = _json_request(f"{base_url.rstrip('/')}/health")
        api_reachable = True
    except ScreenpipeError as exc:
        error = str(exc)

    mcp_config = (config.get("mcp_servers") or {}).get(DEFAULT_MCP_SERVER_NAME)
    return {
        "success": True,
        "api_base": base_url,
        "api_reachable": api_reachable,
        "health": health_payload or {},
        "health_error": error,
        "tools": {
            "npx": bool(shutil.which("npx")),
        },
        "mcp_configured": isinstance(mcp_config, dict),
        "mcp_server": mcp_config or {},
        "config_path": str(_config_path(config_path)),
    }


def install_mcp_server(
    *,
    config_path: str | Path | None = None,
    server_name: str = DEFAULT_MCP_SERVER_NAME,
    force: bool = False,
) -> dict[str, Any]:
    config = _load_config(config_path)
    servers = config.setdefault("mcp_servers", {})
    if not isinstance(servers, dict):
        raise ScreenpipeError("config.yaml has a non-dict `mcp_servers` value")

    existing = servers.get(server_name)
    if existing == DEFAULT_MCP_CONFIG:
        path = _save_config(config, config_path)
        return {
            "success": True,
            "changed": False,
            "config_path": str(path),
            "server_name": server_name,
            "server": existing,
        }
    if existing and not force:
        raise ScreenpipeError(
            f"MCP server '{server_name}' already exists. Re-run with force=True to replace it."
        )

    servers[server_name] = dict(DEFAULT_MCP_CONFIG)
    path = _save_config(config, config_path)
    return {
        "success": True,
        "changed": True,
        "config_path": str(path),
        "server_name": server_name,
        "server": servers[server_name],
    }


def search(
    query: str,
    *,
    base_url: str = DEFAULT_API_BASE,
    content_type: str = "",
    limit: int = 10,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "q": query,
        "limit": max(1, int(limit)),
    }
    if content_type:
        params["content_type"] = content_type
    payload = _json_request(f"{base_url.rstrip('/')}/search", params=params)
    return {
        "success": True,
        "query": query,
        "content_type": content_type,
        "limit": params["limit"],
        "results": payload,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hermes Screenpipe helper")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="Check local API health and MCP config")
    doctor_p.add_argument("--base-url", default=DEFAULT_API_BASE)
    doctor_p.add_argument("--config", default="")

    install_p = sub.add_parser("install-mcp", help="Add Screenpipe MCP config to Hermes")
    install_p.add_argument("--config", default="")
    install_p.add_argument("--server-name", default=DEFAULT_MCP_SERVER_NAME)
    install_p.add_argument("--force", action="store_true")

    search_p = sub.add_parser("search", help="Query the local Screenpipe REST API")
    search_p.add_argument("--query", required=True)
    search_p.add_argument("--content-type", default="")
    search_p.add_argument("--limit", type=int, default=10)
    search_p.add_argument("--base-url", default=DEFAULT_API_BASE)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            payload = doctor(base_url=args.base_url, config_path=args.config or None)
        elif args.command == "install-mcp":
            payload = install_mcp_server(
                config_path=args.config or None,
                server_name=args.server_name,
                force=args.force,
            )
        elif args.command == "search":
            payload = search(
                args.query,
                base_url=args.base_url,
                content_type=args.content_type,
                limit=args.limit,
            )
        else:
            raise ScreenpipeError(f"Unknown command: {args.command}")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except ScreenpipeError as exc:
        print(json.dumps({"success": False, "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
