#!/usr/bin/env python3
"""Composio CLI helper for the Hermes composio skill.

Thin wrapper over the Composio Python SDK (https://docs.composio.dev) that
gives the agent a shell-friendly JSON interface to 1000+ SaaS toolkits
(Gmail, GitHub, Slack, Notion, Linear, Google Drive/Calendar, Airtable, ...).

Composio hosts the OAuth flows and stores connected accounts, so Hermes
never handles third-party refresh tokens — only the single COMPOSIO_API_KEY.

Usage:
    python3 composio_cli.py toolkits                      # connected accounts
    python3 composio_cli.py tools <toolkit>               # list a toolkit's tools
    python3 composio_cli.py search <query>                # search tools by text
    python3 composio_cli.py schema <TOOL_SLUG>            # input schema for a tool
    python3 composio_cli.py execute <TOOL_SLUG> --args '{"k": "v"}'
    python3 composio_cli.py connect <toolkit>             # start OAuth, prints URL
    python3 composio_cli.py wait <connection_request_id>  # poll until connected

All commands print a single JSON object to stdout. Errors come back as
{"successful": false, "error": "..."} with exit code 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import NoReturn

DEFAULT_USER_ID = os.environ.get("COMPOSIO_USER_ID", "hermes")
MAX_SEARCH_RESULTS = 20


def _fail(msg: str, code: int = 1) -> NoReturn:
    print(json.dumps({"successful": False, "error": msg}))
    sys.exit(code)


def _get_client():
    api_key = (os.environ.get("COMPOSIO_API_KEY") or "").strip()
    if not api_key:
        _fail(
            "COMPOSIO_API_KEY is not set. Add it to ~/.hermes/.env "
            "(create a key at https://app.composio.dev)."
        )
    try:
        from composio import Composio  # type: ignore[attr-defined]
    except ImportError:
        _fail(
            "The 'composio' package is not installed. "
            f"Install with: {sys.executable} -m pip install composio"
        )
    return Composio(api_key=api_key)


def _tool_summaries(raw: object) -> list[dict]:
    """Reduce Composio tool entries to name/description/required params."""
    out: list[dict] = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        params = fn.get("parameters") or {}
        out.append(
            {
                "name": fn.get("name", ""),
                "description": (fn.get("description") or "")[:300],
                "required_params": params.get("required", []),
            }
        )
    return out


def cmd_toolkits(client, args) -> dict:
    """List connected accounts (which toolkits are usable right now)."""
    accounts = client.connected_accounts.list(user_ids=[args.user])
    items = getattr(accounts, "items", None) or []
    out = []
    for acct in items:
        toolkit = getattr(acct, "toolkit", None)
        out.append(
            {
                "id": getattr(acct, "id", ""),
                "toolkit": getattr(toolkit, "slug", "") or str(toolkit or ""),
                "status": str(getattr(acct, "status", "")),
            }
        )
    return {"successful": True, "connected_accounts": out, "count": len(out)}


def cmd_tools(client, args) -> dict:
    raw = client.tools.get(args.user, toolkits=[args.toolkit.strip()])
    tools = _tool_summaries(raw)
    return {
        "successful": True,
        "toolkit": args.toolkit,
        "tools": tools,
        "count": len(tools),
    }


def cmd_search(client, args) -> dict:
    raw = client.tools.get(args.user, search=args.query.strip())
    items = raw[:MAX_SEARCH_RESULTS] if isinstance(raw, list) else []
    tools = _tool_summaries(items)
    return {"successful": True, "query": args.query, "tools": tools, "count": len(tools)}


def cmd_schema(client, args) -> dict:
    slug = args.slug.strip()
    raw = client.tools.get(args.user, search=slug)
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function") or {}
        if fn.get("name") == slug:
            return {
                "successful": True,
                "name": fn.get("name"),
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
    return {"successful": False, "error": f"tool {slug!r} not found"}


def cmd_execute(client, args) -> dict:
    try:
        arguments = json.loads(args.args) if args.args else {}
    except json.JSONDecodeError as exc:
        return {"successful": False, "error": f"--args is not valid JSON: {exc}"}
    if not isinstance(arguments, dict):
        return {"successful": False, "error": "--args must be a JSON object"}

    # Version pinning is per-toolkit; callers here don't know toolkit versions
    # in advance, so ask the SDK to run the latest.
    result = client.tools.execute(
        args.slug.strip(),
        user_id=args.user,
        arguments=arguments,
        dangerously_skip_version_check=True,
    )
    if isinstance(result, dict):
        return {
            "successful": bool(result.get("successful", False)),
            "error": result.get("error"),
            "data": result.get("data", {}),
        }
    return {
        "successful": bool(getattr(result, "successful", False)),
        "error": getattr(result, "error", None),
        "data": getattr(result, "data", {}),
    }


def cmd_connect(client, args) -> dict:
    """Start an OAuth connection for a toolkit; user opens the printed URL."""
    toolkit = args.toolkit.strip().lower()
    # Preferred: one-call authorize (Composio-managed auth config).
    try:
        req = client.toolkits.authorize(user_id=args.user, toolkit=toolkit)
        return {
            "successful": True,
            "toolkit": toolkit,
            "redirect_url": getattr(req, "redirect_url", None),
            "connection_request_id": getattr(req, "id", None),
            "note": "Open redirect_url in a browser to grant access, then run "
            "'wait <connection_request_id>' or retry your command.",
        }
    except Exception:
        pass
    # Fallback: explicit auth config + connected account initiation.
    auth_config = client.auth_configs.create(
        toolkit=toolkit, options={"type": "use_composio_managed_auth"}
    )
    req = client.connected_accounts.initiate(
        user_id=args.user, auth_config_id=getattr(auth_config, "id", auth_config)
    )
    return {
        "successful": True,
        "toolkit": toolkit,
        "redirect_url": getattr(req, "redirect_url", None),
        "connection_request_id": getattr(req, "id", None),
        "note": "Open redirect_url in a browser to grant access.",
    }


def cmd_wait(client, args) -> dict:
    acct = client.connected_accounts.wait_for_connection(
        args.request_id, timeout=float(args.timeout)
    )
    return {
        "successful": True,
        "connected_account_id": getattr(acct, "id", ""),
        "status": str(getattr(acct, "status", "")),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Composio CLI helper for the Hermes composio skill."
    )
    p.add_argument(
        "--user",
        default=DEFAULT_USER_ID,
        help="Composio user_id scoping connected accounts (default: %(default)s)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("toolkits", help="List connected accounts")

    sp = sub.add_parser("tools", help="List tools in a toolkit")
    sp.add_argument("toolkit")

    sp = sub.add_parser("search", help="Search tools across all toolkits")
    sp.add_argument("query")

    sp = sub.add_parser("schema", help="Show a tool's input schema")
    sp.add_argument("slug")

    sp = sub.add_parser("execute", help="Execute a tool by slug")
    sp.add_argument("slug")
    sp.add_argument("--args", default="", help="JSON object of tool arguments")

    sp = sub.add_parser("connect", help="Start OAuth for a toolkit")
    sp.add_argument("toolkit")

    sp = sub.add_parser("wait", help="Wait for a pending connection")
    sp.add_argument("request_id")
    sp.add_argument("--timeout", default="120", help="Seconds to wait (default 120)")

    return p


HANDLERS = {
    "toolkits": cmd_toolkits,
    "tools": cmd_tools,
    "search": cmd_search,
    "schema": cmd_schema,
    "execute": cmd_execute,
    "connect": cmd_connect,
    "wait": cmd_wait,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = _get_client()
    try:
        result = HANDLERS[args.cmd](client, args)
    except Exception as exc:  # surface the wire error, never swallow it
        result = {"successful": False, "error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("successful") else 1


if __name__ == "__main__":
    sys.exit(main())
