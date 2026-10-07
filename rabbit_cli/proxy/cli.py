"""CLI handlers for the ``rabbit proxy`` subcommand."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

from rabbit_cli.proxy.adapters import ADAPTERS, get_adapter
from rabbit_cli.proxy.server import (
    AIOHTTP_AVAILABLE, DEFAULT_HOST, DEFAULT_PORT, run_server
)

logger = logging.getLogger(__name__)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def cmd_proxy_start(args: Any) -> int:
    """Run the proxy server in the foreground."""
    if not AIOHTTP_AVAILABLE:
        _err("rabbit proxy requires aiohttp. Run `rabbit setup` to install it.")
        return 1
    provider = getattr(args, "provider", None) or "xai"
    try:
        adapter = get_adapter(provider)
    except ValueError as exc:
        _err(f"Error: {exc}")
        return 2
    if not adapter.is_authenticated():
        auth_hint = getattr(adapter, "auth_hint", f"rabbit auth add {adapter.name}")
        _err(f"Not logged into {adapter.display_name}. Run `{auth_hint}` first.")
        return 2
    host = getattr(args, "host", None) or DEFAULT_HOST
    port = getattr(args, "port", None) or DEFAULT_PORT
    _err(
        f"Starting Rabbit proxy for {adapter.display_name}\n"
        f"  Listening on:  http://{host}:{port}/v1\n"
        f"  Forwarding to: (resolved per-request from your subscription)\n"
        f"  Use any bearer token in the client — the proxy attaches your real credential.\n"
        f"\n"
        f"Press Ctrl+C to stop."
    )
    try:
        asyncio.run(run_server(adapter, host=host, port=port))
    except KeyboardInterrupt:
        _err("\nproxy: stopped")
    except OSError as exc:
        _err(f"proxy: failed to bind {host}:{port}: {exc}")
        return 1
    return 0


def cmd_proxy_status(args: Any) -> int:
    """Print the status of each configured upstream adapter."""
    print("Rabbit proxy upstream adapters\n")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        if not adapter.is_authenticated():
            print(f"  [{name:8s}] {adapter.display_name} — not logged in")
            continue
        try:
            cred = adapter.get_credential()
        except Exception as exc:
            print(f"  [{name:8s}] {adapter.display_name} — credentials need attention ({exc})")
            continue
        expires = f" (bearer expires {cred.expires_at})" if cred.expires_at else ""
        print(f"  [{name:8s}] {adapter.display_name} — ready{expires}")
    print("\nStart the proxy with: rabbit proxy start [--provider <name>]")
    return 0


def cmd_proxy_list_providers(args: Any) -> int:
    """List available proxy upstream providers."""
    print("Available proxy upstream providers:")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        print(f"  {name}  — {adapter.display_name}")
    return 0


_SUBCOMMANDS = {
    "start": cmd_proxy_start,
    "status": cmd_proxy_status,
    "providers": cmd_proxy_list_providers,
    "list": cmd_proxy_list_providers,
}


def cmd_proxy(args: Any) -> int:
    """Dispatch ``rabbit proxy <subcommand>``; no/unknown subcommand prints the short help."""
    handler = _SUBCOMMANDS.get(getattr(args, "proxy_command", None))
    if handler is not None:
        return handler(args)
    _err(
        "rabbit proxy — local OpenAI-compatible proxy that attaches your\n"
        "OAuth-authenticated provider credentials to outbound requests.\n"
        "\n"
        "Subcommands:\n"
        "  rabbit proxy start [--provider xai] [--host 127.0.0.1] [--port 8645]\n"
        "      Run the proxy in the foreground.\n"
        "  rabbit proxy status\n"
        "      Show which upstream adapters are ready.\n"
        "  rabbit proxy providers\n"
        "      List available upstream providers.\n"
    )
    return 0


__all__ = ["cmd_proxy", "cmd_proxy_start", "cmd_proxy_status", "cmd_proxy_list_providers"]
