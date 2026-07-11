"""CLI handlers for the ``hermes proxy`` subcommand."""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import sys
from pathlib import Path
from typing import Any

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.server import (
    AIOHTTP_AVAILABLE,
    DEFAULT_HOST,
    DEFAULT_PORT,
    run_server,
)

logger = logging.getLogger(__name__)


def _read_downstream_token_file(path: str) -> str:
    token_path = Path(path).expanduser()
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Proxy auth token file is empty: {token_path}")
    return token


def _is_loopback_host(host: str) -> bool:
    if host.strip().lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _print_aiohttp_missing() -> None:
    print(
        "hermes proxy requires aiohttp. Install one of:\n"
        "  pip install 'hermes-agent[messaging]'\n"
        "  pip install aiohttp",
        file=sys.stderr,
    )


def cmd_proxy_start(args: Any) -> int:
    """Run the proxy server in the foreground.

    Returns process exit code (0 on clean shutdown).
    """
    if not AIOHTTP_AVAILABLE:
        _print_aiohttp_missing()
        return 1

    provider = getattr(args, "provider", None) or "nous"
    try:
        adapter = get_adapter(provider)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if not adapter.is_authenticated():
        auth_hint = getattr(adapter, "auth_hint", f"hermes auth add {adapter.name}")
        print(
            f"Not logged into {adapter.display_name}. "
            f"Run `{auth_hint}` first.",
            file=sys.stderr,
        )
        return 2

    host = getattr(args, "host", None) or DEFAULT_HOST
    port = getattr(args, "port", None) or DEFAULT_PORT
    token_file_value = getattr(args, "auth_token_file", None)
    token_file = token_file_value if isinstance(token_file_value, str) else None
    downstream_token = None
    if token_file:
        try:
            downstream_token = _read_downstream_token_file(token_file)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    if not _is_loopback_host(host) and not downstream_token:
        print(
            "Error: non-loopback proxy binds require --auth-token-file. "
            "Refusing to expose subscription credentials through an unauthenticated proxy.",
            file=sys.stderr,
        )
        return 2

    auth_message = (
        "  Downstream auth: required (token file)\n"
        if downstream_token
        else "  Downstream auth: unrestricted (loopback-only bind)\n"
    )

    print(
        f"Starting Hermes proxy for {adapter.display_name}\n"
        f"  Listening on:  http://{host}:{port}/v1\n"
        f"  Forwarding to: (resolved per-request from your subscription)\n"
        f"{auth_message}"
        f"\n"
        f"Press Ctrl+C to stop.",
        file=sys.stderr,
    )

    try:
        asyncio.run(
            run_server(
                adapter,
                host=host,
                port=port,
                downstream_token=downstream_token,
            )
        )
    except KeyboardInterrupt:
        print("\nproxy: stopped", file=sys.stderr)
    except OSError as exc:
        print(f"proxy: failed to bind {host}:{port}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_proxy_status(args: Any) -> int:
    """Print the status of each configured upstream adapter."""
    print("Hermes proxy upstream adapters\n")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        if not adapter.is_authenticated():
            print(f"  [{name:8s}] {adapter.display_name} — not logged in")
            continue
        try:
            cred = adapter.get_credential()
        except Exception as exc:
            print(
                f"  [{name:8s}] {adapter.display_name} — credentials need attention "
                f"({exc})"
            )
            continue
        expires = f" (bearer expires {cred.expires_at})" if cred.expires_at else ""
        print(f"  [{name:8s}] {adapter.display_name} — ready{expires}")
    print(
        "\nStart the proxy with: hermes proxy start [--provider <name>]"
    )
    return 0


def cmd_proxy_list_providers(args: Any) -> int:
    """List available proxy upstream providers."""
    print("Available proxy upstream providers:")
    for name in sorted(ADAPTERS):
        adapter = get_adapter(name)
        print(f"  {name}  — {adapter.display_name}")
    return 0


def cmd_proxy(args: Any) -> int:
    """Dispatch ``hermes proxy <subcommand>``."""
    sub = getattr(args, "proxy_command", None)
    if sub == "start":
        return cmd_proxy_start(args)
    if sub == "status":
        return cmd_proxy_status(args)
    if sub in {"providers", "list"}:
        return cmd_proxy_list_providers(args)
    # No subcommand → print short help.
    print(
        "hermes proxy — local OpenAI-compatible proxy that attaches your\n"
        "OAuth-authenticated provider credentials to outbound requests.\n"
        "\n"
        "Subcommands:\n"
        "  hermes proxy start [--provider nous|openai-codex|xai] [--host 127.0.0.1] [--port 8645]\n"
        "      [--auth-token-file PATH]\n"
        "      Run the proxy in the foreground.\n"
        "  hermes proxy status\n"
        "      Show which upstream adapters are ready.\n"
        "  hermes proxy providers\n"
        "      List available upstream providers.\n",
        file=sys.stderr,
    )
    return 0


__all__ = [
    "cmd_proxy",
    "cmd_proxy_start",
    "cmd_proxy_status",
    "cmd_proxy_list_providers",
]
