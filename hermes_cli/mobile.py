"""Opt-in Hermes Mobile pairing CLI."""

from __future__ import annotations

import socket
from argparse import Namespace
from urllib.parse import urlencode


def build_mobile_parser(subparsers) -> None:
    parser = subparsers.add_parser("mobile", help="Manage the optional Hermes Mobile extension")
    actions = parser.add_subparsers(dest="mobile_action", required=True)
    pair = actions.add_parser("pair", help="Print a one-time mobile pairing QR code")
    pair.add_argument("--url", help="Public API server URL embedded in the QR code")
    parser.set_defaults(func=mobile_command)


def _lan_address() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"


def _api_url(override: str | None) -> str:
    if override:
        return override.rstrip("/")
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    api = (((cfg.get("platforms") or {}).get("api_server") or {}).get("extra") or {})
    host = str(api.get("host") or "127.0.0.1")
    if host in {"0.0.0.0", "127.0.0.1", "localhost", "::"}:
        host = _lan_address()
    return f"http://{host}:{int(api.get('port') or 8642)}"


def mobile_command(args: Namespace) -> None:
    from gateway.mobile_notifications import MobilePairingStore, mobile_extension_enabled

    if args.mobile_action != "pair":
        return
    if not mobile_extension_enabled():
        raise SystemExit("Hermes Mobile is disabled. Set mobile_notifications.enabled: true in config.yaml.")

    grant = MobilePairingStore().create_grant()
    host_url = _api_url(args.url)
    payload = "hermes://pair?" + urlencode({"url": host_url, "grant": grant.secret})
    from hermes_cli.dingtalk_auth import render_qr_to_terminal

    print("Scan this QR code in Hermes Mobile:")
    if not render_qr_to_terminal(payload):
        print(payload)
    print(f"\nHost: {host_url}")
    print(f"Fallback code: {grant.code}")
    print("Expires in 5 minutes and can be used once.")
