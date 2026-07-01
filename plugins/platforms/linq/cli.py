"""
``hermes linq ...`` CLI subcommands — registered by the plugin via
``ctx.register_cli_command()``.

Subcommands::

    setup            store the Linq API token (+ from-phone) and print
                     webhook-registration instructions
    status           show credential state and run a connectivity probe
    webhook show     print the local webhook URL to register in the Linq
                     dashboard, plus a generated signing-secret reminder
    probe            hit GET /phonenumbers to confirm the token works

Linq webhooks are registered in the Linq dashboard (Linq POSTs to a URL you
control), so there is no API-side ``webhook register`` verb the way Photon
has — the ``webhook show`` helper just tells you exactly what to paste in.
"""
from __future__ import annotations

import argparse
import asyncio
import getpass
import os
import sys

from . import auth as linq_auth
from . import signing
from .adapter import (
    _DEFAULT_WEBHOOK_BIND,
    _DEFAULT_WEBHOOK_PATH,
    _DEFAULT_WEBHOOK_PORT,
)
from .linq_api import DEFAULT_API_BASE, LinqClient


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes linq ...` subcommands."""
    subs = parser.add_subparsers(dest="linq_command", required=False)

    p_setup = subs.add_parser("setup", help="Store Linq API token + from-phone")
    p_setup.add_argument("--token", default=None, help="Linq API token (omit to be prompted securely)")
    p_setup.add_argument("--phone", default=None, help="Your Linq from-phone (E.164, e.g. +15551234567)")

    subs.add_parser("status", help="Show credential state and probe connectivity")
    subs.add_parser("probe", help="Confirm the API token works (GET /phonenumbers)")

    p_hook = subs.add_parser("webhook", help="Webhook registration help")
    hook_subs = p_hook.add_subparsers(dest="linq_webhook_command", required=False)
    p_show = hook_subs.add_parser("show", help="Print the webhook URL to register in the Linq dashboard")
    p_show.add_argument("--public-url", default=None, help="Public base URL the gateway is reachable at")

    parser.set_defaults(func=dispatch)


# ---------------------------------------------------------------------------
# Dispatch

def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "linq_command", None)
    if sub is None:
        return _cmd_status(args)
    if sub == "setup":
        return _cmd_setup(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "probe":
        return _cmd_probe(args)
    if sub == "webhook":
        return _cmd_webhook(args)
    print(f"unknown subcommand: {sub}", file=sys.stderr)
    return 2


# ---------------------------------------------------------------------------
# Subcommand handlers

def _cmd_setup(args: argparse.Namespace) -> int:
    print("[1/2] Linq API token")
    token = args.token or os.getenv(linq_auth.ENV_TOKEN) or _prompt(
        "  Paste your Linq API token (from linqapp.com dashboard): ", secret=True
    )
    if not token:
        print("no token provided — aborting", file=sys.stderr)
        return 1

    print("[2/2] Linq from-phone (optional)")
    phone = args.phone or _prompt(
        "  Linq phone number this agent sends from (E.164, blank to skip): "
    )

    linq_auth.store_credentials(token, from_phone=phone or None)
    print(f"  ✓ saved to {linq_auth._auth_json_path()}")
    print()
    print("✓ Linq setup complete.")
    print("  Next: register a webhook URL in your Linq dashboard so inbound")
    print("        iMessages reach this gateway:")
    print(f"        hermes linq webhook show --public-url https://YOUR-PUBLIC-HOST")
    print("  Then start the gateway:")
    print("        hermes gateway start --platform linq")
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    linq_auth.print_credential_summary(print)
    return _cmd_probe(_args)


def _cmd_probe(_args: argparse.Namespace) -> int:
    token = linq_auth.load_token()
    if not token:
        print("  connectivity         : ✗ no token (run `hermes linq setup`)")
        return 1
    api_base = os.getenv("LINQ_API_BASE") or DEFAULT_API_BASE

    async def _run() -> int:
        async with LinqClient(token, api_base=api_base) as client:
            try:
                numbers = await client.list_phone_numbers(timeout=8.0)
            except Exception as exc:
                print(f"  connectivity         : ✗ {exc}")
                return 1
        count = len(numbers)
        print(f"  connectivity         : ✓ token valid ({count} number(s) on account)")
        return 0

    try:
        return asyncio.run(_run())
    except RuntimeError as exc:
        # Already inside an event loop (rare for a CLI) — fall back.
        print(f"  connectivity         : ? could not run probe: {exc}")
        return 1


def _cmd_webhook(args: argparse.Namespace) -> int:
    public = getattr(args, "public_url", None) or "https://YOUR-PUBLIC-HOST"
    path = os.getenv("LINQ_WEBHOOK_PATH") or _DEFAULT_WEBHOOK_PATH
    port = signing.coerce_port(os.getenv("LINQ_WEBHOOK_PORT"), _DEFAULT_WEBHOOK_PORT)
    bind = os.getenv("LINQ_WEBHOOK_BIND") or _DEFAULT_WEBHOOK_BIND
    url = f"{public.rstrip('/')}{path}"
    print("Linq webhook registration")
    print(f"  Local listener   : http://{bind}:{port}{path}")
    print(f"  Register this URL in your Linq dashboard:")
    print(f"      {url}")
    print()
    print("  Then export the signing secret your dashboard shows so the adapter")
    print("  can verify deliveries:")
    print("      export LINQ_WEBHOOK_SECRET=<secret-from-linq-dashboard>")
    return 0


# ---------------------------------------------------------------------------
# Gateway-setup entry point
#
# `hermes gateway setup` discovers platforms via the registry and calls each
# entry's zero-arg ``setup_fn``.  Linq registers this so it appears in the
# unified setup wizard alongside every other channel.

def gateway_setup() -> None:
    """Run Linq first-time setup from the `hermes gateway setup` wizard."""
    args = argparse.Namespace(linq_command="setup", token=None, phone=None)
    _cmd_setup(args)


# ---------------------------------------------------------------------------
# Small interactive helper

def _prompt(prompt: str, *, secret: bool = False) -> str:
    if not sys.stdin.isatty():
        return ""
    try:
        if secret:
            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""
