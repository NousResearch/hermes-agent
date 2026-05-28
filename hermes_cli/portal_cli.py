"""``hermes portal`` — small CLI surface for Nous Portal users.

Subcommands:
  status   Show Portal auth state + which Tool Gateway tools are routed.
  open     Open the Portal subscription page in the user's default browser.
  tools    List Tool Gateway tools and which are active in the current config.

This command is intentionally minimal — it does not duplicate functionality
already in ``hermes auth`` or ``hermes tools``. It's a discovery + status
surface for the Portal subscription itself.
"""
from __future__ import annotations

import sys
import webbrowser
from typing import Optional

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config

DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
SUBSCRIPTION_URL = "https://portal.nousresearch.com/manage-subscription"
DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway"


def _nous_portal_base_url() -> str:
    """Resolve the Portal base URL from auth state or default."""
    try:
        from hermes_cli.auth import get_nous_auth_status
        status = get_nous_auth_status() or {}
        url = status.get("portal_base_url")
        if isinstance(url, str) and url.strip():
            return url.rstrip("/")
    except Exception:
        pass
    return DEFAULT_PORTAL_URL


def _cmd_status(args) -> int:
    """Show Portal auth + Tool Gateway routing summary."""
    from hermes_cli.auth import get_nous_auth_status
    from hermes_cli.nous_subscription import get_nous_subscription_features

    config = load_config() or {}

    try:
        auth = get_nous_auth_status() or {}
    except Exception:
        auth = {}

    logged_in = bool(auth.get("logged_in"))

    print()
    print(color("  Nous Portal", Colors.MAGENTA))
    print(color("  ───────────", Colors.MAGENTA))
    if logged_in:
        portal = auth.get("portal_base_url") or DEFAULT_PORTAL_URL
        print(f"  Auth:    {color('✓ logged in', Colors.GREEN)}")
        print(f"  Portal:  {portal}")
        inference = auth.get("inference_base_url")
        if inference:
            print(f"  API:     {inference}")
    else:
        print(f"  Auth:    {color('not logged in', Colors.YELLOW)}")
        print(f"  Sign up: {SUBSCRIPTION_URL}")
        print(f"  Login:   hermes auth add nous --type oauth")

    # Provider selection (independent of auth)
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    provider = str(model_cfg.get("provider") or "").strip().lower()
    if provider == "nous":
        print(f"  Model:   {color('✓ using Nous as inference provider', Colors.GREEN)}")
    elif provider:
        print(f"  Model:   currently {provider} (switch with `hermes model`)")

    # Tool Gateway routing
    print()
    print(color("  Tool Gateway", Colors.MAGENTA))
    print(color("  ────────────", Colors.MAGENTA))
    try:
        features = get_nous_subscription_features(config)
    except Exception:
        features = None

    if features is None:
        print("  (could not resolve subscription state)")
        return 0

    rows = []
    for feat in features.items():
        if feat.managed_by_nous:
            state = color("via Nous Portal", Colors.GREEN)
        elif feat.active and feat.current_provider:
            state = feat.current_provider
        elif feat.active:
            state = "active"
        else:
            state = color("not configured", Colors.DIM)
        rows.append((feat.label, state))

    width = max((len(r[0]) for r in rows), default=0)
    for label, state in rows:
        print(f"  {label:<{width}}   {state}")

    if not logged_in:
        print()
        print(color(f"  Docs: {DOCS_URL}", Colors.DIM))
    return 0


def _cmd_open(args) -> int:
    """Open the Portal subscription page in the default browser."""
    target = SUBSCRIPTION_URL
    print(f"Opening {target}")
    try:
        opened = webbrowser.open(target)
    except Exception:
        opened = False
    if not opened:
        print()
        print("Could not launch a browser. Visit the URL above manually.")
        return 1
    return 0


def _cmd_tools(args) -> int:
    """List the Tool Gateway catalog + current routing."""
    from hermes_cli.nous_subscription import get_nous_subscription_features

    config = load_config() or {}
    try:
        features = get_nous_subscription_features(config)
    except Exception:
        print("Could not resolve Tool Gateway state.", file=sys.stderr)
        return 1

    # Static catalog — the partners Tool Gateway routes to today.
    catalog = [
        ("web",       "Web search & extract",  "Firecrawl"),
        ("image_gen", "Image generation",      "FAL"),
        ("tts",       "Text-to-speech",        "OpenAI TTS"),
        ("browser",   "Browser automation",    "Browser Use"),
        ("modal",     "Cloud terminal",        "Modal"),
    ]

    print()
    print(color("  Tool Gateway catalog", Colors.MAGENTA))
    print(color("  ────────────────────", Colors.MAGENTA))

    if not features.nous_auth_present:
        print(color("  Not logged into Nous Portal — sign in with `hermes auth add nous --type oauth`.", Colors.YELLOW))
        print()

    label_width = max(len(label) for _, label, _ in catalog)
    for key, label, partner in catalog:
        feat = features.features.get(key)
        if feat is None:
            state = color("unknown", Colors.DIM)
        elif feat.managed_by_nous:
            state = color("✓ via Nous Portal", Colors.GREEN)
        elif feat.active and feat.current_provider:
            state = feat.current_provider
        elif feat.active:
            state = "active"
        else:
            state = color("not configured", Colors.DIM)
        print(f"  {label:<{label_width}}  partner: {partner:<14} {state}")

    print()
    print(color(f"  Manage your subscription: {SUBSCRIPTION_URL}", Colors.DIM))
    print(color(f"  Docs: {DOCS_URL}", Colors.DIM))
    return 0


def portal_command(args) -> int:
    """Top-level dispatch for `hermes portal <subcommand>`."""
    sub = getattr(args, "portal_command", None)
    if sub in {None, ""}:
        # Default to status — matches gh / kubectl conventions where the
        # subcommand-less form gives a useful overview.
        return _cmd_status(args)
    if sub == "status":
        return _cmd_status(args)
    if sub == "open":
        return _cmd_open(args)
    if sub == "tools":
        return _cmd_tools(args)
    print(f"Unknown portal subcommand: {sub}", file=sys.stderr)
    print("Run `hermes portal -h` for usage.", file=sys.stderr)
    return 1


def add_parser(subparsers) -> None:
    """Register `hermes portal` on the given argparse subparsers object."""
    portal_parser = subparsers.add_parser(
        "portal",
        help="Nous Portal status, subscription, and Tool Gateway routing",
        description=(
            "Inspect Nous Portal auth, Tool Gateway routing, and open the "
            "Portal subscription page. Subcommands: status (default), "
            "open, tools."
        ),
    )
    portal_sub = portal_parser.add_subparsers(dest="portal_command")

    portal_sub.add_parser(
        "status",
        help="Show Portal auth + Tool Gateway routing summary (default)",
    )
    portal_sub.add_parser(
        "open",
        help="Open the Portal subscription page in your default browser",
    )
    portal_sub.add_parser(
        "tools",
        help="List Tool Gateway tools and which are routed via Nous",
    )

    portal_parser.set_defaults(func=portal_command)
