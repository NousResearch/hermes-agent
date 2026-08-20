"""``hermes portal`` — the human-readable entry point for Nous Portal.

Running ``hermes portal`` with no subcommand performs the one-shot Portal
onboarding: OAuth login, pick a Nous model, switch the inference provider to
Nous, and offer to enable the Tool Gateway. It is the friendly alias for
``hermes auth add nous --type oauth`` (which still works), is identical to
``hermes setup --portal``, and runs the same Nous flow as the first-time quick
setup.

Subcommands:
  (none)   Log in to Nous Portal + set it up (one-shot onboarding).
  login    Explicit alias for the default one-shot onboarding.
  info     Show Portal auth state + which Tool Gateway tools are routed.
  open     Open the Portal subscription page in the user's default browser.
  tools    List Tool Gateway tools and which are active in the current config.
  usage    Show sanitized balance/allowance telemetry; add --json for automation.

This command is intentionally minimal — it does not duplicate functionality
already in ``hermes auth`` or ``hermes tools``. It's the onboarding + discovery
surface for the Portal subscription itself.
"""
from __future__ import annotations

import json
import sys
import webbrowser

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config
from agent.billing_usage import build_usage_model

DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
SUBSCRIPTION_URL = "https://portal.nousresearch.com/manage-subscription"
DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway"


def _cmd_status(args) -> int:
    """Show Portal auth + Tool Gateway routing summary."""
    from hermes_cli.auth import get_nous_auth_status_local
    from hermes_cli.nous_subscription import get_nous_subscription_features

    config = load_config() or {}

    try:
        # Read-only status display: refresh-free snapshot (no OAuth refresh).
        auth = get_nous_auth_status_local() or {}
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
        print("  Login:   hermes portal")

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


_USAGE_SCHEMA_VERSION = 1


def build_usage_payload(model) -> dict[str, object]:
    """Return the stable, sanitized ``portal usage --json`` payload.

    The payload deliberately exposes only account-level display values. It never
    serializes raw Portal responses, identifiers, OAuth credentials, tool state,
    or inference routes. A percentage is emitted only when Portal supplied a
    positive monthly allowance; top-ups have no denominator and remain a dollar
    balance rather than a fabricated percentage.
    """
    unavailable = not bool(model and getattr(model, "available", False))
    if unavailable:
        return {
            "schema_version": _USAGE_SCHEMA_VERSION,
            "available": False,
            "status": "unavailable",
            "plan": None,
            "renews_at": None,
            "subscription": None,
            "top_up": None,
            "total_usable_usd": None,
        }

    plan_bar = getattr(model, "plan_bar", None)
    subscription = None
    if plan_bar is not None:
        monthly_allowance = _finite_number(getattr(plan_bar, "total_usd", None))
        remaining = _finite_number(getattr(plan_bar, "remaining_usd", None))
        used_percent = getattr(plan_bar, "pct_used", None)
        if monthly_allowance is not None and monthly_allowance > 0 and remaining is not None:
            subscription = {
                "remaining_usd": remaining,
                "monthly_allowance_usd": monthly_allowance,
                "used_percent": used_percent if isinstance(used_percent, int) else None,
            }

    topup_remaining = _finite_number(getattr(model, "topup_remaining_usd", None))
    top_up = {"remaining_usd": topup_remaining} if topup_remaining is not None else None

    return {
        "schema_version": _USAGE_SCHEMA_VERSION,
        "available": True,
        "status": getattr(model, "status", None) or "unknown",
        "plan": _clean_text(getattr(model, "plan_name", None)),
        "renews_at": _clean_text(getattr(model, "renews_at", None)),
        "subscription": subscription,
        "top_up": top_up,
        "total_usable_usd": _finite_number(getattr(model, "total_spendable_usd", None)),
    }


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    import math

    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _clean_text(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _cmd_usage(args) -> int:
    """Print safe Portal balance/allowance telemetry without exposing OAuth state."""
    model = build_usage_model(timeout=8.0)
    payload = build_usage_payload(model)
    if getattr(args, "json", False):
        print(json.dumps(payload, separators=(",", ":"), sort_keys=True))
    elif not payload["available"]:
        print("Nous Portal usage is unavailable. Log in with `hermes portal` and try again.")
    else:
        print("Nous Portal usage")
        if payload["plan"]:
            print(f"Plan: {payload['plan']}")
        if payload["subscription"]:
            subscription = payload["subscription"]
            print(
                "Subscription allowance: "
                f"${subscription['remaining_usd']:.2f} remaining of "
                f"${subscription['monthly_allowance_usd']:.2f}"
            )
        if payload["top_up"]:
            print(f"Top-up balance: ${payload['top_up']['remaining_usd']:.2f}")
        if payload["total_usable_usd"] is not None:
            print(f"Total usable: ${payload['total_usable_usd']:.2f}")
        if payload["renews_at"]:
            print(f"Renews: {payload['renews_at']}")
    return 0 if payload["available"] else 1


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
        print(color("  Not logged into Nous Portal — sign in with `hermes portal`.", Colors.YELLOW))
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


def _cmd_login(args) -> int:
    """Run the one-shot Nous Portal onboarding (login + model + provider + tools).

    This is the human-readable front door for `hermes auth add nous --type
    oauth`. It reuses the exact wiring behind `hermes setup --portal` (which in
    turn runs the same Nous flow as the first-time quick setup), so the
    commands stay in lockstep: device-code login, pick a Nous model, switch the
    inference provider to Nous, then offer the Tool Gateway opt-in.
    """
    from hermes_cli.setup import _run_portal_one_shot

    config = load_config() or {}
    try:
        _run_portal_one_shot(config)
    except (KeyboardInterrupt, EOFError):
        print()
        print("Portal setup cancelled.")
        return 1
    return 0


def portal_command(args) -> int:
    """Top-level dispatch for `hermes portal <subcommand>`."""
    sub = getattr(args, "portal_command", None)
    if sub in {None, "", "login"}:
        # Default to the one-shot onboarding — `hermes portal` is the
        # human-readable alias for `hermes auth add nous --type oauth` /
        # `hermes setup --portal`.
        return _cmd_login(args)
    if sub in {"info", "status"}:
        # `status` kept as a back-compat alias for the prior default.
        return _cmd_status(args)
    if sub == "open":
        return _cmd_open(args)
    if sub == "tools":
        return _cmd_tools(args)
    if sub == "usage":
        return _cmd_usage(args)
    print(f"Unknown portal subcommand: {sub}", file=sys.stderr)
    print("Run `hermes portal -h` for usage.", file=sys.stderr)
    return 1


def add_parser(subparsers) -> None:
    """Register `hermes portal` on the given argparse subparsers object."""
    portal_parser = subparsers.add_parser(
        "portal",
        help="Set up Nous Portal (login, model pick, Tool Gateway); see also `portal info`",
        description=(
            "Run `hermes portal` with no subcommand to log in to Nous Portal "
            "and set it up — pick a model, set Nous as your provider, and offer "
            "the Tool Gateway (the human-readable alias for `hermes auth add "
            "nous --type oauth`, identical to `hermes setup --portal`). "
            "Subcommands: login (default), info, open, tools, usage."
        ),
    )
    portal_sub = portal_parser.add_subparsers(dest="portal_command")

    portal_sub.add_parser(
        "login",
        help="Log in to Nous Portal + set it up (default; one-shot onboarding)",
    )
    portal_sub.add_parser(
        "info",
        help="Show Portal auth + Tool Gateway routing summary",
    )
    # `status` retained as a hidden back-compat alias for `info`.
    portal_sub.add_parser("status")
    portal_sub.add_parser(
        "open",
        help="Open the Portal subscription page in your default browser",
    )
    portal_sub.add_parser(
        "tools",
        help="List Tool Gateway tools and which are routed via Nous",
    )
    usage_parser = portal_sub.add_parser(
        "usage",
        help="Show Portal balance/allowance telemetry (use --json for scripts)",
    )
    usage_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the stable, sanitized JSON usage contract",
    )

    portal_parser.set_defaults(func=portal_command)
