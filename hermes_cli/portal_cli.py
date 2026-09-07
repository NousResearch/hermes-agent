"""``hermes portal`` — the human-readable entry point for Nous Portal."""
from __future__ import annotations

import sys
import webbrowser

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config

DEFAULT_PORTAL_URL = "https://portal.nousresearch.com"
SUBSCRIPTION_URL = "https://portal.nousresearch.com/manage-subscription"
DOCS_URL = "https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway"
# Static `portal tools` catalog — the partners Tool Gateway routes to today: (key, label, partner).
_CATALOG = [
    ("web", "Web search & extract", "Firecrawl"),
    ("image_gen", "Image generation", "FAL"),
    ("tts", "Text-to-speech", "OpenAI TTS"),
    ("browser", "Browser automation", "Browser Use"),
    ("modal", "Cloud terminal", "Modal"),
]


def _feature_state(feat, *, via_nous: str) -> str:
    """Routing column shared by `portal info` and `portal tools`."""
    if feat.managed_by_nous:
        return color(via_nous, Colors.GREEN)
    if feat.active:
        return feat.current_provider or "active"
    return color("not configured", Colors.DIM)


def _heading(title: str) -> None:
    print()
    print(color(f"  {title}", Colors.MAGENTA))
    print(color("  " + "─" * len(title), Colors.MAGENTA))


def _cmd_status(args) -> int:
    """Show Portal auth + Tool Gateway routing summary."""
    from hermes_cli.auth import get_nous_auth_status_local
    from hermes_cli.nous_subscription import get_nous_subscription_features

    config = load_config() or {}
    try:
        auth = get_nous_auth_status_local() or {}  # refresh-free snapshot
    except Exception:
        auth = {}
    logged_in = bool(auth.get("logged_in"))
    _heading("Nous Portal")
    if logged_in:
        print(f"  Auth:    {color('✓ logged in', Colors.GREEN)}")
        print(f"  Portal:  {auth.get('portal_base_url') or DEFAULT_PORTAL_URL}")
        if auth.get("inference_base_url"):
            print(f"  API:     {auth['inference_base_url']}")
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

    _heading("Tool Gateway")
    try:
        features = get_nous_subscription_features(config)
    except Exception:
        print("  (could not resolve subscription state)")
        return 0
    rows = [(feat.label, _feature_state(feat, via_nous="via Nous Portal")) for feat in features.items()]
    width = max((len(r[0]) for r in rows), default=0)
    for label, state in rows:
        print(f"  {label:<{width}}   {state}")
    if not logged_in:
        print()
        print(color(f"  Docs: {DOCS_URL}", Colors.DIM))
    return 0


def _cmd_open(args) -> int:
    """Open the Portal subscription page in the default browser."""
    print(f"Opening {SUBSCRIPTION_URL}")
    try:
        opened = webbrowser.open(SUBSCRIPTION_URL)
    except Exception:
        opened = False
    if opened:
        return 0
    print()
    print("Could not launch a browser. Visit the URL above manually.")
    return 1


def _cmd_tools(args) -> int:
    """List the Tool Gateway catalog + current routing."""
    from hermes_cli.nous_subscription import get_nous_subscription_features

    config = load_config() or {}
    try:
        features = get_nous_subscription_features(config)
    except Exception:
        print("Could not resolve Tool Gateway state.", file=sys.stderr)
        return 1

    _heading("Tool Gateway catalog")
    if not features.nous_auth_present:
        print(color("  Not logged into Nous Portal — sign in with `hermes portal`.", Colors.YELLOW))
        print()

    label_width = max(len(label) for _, label, _ in _CATALOG)
    for key, label, partner in _CATALOG:
        feat = features.features.get(key)
        state = color("unknown", Colors.DIM) if feat is None else _feature_state(feat, via_nous="✓ via Nous Portal")
        print(f"  {label:<{label_width}}  partner: {partner:<14} {state}")

    print()
    print(color(f"  Manage your subscription: {SUBSCRIPTION_URL}", Colors.DIM))
    print(color(f"  Docs: {DOCS_URL}", Colors.DIM))
    return 0


def _cmd_login(args) -> int:
    """One-shot Nous Portal onboarding (login + model + provider + tools).

    Reuses the exact wiring behind ``hermes setup --portal`` so the commands stay in lockstep.
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


# Default (None/"") is the one-shot onboarding (alias for `hermes auth add nous --type oauth` /
# `hermes setup --portal`). `status` kept as a back-compat alias for `info`.
_SUBCOMMANDS = {
    None: _cmd_login,
    "": _cmd_login,
    "login": _cmd_login,
    "info": _cmd_status,
    "status": _cmd_status,
    "open": _cmd_open,
    "tools": _cmd_tools,
}


def portal_command(args) -> int:
    """Top-level dispatch for `hermes portal <subcommand>`."""
    sub = getattr(args, "portal_command", None)
    handler = _SUBCOMMANDS.get(sub)
    if handler is not None:
        return handler(args)
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
            "Subcommands: login (default), info, open, tools."
        ),
    )
    portal_sub = portal_parser.add_subparsers(dest="portal_command")

    # `status` is a hidden (no help) back-compat alias; registration order = `hermes portal -h` order.
    for name, help_text in (
        ("login", "Log in to Nous Portal + set it up (default; one-shot onboarding)"),
        ("info", "Show Portal auth + Tool Gateway routing summary"),
        ("status", None),
        ("open", "Open the Portal subscription page in your default browser"),
        ("tools", "List Tool Gateway tools and which are routed via Nous"),
    ):
        portal_sub.add_parser(name, **({} if help_text is None else {"help": help_text}))

    portal_parser.set_defaults(func=portal_command)
