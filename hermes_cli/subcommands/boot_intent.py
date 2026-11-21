"""``hermes boot-intent``: register a next step to run after a planned gateway restart.

The intent is consumed once at boot by ``GatewayRunner._schedule_boot_intents``. A planned
restart is the only covered case — a crash or OOM has no marker and behaves as before.

Write the prompt as *precondition + step + stop point*, not as a bare action: a bare action
re-runs work that already finished before the restart.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime


def _parse_ttl(raw: str) -> float:
    text = str(raw).strip().lower()
    mult = 1.0
    if text.endswith("h"):
        mult, text = 3600.0, text[:-1]
    elif text.endswith("m"):
        mult, text = 60.0, text[:-1]
    elif text.endswith("s"):
        mult, text = 1.0, text[:-1]
    try:
        value = float(text) * mult
    except ValueError:
        raise SystemExit(f"boot-intent: invalid --ttl {raw!r} (use e.g. 24h, 90m, 3600)")
    if value <= 0:
        raise SystemExit("boot-intent: --ttl must be positive")
    return value


def cmd_boot_intent_add(args) -> int:
    from gateway.boot_intents import add_intent

    session_key = (args.session_key or os.environ.get("HERMES_SESSION_KEY") or "").strip()
    if not session_key:
        print(
            "boot-intent: no --session-key given and HERMES_SESSION_KEY is unset. "
            "Pass --session-key explicitly.",
            file=sys.stderr,
        )
        return 2
    try:
        item = add_intent(
            session_key,
            args.prompt,
            profile=args.profile,
            ttl_seconds=_parse_ttl(args.ttl),
        )
    except ValueError as exc:
        print(f"boot-intent: {exc}", file=sys.stderr)
        return 2
    print(f"registered {item['id']} -> {session_key} (ttl={args.ttl})")
    return 0


def cmd_boot_intent_list(args) -> int:
    from gateway.boot_intents import list_intents

    items = list_intents(include_consumed=args.all)
    if not items:
        print("no boot intents")
        return 0
    for item in items:
        created = datetime.fromtimestamp(float(item.get("created_at") or 0)).strftime("%Y-%m-%d %H:%M")
        if item.get("consumed_at"):
            state = "done:" + str(item.get("result") or "?")
        elif item.get("claimed_at"):
            state = "claimed"
        else:
            state = "pending"
        print(f"{item.get('id', '?')}  {state:16s}  {created}  {item.get('session_key', '?')}")
        prompt = " ".join(str(item.get("prompt") or "").split())
        print(f"    {prompt[:300]}")
    return 0


def cmd_boot_intent_clear(args) -> int:
    from gateway.boot_intents import clear_intents

    removed = clear_intents(intent_id=args.id)
    print(f"cleared {removed} intent(s)")
    return 0


def build_boot_intent_parser(subparsers) -> None:
    """Attach the ``boot-intent`` subcommand to ``subparsers``."""
    parser = subparsers.add_parser(
        "boot-intent",
        help="Register a next step to run automatically after a planned gateway restart",
        description=(
            "Boot intents make the gateway continue work after a planned restart without "
            "waiting for the user to send a message. The intent is claimed once at startup, "
            "run as an internal turn, then marked consumed. Only planned restarts are covered."
        ),
    )
    sub = parser.add_subparsers(dest="boot_intent_command", required=True)

    add = sub.add_parser("add", help="Register an intent for the next boot")
    add.add_argument(
        "--prompt", "-p", required=True,
        help="Precondition + step + stop point, e.g. 'config already edited; verify PID changed, "
             "grep ERROR in last 50 log lines; report and stop'",
    )
    add.add_argument("--session-key", default=None, help="Defaults to $HERMES_SESSION_KEY")
    add.add_argument("--profile", default=None, help="Owning profile (audit only)")
    add.add_argument("--ttl", default="24h", help="Expiry, e.g. 24h / 90m / 3600 (default 24h)")
    add.set_defaults(func=cmd_boot_intent_add)

    listing = sub.add_parser("list", help="List intents")
    listing.add_argument("--all", action="store_true", help="Include consumed and expired rows")
    listing.set_defaults(func=cmd_boot_intent_list)

    clear = sub.add_parser("clear", help="Remove all intents (or one with --id)")
    clear.add_argument("--id", default=None, help="Remove just this intent id")
    clear.set_defaults(func=cmd_boot_intent_clear)
