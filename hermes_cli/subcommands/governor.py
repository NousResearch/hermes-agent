"""``hermes governor`` subcommand parser: certified context-engine key lifecycle."""

from __future__ import annotations

from typing import Callable


def build_governor_parser(subparsers, *, cmd_governor: Callable) -> None:
    """Attach the ``governor`` subcommand group."""
    parser = subparsers.add_parser(
        "governor",
        help="Certified context engine (ri-context-governor) key lifecycle",
        description=(
            "Initialize, inspect, and rotate the governed HMAC key used by the "
            "receipt-carrying context engine (exact-fallback recovery + lineage).\n\n"
            "Binary: cargo install context-governor (>=0.2.0).\n"
            "Activate: context.engine: ri-context-governor in config.yaml."
        ),
    )
    sub = parser.add_subparsers(dest="governor_command")
    sub.add_parser("init", help="Provision the governed key (idempotent first install)")
    sub.add_parser("status", help="Show binary, key binding, and engine availability")
    sub.add_parser("rotate", help="Retire the active key and bind a fresh one")
    parser.set_defaults(func=cmd_governor)