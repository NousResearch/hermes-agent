"""``hermes usage`` account-limit snapshot command."""

from __future__ import annotations

from typing import Callable


def build_usage_parser(subparsers, *, cmd_usage: Callable) -> None:
    parser = subparsers.add_parser(
        "usage",
        help="Show provider-reported account limits",
        description="Fetch subscription/account limits for the active or selected provider",
    )
    parser.add_argument(
        "--provider",
        help="Provider to inspect (defaults to the active runtime provider)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a stable machine-readable snapshot",
    )
    parser.add_argument(
        "--output",
        help="Atomically write the JSON snapshot to this local path",
    )
    parser.set_defaults(func=cmd_usage)
