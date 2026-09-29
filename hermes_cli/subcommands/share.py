"""``hermes share`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_share_parser(subparsers, *, cmd_share: Callable) -> None:
    """Attach ``share``: connection codes and paired devices for `hermes serve --share tailcat`."""
    parser = subparsers.add_parser(
        "share", help="Pair devices with this backend over tailcat",
        description="Manage sharing this Hermes backend over tailcat. Start sharing with "
            "`hermes serve --share tailcat` (or dashboard.share: tailcat), then give a "
            "device a one-time connection code.")
    sub = parser.add_subparsers(dest="share_action")
    sub.add_parser("status", help="Show the share address and paired devices")
    sub.add_parser("code", help="Print a one-time connection code (valid 5 minutes)")
    sub.add_parser("devices", help="List paired devices")
    revoke = sub.add_parser("revoke", help="Revoke a paired device")
    revoke.add_argument("device_id", help="Device id from `hermes share devices`")
    reset = sub.add_parser(
        "reset", help="New tailcat address; forget every paired device and code")
    reset.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    parser.set_defaults(func=cmd_share)
