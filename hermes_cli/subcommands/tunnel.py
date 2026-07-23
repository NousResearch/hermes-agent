"""``hermes tunnel`` subcommand parser builder.

Exposes user-built local services to the internet on per-user noit2.com
subdomains via a Cloudflare named tunnel, with an idle-reset 30-minute
dead-man's switch and admin-approved hold-open. Handler ``cmd_tunnel``
lives in ``hermes_cli/main.py`` and is injected here.
"""

from __future__ import annotations

from typing import Callable


def build_tunnel_parser(subparsers, *, cmd_tunnel: Callable) -> None:
    tunnel_parser = subparsers.add_parser(
        "tunnel",
        help="Expose a local service to the internet via a Cloudflare Tunnel",
        description=(
            "Expose a user-built local app/API to the internet on a per-user "
            "noit2.com subdomain via Cloudflare Tunnel. Ephemeral by default: "
            "an idle-reset 30-minute dead-man's switch closes the tunnel when "
            "traffic stops. Use 'hold' + admin 'approve' for longer exposure."
        ),
    )
    sub = tunnel_parser.add_subparsers(dest="tunnel_command")

    up = sub.add_parser("up", help="Start the tunnel for one or more origins")
    up.add_argument("--origin", action="append", dest="origins", default=[],
                    metavar="SUB=HOST:PORT",
                    help="e.g. --origin alice=127.0.0.1:3000 (repeatable)")
    up.add_argument("--hold-request", action="store_true", dest="hold_request",
                    help="File a hold-open request immediately on start")
    up.add_argument("--reason", default="", help="Reason for the hold request")
    up.add_argument("--until", default="", help="Requested hold duration, e.g. 4h")
    up.set_defaults(func=cmd_tunnel)

    down = sub.add_parser("down", help="Stop the running tunnel")
    down.add_argument("--kill-origins", action="store_true",
                      help="Also stop the local origin services (default: leave them running)")
    down.set_defaults(func=cmd_tunnel)

    for name, help_ in (("status", "Show running tunnel state"),
                        ("doctor", "Health-check cloudflared, creds, origins, DNS"),
                        ("requests", "List pending hold requests")):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(func=cmd_tunnel)

    hold = sub.add_parser("hold", help="File a hold-open request for the running tunnel")
    hold.add_argument("--reason", default="")
    hold.add_argument("--until", default="")
    hold.set_defaults(func=cmd_tunnel)

    approve = sub.add_parser("approve", help="Approve a hold request (admin)")
    approve.add_argument("id", nargs="?", help="Hold request id")
    approve.add_argument("--until", default="", help="Approved duration, e.g. 6h")
    approve.set_defaults(func=cmd_tunnel)

    deny = sub.add_parser("deny", help="Deny a hold request (admin)")
    deny.add_argument("id", nargs="?", help="Hold request id")
    deny.add_argument("--reason", default="")
    deny.set_defaults(func=cmd_tunnel)

    tunnel_parser.set_defaults(func=cmd_tunnel)
