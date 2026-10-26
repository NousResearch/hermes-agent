"""``hermes login`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_login_parser(subparsers, *, cmd_login: Callable) -> None:
    """Attach the deprecated ``login`` subcommand (handler only prints a deprecation notice).

    Kept registered so old scripts get the actionable message instead of argparse's
    ``invalid choice``. Registered WITHOUT ``help=`` so it is omitted from ``hermes --help``
    (``help=SUPPRESS`` leaks ``==SUPPRESS==`` for top-level subparsers on 3.12+). ``--provider``
    takes ANY value (no ``choices=``) so the handler is reached rather than argparse erroring.

    This hides a command that no longer works (#24756) without the ``help=argparse.SUPPRESS``
    ``==SUPPRESS==`` leak that argparse emits for a top-level subparser on Python 3.12+.
    """
    login_parser = subparsers.add_parser(
        "login",
        description="Deprecated. Use `hermes auth` to manage credentials, "
            "`hermes model` to select a provider, or `hermes setup` for full setup.")
    # No ``choices=`` on purpose — the handler is a deprecation notice that
    # ignores the value, and a restrictive list would reject providers the user
    # legitimately wants (e.g. ``anthropic``) with an argparse error before the
    # friendly redirect message is ever printed.
    login_parser.add_argument(
        "--provider", default=None, help="(deprecated) Provider name; ignored — see `hermes model`")
    login_parser.add_argument("--portal-url", help="Portal base URL (default: production portal)")
    login_parser.add_argument(
        "--inference-url", help="Inference API base URL (default: production inference API)")
    login_parser.add_argument(
        "--client-id", default=None, help="OAuth client id to use (default: hermes-cli)")
    login_parser.add_argument("--scope", default=None, help="OAuth scope to request")
    login_parser.add_argument(
        "--no-browser", action="store_true",
        help="Do not attempt to open the browser automatically")
    login_parser.add_argument(
        "--timeout", type=float, default=15.0, help="HTTP request timeout in seconds (default: 15)")
    login_parser.add_argument("--ca-bundle", help="Path to CA bundle PEM file for TLS verification")
    login_parser.add_argument(
        "--insecure", action="store_true", help="Disable TLS verification (testing only)")
    login_parser.set_defaults(func=cmd_login)
