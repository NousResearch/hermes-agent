"""``rabbit login`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_login_parser(subparsers, *, cmd_login: Callable) -> None:
    """Attach the deprecated ``login`` subcommand (handler only prints a deprecation notice).

    Kept registered so old scripts get the actionable message instead of argparse's
    ``invalid choice``. Registered WITHOUT ``help=`` so it is omitted from ``rabbit --help``
    (``help=SUPPRESS`` leaks ``==SUPPRESS==`` for top-level subparsers on 3.12+). ``--provider``
    takes ANY value (no ``choices=``) so the handler is reached rather than argparse erroring.

    This hides a command that no longer works (#24756) without the ``help=argparse.SUPPRESS``
    ``==SUPPRESS==`` leak that argparse emits for a top-level subparser on Python 3.12+.
    """
    login_parser = subparsers.add_parser(
        "login",
        description="Deprecated. Use `rabbit auth` to manage credentials, "
            "`rabbit model` to select a provider, or `rabbit setup` for full setup.")
    # No ``choices=`` on purpose — the handler is a deprecation notice that
    # ignores the value, and a restrictive list would reject providers the user
    # legitimately wants (e.g. ``anthropic``) with an argparse error before the
    # friendly redirect message is ever printed.
    login_parser.add_argument(
        "--provider", default=None, help="(deprecated) Provider name; ignored — see `rabbit model`")
    login_parser.set_defaults(func=cmd_login)
