"""``hermes fallback`` subcommand parser."""

from __future__ import annotations


def build_fallback_parser(subparsers) -> None:
    """Attach the ``fallback`` subcommand to ``subparsers``."""
    from hermes_cli.fallback_cmd import cmd_fallback

    fallback_parser = subparsers.add_parser(
        "fallback", help="Manage fallback providers (tried when the primary model fails)",
        description="Manage fallback providers and automatic or manual activation "
            "when the primary model fails. See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers")
    fallback_subparsers = fallback_parser.add_subparsers(dest="fallback_command")
    fallback_subparsers.add_parser(
        "list", aliases=["ls"], help="Show the current fallback chain (default when no subcommand)")
    fallback_subparsers.add_parser(
        "add",
        help="Pick a provider + model (same picker as `hermes model`) and append to the chain")
    fallback_subparsers.add_parser(
        "remove", aliases=["rm"], help="Pick an entry to delete from the chain")
    fallback_subparsers.add_parser("clear", help="Remove all fallback entries")
    auto_parser = fallback_subparsers.add_parser("auto", help="Enable or disable automatic fallback")
    auto_parser.add_argument("fallback_auto", choices=("on", "off"), help="off requires an interactive choice each turn")
    fallback_parser.set_defaults(func=cmd_fallback)
