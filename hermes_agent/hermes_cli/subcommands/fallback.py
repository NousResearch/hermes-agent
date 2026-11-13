"""``hermes fallback`` subcommand parser."""

from __future__ import annotations


def build_fallback_parser(subparsers) -> None:
    """Attach the ``fallback`` subcommand to ``subparsers``."""
    from hermes_agent.hermes_cli.fallback_cmd import cmd_fallback
    from hermes_agent.hermes_cli.subcommands._shared import add_json_flag

    fallback_parser = subparsers.add_parser(
        "fallback", help="Manage fallback providers (tried when the primary model fails)",
        description="Manage the fallback provider chain.  Fallback providers are tried "
            "in order when the primary model fails with rate-limit, overload, or "
            "connection errors.  See: "
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
    check = fallback_subparsers.add_parser(
        "check", help="Check each fallback entry is usable, without starting an agent session (exit 1 if not)")
    check.add_argument("--live", action="store_true",
                       help="Also send one minimal inference request per entry (on that entry's own provider/model)")
    add_json_flag(check, "Print the results as JSON")
    fallback_parser.set_defaults(func=cmd_fallback)
