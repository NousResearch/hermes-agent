"""``hermes fallback`` subcommand parser."""

from __future__ import annotations


def build_fallback_parser(subparsers) -> None:
    """Attach the ``fallback`` subcommand to ``subparsers``."""
    from hermes_cli.fallback_cmd import cmd_fallback

    fallback_parser = subparsers.add_parser(
        "fallback", help="Manage fallback providers (tried when the primary model fails)",
        description="Manage the fallback provider chain.  Fallback providers are tried "
            "in order when the primary model fails with rate-limit, overload, or "
            "connection errors.  See: "
            "https://hermes-agent.nousresearch.com/docs/user-guide/features/fallback-providers")
    fallback_subparsers = fallback_parser.add_subparsers(dest="fallback_command")
    fallback_subparsers.add_parser(
        "list", aliases=["ls"], help="Show the current fallback chain (default when no subcommand)")
    fallback_add = fallback_subparsers.add_parser(
        "add",
        help="Pick a provider + model, or pass --provider and --model, then append to the chain")
    fallback_add.add_argument("--provider", dest="fallback_provider", metavar="PROVIDER",
                              help="Provider for a non-interactive fallback entry")
    fallback_add.add_argument("--model", dest="fallback_model", metavar="MODEL",
                              help="Model for a non-interactive fallback entry")
    fallback_add.add_argument("--base-url", dest="fallback_base_url", metavar="URL",
                              help="Optional endpoint URL for a non-interactive fallback entry")
    fallback_add.add_argument("--api-mode", dest="fallback_api_mode", metavar="MODE",
                              help="Optional API mode for a non-interactive fallback entry")
    fallback_subparsers.add_parser(
        "remove", aliases=["rm"], help="Pick an entry to delete from the chain")
    fallback_subparsers.add_parser("clear", help="Remove all fallback entries")
    fallback_parser.set_defaults(func=cmd_fallback)
