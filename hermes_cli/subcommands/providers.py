"""``hermes providers`` subcommand parser (handler: :mod:`hermes_cli.providers_cmd`)."""

from __future__ import annotations


def _cmd_providers(args) -> int:
    from hermes_cli.providers_cmd import cmd_providers

    return cmd_providers(args)


def build_providers_parser(subparsers) -> None:
    """Register ``hermes providers list``."""
    providers_parser = subparsers.add_parser(
        "providers",
        help="List inference providers and their models without opening the picker",
        description="List inference providers in a scriptable format.",
    )
    providers_subparsers = providers_parser.add_subparsers(dest="providers_action")
    providers_subparsers.required = True
    providers_list = providers_subparsers.add_parser("list", aliases=["ls"], help="List inference providers")
    providers_list.add_argument("--provider", metavar="NAME", help="Limit output to one provider id")
    providers_list.add_argument("--all", action="store_true", help="Include supported but unconfigured providers")
    providers_list.add_argument(
        "--offline", "--no-live", dest="offline", action="store_true",
        help="Use local configuration and bundled catalogs only; never contact a provider")
    providers_list.add_argument("--json", action="store_true", help="Print one JSON document (schema_version 1)")
    providers_list.set_defaults(func=_cmd_providers)
