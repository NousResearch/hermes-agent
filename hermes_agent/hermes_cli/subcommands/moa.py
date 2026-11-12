"""``hermes moa`` subcommand parser."""

from __future__ import annotations


def build_moa_parser(subparsers) -> None:
    """Attach the ``moa`` subcommand to ``subparsers``."""
    from hermes_cli.moa_cmd import cmd_moa

    moa_parser = subparsers.add_parser(
        "moa", help="Configure Mixture of Agents provider/model slots",
        description="Configure the provider/model set used by /moa <prompt>.")
    moa_subparsers = moa_parser.add_subparsers(dest="moa_command")
    moa_subparsers.add_parser("list", aliases=["ls"], help="Show current MoA model slots")
    moa_configure = moa_subparsers.add_parser("configure", aliases=["config"], help="Interactively pick MoA models")
    moa_configure.add_argument("name", nargs="?", help="Preset name to create or update")
    moa_delete = moa_subparsers.add_parser("delete", aliases=["rm"], help="Delete a MoA preset")
    moa_delete.add_argument("name", help="Preset name to delete")
    moa_parser.set_defaults(func=cmd_moa)
