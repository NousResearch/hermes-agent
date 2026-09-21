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
    moa_configure = moa_subparsers.add_parser(
        "configure", aliases=["config"],
        help="Pick MoA models interactively, or declare them with --slots")
    moa_configure.add_argument("name", nargs="?", help="Preset name to create or update")
    # Declarative path (#102265): a preset is deployment state, not picker state, so fleet setups
    # must be able to converge it from a file. Any of these flags skips the picker entirely.
    moa_configure.add_argument(
        "--slots", metavar="PROVIDER/MODEL[,PROVIDER/MODEL...]",
        help="Reference models to declare in the preset (skips the picker)")
    moa_configure.add_argument(
        "--slots-file", dest="slots_file", metavar="PATH",
        help="JSON file declaring the preset: a list of 'provider/model' strings, or an object "
             "with 'reference_models' and an optional 'aggregator'")
    moa_configure.add_argument(
        "--aggregator", metavar="PROVIDER/MODEL",
        help="Aggregator (acting) model; required to create a preset non-interactively, while an "
             "existing preset keeps its current aggregator when it is omitted")
    moa_delete = moa_subparsers.add_parser("delete", aliases=["rm"], help="Delete a MoA preset")
    moa_delete.add_argument("name", help="Preset name to delete")
    moa_parser.set_defaults(func=cmd_moa)
