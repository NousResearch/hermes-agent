"""``hermes checkpoints`` subcommand parser."""

from __future__ import annotations


def build_checkpoints_parser(subparsers) -> None:
    """Attach the ``checkpoints`` subcommand to ``subparsers``."""
    checkpoints_parser = subparsers.add_parser(
        "checkpoints", help="Inspect / prune / clear ~/.hermes/checkpoints/",
        description="Manage the filesystem checkpoint store — the shadow git "
        "repo hermes uses to snapshot working directories before "
        "write_file/patch/terminal calls. Lets you see how much "
        "space checkpoints occupy, force a prune, or wipe the base.")
    from hermes_cli.checkpoints import register_cli as _register_checkpoints_cli
    _register_checkpoints_cli(checkpoints_parser)
