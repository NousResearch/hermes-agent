"""``hermes clipboard`` subcommand parser."""

from __future__ import annotations

from hermes_agent.hermes_cli.subcommands._shared import add_json_flag


def _cmd_clipboard(args):
    from hermes_agent.hermes_cli.clipboard_cmd import cmd_clipboard

    return cmd_clipboard(args)


def build_clipboard_parser(subparsers) -> None:
    """Attach the ``clipboard`` subcommand to ``subparsers``."""
    clipboard_parser = subparsers.add_parser(
        "clipboard", help="Read this machine's clipboard (device-local)",
        description="Read this machine's clipboard with the extractor the CLI and TUI use. "
            "Device-local: no gateway, no session store.")
    actions = clipboard_parser.add_subparsers(dest="clipboard_action", required=True)
    image_parser = actions.add_parser(
        "image", help="Write the clipboard image to a file as PNG",
        description="Write the clipboard image to --out (overwritten). Exit 0: written; "
            "1: no image; 3: no clipboard tool installed; 4: --out not writable "
            "(an existing --out is kept on any failure).")
    image_parser.add_argument("--out", required=True, metavar="PATH", help="File to write")
    add_json_flag(image_parser, "Print one JSON object on stdout")
    image_parser.set_defaults(func=_cmd_clipboard)
