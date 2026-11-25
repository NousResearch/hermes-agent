"""``hermes import`` subcommand parser."""

from __future__ import annotations

from typing import Callable


def build_import_cmd_parser(subparsers, *, cmd_import: Callable) -> None:
    """Attach the ``import`` subcommand to ``subparsers``."""
    import_parser = subparsers.add_parser(
        "import", help="Restore a Hermes backup from a zip file",
        description="Extract a previously created Hermes backup into your "
        "Hermes home directory, restoring configuration, skills, "
        "sessions, and data")
    import_parser.add_argument("zipfile", help="Path to the backup zip file")
    import_parser.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing files without confirmation")
    import_parser.add_argument(
        "--keep-host", action="store_true",
        help="Replace this agent with the backup but keep what its host set up: its Nous Portal "
        "sign-in, model/provider, tool-gateway, dashboard and scheduler settings, and NOUS_*/HERMES_* "
        "keys. A full backup of this agent is written first (the undo point). For Hermes Cloud agents")
    import_parser.add_argument(
        "--messaging-off", action="store_true",
        help="With --keep-host: every messaging bot the backup runs (Telegram, Discord, ...) arrives "
        "switched off, token and settings kept, so the computer it came from keeps answering. Turn "
        "each on on the Messaging page. The Desktop transfer passes it; an undo does not")
    import_parser.set_defaults(func=cmd_import)
