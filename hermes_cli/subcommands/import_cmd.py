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
        "--verify-only", action="store_true",
        help="Verify the archive by restoring it into a throwaway home; never writes to "
             "this profile and exits nonzero when the restore cannot be proven")
    import_parser.add_argument(
        "--keep-candidate", action="store_true",
        help="With --verify-only: keep the temporary restore for inspection instead of "
             "removing it on success")
    import_parser.set_defaults(func=cmd_import)
