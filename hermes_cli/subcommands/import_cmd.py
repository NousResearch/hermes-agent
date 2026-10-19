"""``hermes import`` subcommand parser and handler."""

from __future__ import annotations

from typing import Callable, Optional


def cmd_import(args) -> Optional[int]:
    """``hermes import <archive>``; ``--verify-only`` drills the archive instead of restoring.

    The mode lives here, at the command boundary, so ``hermes_cli.backup.run_import`` stays the
    one restore implementation: a verification run is the same pipeline pointed at a throwaway
    home (#117005), not a second restore path. Both imports are lazy — the CLI fast path and
    ``patch("hermes_cli.backup.run_import")`` both keep working.
    """
    if getattr(args, "verify_only", False):
        from hermes_cli.backup_verify import run_verify_import

        run_verify_import(args)
        return None
    from hermes_cli.backup import run_import

    return run_import(args)


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
