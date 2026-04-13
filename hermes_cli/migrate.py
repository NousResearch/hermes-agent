"""
Hermes unified migration command.

Supports cross-machine, cross-platform migration between
Linux, macOS, and WSL2 with automatic path remapping.

Usage::

    hermes migrate export                    Export to hermes-migration-{timestamp}.tar.gz
    hermes migrate export --preset full     Include secrets
    hermes migrate export -o backup.tar.gz  Custom output path
    hermes migrate import -i backup.tar.gz  Import from bundle
    hermes migrate import -i backup.tar.gz --dry-run   Preview without applying
    hermes migrate import -i backup.tar.gz --interactive  Guided import
    hermes migrate verify -i backup.tar.gz  Verify bundle
    hermes migrate verify                   Verify current installation
    hermes migrate doctor                   Check environment health
"""

import sys

from hermes_cli.colors import Colors, color
from hermes_cli.migrate_export import export_bundle

# Lazy imports — stub modules raise NotImplementedError until their PR lands
def _lazy_import_import():
    from hermes_cli.migrate_import import import_bundle
    return import_bundle

def _lazy_import_verify():
    from hermes_cli.migrate_verify import run_doctor, verify_bundle
    return run_doctor, verify_bundle


def run_migrate(args):
    """Entry point called by main.py's cmd_migrate handler."""
    action = getattr(args, "action", None)

    if action is None:
        print("Run 'hermes migrate --help' to see available subcommands.")
        return

    interactive = getattr(args, "interactive", False)

    try:
        if action == "export":
            export_bundle(getattr(args, "output", None), getattr(args, "preset", "safe"))
            return True
        elif action == "import":
            try:
                import_bundle_fn = _lazy_import_import()
                import_bundle_fn(
                    getattr(args, "input", None),
                    dry_run=getattr(args, "dry_run", False),
                    interactive=interactive,
                )
                return True
            except NotImplementedError:
                print(color(
                    "\n  migrate import is not yet available in this PR.\n"
                    "  It will be included in a separate PR: feat/hermes-migrate-import",
                    Colors.YELLOW,
                ))
                return False
        elif action == "verify":
            _, verify_bundle = _lazy_import_verify()
            return verify_bundle(getattr(args, "input", None))
        elif action == "doctor":
            run_doctor, _ = _lazy_import_verify()
            return run_doctor()
    except KeyboardInterrupt:
        print(color("\n\nCancelled.", Colors.YELLOW))
        sys.exit(130)
    except Exception as e:
        print(color(f"\n\nError: {e}", Colors.RED))
        sys.exit(1)
