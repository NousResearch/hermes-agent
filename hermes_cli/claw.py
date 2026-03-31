"""hermes claw — OpenClaw migration commands.

Usage:
    hermes claw migrate              # Interactive migration from ~/.openclaw
    hermes claw migrate --dry-run    # Preview what would be migrated
    hermes claw migrate --preset full --overwrite  # Full migration, overwrite conflicts
"""

import importlib.util
import logging
import sys
from pathlib import Path

from hermes_cli.config import get_hermes_home, get_config_path, load_config, save_config
from hermes_constants import get_optional_skills_dir
from hermes_cli.setup import (
    Colors,
    color,
    print_header,
    print_info,
    print_success,
    print_error,
    prompt_yes_no,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

_OPENCLAW_SCRIPT = (
    get_optional_skills_dir(PROJECT_ROOT / "optional-skills")
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)

# Fallback: user may have installed the skill from the Hub
_OPENCLAW_SCRIPT_INSTALLED = (
    get_hermes_home()
    / "skills"
    / "migration"
    / "openclaw-migration"
    / "scripts"
    / "openclaw_to_hermes.py"
)


def _find_migration_script() -> Path | None:
    """Find the openclaw_to_hermes.py script in known locations."""
    for candidate in [_OPENCLAW_SCRIPT, _OPENCLAW_SCRIPT_INSTALLED]:
        if candidate.exists():
            return candidate
    return None


def _load_migration_module(script_path: Path):
    """Dynamically load the migration script as a module."""
    spec = importlib.util.spec_from_file_location("openclaw_to_hermes", script_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules so @dataclass can resolve the module
    # (Python 3.11+ requires this for dynamically loaded modules)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return mod


def claw_command(args):
    """Route hermes claw subcommands."""
    action = getattr(args, "claw_action", None)

    if action == "migrate":
        _cmd_migrate(args)
    else:
        print("Usage: hermes claw migrate [options]")
        print()
        print("Commands:")
        print("  migrate          Migrate settings from OpenClaw to Hermes")
        print()
        print("Run 'hermes claw migrate --help' for migration options.")


def _cmd_migrate(args):
    """Run the OpenClaw → Hermes migration."""
    # Check current and legacy OpenClaw directories
    explicit_source = getattr(args, "source", None)
    if explicit_source:
        source_dir = Path(explicit_source)
    else:
        source_dir = Path.home() / ".openclaw"
        if not source_dir.is_dir():
            # Try legacy directory names
            for legacy in (".clawdbot", ".moldbot"):
                candidate = Path.home() / legacy
                if candidate.is_dir():
                    source_dir = candidate
                    break
    dry_run = getattr(args, "dry_run", False)
    preset = getattr(args, "preset", "full")
    overwrite = getattr(args, "overwrite", False)
    migrate_secrets = getattr(args, "migrate_secrets", False)
    workspace_target = getattr(args, "workspace_target", None)
    skill_conflict = getattr(args, "skill_conflict", "skip")

    # If using the "full" preset, secrets are included by default
    if preset == "full":
        migrate_secrets = True

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│          ⚕ Hermes — OpenClaw Migration                 │",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )

    # Check source directory
    if not source_dir.is_dir():
        print()
        print_error(f"OpenClaw directory not found: {source_dir}")
        print_info("Make sure your OpenClaw installation is at the expected path.")
        print_info("You can specify a custom path: hermes claw migrate --source /path/to/.openclaw")
        return

    # Find the migration script
    script_path = _find_migration_script()
    if not script_path:
        print()
        print_error("Migration script not found.")
        print_info("Expected at one of:")
        print_info(f"  {_OPENCLAW_SCRIPT}")
        print_info(f"  {_OPENCLAW_SCRIPT_INSTALLED}")
        print_info("Make sure the openclaw-migration skill is installed.")
        return

    # Show what we're doing
    hermes_home = get_hermes_home()
    print()
    print_header("Migration Settings")
    print_info(f"Source:      {source_dir}")
    print_info(f"Target:      {hermes_home}")
    print_info(f"Preset:      {preset}")
    print_info(f"Mode:        {'dry run (preview only)' if dry_run else 'execute'}")
    print_info(f"Overwrite:   {'yes' if overwrite else 'no (skip conflicts)'}")
    print_info(f"Secrets:     {'yes (allowlisted only)' if migrate_secrets else 'no'}")
    if skill_conflict != "skip":
        print_info(f"Skill conflicts: {skill_conflict}")
    if workspace_target:
        print_info(f"Workspace:   {workspace_target}")
    print()

    # For execute mode (non-dry-run), confirm unless --yes was passed
    if not dry_run and not getattr(args, "yes", False):
        if not prompt_yes_no("Proceed with migration?", default=True):
            print_info("Migration cancelled.")
            return

    # Ensure config.yaml exists before migration tries to read it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # Load and run the migration
    try:
        mod = _load_migration_module(script_path)
        if mod is None:
            print_error("Could not load migration script.")
            return

        selected = mod.resolve_selected_options(None, None, preset=preset)
        ws_target = Path(workspace_target).resolve() if workspace_target else None

        migrator = mod.Migrator(
            source_root=source_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=not dry_run,
            workspace_target=ws_target,
            overwrite=overwrite,
            migrate_secrets=migrate_secrets,
            output_dir=None,
            selected_options=selected,
            preset_name=preset,
            skill_conflict_mode=skill_conflict,
        )
        report = migrator.migrate()
    except Exception as e:
        print()
        print_error(f"Migration failed: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return

    # Print results
    _print_migration_report(report, dry_run)


def _print_migration_report(report: dict, dry_run: bool):
    """Print a formatted migration report."""
    summary = report.get("summary", {})
    migrated = summary.get("migrated", 0)
    skipped = summary.get("skipped", 0)
    conflicts = summary.get("conflict", 0)
    errors = summary.get("error", 0)

    print()
    if dry_run:
        print_header("Dry Run Results")
        print_info("No files were modified. This is a preview of what would happen.")
    else:
        print_header("Migration Results")

    print()

    # Detailed items
    items = report.get("items", [])
    if items:
        # Group by status
        migrated_items = [i for i in items if i.get("status") == "migrated"]
        skipped_items = [i for i in items if i.get("status") == "skipped"]
        conflict_items = [i for i in items if i.get("status") == "conflict"]
        error_items = [i for i in items if i.get("status") == "error"]

        if migrated_items:
            label = "Would migrate" if dry_run else "Migrated"
            print(color(f"  ✓ {label}:", Colors.GREEN))
            for item in migrated_items:
                kind = item.get("kind", "unknown")
                dest = item.get("destination", "")
                if dest:
                    dest_short = str(dest).replace(str(Path.home()), "~")
                    print(f"      {kind:<22s} → {dest_short}")
                else:
                    print(f"      {kind}")
            print()

        if conflict_items:
            print(color("  ⚠ Conflicts (skipped — use --overwrite to force):", Colors.YELLOW))
            for item in conflict_items:
                kind = item.get("kind", "unknown")
                reason = item.get("reason", "already exists")
                print(f"      {kind:<22s}  {reason}")
            print()

        if skipped_items:
            print(color("  ─ Skipped:", Colors.DIM))
            for item in skipped_items:
                kind = item.get("kind", "unknown")
                reason = item.get("reason", "")
                print(f"      {kind:<22s}  {reason}")
            print()

        if error_items:
            print(color("  ✗ Errors:", Colors.RED))
            for item in error_items:
                kind = item.get("kind", "unknown")
                reason = item.get("reason", "unknown error")
                print(f"      {kind:<22s}  {reason}")
            print()

    # Summary line
    parts = []
    if migrated:
        action = "would migrate" if dry_run else "migrated"
        parts.append(f"{migrated} {action}")
    if conflicts:
        parts.append(f"{conflicts} conflict(s)")
    if skipped:
        parts.append(f"{skipped} skipped")
    if errors:
        parts.append(f"{errors} error(s)")

    if parts:
        print_info(f"Summary: {', '.join(parts)}")
    else:
        print_info("Nothing to migrate.")

    # Output directory
    output_dir = report.get("output_dir")
    if output_dir:
        print_info(f"Full report saved to: {output_dir}")

    if dry_run:
        print()
        print_info("To execute the migration, run without --dry-run:")
        print_info(f"  hermes claw migrate --preset {report.get('preset', 'full')}")
    elif migrated:
        print()
        print_success("Migration complete!")
        # Warn if API keys were skipped (migrate_secrets not enabled)
        skipped_keys = [
            i for i in report.get("items", [])
            if i.get("kind") == "provider-keys" and i.get("status") == "skipped"
        ]
        if skipped_keys:
            print()
            print(color("  ⚠ API keys were NOT migrated (secrets migration is disabled by default).", Colors.YELLOW))
            print(color("  Your OPENROUTER_API_KEY and other provider keys must be added manually.", Colors.YELLOW))
            print()
            print_info("To migrate API keys, re-run with:")
            print_info("  hermes claw migrate --migrate-secrets")
            print()
            print_info("Or add your key manually:")
            print_info("  hermes config set OPENROUTER_API_KEY sk-or-v1-...")
