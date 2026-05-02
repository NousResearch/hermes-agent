"""hermes claw — OpenClaw migration commands.

Usage:
    hermes claw migrate              # Preview then migrate (always shows preview first)
    hermes claw migrate --dry-run    # Preview only, no changes
    hermes claw migrate --yes        # Skip confirmation prompt
    hermes claw migrate --preset full --overwrite --migrate-secrets  # Full run w/ secrets
    hermes claw migrate --no-backup  # Skip pre-migration snapshot
    hermes claw cleanup              # Archive leftover OpenClaw directories
    hermes claw cleanup --dry-run    # Preview what would be archived
"""

import importlib.util
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

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

# Known OpenClaw directory names (current + legacy)
_OPENCLAW_DIR_NAMES = (".openclaw", ".clawdbot", ".moltbot")

def _detect_openclaw_processes() -> list[str]:
    """Detect running OpenClaw processes and services.

    Returns a list of human-readable descriptions of what was found.
    An empty list means nothing was detected.
    """
    found: list[str] = []

    # -- systemd service (Linux) ------------------------------------------
    if sys.platform != "win32":
        try:
            result = subprocess.run(
                ["systemctl", "--user", "is-active", "openclaw-gateway.service"],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip() == "active":
                found.append("systemd service: openclaw-gateway.service")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    # -- process scan ------------------------------------------------------
    if sys.platform == "win32":
        try:
            for exe in ("openclaw.exe", "clawd.exe"):
                result = subprocess.run(
                    ["tasklist", "/FI", f"IMAGENAME eq {exe}"],
                    capture_output=True, text=True, timeout=5,
                )
                if exe in result.stdout.lower():
                    found.append(f"process: {exe}")

            # Node.js-hosted OpenClaw — tasklist doesn't show command lines,
            # so fall back to PowerShell.
            ps_cmd = (
                'Get-CimInstance Win32_Process -Filter "Name = \'node.exe\'" | '
                'Where-Object { $_.CommandLine -match "openclaw|clawd" } | '
                'Select-Object -First 1 ProcessId'
            )
            result = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5,
            )
            if result.stdout.strip():
                found.append(f"node.exe process with openclaw in command line (PID {result.stdout.strip()})")
        except Exception:
            pass
    else:
        try:
            result = subprocess.run(
                ["pgrep", "-f", "openclaw"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0:
                pids = result.stdout.strip().split()
                found.append(f"openclaw process(es) (PIDs: {', '.join(pids)})")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return found


def _warn_if_openclaw_running(auto_yes: bool) -> None:
    """Warn if OpenClaw is still running before migration.

    Telegram, Discord, and Slack only allow one active connection per bot
    token. Migrating while OpenClaw is running causes both to fight for the
    same token.
    """
    running = _detect_openclaw_processes()
    if not running:
        return

    print()
    print_error("OpenClaw appears to be running:")
    for detail in running:
        print_info(f"  * {detail}")
    print_info(
        "Messaging platforms (Telegram, Discord, Slack) only allow one "
        "active session per bot token. If you continue, both OpenClaw and "
        "Hermes may try to use the same token, causing disconnects."
    )
    print_info("Recommendation: stop OpenClaw before migrating.")
    print()
    if auto_yes:
        return
    if not sys.stdin.isatty():
        print_info("Non-interactive session — continuing to preview only.")
        return
    if not prompt_yes_no("Continue anyway?", default=False):
        print_info("Migration cancelled. Stop OpenClaw and try again.")
        sys.exit(0)


def _warn_if_gateway_running(auto_yes: bool) -> None:
    """Check if a Hermes gateway is running with connected platforms.

    Migrating bot tokens while the gateway is polling will cause conflicts
    (e.g. Telegram 409 "terminated by other getUpdates request"). Warn the
    user and let them decide whether to continue.
    """
    from gateway.status import get_running_pid, read_runtime_status

    if not get_running_pid():
        return

    data = read_runtime_status() or {}
    platforms = data.get("platforms") or {}
    connected = [name for name, info in platforms.items()
                 if isinstance(info, dict) and info.get("state") == "connected"]
    if not connected:
        return

    print()
    print_error(
        "Hermes gateway is running with active connections: "
        + ", ".join(connected)
    )
    print_info(
        "Migrating bot tokens while the gateway is active will cause "
        "conflicts (Telegram, Discord, and Slack only allow one active "
        "session per token)."
    )
    print_info("Recommendation: stop the gateway first with 'hermes stop'.")
    print()
    if not auto_yes and not prompt_yes_no("Continue anyway?", default=False):
        print_info("Migration cancelled. Stop the gateway and try again.")
        sys.exit(0)

# State files commonly found in OpenClaw workspace directories — listed
# during cleanup to help the user decide whether to archive
_WORKSPACE_STATE_GLOBS = (
    "*/todo.json",
    "*/sessions/*",
    "*/memory/*.json",
    "*/logs/*",
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


def _find_openclaw_dirs() -> list[Path]:
    """Find all OpenClaw directories on disk."""
    found = []
    for name in _OPENCLAW_DIR_NAMES:
        candidate = Path.home() / name
        if candidate.is_dir():
            found.append(candidate)
    return found


def _scan_workspace_state(source_dir: Path) -> list[tuple[Path, str]]:
    """Scan an OpenClaw directory for workspace state files.

    Returns a list of (path, description) tuples.
    """
    findings: list[tuple[Path, str]] = []

    # Direct state files in the root
    for name in ("todo.json", "sessions", "logs"):
        candidate = source_dir / name
        if candidate.exists():
            kind = "directory" if candidate.is_dir() else "file"
            findings.append((candidate, f"Root {kind}: {name}"))

    # State files inside workspace directories
    for child in sorted(source_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        # Check for workspace-like subdirectories
        for state_name in ("todo.json", "sessions", "logs", "memory"):
            state_path = child / state_name
            if state_path.exists():
                kind = "directory" if state_path.is_dir() else "file"
                rel = state_path.relative_to(source_dir).as_posix()
                findings.append((state_path, f"Workspace {kind}: {rel}"))

    return findings


def _archive_directory(source_dir: Path, dry_run: bool = False) -> Path:
    """Rename an OpenClaw directory to .pre-migration.

    Returns the archive path.
    """
    timestamp = datetime.now().strftime("%Y%m%d")
    archive_name = f"{source_dir.name}.pre-migration"
    archive_path = source_dir.parent / archive_name

    # If archive already exists, add timestamp
    if archive_path.exists():
        archive_name = f"{source_dir.name}.pre-migration-{timestamp}"
        archive_path = source_dir.parent / archive_name

    # If still exists (multiple runs same day), add counter
    counter = 2
    while archive_path.exists():
        archive_name = f"{source_dir.name}.pre-migration-{timestamp}-{counter}"
        archive_path = source_dir.parent / archive_name
        counter += 1

    if not dry_run:
        source_dir.rename(archive_path)

    return archive_path


def claw_command(args):
    """Route hermes claw subcommands."""
    action = getattr(args, "claw_action", None)

    if action == "migrate":
        _cmd_migrate(args)
    elif action in ("cleanup", "clean"):
        _cmd_cleanup(args)
    else:
        print("Usage: hermes claw <command> [options]")
        print()
        print("Commands:")
        print("  migrate          Migrate settings from OpenClaw to Hermes")
        print("  cleanup          Archive leftover OpenClaw directories after migration")
        print()
        print("Run 'hermes claw <command> --help' for options.")


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
            for legacy in (".clawdbot", ".moltbot"):
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
    no_backup = getattr(args, "no_backup", False)

    # Secrets are never included implicitly — they must be explicitly requested
    # via --migrate-secrets, even under --preset full.  This mirrors OpenClaw's
    # migrate-hermes posture (two-phase: run once without secrets, rerun with
    # --include-secrets) and prevents a --preset full invocation from silently
    # importing API keys that the user may not have intended to copy.

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
    auto_yes = getattr(args, "yes", False)
    print()
    print_header("Migration Settings")
    print_info(f"Source:      {source_dir}")
    print_info(f"Target:      {hermes_home}")
    print_info(f"Preset:      {preset}")
    print_info(f"Overwrite:   {'yes' if overwrite else 'no (skip conflicts)'}")
    print_info(f"Secrets:     {'yes (allowlisted only)' if migrate_secrets else 'no'}")
    if skill_conflict != "skip":
        print_info(f"Skill conflicts: {skill_conflict}")
    if workspace_target:
        print_info(f"Workspace:   {workspace_target}")
    print()

    # Check if OpenClaw is still running — migrating tokens while both are
    # active will cause conflicts (e.g. Telegram 409).
    _warn_if_openclaw_running(auto_yes)

    # Check if a Hermes gateway is running with connected platforms.
    _warn_if_gateway_running(auto_yes)

    # Ensure config.yaml exists before migration tries to read it
    config_path = get_config_path()
    if not config_path.exists():
        save_config(load_config())

    # Load the migration module
    try:
        mod = _load_migration_module(script_path)
        if mod is None:
            print_error("Could not load migration script.")
            return
    except Exception as e:
        print()
        print_error(f"Could not load migration script: {e}")
        logger.debug("OpenClaw migration error", exc_info=True)
        return

    selected = mod.resolve_selected_options(None, None, preset=preset)
    ws_target = Path(workspace_target).resolve() if workspace_target else None

    # ── Phase 1: Always preview first ──────────────────────────
    try:
        preview = mod.Migrator(
            source_root=source_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=False,
            workspace_target=ws_target,
            overwrite=overwrite,
            migrate_secrets=migrate_secrets,
            output_dir=None,
            selected_options=selected,
            preset_name=preset,
            skill_conflict_mode=skill_conflict,
        )
        preview_report = preview.migrate()
    except Exception as e:
        print()
        print_error(f"Migration preview failed: {e}")
        logger.debug("OpenClaw migration preview error", exc_info=True)
        return

    preview_summary = preview_report.get("summary", {})
    preview_count = preview_summary.get("migrated", 0)
    preview_conflicts = preview_summary.get("conflict", 0)

    # "Nothing to migrate" means nothing migrated AND nothing blocked by
    # conflicts.  If there are conflicts, we still want to show the plan and
    # surface the refusal/--overwrite guidance instead of silently bailing.
    if preview_count == 0 and preview_conflicts == 0:
        print()
        print_info("Nothing to migrate from OpenClaw.")
        _print_migration_report(preview_report, dry_run=True)
        return

    print()
    if preview_count > 0:
        print_header(f"Migration Preview — {preview_count} item(s) would be imported")
    else:
        print_header(
            f"Migration Preview — {preview_conflicts} conflict(s), nothing would be imported"
        )
    print_info("No changes have been made yet. Review the list below:")
    _print_migration_report(preview_report, dry_run=True)

    # If --dry-run, stop here
    if dry_run:
        return

    # ── Phase 1b: Refuse if the plan has conflicts and --overwrite is not set ─
    # Modelled on OpenClaw's assertConflictFreePlan() — apply is a safe no-op
    # on conflicts unless the user explicitly opts in to overwriting.  Without
    # this guard, the user would answer "yes, proceed" and silently end up
    # with a migration that skipped every conflicting item.
    if preview_conflicts > 0 and not overwrite:
        print()
        print_error(
            f"Plan has {preview_conflicts} conflict(s). Refusing to apply."
        )
        print_info(
            "Each conflict is an item whose target already exists in ~/.hermes/. "
            "Re-run with --overwrite to replace conflicting targets (item-level "
            "backups are written to the migration report directory)."
        )
        print_info("Or re-run with --dry-run to review the full plan.")
        return

    # ── Phase 2: Confirm and execute ───────────────────────────
    print()
    if not auto_yes:
        if not sys.stdin.isatty():
            print_info("Non-interactive session — preview only.")
            print_info("To execute, re-run with: hermes claw migrate --yes")
            return
        if not prompt_yes_no("Proceed with migration?", default=True):
            print_info("Migration cancelled.")
            return

    # ── Phase 2b: Pre-apply backup of the Hermes home ─────────
    # Delegates to hermes_cli.backup.create_pre_migration_backup(), which
    # shares implementation with the pre-update backup (same exclusion
    # rules, same SQLite safe-copy, zip format) so the archive is
    # restorable with `hermes import`.  Mirrors OpenClaw's
    # createPreMigrationBackup posture — one atomic restore point before
    # any mutation, auto-pruned to the last 5 pre-migration zips.
    backup_archive: Optional[Path] = None
    if not no_backup:
        try:
            from hermes_cli.backup import create_pre_migration_backup, _format_size
            backup_archive = create_pre_migration_backup(hermes_home=hermes_home)
            if backup_archive:
                size_str = _format_size(backup_archive.stat().st_size)
                print()
                print_success(f"Pre-migration backup: {backup_archive} ({size_str})")
                print_info(f"Restore with: hermes import {backup_archive.name}")
        except Exception as e:
            print()
            print_error(f"Could not create pre-migration backup: {e}")
            print_info(
                "Re-run with --no-backup to skip, or free up disk space under the Hermes home."
            )
            logger.debug("Pre-migration backup error", exc_info=True)
            return

    try:
        migrator = mod.Migrator(
            source_root=source_dir.resolve(),
            target_root=hermes_home.resolve(),
            execute=True,
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
        if backup_archive:
            print_info(f"A pre-migration backup is available at: {backup_archive}")
            print_info(f"Restore with: hermes import {backup_archive.name}")
        return

    # Print results
    _print_migration_report(report, dry_run=False)

    # Source directory is left untouched — archiving is not the migration
    # tool's responsibility.  Users who want to clean up can run
    # 'hermes claw cleanup' separately.


def _cmd_cleanup(args):
    """Archive leftover OpenClaw directories after migration.

    Scans for OpenClaw directories that still exist after migration and offers
    to rename them to .pre-migration to free disk space.
    """
    dry_run = getattr(args, "dry_run", False)
    auto_yes = getattr(args, "yes", False)
    explicit_source = getattr(args, "source", None)

    print()
    print(
        color(
            "┌─────────────────────────────────────────────────────────┐",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "│          ⚕ Hermes — OpenClaw Cleanup                   │",
            Colors.MAGENTA,
        )
    )
    print(
        color(
            "└─────────────────────────────────────────────────────────┘",
            Colors.MAGENTA,
        )
    )

    # Find OpenClaw directories
    if explicit_source:
        dirs_to_check = [Path(explicit_source)]
    else:
        dirs_to_check = _find_openclaw_dirs()

    if not dirs_to_check:
        print()
        print_success("No OpenClaw directories found. Nothing to clean up.")
        return

    # Warn if OpenClaw is still running — archiving while the service is
    # active causes it to recreate an empty skeleton directory (#8502).
    running = _detect_openclaw_processes()
    if running:
        print()
        print_error("OpenClaw appears to be still running:")
        for detail in running:
            print_info(f"  * {detail}")
        print_info(
            "Archiving .openclaw/ while the service is active may cause it to "
            "immediately recreate an empty skeleton directory, destroying your config."
        )
        print_info("Stop OpenClaw first: systemctl --user stop openclaw-gateway.service")
        print()
        if not auto_yes:
            if not sys.stdin.isatty():
                print_info("Non-interactive session — aborting. Stop OpenClaw and re-run.")
                return
            if not prompt_yes_no("Proceed anyway?", default=False):
                print_info("Aborted. Stop OpenClaw first, then re-run: hermes claw cleanup")
                return

    total_archived = 0

    for source_dir in dirs_to_check:
        print()
        print_header(f"Found: {source_dir}")

        # Scan for state files
        state_files = _scan_workspace_state(source_dir)

        # Show directory stats
        try:
            workspace_dirs = [
                d for d in source_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
                and any((d / name).exists() for name in ("todo.json", "SOUL.md", "MEMORY.md", "USER.md"))
            ]
        except OSError:
            workspace_dirs = []

        if workspace_dirs:
            print_info(f"Workspace directories: {len(workspace_dirs)}")
            for ws in workspace_dirs[:5]:
                items = []
                if (ws / "todo.json").exists():
                    items.append("todo.json")
                if (ws / "sessions").is_dir():
                    items.append("sessions/")
                if (ws / "SOUL.md").exists():
                    items.append("SOUL.md")
                if (ws / "MEMORY.md").exists():
                    items.append("MEMORY.md")
                detail = ", ".join(items) if items else "empty"
                print(f"      {ws.name}/  ({detail})")
            if len(workspace_dirs) > 5:
                print(f"      ... and {len(workspace_dirs) - 5} more")

        if state_files:
            print()
            print(color(f"  {len(state_files)} state file(s) found:", Colors.YELLOW))
            for path, desc in state_files[:8]:
                print(f"      {desc}")
            if len(state_files) > 8:
                print(f"      ... and {len(state_files) - 8} more")

        print()

        if dry_run:
            archive_path = _archive_directory(source_dir, dry_run=True)
            print_info(f"Would archive: {source_dir} → {archive_path}")
        elif not auto_yes and not sys.stdin.isatty():
            print_info(f"Non-interactive session — would archive: {source_dir}")
            print_info("To execute, re-run with: hermes claw cleanup --yes")
        else:
            if auto_yes or prompt_yes_no(f"Archive {source_dir}?", default=True):
                try:
                    archive_path = _archive_directory(source_dir)
                    print_success(f"Archived: {source_dir} → {archive_path}")
                    total_archived += 1
                except OSError as e:
                    print_error(f"Could not archive: {e}")
                    print_info(f"Try manually: mv {source_dir} {source_dir}.pre-migration")
            else:
                print_info("Skipped.")

    # Summary
    print()
    if dry_run:
        print_info(f"Dry run complete. {len(dirs_to_check)} directory(ies) would be archived.")
        print_info("Run without --dry-run to archive them.")
    elif total_archived:
        print_success(f"Cleaned up {total_archived} OpenClaw directory(ies).")
        print_info("Directories were renamed, not deleted. You can undo by renaming them back.")
    else:
        print_info("No directories were archived.")


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
