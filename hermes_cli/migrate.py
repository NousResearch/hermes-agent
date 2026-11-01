"""CLI handlers for ``hermes migrate ...``.

``hermes migrate`` (bare) and ``hermes migrate <agent>`` move a user over from another agent through
the agent-importers plugin; ``hermes migrate xai`` diagnoses and (with --apply) rewrites references to
xAI models retired on May 15, 2026.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hermes_cli.colors import Colors, color
from hermes_cli.config import load_config


def cmd_migrate(args: Any) -> int:
    """Dispatcher for ``hermes migrate <subtype>``."""
    if getattr(args, "migrate_type", None) == "xai":
        return cmd_migrate_xai(args)
    if getattr(args, "migrate_type", None) is None:
        from hermes_cli.config import get_hermes_home
        from hermes_cli.setup_agent_import import migrate_from_agents
        return migrate_from_agents(get_hermes_home())

    print("usage: hermes migrate [<agent> | xai [--apply] [--no-backup]]", file=sys.stderr)
    return 2


# ``hermes migrate <agent>`` flags forwarded to ``hermes import-agent <agent>``, as (dest, flag).
_AGENT_IMPORT_FLAGS = (("dry_run", "--dry-run"), ("overwrite", "--overwrite"), ("yes", "--yes"),
                       ("activate_routines", "--activate-routines"))


def cmd_migrate_agent(args: Any) -> int:
    """``hermes migrate <agent>``: install the agent-importers plugin when missing, then run its
    import for that agent with the flags given here."""
    from hermes_cli.config import get_hermes_home
    from hermes_cli.setup_agent_import import migrate_from_agents
    extra = [flag for dest, flag in _AGENT_IMPORT_FLAGS if getattr(args, dest, False)]
    if getattr(args, "source", None):
        extra += ["--source", str(Path(args.source).expanduser())]
    for name in getattr(args, "bot", None) or ():
        extra += ["--bot", name]
    return migrate_from_agents(get_hermes_home(), args.migrate_agent, tuple(extra))


def _fail(message: str) -> int:
    print(f"  {color('✗', Colors.RED)} {message}", file=sys.stderr)
    return 1


def cmd_migrate_xai(args: Any) -> int:
    """Run xAI May-15 model migration in dry-run or apply mode."""
    from hermes_cli.config_backend import config_exists
    from hermes_cli.xai_retirement import (
        MIGRATION_GUIDE_URL, RETIREMENT_DATE, apply_migration, find_retired_xai_refs, format_issue)

    apply = bool(getattr(args, "apply", False))
    no_backup = bool(getattr(args, "no_backup", False))
    issues = find_retired_xai_refs(load_config())

    print()
    print(color(f"◆ xAI Model Retirement Migration ({RETIREMENT_DATE})", Colors.CYAN, Colors.BOLD))
    print()

    if not issues:
        print(f"  {color('✓', Colors.GREEN)} No retired xAI models in config — nothing to migrate.")
        return 0

    print(f"  Found {len(issues)} retired xAI model reference(s):")
    print()
    for issue in issues:
        print(f"    {color('⚠', Colors.YELLOW)} {format_issue(issue)}")
    print()
    print(f"    {color('→', Colors.CYAN)} Migration guide: {MIGRATION_GUIDE_URL}")
    print()

    config_path = _resolve_config_path()

    if not apply:
        print(color("Dry-run mode — no changes written.", Colors.DIM))
        print(color(
            "Re-run with `hermes migrate xai --apply` to rewrite "
            f"{config_path} in-place (backup created automatically).",
            Colors.DIM))
        return 0

    if not config_path or not config_exists(config_path):
        return _fail(f"Could not locate config.yaml (looked at: {config_path})")

    try:
        result = apply_migration(config_path=config_path, issues=issues, backup=not no_backup)
    except Exception as exc:
        return _fail(f"Migration failed: {exc}")

    if not result.config_changed:
        print(f"  {color('⚠', Colors.YELLOW)} No changes written.")
        return 0

    if result.backup_path is not None:
        print(f"  {color('✓', Colors.GREEN)} Backup: {result.backup_path}")
    print(
        f"  {color('✓', Colors.GREEN)} Updated {len(result.issues_resolved)} "
        f"slot(s) in {result.file_path}")
    print()
    print(color("Run `hermes doctor` to confirm no retired xAI models remain.", Colors.DIM))
    return 0


def _resolve_config_path() -> Path:
    """Best-effort: locate the active config.yaml on disk."""
    from hermes_cli.config import get_hermes_home
    return get_hermes_home() / "config.yaml"
