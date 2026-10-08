"""CLI handlers for ``hermes migrate ...``.

Currently exposes only ``hermes migrate xai`` — diagnoses and (with --apply) rewrites references to
retired xAI models.
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

    print("usage: hermes migrate xai [--apply] [--no-backup]", file=sys.stderr)
    return 2


def _fail(message: str) -> int:
    print(f"  {color('✗', Colors.RED)} {message}", file=sys.stderr)
    return 1


def cmd_migrate_xai(args: Any) -> int:
    """Run xAI model-retirement migration in dry-run or apply mode."""
    from hermes_cli.xai_retirement import (
        apply_migration, find_retired_xai_refs, format_issue, guide_urls)

    apply = bool(getattr(args, "apply", False))
    no_backup = bool(getattr(args, "no_backup", False))
    issues = find_retired_xai_refs(load_config())

    print()
    print(color("◆ xAI model retirements", Colors.CYAN, Colors.BOLD))
    print()

    if not issues:
        print(f"  {color('✓', Colors.GREEN)} No retired xAI models in config — nothing to migrate.")
        return 0

    print(f"  Found {len(issues)} retired xAI model reference(s):")
    print()
    for issue in issues:
        print(f"    {color('⚠', Colors.YELLOW)} {format_issue(issue)}")
    print()
    for guide_url in guide_urls(issues):
        print(f"    {color('→', Colors.CYAN)} Migration guide: {guide_url}")
    print()

    config_path = _resolve_config_path()

    if not apply:
        print(color("Dry-run mode — no changes written.", Colors.DIM))
        print(color(
            "Re-run with `hermes migrate xai --apply` to rewrite "
            f"{config_path} in-place (backup created automatically).",
            Colors.DIM))
        return 0

    if not config_path or not config_path.exists():
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
