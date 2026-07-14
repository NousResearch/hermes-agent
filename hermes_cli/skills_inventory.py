"""Skill inventory and active-skill audit helpers for ``hermes skills``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table


@dataclass
class SkillInventoryEntry:
    name: str
    category: str
    path: Path
    real_path: Path
    root: Path
    source: str
    active: bool
    disabled: bool = False
    archived: bool = False
    quarantined: bool = False
    external: bool = False
    symlinked: bool = False
    shadowed: bool = False
    duplicate: bool = False
    skipped_reason: str = ""
    error: str = ""


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _path_contains_symlink(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    current = root
    if root.is_symlink():
        return True
    for part in rel_parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def _skill_category(root: Path, skill_dir: Path) -> str:
    try:
        rel = skill_dir.relative_to(root)
    except ValueError:
        return ""
    if len(rel.parts) > 1:
        return rel.parts[0]
    return ""


def _entry_to_dict(entry: SkillInventoryEntry) -> dict[str, Any]:
    return {
        "name": entry.name,
        "category": entry.category,
        "path": str(entry.path),
        "real_path": str(entry.real_path),
        "root": str(entry.root),
        "source": entry.source,
        "active": entry.active,
        "disabled": entry.disabled,
        "archived": entry.archived,
        "quarantined": entry.quarantined,
        "external": entry.external,
        "symlinked": entry.symlinked,
        "shadowed": entry.shadowed,
        "duplicate": entry.duplicate,
        "skipped_reason": entry.skipped_reason,
        "error": entry.error,
    }


def _iter_status_skill_files(status_root: Path):
    if not status_root.is_dir():
        return
    for skill_md in sorted(status_root.rglob("SKILL.md")):
        yield skill_md


def _read_skill_entry(
    *,
    skill_md: Path,
    root: Path,
    source: str,
    disabled_names: set[str],
    archived: bool = False,
    quarantined: bool = False,
    external: bool = False,
) -> SkillInventoryEntry:
    from agent.skill_utils import (
        parse_frontmatter,
        skill_matches_environment,
        skill_matches_platform,
    )

    skill_dir = skill_md.parent
    real_path = _safe_resolve(skill_dir)
    symlinked = _path_contains_symlink(skill_dir, root)
    if symlinked:
        try:
            external = external or not real_path.is_relative_to(_safe_resolve(root))
        except (OSError, RuntimeError):
            external = True

    try:
        raw = skill_md.read_text(encoding="utf-8")
        frontmatter, _body = parse_frontmatter(raw)
        name = str(frontmatter.get("name") or skill_dir.name)
    except Exception as exc:
        return SkillInventoryEntry(
            name=skill_dir.name,
            category=_skill_category(root, skill_dir),
            path=skill_dir,
            real_path=real_path,
            root=root,
            source=source,
            active=False,
            archived=archived,
            quarantined=quarantined,
            external=external,
            symlinked=symlinked,
            skipped_reason="error",
            error=str(exc),
        )

    disabled = name in disabled_names
    skipped_reason = ""
    active = not archived and not quarantined and not disabled
    if disabled:
        skipped_reason = "disabled"
    elif archived:
        skipped_reason = "archived"
    elif quarantined:
        skipped_reason = "quarantined"
    elif not skill_matches_platform(frontmatter):
        active = False
        skipped_reason = "platform"
    elif not skill_matches_environment(frontmatter):
        active = False
        skipped_reason = "environment"

    return SkillInventoryEntry(
        name=name,
        category=_skill_category(root, skill_dir),
        path=skill_dir,
        real_path=real_path,
        root=root,
        source=source,
        active=active,
        disabled=disabled,
        archived=archived,
        quarantined=quarantined,
        external=external,
        symlinked=symlinked,
        skipped_reason=skipped_reason,
    )


def collect_skill_inventory() -> dict[str, Any]:
    """Return resolver-backed skill inventory for the active profile."""
    from agent.skill_utils import (
        get_all_skills_dirs,
        get_disabled_skill_names,
        get_external_skills_dirs,
        iter_skill_index_files,
    )
    from hermes_constants import get_skills_dir

    disabled_names = get_disabled_skill_names()
    external_roots = {_safe_resolve(p) for p in get_external_skills_dirs()}
    local_root = get_skills_dir()
    entries: list[SkillInventoryEntry] = []

    for root in get_all_skills_dirs():
        if not root.is_dir():
            continue
        resolved_root = _safe_resolve(root)
        is_external_root = resolved_root in external_roots
        source = "external" if is_external_root else "local"
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            entries.append(
                _read_skill_entry(
                    skill_md=skill_md,
                    root=root,
                    source=source,
                    disabled_names=disabled_names,
                    external=is_external_root,
                )
            )

    for skill_md in _iter_status_skill_files(local_root / ".archive"):
        entries.append(
            _read_skill_entry(
                skill_md=skill_md,
                root=local_root / ".archive",
                source="archive",
                disabled_names=disabled_names,
                archived=True,
            )
        )

    for skill_md in _iter_status_skill_files(local_root / ".hub" / "quarantine"):
        entries.append(
            _read_skill_entry(
                skill_md=skill_md,
                root=local_root / ".hub" / "quarantine",
                source="quarantine",
                disabled_names=disabled_names,
                quarantined=True,
            )
        )

    by_name: dict[str, list[SkillInventoryEntry]] = {}
    for entry in entries:
        by_name.setdefault(entry.name, []).append(entry)
    for group in by_name.values():
        if len(group) <= 1:
            continue
        for entry in group:
            entry.duplicate = True
        winner_seen = False
        for entry in group:
            if entry.active and not winner_seen:
                winner_seen = True
                continue
            if entry.active:
                entry.active = False
                entry.shadowed = True
                entry.skipped_reason = "shadowed"
            elif not entry.archived and not entry.quarantined:
                entry.shadowed = True
                if not entry.skipped_reason:
                    entry.skipped_reason = "shadowed"

    counts = {
        "total": len(entries),
        "active": sum(1 for e in entries if e.active),
        "disabled": sum(1 for e in entries if e.disabled),
        "archived": sum(1 for e in entries if e.archived),
        "quarantined": sum(1 for e in entries if e.quarantined),
        "external": sum(1 for e in entries if e.external),
        "symlinked": sum(1 for e in entries if e.symlinked),
        "shadowed": sum(1 for e in entries if e.shadowed),
        "duplicate": sum(1 for e in entries if e.duplicate),
        "errors": sum(1 for e in entries if e.error),
    }

    return {
        "counts": counts,
        "entries": [_entry_to_dict(e) for e in entries],
    }


def print_skill_inventory(
    report: dict[str, Any],
    *,
    console: Console,
    as_json: bool = False,
) -> None:
    if as_json:
        # Rich wraps long strings to terminal width, which inserts literal
        # newlines inside JSON strings and corrupts machine-readable output.
        print(json.dumps(report, indent=2, sort_keys=True), file=console.file)
        return

    counts = report["counts"]
    table = Table(title="Skills Inventory")
    table.add_column("Name", style="bold cyan")
    table.add_column("Status", style="dim")
    table.add_column("Source", style="dim")
    table.add_column("Path", style="dim")
    table.add_column("Real Path", style="dim")

    for entry in sorted(report["entries"], key=lambda e: (e["name"], e["path"])):
        if entry["active"]:
            status = "active"
        else:
            status = entry["skipped_reason"] or "inactive"
        table.add_row(
            entry["name"],
            status,
            entry["source"],
            entry["path"],
            entry["real_path"],
        )

    console.print(table)
    console.print(
        "[dim]"
        f"{counts['active']} active, {counts['disabled']} disabled, "
        f"{counts['archived']} archived, {counts['quarantined']} quarantined, "
        f"{counts['external']} external, {counts['symlinked']} symlinked, "
        f"{counts['shadowed']} shadowed, {counts['duplicate']} duplicate"
        "[/]\n"
    )
