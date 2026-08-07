"""Skill groups — organize installed skills into named groups.

Groups keep large skill installs navigable. They are stored in
``config.yaml`` under ``skills.groups`` as a mapping of group name to a
list of skill names::

    skills:
      disabled: [skill-a]
      groups:
        security: [web-pentest, godmode]
        writing:  [humanizer]

Groups are purely organizational: they do not change how skills load or
run. They power the ``hermes skills group ...`` subcommands and the
``--group`` filter on ``hermes skills list``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

from rich.console import Console
from rich.table import Table

from hermes_cli.config import load_config, save_config

_console = Console()


def get_skill_groups(config: Optional[Dict[str, Any]] = None) -> Dict[str, List[str]]:
    """Read ``skills.groups`` from config and normalize it.

    Args:
        config: Optional already-loaded config dict (test hook). When
            omitted, the active profile's config is loaded — profile
            switches (``-p``) swap ``HERMES_HOME`` at process start, so no
            explicit profile flag is needed here, mirroring
            ``get_disabled_skill_names``.

    Returns a dict of group name -> sorted list of unique skill names.
    Tolerates missing/null/malformed sections like ``get_disabled_skills``
    in ``hermes_cli/skills_config.py``.
    """
    if config is None:
        config = load_config()
    skills_cfg = config.get("skills")
    if not isinstance(skills_cfg, dict):
        return {}
    raw = skills_cfg.get("groups")
    if not isinstance(raw, dict):
        return {}
    groups: Dict[str, List[str]] = {}
    for name, members in raw.items():
        if not isinstance(name, str) or not name.strip():
            continue
        normalized = _normalize_members(members)
        if normalized:
            groups[name] = normalized
    return groups


def save_skill_groups(config: Dict[str, Any], groups: Dict[str, List[str]]) -> None:
    """Persist ``skills.groups`` into *config* and write it to disk."""
    config.setdefault("skills", {})
    config["skills"]["groups"] = {
        name: sorted(set(members)) for name, members in groups.items()
    }
    save_config(config)


def add_skills_to_group(
    config: Dict[str, Any], group: str, skills: List[str]
) -> Dict[str, Any]:
    """Add skill names to *group*, creating it when missing.

    Returns a result dict with counts so the CLI can report precisely:
    ``{"created": bool, "added": [...], "duplicates": [...], "unknown": [...]}``.
    """
    groups = get_skill_groups(config)
    created = group not in groups
    members = set(groups.get(group, []))
    added: List[str] = []
    duplicates: List[str] = []
    for skill in skills:
        (duplicates if skill in members else added).append(skill)
        members.add(skill)
    groups[group] = sorted(members)
    save_skill_groups(config, groups)
    return {
        "created": created,
        "added": added,
        "duplicates": duplicates,
        "unknown": _unknown_skill_names(added),
    }


def remove_skills_from_group(
    config: Dict[str, Any],
    group: str,
    skills: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Remove skill names from *group*.

    With no *skills*, the whole group is deleted. A group that ends up
    empty is removed too. Returns ``{"removed": [...], "group_deleted":
    bool, "missing": [...]}``.
    """
    groups = get_skill_groups(config)
    if group not in groups:
        return {"removed": [], "group_deleted": False, "missing": list(skills or [])}
    members = set(groups[group])
    if not skills:
        del groups[group]
        save_skill_groups(config, groups)
        return {"removed": sorted(members), "group_deleted": True, "missing": []}
    removed = [s for s in skills if s in members]
    missing = [s for s in skills if s not in members]
    members.difference_update(removed)
    if members:
        groups[group] = sorted(members)
    else:
        del groups[group]
    save_skill_groups(config, groups)
    return {"removed": removed, "group_deleted": not members, "missing": missing}


def _normalize_members(members) -> List[str]:
    """Normalize a raw group value (list, scalar, or garbage) to a sorted
    list of unique, non-empty strings."""
    if members is None:
        return []
    if isinstance(members, str):
        members = [members]
    if not isinstance(members, (list, tuple, set)):
        return []
    seen: Set[str] = set()
    out: List[str] = []
    for member in members:
        name = str(member).strip() if member is not None else ""
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return sorted(out)


def _validate_group_name(name: str) -> Optional[str]:
    """Return an error message if *name* is not usable as a group name."""
    if not name or not name.strip():
        return "Group name cannot be empty."
    if any(ch.isspace() for ch in name):
        return f"Group name {name!r} must not contain whitespace."
    if name.startswith("-"):
        return f"Group name {name!r} must not start with '-' (looks like a flag)."
    return None


def _installed_skill_names() -> Set[str]:
    """Best-effort set of installed skill names (empty on discovery failure)."""
    try:
        from tools.skills_tool import _find_all_skills

        return {
            skill.get("name")
            for skill in _find_all_skills(skip_disabled=True)
            if skill.get("name")
        }
    except Exception:
        return set()


def _unknown_skill_names(names: List[str]) -> List[str]:
    """Names in *names* that are not currently installed.

    Returns [] when skill discovery fails (config-only associations are
    still allowed — the skill may be installed later).
    """
    installed = _installed_skill_names()
    if not installed:
        return []
    return [name for name in names if name not in installed]


def group_command(args) -> None:
    """Router for ``hermes skills group <action>`` — called from main.py."""
    action = getattr(args, "group_action", None)
    if action in ("list", "ls"):
        _cmd_group_list(as_json=getattr(args, "json", False))
    elif action == "add":
        _cmd_group_add(args.group, args.skills)
    elif action in ("remove", "rm"):
        _cmd_group_remove(args.group, getattr(args, "skills", None) or [])
    else:
        _console.print("Usage: hermes skills group [list|add|remove]\n")


def _cmd_group_list(*, as_json: bool = False) -> None:
    c = _console
    groups = get_skill_groups()
    if not groups:
        c.print(
            "[dim]No skill groups configured. Create one with:[/] "
            "hermes skills group add <group> <skill> [skill ...]\n"
        )
        return
    if as_json:
        c.print(json.dumps(groups, indent=2))
        return
    table = Table(title="Skill Groups")
    table.add_column("Group", style="bold cyan")
    table.add_column("Skills", style="dim")
    for name in sorted(groups):
        table.add_row(name, ", ".join(groups[name]))
    c.print(table)
    total = sum(len(members) for members in groups.values())
    c.print(f"[dim]{len(groups)} group(s), {total} skill assignment(s)[/]\n")


def _cmd_group_add(group: str, skills: List[str]) -> None:
    c = _console
    error = _validate_group_name(group)
    if error:
        c.print(f"[bold red]Error:[/] {error}\n")
        return
    if not skills:
        c.print(
            "[bold red]Error:[/] At least one skill name is required. "
            "Usage: hermes skills group add <group> <skill> [skill ...]\n"
        )
        return
    config = load_config()
    result = add_skills_to_group(config, group, skills)
    verb = "Created group" if result["created"] else "Updated group"
    c.print(f"[bold green]{verb}:[/] {group}")
    if result["added"]:
        c.print(f"[dim]Added: {', '.join(result['added'])}[/]")
    if result["duplicates"]:
        c.print(f"[dim]Already present: {', '.join(result['duplicates'])}[/]")
    if result["unknown"]:
        c.print(
            f"[yellow]Not installed (added anyway): {', '.join(result['unknown'])}[/]"
        )
    c.print()


def _cmd_group_remove(group: str, skills: List[str]) -> None:
    c = _console
    config = load_config()
    result = remove_skills_from_group(config, group, skills or None)
    if result["group_deleted"]:
        c.print(f"[bold green]Deleted group:[/] {group}\n")
        return
    if result["removed"]:
        c.print(
            f"[bold green]Removed from {group}:[/] {', '.join(result['removed'])}"
        )
    if result["missing"]:
        c.print(f"[dim]Not in group: {', '.join(result['missing'])}[/]")
    if not result["removed"]:
        current = get_skill_groups(config).get(group, [])
        c.print(
            f"[yellow]No skills removed. Group '{group}' currently has:[/] "
            f"{', '.join(current) if current else '(none)'}"
        )
    c.print()
