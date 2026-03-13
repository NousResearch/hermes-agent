"""Skill slash commands — scan installed skills and build invocation messages.

Shared between CLI (cli.py) and gateway (gateway/run.py) so both surfaces
can invoke skills via /skill-name commands.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_skill_commands: Dict[str, Dict[str, Any]] = {}


def scan_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Scan ~/.hermes/skills/ and return a mapping of /command -> skill info.

    Returns:
        Dict mapping "/skill-name" to {name, description, skill_md_path, skill_dir}.
    """
    global _skill_commands
    _skill_commands = {}
    try:
        from tools.skills_tool import SKILLS_DIR, _parse_frontmatter, skill_matches_platform
        if not SKILLS_DIR.exists():
            return _skill_commands
        for skill_md in SKILLS_DIR.rglob("SKILL.md"):
            if any(part in ('.git', '.github', '.hub') for part in skill_md.parts):
                continue
            try:
                content = skill_md.read_text(encoding='utf-8')
                frontmatter, body = _parse_frontmatter(content)
                # Skip skills incompatible with the current OS platform
                if not skill_matches_platform(frontmatter):
                    continue
                name = frontmatter.get('name', skill_md.parent.name)
                description = frontmatter.get('description', '')
                if not description:
                    for line in body.strip().split('\n'):
                        line = line.strip()
                        if line and not line.startswith('#'):
                            description = line[:80]
                            break
                cmd_name = name.lower().replace(' ', '-').replace('_', '-')
                _skill_commands[f"/{cmd_name}"] = {
                    "name": name,
                    "description": description or f"Invoke the {name} skill",
                    "skill_md_path": str(skill_md),
                    "skill_dir": str(skill_md.parent),
                }
            except Exception:
                continue
    except Exception:
        pass
    return _skill_commands


def get_skill_commands() -> Dict[str, Dict[str, Any]]:
    """Return the current skill commands mapping (scan first if empty)."""
    if not _skill_commands:
        scan_skill_commands()
    return _skill_commands


def build_skill_invocation_message(
    cmd_key: str,
    user_instruction: str = "",
    task_id: str | None = None,
) -> Optional[str]:
    """Build the user message content for a skill slash command invocation.

    Args:
        cmd_key: The command key including leading slash (e.g., "/gif-search").
        user_instruction: Optional text the user typed after the command.

    Returns:
        The formatted message string, or None if the skill wasn't found.
    """
    commands = get_skill_commands()
    skill_info = commands.get(cmd_key)
    if not skill_info:
        return None

    skill_name = skill_info["name"]
    skill_path = skill_info["skill_dir"]

    try:
        from tools.skills_tool import SKILLS_DIR, skill_view

        loaded_skill = json.loads(skill_view(skill_path, task_id=task_id))
    except Exception:
        return f"[Failed to load skill: {skill_name}]"

    if not loaded_skill.get("success"):
        return f"[Failed to load skill: {skill_name}]"

    content = str(loaded_skill.get("content") or "")
    skill_dir = Path(skill_info["skill_dir"])

    parts = [
        f'[SYSTEM: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
        "",
        content.strip(),
    ]

    if loaded_skill.get("setup_skipped"):
        parts.extend(
            [
                "",
                "[Skill setup note: Required environment setup was skipped. Continue loading the skill and explain any reduced functionality if it matters.]",
            ]
        )
    elif loaded_skill.get("gateway_setup_hint"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['gateway_setup_hint']}]",
            ]
        )
    elif loaded_skill.get("setup_needed") and loaded_skill.get("setup_note"):
        parts.extend(
            [
                "",
                f"[Skill setup note: {loaded_skill['setup_note']}]",
            ]
        )

    supporting = []
    linked_files = loaded_skill.get("linked_files") or {}
    for entries in linked_files.values():
        if isinstance(entries, list):
            supporting.extend(entries)

    if not supporting:
        for subdir in ("references", "templates", "scripts", "assets"):
            subdir_path = skill_dir / subdir
            if subdir_path.exists():
                for f in sorted(subdir_path.rglob("*")):
                    if f.is_file():
                        rel = str(f.relative_to(skill_dir))
                        supporting.append(rel)

    if supporting:
        skill_view_target = str(Path(skill_path).relative_to(SKILLS_DIR))
        parts.append("")
        parts.append("[This skill has supporting files you can load with the skill_view tool:]")
        for sf in supporting:
            parts.append(f"- {sf}")
        parts.append(
            f'\nTo view any of these, use: skill_view(name="{skill_view_target}", file_path="<path>")'
        )

    if user_instruction:
        parts.append("")
        parts.append(f"The user has provided the following instruction alongside the skill invocation: {user_instruction}")

    return "\n".join(parts)
