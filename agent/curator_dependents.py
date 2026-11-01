"""Skills that a live skill depends on. The curator treats them like cron-referenced skills: usage only bumps when a
skill is loaded, so a dependency can age out while its dependent is still served, and archiving it leaves the dependent
pointing at nothing. A dependency is a name the dependent declares in ``related_skills`` or names as a skill token in
its own text (``SKILL.md`` or a ``references/*.md`` support file): ``skill_view("name")`` or a backticked ``name``
followed by "skill". Live means visible to the skill scanners (``.archive/``, ``.hub/`` and ``.curator_backups/``
copies are pruned by ``iter_skill_index_files``) and not disabled."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SKILL_TOKEN = re.compile(r"""skill_view\(\s*["']([\w.-]+)["']|`([\w.-]+)`\s+skills?\b""", re.IGNORECASE)


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        logger.debug("Curator could not read %s for skill dependencies: %s", path, e)
        return ""


def skill_tokens(text: str) -> set[str]:
    """Skill names a text points at as skills: ``skill_view("name")`` calls and backticked ``name`` + "skill"."""
    return {a or b for a, b in _SKILL_TOKEN.findall(text)}


def related_skill_dependencies() -> set[str]:
    """Names declared as related, or named as skill tokens, by any live skill other than themselves. An unreadable
    file contributes nothing (``parse_frontmatter`` already falls back on malformed YAML, so only the read can fail)."""
    from agent.skill_utils import (
        extract_related_skills, get_all_skills_dirs, get_disabled_skill_names, iter_skill_index_files,
        parse_frontmatter,
    )

    disabled = get_disabled_skill_names()
    referenced: set[str] = set()
    for root in get_all_skills_dirs():
        if not root.is_dir():
            continue
        for skill_md in iter_skill_index_files(root, "SKILL.md"):
            text = _read(skill_md)
            frontmatter = parse_frontmatter(text)[0] or {}
            owner = str(frontmatter.get("name") or skill_md.parent.name).strip()
            if not text or owner in disabled:
                continue
            names = set(extract_related_skills(frontmatter)) | skill_tokens(text)
            for support in sorted((skill_md.parent / "references").glob("*.md")):
                names |= skill_tokens(_read(support))
            referenced |= names - {owner}
    return referenced
