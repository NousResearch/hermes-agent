"""
skill_trigger_loader.py — Auto-load skills based on frontmatter trigger patterns.

When `get_triggered_skills(text)` is called at the start of a conversation turn,
it scans all installed SKILL.md files for a `triggers` frontmatter key, matches
each regex pattern against the incoming text, and returns the full content of
every matched skill.

Frontmatter format:
    triggers:
      - 'PHP Parse error'
      - 'syntax error, unexpected'
      - 'mergeable.*CONFLICTING'

Patterns are matched case-insensitively. If a skill matches, its full SKILL.md
content is loaded and injected before the agent's reasoning loop begins.

Design decisions:
- Lazy import: no import-time overhead when the feature is unused.
- Fail-safe: any exception in this module is caught by the caller — a trigger
  failure must never crash the agent's main conversation loop.
- Performance: results are cached for the process lifetime keyed on the skill
  file's mtime, so repeated turns in the same session don't re-scan disk.
- Limit: at most MAX_AUTO_LOADED_SKILLS skills are auto-loaded per turn to
  avoid overwhelming the context with irrelevant content.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

MAX_AUTO_LOADED_SKILLS = 5
_SKILL_FILENAME = "SKILL.md"

# Module-level cache: skill_path -> (mtime, triggers_list | None, full_content)
_trigger_cache: dict[str, Tuple[float, Optional[List[str]], Optional[str]]] = {}


def _get_skills_home() -> Path:
    """Return the root skills directory (profile-aware)."""
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "skills"
    except Exception:
        return Path(os.path.expanduser("~/.hermes/skills"))


def _iter_skill_paths(skills_home: Path):
    """Yield paths to all SKILL.md files under skills_home."""
    if not skills_home.exists():
        return
    for skill_md in skills_home.rglob(_SKILL_FILENAME):
        # Skip worktrees, hidden dirs, and vendor paths
        parts = skill_md.parts
        if any(p.startswith(".") for p in parts):
            continue
        yield skill_md


def _parse_triggers_and_content(skill_path: Path) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Read a SKILL.md file and return (triggers, full_content).
    Returns (None, None) if the file has no 'triggers' key or is unreadable.
    """
    try:
        content = skill_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, None

    # Fast path: skip files without '---' frontmatter delimiter
    if not content.startswith("---"):
        return None, None

    # Extract frontmatter block
    end = content.find("\n---", 3)
    if end == -1:
        return None, None
    frontmatter_text = content[3:end]

    # Only parse YAML if 'triggers' appears in the frontmatter text (cheap check)
    if "triggers" not in frontmatter_text:
        return None, None

    try:
        import yaml
        fm = yaml.safe_load(frontmatter_text) or {}
    except Exception:
        return None, None

    raw_triggers = fm.get("triggers")
    if not raw_triggers or not isinstance(raw_triggers, list):
        return None, None

    # Normalise: each trigger must be a non-empty string
    triggers = [str(t).strip() for t in raw_triggers if t and str(t).strip()]
    if not triggers:
        return None, None

    return triggers, content


def _get_cached_skill(skill_path: Path) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return (triggers, content) from cache, refreshing if the file changed."""
    path_str = str(skill_path)
    try:
        mtime = skill_path.stat().st_mtime
    except OSError:
        return None, None

    cached = _trigger_cache.get(path_str)
    if cached and cached[0] == mtime:
        return cached[1], cached[2]

    triggers, content = _parse_triggers_and_content(skill_path)
    _trigger_cache[path_str] = (mtime, triggers, content)
    return triggers, content


def get_triggered_skills(text: str) -> List[str]:
    """
    Return a list of full SKILL.md contents whose trigger patterns match *text*.

    Args:
        text: The user message or task description for this turn.

    Returns:
        List of SKILL.md file contents (strings) for every matched skill,
        capped at MAX_AUTO_LOADED_SKILLS. Empty list if nothing matches or
        on any error.
    """
    if not text or not text.strip():
        return []

    try:
        skills_home = _get_skills_home()
        matched: List[str] = []

        for skill_path in _iter_skill_paths(skills_home):
            if len(matched) >= MAX_AUTO_LOADED_SKILLS:
                break
            try:
                triggers, content = _get_cached_skill(skill_path)
                if not triggers or not content:
                    continue
                for pattern in triggers:
                    try:
                        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
                            matched.append(content)
                            logger.debug(
                                "skill_trigger_loader: auto-loaded %s (pattern=%r)",
                                skill_path.parent.name,
                                pattern,
                            )
                            break  # one match per skill is enough
                    except re.error:
                        # Invalid regex in frontmatter — skip silently
                        continue
            except Exception as e:
                logger.debug("skill_trigger_loader: error checking %s: %s", skill_path, e)
                continue

        return matched

    except Exception as e:
        logger.debug("skill_trigger_loader: unexpected error: %s", e)
        return []


def format_triggered_skills_block(skill_contents: List[str]) -> str:
    """
    Format auto-loaded skill contents into a single context block for injection.
    """
    if not skill_contents:
        return ""
    parts = ["<!-- Auto-loaded skills (trigger match) -->"]
    for i, content in enumerate(skill_contents, 1):
        parts.append(f"\n--- Auto-loaded skill {i} ---\n{content.strip()}")
    return "\n".join(parts)
