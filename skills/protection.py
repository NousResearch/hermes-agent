"""Declarative Skill Protection Policy — tool-level guard for skill mutation safety.

Prevents agent from silently deleting/editing/patching bundled and hub-installed skills.
The curator prompt provides soft guidance; this module provides hard tool-level enforcement.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# --- Protection level constants ------------------------------------------------

PROTECTION_NONE = "none"         # user/agent-created skills: fully editable
PROTECTION_IMMUTABLE = "immutable"  # bundled/hub skills: no mutations allowed


def _provenance_name(skill_dir: Optional[Path], skill_name: str) -> str:
    """Resolve the canonical provenance key for a skill.

    ``is_bundled`` / ``is_hub_installed`` are keyed on the skill's frontmatter ``name``
    (e.g. ``docx``), but ``skill_manage`` resolves names in several forms — bare dir name,
    frontmatter name, or a category path (``productivity/docx``). Keying the guard on the
    caller's string lets a model bypass it by passing the path form: ``check_mutable('docx')``
    would block while ``check_mutable('productivity/docx')`` resolves to the same dir and
    slips through. Derive provenance from the LOCATED skill dir instead: read the frontmatter
    ``name`` out of its SKILL.md (falling back to the directory name) so every accepted form
    maps back to the manifest key the guard is keyed on.
    """
    if skill_dir is not None:
        try:
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                from agent.skill_utils import parse_frontmatter
                frontmatter, _ = parse_frontmatter(skill_md.read_text(encoding="utf-8", errors="replace"))
                if isinstance(frontmatter, dict) and frontmatter.get("name"):
                    return str(frontmatter["name"])
            if skill_dir.name:
                return skill_dir.name
        except (OSError, ValueError) as e:  # pragma: no cover — defensive
            logger.debug("Failed to read provenance from %s: %s", skill_dir, e)
    return skill_name


def _resolve_protection_level(skill_dir: Optional[Path], skill_name: str) -> str:
    """Determine protection level for a skill based on its provenance.

    Currently a simple ALL-OR-NOTHING rule: bundled and hub-installed skills are
    immutable; user/agent-created skills are editable. Future iterations can make
    this config-declarable (``config.yaml`` → ``skills.protection.levels``).

    Returns ``PROTECTION_IMMUTABLE`` or ``PROTECTION_NONE``.
    """
    from tools.skill_usage import is_bundled, is_hub_installed

    name = _provenance_name(skill_dir, skill_name)
    if is_bundled(name) or is_hub_installed(name):
        return PROTECTION_IMMUTABLE
    return PROTECTION_NONE


def _refusal(skill_name: str, action: str, level: str) -> Dict:
    """Structured error dict for a blocked mutation, matching the existing
    ``_err()`` schema used across skill_manager_tool.py::

        _err("message") -> {"success": False, "error": "..."}

    (The caller's ``_refusal``/``_err`` returns ``{"success": False, "error": msg}``;
    returning the same keys keeps the tool result parseable as a failed call with
    no ``success`` key and a consistent error field.)
    """
    return {
        "success": False,
        "error": (
            f"Skill '{skill_name}' is {level} ({action} blocked). "
            f"Bundled and hub-installed skills are protected from agent mutation. "
            f"To modify, fork the skill first with `skill_manage(action='create', ...)`."
        ),
    }


def check_mutable(skill_dir: Optional[Path], skill_name: str, action: str) -> Optional[Dict]:
    """Check whether a skill can be mutated by the given action.

    ``skill_dir`` is the located skill path resolved by the caller (so category-path /
    bare-dir/name forms all map to one provenance); ``skill_name`` is the caller's string,
    used only for the error message. Returns ``None`` on success (mutation allowed), or an
    error dict on failure (mutation blocked) matching the ``_err()`` schema from
    skill_manager_guards.py.

    Call before every mutation operation (delete, edit, patch, write_file, remove_file).
    """
    level = _resolve_protection_level(skill_dir, skill_name)
    if level == PROTECTION_IMMUTABLE:
        logger.debug("Blocked %s on protected skill '%s' (%s)", action, skill_name, level)
        return _refusal(skill_name, action, level)
    return None


def check_deletable(skill_dir: Optional[Path], skill_name: str) -> Optional[Dict]:
    """Check whether a skill can be deleted. Same contract as ``check_mutable()``.

    Separated so the ``delete`` surface can carry a specific message if needed
    (e.g. ``absorbed_into`` context).
    """
    return check_mutable(skill_dir, skill_name, "delete")