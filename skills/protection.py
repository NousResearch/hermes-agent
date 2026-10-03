"""Declarative Skill Protection Policy — tool-level guard for skill mutation safety.

Prevents agent from silently deleting/editing/patching bundled and hub-installed skills.
The curator prompt provides soft guidance; this module provides hard tool-level enforcement.
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# --- Protection level constants ------------------------------------------------

PROTECTION_NONE = "none"         # user/agent-created skills: fully editable
PROTECTION_IMMUTABLE = "immutable"  # bundled/hub skills: no mutations allowed


def _resolve_protection_level(skill_name: str) -> str:
    """Determine protection level for a skill based on its provenance.

    Currently a simple ALL-OR-NOTHING rule: bundled and hub-installed skills are
    immutable; user/agent-created skills are editable. Future iterations can make
    this config-declarable (``config.yaml`` → ``skills.protection.levels``).

    Returns ``PROTECTION_IMMUTABLE`` or ``PROTECTION_NONE``.
    """
    from tools.skill_usage import is_bundled, is_hub_installed

    if is_bundled(skill_name) or is_hub_installed(skill_name):
        return PROTECTION_IMMUTABLE
    return PROTECTION_NONE


def _refusal(skill_name: str, action: str, level: str) -> Dict:
    """Structured error dict for a blocked mutation, matching the existing
    ``_err()`` idiom in skill_manager_tool.py::

        _err("message") -> {"error": True, "message": "..."}
    """
    return {
        "error": True,
        "message": (
            f"Skill '{skill_name}' is {level} ({action} blocked). "
            f"Bundled and hub-installed skills are protected from agent mutation. "
            f"To modify, fork the skill first with `skill_manage(action='create', ...)`."
        ),
    }


def check_mutable(skill_name: str, action: str) -> Optional[Dict]:
    """Check whether a skill can be mutated by the given action.

    Returns ``None`` on success (mutation allowed), or an error dict on failure
    (mutation blocked) matching the ``_err()`` schema from skill_manager_tool.py.

    Call before every mutation operation (delete, edit, patch, write_file, remove_file).
    """
    level = _resolve_protection_level(skill_name)
    if level == PROTECTION_IMMUTABLE:
        logger.debug("Blocked %s on protected skill '%s' (%s)", action, skill_name, level)
        return _refusal(skill_name, action, level)
    return None


def check_deletable(skill_name: str) -> Optional[Dict]:
    """Check whether a skill can be deleted. Same contract as ``check_mutable()``.

    Separated so the ``delete`` surface can carry a specific message if needed
    (e.g. ``absorbed_into`` context).
    """
    return check_mutable(skill_name, "delete")