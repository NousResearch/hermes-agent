"""
Hermes Core Skill System Package.
Manages skills metadata, permissions, tools, and custom skill creation.
"""

from typing import Any, Dict, List, Optional


class SkillManager:
    def __init__(self) -> None:
        self._skills: Dict[str, Dict[str, Any]] = {}

    def register_skill(self, name: str, skill_data: Dict[str, Any]) -> None:
        self._skills[name] = skill_data

    def list_skills(self) -> List[Dict[str, Any]]:
        return list(self._skills.values())


__all__ = ["SkillManager"]
