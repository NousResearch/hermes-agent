from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .models import SkillRequirement


_REPO_ROOT = Path(__file__).resolve().parents[1]
_SKILLS_ROOT = _REPO_ROOT / "skills" / "cory"

_SKILL_FILE_MAP = {
    "coding-agent-delegation": _SKILLS_ROOT / "coding-agent-delegation" / "SKILL.md",
    "discussion-interpretation": _SKILLS_ROOT / "discussion-interpretation" / "SKILL.md",
    "execution-planning": _SKILLS_ROOT / "execution-planning" / "SKILL.md",
    "governed-change-analysis": _SKILLS_ROOT / "governed-change-analysis" / "SKILL.md",
    "knowledge-compounding": _SKILLS_ROOT / "knowledge-compounding" / "SKILL.md",
    "repo-resolution": _SKILLS_ROOT / "repo-resolution" / "SKILL.md",
    "requirement-clarification": _SKILLS_ROOT / "requirement-clarification" / "SKILL.md",
    "review-synthesis": _SKILLS_ROOT / "review-synthesis" / "SKILL.md",
    "spec-reasoning": _SKILLS_ROOT / "spec-reasoning" / "SKILL.md",
    "structured-event-triage": _SKILLS_ROOT / "structured-event-triage" / "SKILL.md",
    "technical-option-analysis": _SKILLS_ROOT / "technical-option-analysis" / "SKILL.md",
}


@dataclass(frozen=True, slots=True)
class LoadedSkill:
    id: str
    required: bool
    why: str
    content: str


def _strip_frontmatter(markdown: str) -> str:
    if markdown.startswith("---\n"):
        end = markdown.find("\n---\n", 4)
        if end != -1:
            return markdown[end + 5 :].strip()
    return markdown.strip()


def load_skill_text(skill_id: str) -> str:
    path = _SKILL_FILE_MAP.get(skill_id)
    if path is None:
        raise KeyError(f"unknown Cory skill: {skill_id}")
    if not path.is_file():
        raise FileNotFoundError(f"missing Cory skill file: {path}")
    return _strip_frontmatter(path.read_text(encoding="utf-8"))


def load_skill_bundle(requirements: list[SkillRequirement]) -> list[LoadedSkill]:
    return [
        LoadedSkill(
            id=requirement.id,
            required=requirement.required,
            why=requirement.why,
            content=load_skill_text(requirement.id),
        )
        for requirement in requirements
    ]
