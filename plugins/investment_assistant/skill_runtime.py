"""PydanticAI Agent Skills integration for investment assistant agents."""

from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

AGENT_SKILLS_DIR = Path(__file__).resolve().parent / "agent_skills"


def ensure_pydantic_ai_skills_available() -> str:
    """Ensure pydantic-ai-skills is importable, using Hermes lazy deps if needed."""

    if importlib.util.find_spec("pydantic_ai_skills") is None:
        try:
            from tools.lazy_deps import ensure

            ensure("investment.pydantic_ai_skills", prompt=False)
        except Exception as exc:
            raise RuntimeError(
                "pydantic-ai-skills is required for investment assistant agent skills. "
                "Install pydantic-ai-skills or enable Hermes lazy installs."
            ) from exc

    if importlib.util.find_spec("pydantic_ai_skills") is None:
        raise RuntimeError("pydantic-ai-skills is required but not importable.")
    try:
        return importlib.metadata.version("pydantic-ai-skills")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def pydantic_ai_skills_status() -> dict[str, Any]:
    """Return dependency and local skill discovery status."""

    try:
        version = ensure_pydantic_ai_skills_available()
        skills = discover_local_agent_skills()
    except Exception as exc:
        return {
            "available": False,
            "mode": "pydantic_ai_skills_unavailable",
            "reason": str(exc),
            "skills_dir": str(AGENT_SKILLS_DIR),
        }
    return {
        "available": True,
        "mode": "pydantic_ai_skills",
        "package_version": version,
        "skills_dir": str(AGENT_SKILLS_DIR),
        "skills": skills,
    }


def discover_local_agent_skills() -> list[str]:
    """Return skill names discoverable from the plugin-local skills directory."""

    ensure_pydantic_ai_skills_available()
    from pydantic_ai_skills.directory import discover_skills

    return sorted(skill.name for skill in discover_skills(AGENT_SKILLS_DIR, validate=True, max_depth=2))


def create_agent_skills_capability(
    skill_names: list[str] | None = None,
    *,
    allow_scripts: bool = False,
):
    """Create a SkillsCapability scoped to plugin-local investment skills.

    By default script execution is disabled. The first investment assistant
    skills are intended to provide methodology and references, while Python
    workflow code remains responsible for data collection and validation.
    """

    ensure_pydantic_ai_skills_available()
    from pydantic_ai_skills import SkillsCapability

    directories = _skill_directories(skill_names)
    exclude_tools: set[str] = set()
    if not allow_scripts:
        exclude_tools.add("run_skill_script")
    return SkillsCapability(
        directories=directories,
        validate=True,
        max_depth=2,
        exclude_tools=exclude_tools,
    )


def _skill_directories(skill_names: list[str] | None) -> list[Path]:
    if not skill_names:
        return [AGENT_SKILLS_DIR]
    directories: list[Path] = []
    for name in skill_names:
        normalized = str(name or "").strip().lower()
        if not normalized:
            continue
        skill_dir = AGENT_SKILLS_DIR / normalized
        if not skill_dir.is_dir():
            raise ValueError(f"Unknown investment assistant agent skill: {normalized}")
        directories.append(skill_dir)
    return directories or [AGENT_SKILLS_DIR]
