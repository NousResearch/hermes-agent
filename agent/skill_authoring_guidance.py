"""Safe, deterministic loading of bundled skill-authoring guidance."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional


logger = logging.getLogger(__name__)

AUTHORING_SKILL_NAME = "hermes-agent-skill-authoring"
AUTHORING_SKILL_MAJOR_VERSION = "2"
AUTHORING_SKILL_RELATIVE_DIR = (
    Path("software-development") / AUTHORING_SKILL_NAME
)
AUTHORING_CONTRACT_RELATIVE_PATH = Path("references") / "authoring-contract.md"

_MAX_GUIDANCE_CHARS = 100_000
_MAX_GUIDANCE_BYTES = 400_000


@dataclass(frozen=True)
class SkillAuthoringGuidance:
    """Raw bundled authoring files, kept distinct for caller fallbacks."""

    skill_content: str
    contract_content: Optional[str]


def _frontmatter_scalars(content: str) -> dict[str, str]:
    """Parse plain scalar fields from closed leading YAML frontmatter."""
    lines = content.lstrip("\ufeff").splitlines()
    if not lines or lines[0].strip() != "---":
        return {}

    values: dict[str, str] = {}
    closed = False
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            closed = True
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip().strip("\"'")
        if key and value:
            values[key.strip()] = value
    return values if closed else {}


def _read_raw_member(
    base_dir: Path,
    relative_path: Path,
) -> Optional[str]:
    """Read one bounded UTF-8 file without following it outside ``base_dir``."""
    try:
        resolved_base = base_dir.resolve(strict=True)
        resolved_path = (resolved_base / relative_path).resolve(strict=True)
        resolved_path.relative_to(resolved_base)
        if not resolved_path.is_file():
            return None
        if resolved_path.stat().st_size > _MAX_GUIDANCE_BYTES:
            return None
        content = resolved_path.read_text(encoding="utf-8")
        if len(content) > _MAX_GUIDANCE_CHARS:
            return None
        return content
    except (OSError, RuntimeError, ValueError):
        return None


def load_bundled_skill_authoring_guidance(
    platform: Optional[str] = None,
) -> Optional[SkillAuthoringGuidance]:
    """Load the exact bundled v2 skill and optional contract as raw text.

    The active profile's installed skills are never searched, so a user-local
    same-name skill cannot shadow this contract. No skill preprocessing or
    inline-shell rendering occurs.

    Returns ``None`` when bundled skills are opted out, the authoring skill is
    disabled, or the required v2 ``SKILL.md`` fails validation. A missing,
    escaped, oversized, or unreadable contract yields a valid result with
    ``contract_content=None`` so each caller can apply its compact fallback.
    """
    try:
        from agent.skill_utils import get_disabled_skill_names
        from hermes_constants import get_bundled_skills_dir, get_hermes_home

        if (get_hermes_home() / ".no-bundled-skills").exists():
            return None
        if AUTHORING_SKILL_NAME in get_disabled_skill_names(platform=platform):
            return None

        bundled_root = get_bundled_skills_dir(
            Path(__file__).resolve().parent.parent / "skills"
        ).resolve(strict=True)
        skill_dir = (bundled_root / AUTHORING_SKILL_RELATIVE_DIR).resolve(
            strict=True
        )
        skill_dir.relative_to(bundled_root)

        skill_content = _read_raw_member(skill_dir, Path("SKILL.md"))
        if not skill_content:
            return None
        frontmatter = _frontmatter_scalars(skill_content)
        if frontmatter.get("name") != AUTHORING_SKILL_NAME:
            return None
        version = frontmatter.get("version")
        if (
            not version
            or version.split(".", 1)[0] != AUTHORING_SKILL_MAJOR_VERSION
        ):
            return None

        contract_content = _read_raw_member(
            skill_dir,
            AUTHORING_CONTRACT_RELATIVE_PATH,
        )
        return SkillAuthoringGuidance(
            skill_content=skill_content,
            contract_content=contract_content,
        )
    except (OSError, RuntimeError, ValueError):
        logger.debug(
            "Bundled %s unavailable; callers will use fallback guidance",
            AUTHORING_SKILL_NAME,
            exc_info=True,
        )
        return None
    except Exception:
        # Prompt construction and turn finalization are best-effort. Config or
        # import failures must not break either caller.
        logger.debug(
            "Could not load bundled %s; callers will use fallback guidance",
            AUTHORING_SKILL_NAME,
            exc_info=True,
        )
        return None


__all__ = [
    "AUTHORING_CONTRACT_RELATIVE_PATH",
    "AUTHORING_SKILL_MAJOR_VERSION",
    "AUTHORING_SKILL_NAME",
    "AUTHORING_SKILL_RELATIVE_DIR",
    "SkillAuthoringGuidance",
    "load_bundled_skill_authoring_guidance",
]
