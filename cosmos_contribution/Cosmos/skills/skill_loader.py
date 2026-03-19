"""
Cosmos Skill Loader
=======================

Loads and manages skills for the collective.
Compatible with HermesAgent Skill Document format.
"""

import os
import yaml
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from loguru import logger


@dataclass
class Skill:
    """A skill definition."""
    name: str
    description: str
    content: str  # The skill instructions
    metadata: Dict[str, Any] = field(default_factory=dict)
    requires: Dict[str, List[str]] = field(default_factory=dict)
    mcp_config: Optional[Dict[str, Any]] = None

    @property
    def emoji(self) -> str:
        """Get skill emoji from metadata."""
        for platform in ['cosmos', 'hermes', 'clawdbot']:
            if platform in self.metadata:
                return self.metadata[platform].get('emoji', '🦞')
        return '🦞'

    @property
    def homepage(self) -> str:
        """Get skill homepage from metadata."""
        for platform in ['cosmos', 'hermes', 'clawdbot']:
            if platform in self.metadata:
                return self.metadata[platform].get('homepage', '')
        return ''


class SkillLoader:
    """
    Loads skills from various sources.

    Supports:
    - Local SKILL.md files
    - GitHub repos (like hermes-skills)
    - Built-in Cosmos skills
    """

    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or Path(__file__).parent
        self.loaded_skills: Dict[str, Skill] = {}
        self._load_builtin_skills()

    def _load_builtin_skills(self):
        """Load built-in Cosmos skills."""
        from .cosmos_skills import COSMOS_SKILLS
        for skill in COSMOS_SKILLS:
            self.loaded_skills[skill.name] = skill
            logger.info(f"Loaded built-in skill: {skill.name} {skill.emoji}")

    def load_from_file(self, skill_path: Path) -> Optional[Skill]:
        """Load a skill from a SKILL.md file."""
        try:
            content = skill_path.read_text(encoding='utf-8')
            return self._parse_skill_md(content)
        except Exception as e:
            logger.error(f"Failed to load skill from {skill_path}: {e}")
            return None

    def load_from_github(self, repo_url: str, skill_name: str) -> Optional[Skill]:
        """
        Load a skill from a GitHub repository.

        Example: load_from_github("https://github.com/NousResearch/hermes-skills", "1ly-payments")
        """
        import httpx

        # Convert to raw GitHub URL
        if "github.com" in repo_url:
            parts = repo_url.replace("https://github.com/", "").split("/")
            if len(parts) >= 2:
                owner, repo = parts[0], parts[1]
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/main/{skill_name}/SKILL.md"

                try:
                    response = httpx.get(raw_url, timeout=10.0)
                    if response.status_code == 200:
                        skill = self._parse_skill_md(response.text)
                        if skill:
                            self.loaded_skills[skill.name] = skill
                            logger.info(f"Loaded skill from GitHub: {skill.name} {skill.emoji}")
                            return skill
                except Exception as e:
                    logger.error(f"Failed to load skill from GitHub: {e}")

        return None

    def _parse_skill_md(self, content: str) -> Optional[Skill]:
        """Parse a SKILL.md file into a Skill object."""
        try:
            # Extract YAML frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    frontmatter = yaml.safe_load(parts[1])
                    body = parts[2].strip()

                    # Parse metadata
                    metadata = frontmatter.get('metadata', {})
                    if isinstance(metadata, str):
                        metadata = json.loads(metadata)

                    # Extract requires
                    requires = {}
                    for platform in ['cosmos', 'hermes', 'clawdbot']:
                        if platform in metadata and 'requires' in metadata[platform]:
                            requires = metadata[platform]['requires']
                            break

                    return Skill(
                        name=frontmatter.get('name', 'unknown'),
                        description=frontmatter.get('description', ''),
                        content=body,
                        metadata=metadata,
                        requires=requires,
                    )

            # No frontmatter, use content as-is
            return Skill(
                name='unknown',
                description='',
                content=content,
            )

        except Exception as e:
            logger.error(f"Failed to parse skill: {e}")
            return None

    def get_skill(self, name: str) -> Optional[Skill]:
        """Get a loaded skill by name."""
        return self.loaded_skills.get(name)

    def list_skills(self) -> List[str]:
        """List all loaded skill names."""
        return list(self.loaded_skills.keys())

    def get_skill_prompt(self, name: str) -> str:
        """Get the full prompt/instructions for a skill."""
        skill = self.get_skill(name)
        if skill:
            return f"""# {skill.name} {skill.emoji}

{skill.description}

{skill.content}
"""
        return ""


# Global skill loader instance
_skill_loader: Optional[SkillLoader] = None


def get_skill_loader() -> SkillLoader:
    """Get or create the global skill loader."""
    global _skill_loader
    if _skill_loader is None:
        _skill_loader = SkillLoader()
    return _skill_loader


def load_skill(name: str) -> Optional[Skill]:
    """Load a skill by name."""
    return get_skill_loader().get_skill(name)
