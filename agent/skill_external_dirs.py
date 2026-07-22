"""External skill directory filtering with include/exclude glob patterns.

Extends skills.external_dirs config to support glob-based filtering and
category mapping, while maintaining backward compatibility with the simple
list-of-paths format.

Config examples:

    # Simple (backward compatible)
    skills:
      external_dirs:
        - /path/to/skills

    # Advanced (with filtering)
    skills:
      external_dirs:
        - path: /path/to/skills
          include:
            - "anthropic-*"
            - "mcp-*"
          exclude:
            - "*test*"
            - "*sample*"
          category_map:
            "anthropic-*": "anthropic-tools"
"""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExternalDirConfig:
    """Configuration for a single external skill directory."""
    path: str
    include: List[str] = field(default_factory=lambda: ["*"])
    exclude: List[str] = field(default_factory=list)
    category_map: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


def parse_external_dir_config(entry: Any) -> Optional[ExternalDirConfig]:
    """Parse an external_dirs entry into ExternalDirConfig.
    
    Supports both formats:
    - String: "/path/to/skills" → ExternalDirConfig(path="/path/to/skills")
    - Dict: {path: "...", include: [...], exclude: [...]}
    
    Returns None if entry is invalid.
    """
    if isinstance(entry, str):
        entry = entry.strip()
        if not entry:
            return None
        return ExternalDirConfig(path=entry)
    
    if isinstance(entry, dict):
        path = entry.get("path")
        if not path or not isinstance(path, str):
            return None
        
        include = entry.get("include", ["*"])
        if not isinstance(include, list):
            include = [include] if include else ["*"]
        
        exclude = entry.get("exclude", [])
        if not isinstance(exclude, list):
            exclude = [exclude] if exclude else []
        
        category_map = entry.get("category_map", {})
        if not isinstance(category_map, dict):
            category_map = {}
        
        enabled = entry.get("enabled", True)
        
        return ExternalDirConfig(
            path=path,
            include=include,
            exclude=exclude,
            category_map=category_map,
            enabled=enabled,
        )
    
    return None


def matches_any(name: str, patterns: List[str]) -> bool:
    """Check if name matches any glob pattern (fnmatch syntax)."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def matches_all_words(name: str, patterns: List[str]) -> bool:
    """Check if all patterns match using word boundaries for better precision."""
    for pattern in patterns:
        # For patterns with spaces, check if all words appear in name
        if " " in pattern:
            words = pattern.lower().split()
            name_lower = name.lower()
            if not all(word in name_lower for word in words):
                return False
        else:
            # Use fnmatch for glob patterns, or check if pattern appears as substring
            if any(c in pattern for c in "*?["):
                if not fnmatch.fnmatch(name.lower(), pattern.lower()):
                    return False
            else:
                if pattern.lower() not in name.lower():
                    return False
    return True


def determine_category(skill_name: str, category_map: Dict[str, str]) -> str:
    """Determine skill category from category_map patterns."""
    for pattern, category in category_map.items():
        if fnmatch.fnmatch(skill_name, pattern):
            return category
    return "external"


# ── Category helpers for batch operations ─────────────────────────────────────

def get_skill_category(skill_name: str) -> str:
    """Get the category for a skill from its SKILL.md frontmatter.
    
    Searches for the skill in ~/.hermes/skills/ and external_dirs,
    parses the frontmatter, and returns the category field.
    Returns 'general' if not found or if no category is specified.
    """
    from pathlib import Path
    from hermes_constants import get_skills_dir
    
    skills_dir = get_skills_dir()
    if not skills_dir.exists():
        return "general"
    
    # Search for the skill in the skills directory
    for skill_dir in skills_dir.rglob("*"):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        
        try:
            content = skill_md.read_text(encoding="utf-8")
            # Parse YAML frontmatter
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        name = frontmatter.get("name", "")
                        if name == skill_name:
                            return str(frontmatter.get("category", "general"))
        except Exception:
            continue
    
    return "general"


def get_skills_by_category(category_pattern: str) -> list:
    """Get all skills matching a category pattern (supports glob).
    
    Returns a list of skill names whose category matches the pattern.
    """
    import fnmatch
    from pathlib import Path
    from hermes_constants import get_skills_dir
    
    skills_dir = get_skills_dir()
    if not skills_dir.exists():
        return []
    
    matched_skills = []
    
    for skill_dir in skills_dir.rglob("*"):
        if not skill_dir.is_dir():
            continue
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            continue
        
        try:
            content = skill_md.read_text(encoding="utf-8")
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    import yaml
                    frontmatter = yaml.safe_load(parts[1])
                    if isinstance(frontmatter, dict):
                        name = frontmatter.get("name", "")
                        cat = str(frontmatter.get("category", "general"))
                        if name and fnmatch.fnmatch(cat, category_pattern):
                            matched_skills.append(name)
        except Exception:
            continue
    
    return matched_skills
