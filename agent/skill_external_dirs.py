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
            if not fnmatch.fnmatch(name, pattern):
                return False
    return True


def determine_category(skill_name: str, category_map: Dict[str, str]) -> str:
    """Determine skill category from category_map patterns."""
    for pattern, category in category_map.items():
        if fnmatch.fnmatch(skill_name, pattern):
            return category
    return "external"
