"""Trigger-based skill auto-loading.

Scans SKILL.md frontmatter for 'triggers:' keywords and automatically
loads matching skills when user input contains trigger phrases.

Usage:
    This module is integrated into the agent's prompt builder to
    auto-load skills before LLM calls when triggers match.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from hermes_constants import get_skills_dir

logger = logging.getLogger(__name__)

# Cache for the trigger index
_TRIGGER_INDEX: Optional["TriggerIndex"] = None


class TriggerIndex:
    """Maps trigger keywords to skill names for O(1) lookup."""
    
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or get_skills_dir()
        self.index: Dict[str, List[str]] = defaultdict(list)  # trigger -> [skill_names]
        self.skill_triggers: Dict[str, List[str]] = {}  # skill -> [triggers]
        self._built = False
    
    def build(self) -> Dict[str, List[str]]:
        """Scan all SKILL.md files, extract triggers, build reverse index."""
        self.index = defaultdict(list)
        self.skill_triggers = {}
        
        try:
            from agent.skill_utils import parse_frontmatter, is_excluded_skill_path
        except ImportError:
            # Fallback for testing
            parse_frontmatter = self._parse_frontmatter_simple
            is_excluded_skill_path = lambda x: False
        
        for skill_md in self.skills_dir.rglob("SKILL.md"):
            if is_excluded_skill_path(skill_md):
                continue
            
            skill_name = skill_md.parent.name
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, _ = parse_frontmatter(content)
                
                triggers = frontmatter.get("triggers", [])
                if triggers:
                    self.skill_triggers[skill_name] = triggers
                    for trigger in triggers:
                        normalized = self._normalize(trigger)
                        self.index[normalized].append(skill_name)
            except Exception as e:
                logger.debug(f"Failed to parse triggers for {skill_name}: {e}")
        
        self._built = True
        return dict(self.index)
    
    def _parse_frontmatter_simple(self, content: str) -> tuple:
        """Simple frontmatter parser for testing."""
        import yaml
        
        if not content.startswith("---"):
            return {}, content
        
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}, content
        
        try:
            frontmatter = yaml.safe_load(parts[1]) or {}
            return frontmatter, parts[2]
        except Exception:
            return {}, content
    
    def _normalize(self, trigger: str) -> str:
        """Normalize trigger for matching: lowercase, alphanumeric + spaces."""
        return re.sub(r"[^\w\s-]", "", trigger.lower()).strip()
    
    def match(self, user_input: str) -> List[str]:
        """Find skills matching user input. Returns unique skill names."""
        if not self._built:
            self.build()
        
        normalized_input = self._normalize(user_input)
        
        matched: Set[str] = set()
        
        # Check exact phrase matches first
        for trigger, skills in self.index.items():
            if trigger in normalized_input:
                matched.update(skills)
        
        return list(matched)
    
    def get_skill_triggers(self, skill_name: str) -> List[str]:
        """Return triggers for a specific skill."""
        return self.skill_triggers.get(skill_name, [])


def get_trigger_index() -> TriggerIndex:
    """Get or create the global trigger index."""
    global _TRIGGER_INDEX
    if _TRIGGER_INDEX is None:
        _TRIGGER_INDEX = TriggerIndex()
    return _TRIGGER_INDEX


def clear_trigger_index():
    """Clear the global trigger index (for testing)."""
    global _TRIGGER_INDEX
    _TRIGGER_INDEX = None


def match_skills_for_input(user_input: str) -> List[str]:
    """Match user input against skill triggers and return matching skill names."""
    index = get_trigger_index()
    return index.match(user_input)


def auto_load_skills(user_input: str, max_skills: int = 5) -> List[str]:
    """Auto-load skills matching user input triggers.
    
    Returns list of skill names that were loaded.
    """
    matched = match_skills_for_input(user_input)
    
    if not matched:
        return []
    
    # Limit to max_skills
    matched = matched[:max_skills]
    
    # Load each skill
    loaded = []
    for skill_name in matched:
        try:
            # Use skill_view to load the skill
            from hermes_tools import skill_view
            skill_view(name=skill_name)
            loaded.append(skill_name)
            logger.info(f"Auto-loaded skill: {skill_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-load skill {skill_name}: {e}")
    
    return loaded


# ── Integration with conversation loop ─────────────────────────────────────

def auto_load_skills_for_turn(user_message: str, max_skills: int = 5) -> str:
    """Auto-load skills matching user input and return context block.
    
    This function is called from the conversation loop before the LLM call.
    It matches user input against skill triggers, loads matching skills via
    skill_view, and returns a context block that can be injected into the
    user message via the api_content sidecar.
    
    Returns empty string if no skills match or if auto-loading is disabled.
    """
    from hermes_cli.config import cfg_get
    
    # Check if auto-loading is enabled (default: True)
    try:
        from hermes_cli.config import load_config
        config = load_config()
        skills_cfg = config.get("skills", {})
        auto_load_cfg = skills_cfg.get("auto_load", {})
        if isinstance(auto_load_cfg, dict):
            enabled = auto_load_cfg.get("enabled", True)
        else:
            enabled = bool(auto_load_cfg)
        if not enabled:
            return ""
    except Exception:
        pass
    
    # Match triggers
    matched = match_skills_for_input(user_message)
    if not matched:
        return ""
    
    # Limit to max_skills
    matched = matched[:max_skills]
    
    # Load each skill and collect content
    loaded_skills = []
    for skill_name in matched:
        try:
            from tools.skills_tool import skill_view
            result_json = skill_view(skill_name)
            import json
            result = json.loads(result_json)
            if result.get("success"):
                content = result.get("content", "")
                if content:
                    loaded_skills.append(f"## Auto-loaded Skill: {skill_name}\n\n{content[:3000]}")
                    logger.info(f"Auto-loaded skill: {skill_name}")
        except Exception as e:
            logger.warning(f"Failed to auto-load skill {skill_name}: {e}")
    
    if not loaded_skills:
        return ""
    
    # Build context block
    header = "The following skills have been automatically loaded based on your message:\n\n"
    skills_block = "\n\n---\n\n".join(loaded_skills)
    footer = "\n\n---\n\nFollow the instructions in these skills for your task."
    
    return header + skills_block + footer


def get_trigger_index_stats() -> dict:
    """Return statistics about the trigger index."""
    index = get_trigger_index()
    if not index._built:
        index.build()
    
    return {
        "total_triggers": len(index.index),
        "total_skills": len(index.skill_triggers),
        "triggers": dict(index.index),
    }
