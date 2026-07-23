"""Skill dependency resolution with safeguards.

Skills can declare dependencies in their SKILL.md frontmatter using the
``dependencies`` and ``optional_dependencies`` fields.  The ``SkillDependencyResolver``
resolves the full dependency graph while enforcing four safeguards:

1. **Cycle detection** — prevents infinite loops (always on)
2. **Depth limit** — prevents deep chains (default 3, max 5)
3. **Skill count limit** — prevents explosion (default 10, max 20)
4. **Token budget** — prevents context saturation (default 10,000, max 50,000)

All limits are configurable via ``skills.dependencies`` in config.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class LoadedSkill:
    """Represents a loaded dependency skill with metadata."""
    
    name: str
    is_optional: bool = False
    is_parent: bool = False
    depth: int = 0
    tokens_estimated: int = 0
    content_snippet: str = ""
    """First 200 chars of the skill content for loading-time feedback."""


@dataclass
class DependencyResolutionError:
    """Describes why a dependency was skipped or failed to load."""
    
    skill_name: str
    reason: str  # "cycle", "depth_limit", "count_limit", "token_budget", "not_found", "io_error"


@dataclass
class DependencyResolutionResult:
    """Result of resolving a skill's dependencies."""
    
    loaded: List[LoadedSkill] = field(default_factory=list)
    errors: List[DependencyResolutionError] = field(default_factory=list)
    total_tokens: int = 0
    total_skills: int = 0
    max_depth_reached: int = 0
    
    @property
    def was_truncated(self) -> bool:
        """True if any safeguards blocked further dependency loading."""
        return len(self.errors) > 0


class SkillDependencyResolver:
    """Resolves skill dependencies with configurable safeguards.
    
    Usage::
    
        resolver = SkillDependencyResolver()
        result = resolver.resolve("rapidwebs-infra-deployment")
        for skill in result.loaded:
            print(f"Loaded {skill.name} at depth {skill.depth}")
    """
    
    _INSTANCE: "Optional[SkillDependencyResolver]" = None
    
    def __init__(
        self,
        max_depth: int = 3,
        max_skills: int = 10,
        max_tokens: int = 10_000,
        enabled: bool = True,
        lazy_load: bool = True,
    ):
        self.max_depth = max(1, min(5, max_depth))          # 1..5
        self.max_skills = max(1, min(20, max_skills))       # 1..20
        self.max_tokens = max(1_000, min(50_000, max_tokens))  # 1K..50K
        self.enabled = enabled
        self.lazy_load = lazy_load
    
    @classmethod
    def get_instance(cls) -> "SkillDependencyResolver":
        """Get the singleton instance, initialized from config."""
        if cls._INSTANCE is not None:
            return cls._INSTANCE
        
        # Read config
        try:
            from hermes_cli.config import load_config
            config = load_config()
            deps_cfg = config.get("skills", {}).get("dependencies", {})
        except Exception:
            deps_cfg = {}
        
        cls._INSTANCE = cls(
            max_depth=deps_cfg.get("max_depth", 3),
            max_skills=deps_cfg.get("max_skills", 10),
            max_tokens=deps_cfg.get("max_tokens", 10_000),
            enabled=deps_cfg.get("enabled", True),
            lazy_load=deps_cfg.get("lazy_load", True),
        )
        return cls._INSTANCE
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._INSTANCE = None
    
    def resolve(self, skill_name: str) -> DependencyResolutionResult:
        """Resolve dependencies for a skill.
        
        Returns a ``DependencyResolutionResult`` with loaded skills and
        any errors from skipped dependencies.
        
        If the resolver is disabled, returns an empty result.
        If the skill has no dependencies, returns just the skill itself.
        The requested skill is always the first entry in ``loaded``.
        """
        result = DependencyResolutionResult()
        
        if not self.enabled:
            return result
        
        visited: Set[str] = set()
        
        def _resolve(name: str, depth: int, is_optional: bool = False) -> None:
            nonlocal result
            
            # ── 1. Cycle detection ─────────────────────────────
            if name in visited:
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason="cycle",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            # ── 2. Depth limit ─────────────────────────────────
            if depth > self.max_depth:
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason=f"depth_limit: max_depth={self.max_depth}",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            # ── 3. Skill count limit ───────────────────────────
            if result.total_skills >= self.max_skills:
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason=f"count_limit: max_skills={self.max_skills}",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            visited.add(name)
            
            # ── Load the skill ─────────────────────────────────
            try:
                load_result = self._load_skill_content(name)
            except Exception as e:
                if is_optional:
                    # Optional deps fail silently
                    return
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason=f"io_error: {e}",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            if load_result is None:
                if is_optional:
                    return
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason="not_found",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            content, dependencies, optional_deps = load_result
            
            # ── Estimate tokens ────────────────────────────────
            estimated_tokens = _estimate_tokens(content)
            
            # ── 4. Token budget ────────────────────────────────
            if result.total_tokens + estimated_tokens > self.max_tokens:
                result.errors.append(DependencyResolutionError(
                    skill_name=name,
                    reason=f"token_budget: max_tokens={self.max_tokens}, "
                           f"estimated={estimated_tokens + result.total_tokens}",
                ))
                if depth > result.max_depth_reached:
                    result.max_depth_reached = depth
                return
            
            # ── Record the loaded skill ───────────────────────
            loaded = LoadedSkill(
                name=name,
                is_optional=is_optional,
                is_parent=(depth == 0),
                depth=depth,
                tokens_estimated=estimated_tokens,
                content_snippet=content[:200].replace("\n", " ").strip(),
            )
            result.loaded.append(loaded)
            result.total_skills += 1
            result.total_tokens += estimated_tokens
            if depth > result.max_depth_reached:
                result.max_depth_reached = depth
            
            # ── Resolve required dependencies (depth + 1) ────
            for dep_name in dependencies:
                _resolve(dep_name, depth + 1, is_optional=False)
            
            # ── Resolve optional dependencies (depth + 1) ─────
            for dep_name in optional_deps:
                _resolve(dep_name, depth + 1, is_optional=True)
        
        # Start resolution at depth 0
        _resolve(skill_name, 0, is_optional=False)
        
        return result
    
    def _load_skill_content(self, name: str):
        """Load a skill's content and frontmatter.
        
        Returns ``(content, dependencies, optional_deps)`` or ``None``
        if the skill is not found.
        """
        from pathlib import Path
        from hermes_constants import get_skills_dir
        
        skills_dir = get_skills_dir()
        if not skills_dir.exists():
            return None
        
        for skill_dir in skills_dir.rglob("*"):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            
            try:
                content = skill_md.read_text(encoding="utf-8")
                if not content.startswith("---"):
                    continue
                
                parts = content.split("---", 2)
                if len(parts) < 3:
                    continue
                
                import yaml
                frontmatter = yaml.safe_load(parts[1])
                if not isinstance(frontmatter, dict):
                    continue
                
                fm_name = str(frontmatter.get("name", ""))
                if fm_name != name:
                    continue
                
                dependencies = list(frontmatter.get("dependencies", []) or [])
                optional_deps = list(frontmatter.get("optional_dependencies", []) or [])
                
                return (parts[2], dependencies, optional_deps)
            except Exception:
                continue
        
        return None


def _estimate_tokens(text: str) -> int:
    """Estimate token count for a text string.
    
    Uses a simple word-based estimator (words * 1.3 + newlines).
    This is intentionally NOT an exact tokenizer — it's a cheap guard
    that runs during dependency resolution, before any LLM call.
    
    Returns a rough upper-bound estimate.
    """
    if not text:
        return 0
    words = len(text.split())
    lines = text.count("\n")
    return int(words * 1.3 + lines * 0.5)


def resolve_skill_dependencies(
    skill_name: str,
    force: bool = False,
) -> DependencyResolutionResult:
    """Convenience wrapper to resolve skill dependencies.
    
    Args:
        skill_name: The skill name to resolve dependencies for.
        force: If True, re-reads config from disk rather than using
               the singleton's cached config.
    
    Returns:
        A ``DependencyResolutionResult``.
    """
    if force:
        SkillDependencyResolver.reset_instance()
    resolver = SkillDependencyResolver.get_instance()
    return resolver.resolve(skill_name)
