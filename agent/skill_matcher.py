"""
Skill Matcher — automatic skill selection based on user input.

Automatically detects relevant skills from user messages, reducing the need
for manual ``/<skill-name>`` invocation.

Three matching strategies (applied in order):
1. Keyword exact match — user message text × SKILL.md ``trigger_keywords``
2. Context hint match   — file extensions → skill (e.g. ``.pdf`` → ``pdf`` skill)
3. Fuzzy match          — Jaccard similarity on skill descriptions

Usage:
    matcher = SkillMatcher(tier_data={"xbrowser": ("frequent", 42)})
    results = matcher.match("help me edit a PDF")
    # → [MatchResult(skill=SkillDescriptor(name="pdf"), score=0.8, ...)]

The matcher accepts optional ``tier_data`` from SkillTierManager for weighted
scoring — skills in higher tiers get a boost.  No direct dependency on
SkillTierManager; pass data as a plain dict for loose coupling.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Default search paths for skills
SKILL_SEARCH_PATHS = [
    Path(__file__).resolve().parent.parent / "skills",            # repo-level skills/
    Path.home() / ".hermes" / "skills",                           # user skills
]

# File extension → skill name mapping (for context hints)
_EXTENSION_MAP: Dict[str, str] = {
    ".docx": "docx",
    ".doc": "docx",
    ".pdf": "pdf",
    ".xlsx": "xlsx",
    ".xls": "xlsx",
    ".csv": "xlsx",
    ".pptx": "pptx",
    ".ppt": "pptx",
    ".md": "markdown",
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".html": "html",
    ".css": "css",
}


# -- Data model ---------------------------------------------------------------

@dataclass
class SkillDescriptor:
    """Describes a parsed skill and its current tier/usage metadata."""
    name: str
    description: str = ""
    trigger_keywords: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)
    path: Path = field(default_factory=Path)
    tier: str = "archived"
    usage_count: int = 0

    @property
    def match_weight(self) -> float:
        """Compute a scoring weight from tier and usage."""
        tier_weight = {"builtin": 1.0, "frequent": 0.8, "archived": 0.3}
        base = tier_weight.get(self.tier, 0.2)
        usage_bonus = min(self.usage_count / 50.0, 0.3)  # cap at +0.3
        return base + usage_bonus


@dataclass
class MatchResult:
    """Result of a single skill match."""
    skill: SkillDescriptor
    score: float
    matched_keywords: List[str] = field(default_factory=list)
    match_type: str = "keyword"      # keyword | context | fuzzy
    auto_activate: bool = True


# -- YAML frontmatter parser (lightweight, no pyyaml dependency) --------------

def _parse_yaml_frontmatter(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parse YAML frontmatter from a SKILL.md file.

    Only ``name``, ``description``, ``trigger_keywords``, and ``platforms``
    fields are extracted.  No pyyaml dependency.
    """
    try:
        text = filepath.read_text(encoding="utf-8")
    except Exception:
        return None

    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return None

    raw = match.group(1)
    result: Dict[str, Any] = {}

    for key in ["name", "description"]:
        m = re.search(rf"^{key}:\s*\"?(.+?)\"?\s*$", raw, re.MULTILINE)
        if m:
            result[key] = m.group(1).strip()

    # List fields
    for list_key in ["trigger_keywords", "platforms"]:
        block = re.search(
            rf"^{list_key}:\s*\n((?:\s+-\s+.+\n?)*)", raw, re.MULTILINE
        )
        if block:
            items = re.findall(
                r"^\s+-\s*\"?(.+?)\"?\s*$", block.group(1), re.MULTILINE
            )
            result[list_key] = [i.strip().lower() for i in items]

    return result


# -- Matcher ------------------------------------------------------------------

class SkillMatcher:
    """Auto-match skills from user messages.

    Args:
        skill_paths: Additional skill search paths (appended to defaults).
        tier_data:  Optional dict ``{skill_name: (tier, usage_count)}`` from
                    SkillTierManager for weighted scoring.  Keeps coupling loose.
    """

    def __init__(
        self,
        skill_paths: Optional[List[Path]] = None,
        tier_data: Optional[Dict[str, Tuple[str, int]]] = None,
    ):
        self._search_paths = list(SKILL_SEARCH_PATHS)
        if skill_paths:
            self._search_paths.extend(skill_paths)

        self._tier_data = tier_data or {}

        self._skills: Dict[str, SkillDescriptor] = {}
        self._keyword_index: Dict[str, List[str]] = {}
        self._loaded = False

        # Co-occurrence tracking: sorted pair → count
        self._co_occurrence: Dict[Tuple[str, str], int] = {}

    def set_tier_data(self, data: Dict[str, Tuple[str, int]]):
        """Update tier/usage data (e.g. after SkillTierManager reloads)."""
        self._tier_data = data
        self._loaded = False  # force rebuild to re-apply weights

    # -- Index ----------------------------------------------------------------

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._build_index()
        self._loaded = True

    def _build_index(self):
        """Scan all search paths and build the skill index."""
        self._skills.clear()
        self._keyword_index.clear()

        for search_path in self._search_paths:
            if not search_path.exists():
                continue

            for item in search_path.iterdir():
                skill_md = item / "SKILL.md" if item.is_dir() else item

                if not isinstance(skill_md, Path) or not skill_md.name.endswith(".md"):
                    continue

                fm = _parse_yaml_frontmatter(skill_md)
                if fm is None:
                    name = item.name if item.is_dir() else skill_md.stem
                else:
                    name = fm.get("name", skill_md.stem)

                tier, count = self._tier_data.get(name, ("archived", 0))

                desc = SkillDescriptor(
                    name=name,
                    description=fm.get("description", "") if fm else "",
                    trigger_keywords=fm.get("trigger_keywords", []) if fm else [],
                    platforms=fm.get("platforms", []) if fm else [],
                    path=skill_md,
                    tier=tier,
                    usage_count=count,
                )

                self._skills[name] = desc

                for kw in desc.trigger_keywords:
                    kw = kw.lower().strip()
                    self._keyword_index.setdefault(kw, [])
                    if name not in self._keyword_index[kw]:
                        self._keyword_index[kw].append(name)

        logger.debug(
            "Index built: %d skills, %d keywords", len(self._skills), len(self._keyword_index)
        )

    def reload(self):
        """Force re-index (call after skills directory changes)."""
        self._loaded = False
        self._build_index()
        self._loaded = True

    # -- Matching -------------------------------------------------------------

    def match(
        self,
        user_message: str,
        context_hints: Optional[List[str]] = None,
        max_results: int = 5,
        min_score: float = 0.1,
    ) -> List[MatchResult]:
        """Match skills against a user message.

        Args:
            user_message: The user's input text.
            context_hints: e.g. file extensions [".pdf", ".py"].
            max_results: Max results to return.
            min_score: Minimum score threshold (0-1).

        Returns:
            Results sorted by score descending.
        """
        self._ensure_loaded()

        text = user_message.lower().strip()
        results: Dict[str, MatchResult] = {}

        # Strategy 1: keyword exact match
        self._match_by_keyword(text, results)

        # Strategy 2: context hints (file extensions)
        if context_hints:
            self._match_by_context(context_hints, results)

        # Strategy 3: description Jaccard similarity
        self._match_by_fuzzy(text, results)

        filtered = [r for r in results.values() if r.score >= min_score]
        filtered.sort(key=lambda r: (-r.score, -r.skill.match_weight))
        return filtered[:max_results]

    def _match_by_keyword(self, text: str, results: Dict[str, MatchResult]):
        """Direct keyword-to-trigger_keywords match."""
        for keyword, skill_names in self._keyword_index.items():
            if keyword not in text:
                continue
            for name in skill_names:
                desc = self._skills.get(name)
                if not desc:
                    continue
                score = 0.8 + desc.match_weight * 0.2
                if name in results:
                    results[name].score = max(results[name].score, score)
                    results[name].matched_keywords.append(keyword)
                else:
                    results[name] = MatchResult(
                        skill=desc,
                        score=score,
                        matched_keywords=[keyword],
                        match_type="keyword",
                    )

    def _match_by_context(self, hints: List[str], results: Dict[str, MatchResult]):
        """Match file-extension hints to skills."""
        for hint in hints:
            skill_name = _EXTENSION_MAP.get(hint.lower().strip())
            if skill_name and skill_name in self._skills:
                results[skill_name] = MatchResult(
                    skill=self._skills[skill_name],
                    score=0.7,
                    matched_keywords=[f"ext:{hint}"],
                    match_type="context",
                )

    def _match_by_fuzzy(self, text: str, results: Dict[str, MatchResult]):
        """Jaccard similarity between message and skill description."""
        text_words = set(text.split())
        if not text_words:
            return

        for name, desc in self._skills.items():
            if name in results:
                continue  # already matched

            desc_text = (desc.description or "").lower()
            desc_words = set(desc_text.split())
            if not desc_words:
                continue

            intersection = text_words & desc_words
            union = text_words | desc_words
            score = len(intersection) / len(union) * 0.5  # fuzzy has lower weight

            if score > 0.1:
                results[name] = MatchResult(
                    skill=desc,
                    score=score,
                    matched_keywords=list(intersection),
                    match_type="fuzzy",
                )

    # -- Co-occurrence --------------------------------------------------------

    def record_co_occurrence(self, skill_a: str, skill_b: str):
        """Note that two skills were used together."""
        if skill_a == skill_b:
            return
        pair = tuple(sorted([skill_a, skill_b]))
        self._co_occurrence[pair] = self._co_occurrence.get(pair, 0) + 1

    # -- Queries --------------------------------------------------------------

    def list_skills(self, tier_filter: Optional[str] = None) -> List[SkillDescriptor]:
        self._ensure_loaded()
        if tier_filter:
            return [d for d in self._skills.values() if d.tier == tier_filter]
        return list(self._skills.values())

    def get_skill(self, name: str) -> Optional[SkillDescriptor]:
        self._ensure_loaded()
        return self._skills.get(name)

    def search_skills(self, query: str, limit: int = 10) -> List[SkillDescriptor]:
        """Search skills by name or description substring."""
        self._ensure_loaded()
        q = query.lower()
        results = [
            d for d in self._skills.values()
            if q in d.name.lower() or q in d.description.lower()
        ]
        results.sort(key=lambda d: -d.match_weight)
        return results[:limit]

    def get_matching_keywords(self) -> List[str]:
        """Return all indexed trigger keywords (for debugging)."""
        self._ensure_loaded()
        return sorted(self._keyword_index.keys())


# -- Global convenience accessor ---------------------------------------------

_matcher_instance: Optional[SkillMatcher] = None


def get_matcher() -> SkillMatcher:
    """Return the global SkillMatcher singleton."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = SkillMatcher()
    return _matcher_instance


def auto_match_skills(
    user_message: str,
    context_hints: Optional[List[str]] = None,
) -> List[str]:
    """Convenience: match and return skill names, highest-scored first."""
    return [
        r.skill.name
        for r in get_matcher().match(user_message, context_hints)
        if r.auto_activate
    ]
