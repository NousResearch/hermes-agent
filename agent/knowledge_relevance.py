"""Cross-project relevance engine for the 3-Tier Knowledge Center.

Determines whether knowledge created in one project is relevant to other
projects based on stack similarity, domain overlap, and keyword matching.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Keywords that indicate cross-project reusable patterns
_CROSS_PROJECT_KEYWORDS = [
    "pattern", "approach", "workflow", "playbook", "lesson",
    "best practice", "anti-pattern", "gotcha", "workaround",
    "migration", "setup", "configuration", "integration",
    "deployment", "pipeline", "architecture", "design",
]

# Stack similarity weights
_STACK_SIMILARITY = {
    ("node/next", "node/vite"): 0.8,
    ("node/next", "node"): 0.6,
    ("node/vite", "node"): 0.6,
    ("node/next", "node/next"): 1.0,
    ("node/vite", "node/vite"): 1.0,
    ("node", "node"): 1.0,
    ("python", "python"): 1.0,
    ("docker/mixed", "docker/mixed"): 1.0,
    ("docker/mixed", "node"): 0.4,
    ("docker/mixed", "python"): 0.4,
}


class KnowledgeRelevanceEngine:
    """Determines cross-project relevance of knowledge content."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        if vault_path is None:
            vault_path = Path.home() / "ObsidianVault" / "HermesAgent"
        self.vault_path = Path(vault_path)
        self._project_metadata: Dict[str, Dict[str, str]] = {}
        self._load_project_metadata()

    def _load_project_metadata(self) -> None:
        """Load project metadata from vault notes."""
        projects_dir = self.vault_path / "projects"
        if not projects_dir.exists():
            return
        for note_file in projects_dir.glob("*.md"):
            if note_file.name == "README.md":
                continue
            try:
                text = note_file.read_text(encoding="utf-8", errors="ignore")
                slug_match = re.search(r"^project_slug:\s*(.+)$", text, re.MULTILINE)
                stack_match = re.search(r"Stack\s*\|\s*`([^`]+)`", text)
                domain_match = re.search(r"^domain:\s*\[(.+?)\]", text, re.MULTILINE)
                if slug_match:
                    slug = slug_match.group(1).strip()
                    self._project_metadata[slug] = {
                        "stack": stack_match.group(1).strip() if stack_match else "unknown/mixed",
                        "domains": [d.strip().strip("'\"") for d in domain_match.group(1).split(",")] if domain_match else [],
                    }
            except Exception as e:
                logger.debug("Error reading project note %s: %s", note_file, e)

    def is_cross_project_relevant(self, content: str, source_project: str) -> bool:
        """Check if content pattern exists in ≥2 projects.

        Args:
            content: Knowledge text to evaluate
            source_project: Project slug where knowledge originated

        Returns:
            True if the knowledge is likely relevant to other projects
        """
        matching_projects = self.find_matching_projects(content, source_project)
        return len(matching_projects) >= 1  # At least 1 other project

    def find_matching_projects(self, content: str, source_project: str) -> List[str]:
        """Return project slugs that would benefit from this knowledge.

        Uses domain overlap + keyword matching + stack similarity.

        Args:
            content: Knowledge text to evaluate
            source_project: Source project slug

        Returns:
            List of matching project slugs (excluding source)
        """
        from agent.knowledge_domains import DomainRelevanceMatcher
        source_meta = self._project_metadata.get(source_project, {})
        source_domains = set(source_meta.get("domains", []))
        source_stack = source_meta.get("stack", "unknown/mixed")

        if not source_domains:
            # Fallback: infer domains from content
            matcher = DomainRelevanceMatcher(vault_path=self.vault_path)
            source_domains = set(matcher.match_knowledge(content))

        if not source_domains:
            return []

        matching: List[str] = []
        for slug, meta in self._project_metadata.items():
            if slug == source_project:
                continue
            project_domains = set(meta.get("domains", []))
            project_stack = meta.get("stack", "unknown/mixed")

            # Domain overlap score
            domain_overlap = len(source_domains & project_domains) / max(len(source_domains | project_domains), 1)

            # Stack similarity score
            stack_sim = _STACK_SIMILARITY.get((source_stack, project_stack), 0.2)

            # Keyword relevance
            content_lower = content.lower()
            keyword_score = sum(1 for kw in _CROSS_PROJECT_KEYWORDS if kw in content_lower) / len(_CROSS_PROJECT_KEYWORDS)

            # Combined score
            combined = 0.5 * domain_overlap + 0.3 * stack_sim + 0.2 * keyword_score

            if combined > 0.3:  # Threshold for relevance
                matching.append(slug)

        # Sort by domain overlap descending
        matching.sort(key=lambda s: len(source_domains & set(self._project_metadata.get(s, {}).get("domains", []))), reverse=True)
        return matching

    def get_relevance_score(self, content: str, target_project: str) -> float:
        """Score how relevant content is to a target project (0.0–1.0).

        Args:
            content: Knowledge text to evaluate
            target_project: Target project slug

        Returns:
            Float 0.0–1.0
        """
        from agent.knowledge_domains import DomainRelevanceMatcher
        source_meta = self._project_metadata.get(target_project, {})
        project_domains = set(source_meta.get("domains", []))
        project_stack = source_meta.get("stack", "unknown/mixed")

        if not project_domains:
            matcher = DomainRelevanceMatcher(vault_path=self.vault_path)
            project_domains = set(matcher.match_knowledge(content))

        if not project_domains:
            return 0.0

        content_lower = content.lower()
        domain_match = 0
        for domain in project_domains:
            if domain in content_lower:
                domain_match += 1
        domain_score = domain_match / len(project_domains)

        keyword_score = sum(1 for kw in _CROSS_PROJECT_KEYWORDS if kw in content_lower) / len(_CROSS_PROJECT_KEYWORDS)

        return 0.6 * domain_score + 0.4 * keyword_score
