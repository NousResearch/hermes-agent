"""Domain relevance matcher for the 3-Tier Knowledge Center.

Maps projects to knowledge domains and determines which domain notes to load
for a given project or piece of content. Filesystem-only — no network calls.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Domain keywords for heuristic matching
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "frontend": [
        "react", "next.js", "nextjs", "vite", "component", "jsx", "tsx",
        "css", "tailwind", "bootstrap", "responsive", "browser", "ui",
        "frontend", "front-end", "html", "dom", "accessibility", "a11y",
        "webpack", "esbuild", "babel", "styled-components", "material-ui",
    ],
    "backend": [
        "api", "endpoint", "database", "db", "sql", "postgres", "mysql",
        "mongodb", "redis", "auth", "authentication", "server", "backend",
        "back-end", "rest", "graphql", "middleware", "route", "controller",
        "service", "model", "migration", "orm", "prisma", "drizzle",
    ],
    "devops": [
        "docker", "compose", "ci/cd", "cicd", "pipeline", "deploy",
        "deployment", "kubernetes", "k8s", "terraform", "ansible",
        "github actions", "jenkins", "monitoring", "prometheus", "grafana",
        "nginx", "reverse proxy", "ssl", "tls", "certificate",
    ],
    "security": [
        "secret", "api key", "token", "password", "credential", "oauth",
        "jwt", "rbac", "permission", "authorization", "vulnerability",
        "xss", "csrf", "injection", "encryption", "hash", "salt",
        "rate limit", "cors", "csp", "security",
    ],
    "testing": [
        "test", "unit test", "integration test", "e2e", "end-to-end",
        "pytest", "jest", "vitest", "cypress", "playwright", "mock",
        "fixture", "assertion", "coverage", "browser test", "localhost",
        "verification", "smoke test", "regression",
    ],
    "data": [
        "data pipeline", "etl", "ml", "machine learning", "model",
        "training", "inference", "embedding", "vector", "embedding",
        "analysis", "pandas", "numpy", "dataframe", "chart",
        "visualization", "analytics", "dashboard data",
    ],
    "mobile": [
        "mobile", "responsive", "pwa", "progressive web app", "ios",
        "android", "react native", "flutter", "touch", "swipe",
        "viewport", "mobile-first",
    ],
    "infrastructure": [
        "vps", "server", "cloud", "aws", "gcp", "azure", "networking",
        "dns", "load balancer", "cdn", "capacity", "scaling",
        "infrastructure", "provisioning", "ssh", "firewall",
    ],
    "business": [
        "business", "company", "synerry", "owner", "md", "strategy",
        "positioning", "proposal", "pitch", "pitching", "tor",
        "go/no-go", "case study", "client", "customer", "stakeholder",
        "ธุรกิจ", "บริษัท", "ประมูล", "ข้อเสนอ", "ลูกค้า",
    ],
    "marketing": [
        "marketing", "market research", "persona", "fgd", "survey",
        "competitive", "competitor", "positioning map", "sentiment",
        "trend", "concept testing", "brand", "campaign", "content",
        "การตลาด", "แบรนด์", "กลุ่มเป้าหมาย", "คู่แข่ง",
    ],
    "sales": [
        "sales", "pipeline", "lead", "deal", "revenue", "upsell",
        "cross-sell", "crm", "account", "proposal", "quote", "close",
        "win rate", "pitch", "client revenue", "ขาย", "รายได้",
    ],
    "finance": [
        "finance", "margin", "cash", "budget", "cost", "profit",
        "loss", "invoice", "accounting", "pricing", "burn", "runway",
        "เงิน", "บัญชี", "กำไร", "ต้นทุน", "งบประมาณ",
    ],
    "operations": [
        "operations", "production", "qc", "delivery", "workflow",
        "process", "sla", "support", "handoff", "timeline", "resource",
        "capacity", "pm", "project manager", "ส่งมอบ", "กระบวนการ",
    ],
    "people": [
        "hr", "people", "employee", "staff", "culture", "performance",
        "1on1", "swot", "feedback", "hiring", "role", "team",
        "พนักงาน", "ทีม", "วัฒนธรรม", "บุคลากร",
    ],
}


class DomainRelevanceMatcher:
    """Matches projects and content to knowledge domains.

    Uses a combination of:
    1. Explicit domain tags from project frontmatter
    2. Keyword-based heuristic matching for arbitrary content
    3. Stack-based domain inference
    """

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        if vault_path is None:
            vault_path = Path.home() / "ObsidianVault" / "HermesAgent"
        self.vault_path = Path(vault_path)
        self._domain_cache: Dict[str, List[str]] = {}
        self._project_domains: Dict[str, List[str]] = {}
        self._load_project_domains()

    def _load_project_domains(self) -> None:
        """Load domain assignments from project vault notes frontmatter."""
        projects_dir = self.vault_path / "projects"
        if not projects_dir.exists():
            logger.warning("Projects dir not found: %s", projects_dir)
            return
        for note_file in projects_dir.glob("*.md"):
            if note_file.name == "README.md":
                continue
            text = note_file.read_text(encoding="utf-8", errors="ignore")
            # Extract project_slug from frontmatter
            slug_match = re.search(r"^project_slug:\s*(.+)$", text, re.MULTILINE)
            domain_match = re.search(r"^domain:\s*\[(.+?)\]", text, re.MULTILINE)
            if slug_match and domain_match:
                slug = slug_match.group(1).strip()
                domains = [d.strip().strip("'\"") for d in domain_match.group(1).split(",")]
                self._project_domains[slug] = domains

    def classify(self, project_slug: str) -> List[str]:
        """Return domain slugs for a project.

        Args:
            project_slug: The project slug (e.g., 'tech-tools-hermes-agent')

        Returns:
            List of domain slugs (e.g., ['frontend', 'backend'])
        """
        if project_slug in self._project_domains:
            return self._project_domains[project_slug]
        # Fallback: check cache
        if project_slug in self._domain_cache:
            return self._domain_cache[project_slug]
        # Unknown project — return empty (caller should handle)
        logger.debug("Unknown project slug: %s", project_slug)
        return []

    def match_knowledge(self, content: str) -> List[str]:
        """Return relevant domain slugs for arbitrary text content.

        Uses keyword matching against _DOMAIN_KEYWORDS.

        Args:
            content: Text to analyze (e.g., a knowledge note body)

        Returns:
            List of matching domain slugs, sorted by match count descending
        """
        content_lower = content.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in _DOMAIN_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in content_lower)
            if count > 0:
                scores[domain] = count
        # Return domains sorted by score descending
        return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)

    def get_relevance_score(self, content: str, target_project: str) -> float:
        """Score how relevant content is to a target project (0.0–1.0).

        Args:
            content: Knowledge text to evaluate
            target_project: Project slug to check against

        Returns:
            Float 0.0 (no relevance) to 1.0 (highly relevant)
        """
        project_domains = self.classify(target_project)
        if not project_domains:
            return 0.0
        content_domains = self.match_knowledge(content)
        if not content_domains:
            return 0.0
        # Jaccard-like overlap
        overlap = len(set(project_domains) & set(content_domains))
        union = len(set(project_domains) | set(content_domains))
        return overlap / union if union > 0 else 0.0

    def get_domain_notes(self, domains: List[str]) -> List[Path]:
        """Return paths to domain KB notes for the given domains.

        Scans each domain directory for .md files (excluding README.md). The
        current restored Obsidian graph stores domain-scoped knowledge under
        ``knowledge/domain/<domain>/``; the older standalone layout used
        ``domains/<domain>/``. Both are supported for migration compatibility.

        Args:
            domains: List of domain slugs

        Returns:
            List of Path objects to domain KB note files
        """
        notes: List[Path] = []
        roots = [
            self.vault_path / "knowledge" / "domain",
            self.vault_path / "domains",
        ]
        for domain_slug in domains:
            for root in roots:
                domain_dir = root / domain_slug
                if not domain_dir.exists():
                    logger.debug("Domain dir not found: %s", domain_dir)
                    continue
                for note_file in sorted(domain_dir.glob("*.md")):
                    if note_file.name == "README.md":
                        continue
                    notes.append(note_file)
        return notes
