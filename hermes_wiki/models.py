from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional


class PageType(str, Enum):
    ENTITY = "entity"
    CONCEPT = "concept"
    COMPARISON = "comparison"
    QUERY = "query"
    SUMMARY = "summary"


class IssueSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class WikiPage:
    path: Path
    title: str
    page_type: PageType
    tags: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created: date | None = None
    updated: date | None = None
    confidence: str | None = None
    contested: bool = False
    contradictions: list[str] = field(default_factory=list)
    content: str = ""
    frontmatter: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return self.path.stem

    @property
    def wikilink(self) -> str:
        return f"[[{self.slug}]]"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()[:16]


@dataclass
class Source:
    path: Path
    source_url: str | None = None
    ingested: date | None = None
    sha256: str | None = None
    content: str = ""

    def compute_body_hash(self) -> str:
        body = self.content
        if body.startswith("---"):
            end = body.find("---", 3)
            if end != -1:
                body = body[end + 3:].strip()
        return hashlib.sha256(body.encode()).hexdigest()

    @property
    def has_drifted(self) -> bool:
        if not self.sha256:
            return False
        return self.compute_body_hash() != self.sha256


@dataclass
class LogEntry:
    date: date
    action: str
    subject: str
    details: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## [{self.date.isoformat()}] {self.action} | {self.subject}"]
        for detail in self.details:
            lines.append(f"- {detail}")
        return "\n".join(lines)


@dataclass
class LintIssue:
    severity: IssueSeverity
    category: str
    message: str
    file_path: str | None = None
    suggestion: str | None = None

    def __str__(self) -> str:
        prefix = f"[{self.severity.value.upper()}]"
        loc = f" ({self.file_path})" if self.file_path else ""
        sug = f" → {self.suggestion}" if self.suggestion else ""
        return f"{prefix} {self.category}: {self.message}{loc}{sug}"


@dataclass
class IngestResult:
    source_path: str
    pages_created: list[str] = field(default_factory=list)
    pages_updated: list[str] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    concepts_found: list[str] = field(default_factory=list)
    key_facts: list[str] = field(default_factory=list)
    chunks_indexed: int = 0

    @property
    def total_pages_touched(self) -> int:
        return len(self.pages_created) + len(self.pages_updated)

    def summary(self) -> str:
        parts = [f"Ingested: {self.source_path}"]
        if self.pages_created:
            parts.append(f"Created {len(self.pages_created)} pages: {', '.join(self.pages_created)}")
        if self.pages_updated:
            parts.append(f"Updated {len(self.pages_updated)} pages: {', '.join(self.pages_updated)}")
        parts.append(f"Indexed {self.chunks_indexed} chunks for search")
        if self.entities_found:
            parts.append(f"Entities: {', '.join(self.entities_found)}")
        if self.concepts_found:
            parts.append(f"Concepts: {', '.join(self.concepts_found)}")
        return "\n".join(parts)


@dataclass
class QueryResult:
    question: str
    answer: str
    sources_consulted: list[str] = field(default_factory=list)
    confidence: str = "medium"
    filed_as: str | None = None

    def summary(self) -> str:
        parts = [self.answer]
        if self.sources_consulted:
            parts.append(f"\nSources: {', '.join(self.sources_consulted)}")
        if self.filed_as:
            parts.append(f"Filed as: {self.filed_as}")
        return "\n".join(parts)


@dataclass
class LintReport:
    issues: list[LintIssue] = field(default_factory=list)
    total_pages: int = 0
    total_sources: int = 0
    total_links: int = 0
    orphan_pages: int = 0
    broken_links: int = 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == IssueSeverity.WARNING)

    def summary(self) -> str:
        parts = [
            f"Wiki Lint Report: {len(self.issues)} issues "
            f"({self.error_count} errors, {self.warning_count} warnings)",
            f"Pages: {self.total_pages} | Sources: {self.total_sources} | "
            f"Links: {self.total_links} | Orphans: {self.orphan_pages} | "
            f"Broken: {self.broken_links}",
        ]
        for sev in [IssueSeverity.ERROR, IssueSeverity.WARNING, IssueSeverity.INFO]:
            issues = [i for i in self.issues if i.severity == sev]
            if issues:
                parts.append(f"\n{'─' * 40}")
                parts.append(f"{sev.value.upper()} ({len(issues)}):")
                for issue in issues:
                    parts.append(f"  {issue}")
        return "\n".join(parts)
