"""Dual memory framework: personal workspace plus procedural skills.

This module keeps the two memory classes intentionally separate:

* Personal workspace (W): user-visible markdown knowledge assets organized
  with a PARA state machine.
* Procedural memory (S): agent-facing Skill Markdown distilled from repeated
  successful workflows.

The framework is local-file based and profile scoped through HERMES_HOME. It
does not register model tools or mutate the system prompt mid-conversation.
Callers can expose it through CLI commands, cron jobs, or a future background
memory agent without adding permanent core tool schema footprint.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Sequence

from hermes_constants import get_hermes_home, get_skills_dir

ParaBucket = Literal["Projects", "Areas", "Resources", "Archives"]

PARA_BUCKETS: tuple[ParaBucket, ...] = ("Projects", "Areas", "Resources", "Archives")
MANIFEST_NAME = "_manifest.md"


def default_workspace_root() -> Path:
    """Return the default profile-scoped personal workspace root."""
    return get_hermes_home() / "personal_workspace"


def default_procedural_skills_root() -> Path:
    """Return the default profile-scoped procedural skill root."""
    return get_skills_dir() / "procedural-memory"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _slugify(text: str, *, fallback: str = "untitled") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or fallback


def _tokenize(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{1,}", text.lower())}


def _safe_relative_markdown_path(name: str) -> Path:
    """Return a one-segment markdown filename derived from user/model text."""
    slug = _slugify(name)
    return Path(f"{slug}.md")


def _frontmatter(data: dict[str, object]) -> str:
    lines = ["---"]
    for key, value in data.items():
        if isinstance(value, list):
            rendered = "[" + ", ".join(str(v) for v in value) + "]"
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines)


def route_item(title: str, content: str = "", *, status_hint: str = "") -> ParaBucket:
    """Deterministically route a candidate workspace item into PARA.

    A future memory agent can replace this with an LLM classifier that uses the
    same output contract. The heuristic keeps the framework testable and useful
    offline.
    """
    haystack = f"{title}\n{content}\n{status_hint}".lower()
    if re.search(r"\b(done|completed|finished|inactive|archive|archived|retired)\b", haystack):
        return "Archives"
    if re.search(r"\b(deadline|due|milestone|ship|launch|deliver|project|sprint|todo|next step)\b", haystack):
        return "Projects"
    if re.search(r"\b(ongoing|maintain|responsibility|area|habit|routine|standard|policy)\b", haystack):
        return "Areas"
    return "Resources"


@dataclass(frozen=True)
class WorkspaceRecord:
    """Manifest-level metadata for one personal workspace file."""

    bucket: ParaBucket
    path: Path
    title: str
    summary: str = ""
    tags: tuple[str, ...] = ()
    updated_at: str = ""


@dataclass(frozen=True)
class RetrievalResult:
    """One top-k workspace retrieval result."""

    record: WorkspaceRecord
    score: int
    content: str


@dataclass
class WorkspaceItem:
    """A write candidate produced by a memory agent or CLI call."""

    title: str
    content: str
    bucket: ParaBucket | None = None
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    backlinks: list[str] = field(default_factory=list)
    status_hint: str = ""


class PersonalWorkspace:
    """PARA markdown workspace with manifest-based hierarchical retrieval."""

    def __init__(self, root: Path | None = None):
        self.root = root or default_workspace_root()

    def initialize(self) -> None:
        """Create the PARA directory skeleton and manifests."""
        self.root.mkdir(parents=True, exist_ok=True)
        for bucket in PARA_BUCKETS:
            bucket_dir = self.root / bucket
            bucket_dir.mkdir(parents=True, exist_ok=True)
            manifest = bucket_dir / MANIFEST_NAME
            if not manifest.exists():
                manifest.write_text(self._empty_manifest(bucket), encoding="utf-8")

    def write_item(self, item: WorkspaceItem, *, mode: Literal["new", "append", "update"] = "new") -> Path:
        """Write a workspace item and refresh the bucket manifest.

        ``new`` creates a unique file when a slug already exists. ``append``
        appends a dated section to an existing slug. ``update`` replaces the
        body of the existing slug while preserving the same filename.
        """
        if not item.title.strip():
            raise ValueError("Workspace item title cannot be empty")
        if not item.content.strip():
            raise ValueError("Workspace item content cannot be empty")

        self.initialize()
        bucket = item.bucket or route_item(item.title, item.content, status_hint=item.status_hint)
        bucket_dir = self.root / bucket
        rel = _safe_relative_markdown_path(item.title)
        path = bucket_dir / rel
        if mode == "new":
            path = self._unique_path(path)

        now = _utc_now()
        header = _frontmatter(
            {
                "title": item.title.strip(),
                "bucket": bucket,
                "summary": item.summary.strip() or self._summarize(item.content),
                "tags": [t.strip() for t in item.tags if t.strip()],
                "backlinks": [b.strip() for b in item.backlinks if b.strip()],
                "updated_at": now,
                "created_by": "hermes-dual-memory",
            }
        )
        body = item.content.strip() + "\n"

        if mode == "append" and path.exists():
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"\n\n## Update {now}\n\n{body}")
        else:
            path.write_text(f"{header}\n\n# {item.title.strip()}\n\n{body}", encoding="utf-8")

        self.rebuild_manifest(bucket)
        return path

    def read_manifests(self) -> dict[ParaBucket, str]:
        """Read the four PARA manifests without scanning file bodies."""
        self.initialize()
        return {
            bucket: (self.root / bucket / MANIFEST_NAME).read_text(encoding="utf-8")
            for bucket in PARA_BUCKETS
        }

    def retrieve(self, query: str, *, top_k: int = 3, candidate_limit: int = 8) -> list[RetrievalResult]:
        """Two-stage retrieval using manifests first, then top candidate files."""
        if top_k <= 0:
            return []
        manifests = self.read_manifests()
        query_terms = _tokenize(query)
        bucket_scores = [
            (self._score_text(query_terms, f"{bucket}\n{manifest}"), bucket)
            for bucket, manifest in manifests.items()
        ]
        relevant_buckets = [bucket for score, bucket in sorted(bucket_scores, reverse=True) if score > 0]
        if not relevant_buckets:
            relevant_buckets = list(PARA_BUCKETS)

        records: list[WorkspaceRecord] = []
        for bucket in relevant_buckets:
            records.extend(self.parse_manifest(bucket, manifests[bucket]))

        ranked_records = sorted(
            ((self._score_record(query_terms, record), record) for record in records),
            key=lambda pair: pair[0],
            reverse=True,
        )
        results: list[RetrievalResult] = []
        for score, record in ranked_records[:candidate_limit]:
            path = self.root / record.bucket / record.path
            if not path.exists() or path.name == MANIFEST_NAME:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except OSError:
                continue
            full_score = score + self._score_text(query_terms, content)
            if full_score > 0 or not query_terms:
                results.append(RetrievalResult(record=record, score=full_score, content=content))

        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    def rebuild_manifest(self, bucket: ParaBucket) -> Path:
        """Rebuild a bucket manifest from markdown files in that bucket."""
        if bucket not in PARA_BUCKETS:
            raise ValueError(f"Unknown PARA bucket: {bucket}")
        bucket_dir = self.root / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for path in sorted(bucket_dir.glob("*.md")):
            if path.name == MANIFEST_NAME:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            meta = self._parse_frontmatter(text)
            title = str(meta.get("title") or path.stem.replace("-", " ").title())
            summary = str(meta.get("summary") or self._summarize(text))
            tags = tuple(str(t).strip() for t in meta.get("tags", []) if str(t).strip())
            updated = str(meta.get("updated_at") or "")
            records.append(
                WorkspaceRecord(
                    bucket=bucket,
                    path=Path(path.name),
                    title=title,
                    summary=summary,
                    tags=tags,
                    updated_at=updated,
                )
            )
        manifest = bucket_dir / MANIFEST_NAME
        manifest.write_text(self._render_manifest(bucket, records), encoding="utf-8")
        return manifest

    def parse_manifest(self, bucket: ParaBucket, text: str) -> list[WorkspaceRecord]:
        """Parse records from a generated manifest."""
        records: list[WorkspaceRecord] = []
        for line in text.splitlines():
            if not line.startswith("- ["):
                continue
            match = re.match(
                r"- \[(?P<title>.*?)\]\((?P<path>.*?)\) - (?P<summary>.*?)(?: \| tags: (?P<tags>.*?))?(?: \| updated: (?P<updated>.*))?$",
                line,
            )
            if not match:
                continue
            tags = tuple(
                t.strip()
                for t in (match.group("tags") or "").split(",")
                if t.strip()
            )
            records.append(
                WorkspaceRecord(
                    bucket=bucket,
                    path=Path(match.group("path")),
                    title=match.group("title"),
                    summary=match.group("summary").strip(),
                    tags=tags,
                    updated_at=(match.group("updated") or "").strip(),
                )
            )
        return records

    @staticmethod
    def _empty_manifest(bucket: ParaBucket) -> str:
        return (
            f"# {bucket} Manifest\n\n"
            "This manifest is the retrieval entry point for this PARA bucket.\n\n"
            "- Files: none yet\n"
        )

    @staticmethod
    def _render_manifest(bucket: ParaBucket, records: Sequence[WorkspaceRecord]) -> str:
        lines = [
            f"# {bucket} Manifest",
            "",
            "This manifest is the retrieval entry point for this PARA bucket.",
            "",
        ]
        if not records:
            lines.append("- Files: none yet")
            return "\n".join(lines) + "\n"
        for record in records:
            tags = f" | tags: {', '.join(record.tags)}" if record.tags else ""
            updated = f" | updated: {record.updated_at}" if record.updated_at else ""
            lines.append(
                f"- [{record.title}]({record.path.as_posix()}) - {record.summary}{tags}{updated}"
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _summarize(text: str, *, limit: int = 160) -> str:
        squashed = re.sub(r"\s+", " ", text).strip()
        return squashed[: limit - 1] + "..." if len(squashed) > limit else squashed

    @staticmethod
    def _parse_frontmatter(text: str) -> dict[str, object]:
        if not text.startswith("---\n"):
            return {}
        end = text.find("\n---", 4)
        if end == -1:
            return {}
        meta: dict[str, object] = {}
        for line in text[4:end].splitlines():
            if ":" not in line:
                continue
            key, raw = line.split(":", 1)
            value = raw.strip()
            if value.startswith("[") and value.endswith("]"):
                items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
                meta[key.strip()] = items
            else:
                meta[key.strip()] = value
        return meta

    @staticmethod
    def _score_text(query_terms: set[str], text: str) -> int:
        if not query_terms:
            return 0
        terms = _tokenize(text)
        return len(query_terms & terms)

    def _score_record(self, query_terms: set[str], record: WorkspaceRecord) -> int:
        weighted = (
            f"{record.title} {record.title} "
            f"{record.summary} "
            f"{' '.join(record.tags)} {' '.join(record.tags)}"
        )
        return self._score_text(query_terms, weighted)

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(2, 1000):
            candidate = path.with_name(f"{stem}-{idx}{suffix}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"Could not allocate unique path for {path}")


@dataclass
class SkillDraft:
    """A procedural memory candidate distilled from successful work."""

    name: str
    description: str
    triggers: list[str]
    steps: list[str]
    constraints: list[str] = field(default_factory=list)
    recovery: list[str] = field(default_factory=list)
    source: str = ""


class ProceduralMemory:
    """Write procedural memory as normal Hermes Skill Markdown files."""

    def __init__(self, root: Path | None = None):
        self.root = root or default_procedural_skills_root()

    def write_skill(self, draft: SkillDraft, *, overwrite: bool = False) -> Path:
        """Create or update a procedural skill draft."""
        if not draft.name.strip():
            raise ValueError("Skill name cannot be empty")
        if not draft.description.strip():
            raise ValueError("Skill description cannot be empty")
        if not draft.triggers:
            raise ValueError("Skill draft must include at least one trigger")
        if not draft.steps:
            raise ValueError("Skill draft must include at least one step")

        slug = _slugify(draft.name)
        skill_dir = self.root / slug
        skill_path = skill_dir / "SKILL.md"
        if skill_path.exists() and not overwrite:
            raise FileExistsError(f"Procedural skill already exists: {skill_path}")
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path.write_text(self.render_skill(draft), encoding="utf-8")
        return skill_path

    @staticmethod
    def render_skill(draft: SkillDraft) -> str:
        tags = ["procedural-memory", "agent-distilled"]
        front = _frontmatter(
            {
                "name": _slugify(draft.name),
                "description": draft.description.strip(),
                "version": "0.1.0",
                "author": "Hermes Memory Agent",
                "platforms": "[linux, macos, windows]",
                "metadata.hermes.tags": tags,
                "metadata.hermes.created_by": "agent",
            }
        )
        sections = [
            front,
            "",
            f"# {draft.name.strip()}",
            "",
            "## When To Use",
            "",
            *[f"- {item.strip()}" for item in draft.triggers if item.strip()],
            "",
            "## Procedure",
            "",
            *[f"{idx}. {step.strip()}" for idx, step in enumerate(draft.steps, start=1) if step.strip()],
        ]
        if draft.constraints:
            sections.extend(["", "## Constraints", "", *[f"- {c.strip()}" for c in draft.constraints if c.strip()]])
        if draft.recovery:
            sections.extend(["", "## Recovery", "", *[f"- {r.strip()}" for r in draft.recovery if r.strip()]])
        if draft.source.strip():
            sections.extend(["", "## Provenance", "", draft.source.strip()])
        return "\n".join(sections).rstrip() + "\n"


def filter_workspace_candidate(text: str) -> bool:
    """Return True when content has durable user-facing knowledge value."""
    stripped = text.strip()
    if len(stripped) < 40:
        return False
    low = stripped.lower()
    if re.search(r"\b(thanks|ok|sounds good|temporary|scratch|never mind)\b", low):
        return False
    return True


def format_retrieval_results(results: Iterable[RetrievalResult]) -> str:
    """Render retrieval results for CLI output or future context injection."""
    chunks = []
    for result in results:
        record = result.record
        chunks.append(
            f"## {record.title}\n"
            f"- bucket: {record.bucket}\n"
            f"- path: {record.path.as_posix()}\n"
            f"- score: {result.score}\n\n"
            f"{result.content.strip()}"
        )
    return "\n\n".join(chunks)
