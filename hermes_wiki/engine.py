from __future__ import annotations

import hashlib
import logging
import re
from datetime import date
from pathlib import Path

from hermes_wiki.config import WikiConfig
from hermes_wiki.frontmatter import (
    extract_wikilinks,
    read_page,
    slugify,
    write_page,
)
from hermes_wiki.indexer import WikiIndexer
from hermes_wiki.llm import WikiLLM
from hermes_wiki.models import (
    IngestResult,
    IssueSeverity,
    LintIssue,
    LintReport,
    QueryResult,
)
from hermes_wiki.search import WikiSearch

logger = logging.getLogger(__name__)


class WikiEngine:
    """Main orchestrator for the LLM Wiki memory system.

    Implements phased ingest pipeline:
      Phase 1: Save raw source, detect duplicates/drift
      Phase 2: LLM extracts candidate entities/concepts/facts
      Phase 3: Resolve candidates against existing wiki pages
      Phase 4: LLM generates/updates pages (constrained to known links)
      Phase 5: Post-write lint repair (auto-stub or downgrade dangling links)
      Phase 6: Index, update navigation, log
    """

    ALLOWED_SOURCE_TYPES = frozenset({"articles", "papers", "transcripts"})

    def __init__(self, config: WikiConfig | None = None, *, read_only: bool = False):
        self.config = config or WikiConfig()
        self.read_only = read_only
        if not read_only:
            self.config.ensure_dirs()
        self.search = WikiSearch(self.config, ensure_collection=not read_only, read_only=read_only)
        self.llm = WikiLLM(self.config)
        self.indexer = WikiIndexer(self.config)

    @staticmethod
    def _safe_slug(value: str | None) -> str | None:
        """Return a normalized single-path-component slug or None if unsafe."""
        if not value:
            return None
        raw = str(value).strip()
        if not raw or "/" in raw or "\\" in raw or raw in {".", ".."}:
            return None
        slug = slugify(raw)
        if not slug or slug in {".", ".."} or "/" in slug or "\\" in slug:
            return None
        return slug

    @staticmethod
    def _safe_child_path(directory: Path, slug: str | None) -> Path | None:
        safe_slug = WikiEngine._safe_slug(slug)
        if not safe_slug:
            return None
        root = directory.resolve()
        path = (root / f"{safe_slug}.md").resolve()
        if path.parent != root:
            return None
        return path

    def init_wiki(self, domain: str, tags: list[str] | None = None) -> str:
        """Initialize a new wiki with schema, index, and log."""
        self.config.ensure_dirs()

        tag_section = "\n".join(f"- {t}" for t in (tags or ["general"]))
        schema = f"""# Wiki Schema

## Domain
{domain}

## Conventions
- File names: lowercase, hyphens, no spaces (e.g., `transformer-architecture.md`)
- Every wiki page starts with YAML frontmatter
- Use `[[wikilinks]]` to link between pages — ONLY to pages that exist
- When updating a page, always bump the `updated` date
- Every new page must be added to `index.md`
- Every action must be appended to `log.md`
- Provenance markers: `^[raw/articles/source-file.md]` with exact source paths

## Frontmatter
```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | query | summary
tags: [from taxonomy below]
sources: [raw/articles/source-name.md]
confidence: high | medium | low
contested: true
contradictions: [other-page-slug]
---
```

## Tag Taxonomy
{tag_section}

## Page Thresholds
- Create a page when an entity/concept is central to a source (not passing mentions)
- Add to existing page when a source mentions something already covered
- Don't create for passing mentions of well-known concepts (e.g., "Markdown", "Python")
- Split pages exceeding 200 lines
"""
        self.config.schema_path.write_text(schema, encoding="utf-8")
        self.indexer.rebuild_index()

        log_content = (
            "# Wiki Log\n\n"
            "> Chronological record of all wiki actions. Append-only.\n"
            "> Format: `## [YYYY-MM-DD] action | subject`\n\n"
            f"## [{date.today().isoformat()}] create | Wiki initialized\n"
            f"- Domain: {domain}\n"
        )
        self.config.log_path.write_text(log_content, encoding="utf-8")
        self.search.reindex_all()

        return f"Wiki initialized at {self.config.wiki_path}\nDomain: {domain}\nReady for sources."

    def orient(self) -> str:
        """Session orientation: read schema, index, recent log. Call at start of every session."""
        parts = []

        if self.config.schema_path.exists():
            schema = self.config.schema_path.read_text(encoding="utf-8")
            parts.append(f"## Schema\n{schema[:2000]}")

        index_text = self.indexer.read_index()
        if index_text:
            parts.append(f"## Index\n{index_text[:3000]}")

        recent = self.indexer.recent_log_entries(20)
        parts.append(f"## Recent Activity\n{recent}")

        status = self.status()
        parts.append(
            f"## Stats\n"
            f"Pages: {status['total_pages']} | Sources: {status['total_sources']} | "
            f"Chunks: {status['indexed_chunks']}"
        )

        return "\n\n".join(parts)

    # ── Phased Ingest Pipeline ───────────────────────────────────────

    def ingest_text(self, text: str, name: str, source_url: str | None = None,
                    source_type: str = "articles", dry_run: bool = False) -> IngestResult:
        """Ingest a text source into the wiki using the phased pipeline."""
        source_type = str(source_type or "articles").strip().lower()
        if source_type not in self.ALLOWED_SOURCE_TYPES:
            allowed = ", ".join(sorted(self.ALLOWED_SOURCE_TYPES))
            raise ValueError(f"Unsupported source_type {source_type!r}; expected one of: {allowed}")

        slug = slugify(name)
        source_rel = f"raw/{source_type}/{slug}.md"
        result = IngestResult(source_path=source_rel)

        # ── Phase 1: Save raw source, detect duplicates/drift ────────
        source_path = self.config.raw_dir / source_type / f"{slug}.md"
        body_hash = hashlib.sha256(text.encode()).hexdigest()

        if source_path.exists():
            existing_fm, _ = read_page(source_path)
            if existing_fm.get("sha256") == body_hash:
                return IngestResult(
                    source_path=source_rel,
                    key_facts=[f"Source '{name}' already ingested with identical content — skipped"],
                )

        if not dry_run:
            write_page(source_path, {
                "source_url": source_url or "",
                "ingested": date.today().isoformat(),
                "sha256": body_hash,
            }, text)

        # ── Phase 2: LLM extracts candidates ────────────────────────
        schema_text = ""
        if self.config.schema_path.exists():
            schema_text = self.config.schema_path.read_text(encoding="utf-8")

        existing_entities = self.indexer.list_entity_slugs()
        existing_concepts = self.indexer.list_concept_slugs()

        analysis = self.llm.analyze_source(
            source_text=text,
            existing_entities=existing_entities,
            existing_concepts=existing_concepts,
            schema_text=schema_text,
        )

        # ── Phase 3: Resolve candidates against existing pages ───────
        all_known = set(self.indexer.list_all_slugs())
        pages_to_create: list[tuple[str, dict, str]] = []  # (slug, info, type)
        pages_to_update: list[tuple[str, dict, str]] = []

        for entity in analysis.get("entities", []):
            entity_slug = self._safe_slug(entity.get("slug") or entity.get("name", ""))
            if not entity_slug:
                logger.warning("Skipping entity with unsafe or empty slug: %r", entity.get("slug"))
                continue
            entity_path = self._safe_child_path(self.config.entities_dir, entity_slug)
            if entity_path is None:
                logger.warning("Skipping entity with unsafe slug: %r", entity_slug)
                continue
            if entity_path.exists():
                pages_to_update.append((entity_slug, entity, "entity"))
            else:
                pages_to_create.append((entity_slug, entity, "entity"))

        for concept in analysis.get("concepts", []):
            concept_slug = self._safe_slug(concept.get("slug") or concept.get("name", ""))
            if not concept_slug:
                logger.warning("Skipping concept with unsafe or empty slug: %r", concept.get("slug"))
                continue
            concept_path = self._safe_child_path(self.config.concepts_dir, concept_slug)
            if concept_path is None:
                logger.warning("Skipping concept with unsafe slug: %r", concept_slug)
                continue
            if concept_path.exists():
                pages_to_update.append((concept_slug, concept, "concept"))
            else:
                pages_to_create.append((concept_slug, concept, "concept"))

        planned_slugs = all_known | {s for s, _, _ in pages_to_create} | {s for s, _, _ in pages_to_update}

        if dry_run:
            result.pages_created = [s for s, _, _ in pages_to_create]
            result.pages_updated = [s for s, _, _ in pages_to_update]
            result.entities_found = [e.get("name", "") for e in analysis.get("entities", [])]
            result.concepts_found = [c.get("name", "") for c in analysis.get("concepts", [])]
            result.key_facts = analysis.get("key_facts", [])
            return result

        # ── Phase 4: Generate/update pages ───────────────────────────
        source_summary = analysis.get("summary", text[:2000])
        known_pages = sorted(planned_slugs)

        for page_slug, info, page_type in pages_to_create + pages_to_update:
            type_dir = self.config.entities_dir if page_type == "entity" else self.config.concepts_dir
            page_path = self._safe_child_path(type_dir, page_slug)
            if page_path is None:
                logger.warning("Skipping page with unsafe slug: %r", page_slug)
                continue

            existing_content = None
            existing_fm = {}
            if page_path.exists():
                existing_fm, existing_content = read_page(page_path)

            related = self.search.search(info.get("name", ""), limit=3, exclude_sources=True)
            related_context = "\n\n".join(
                f"[[{r.page_path.split('/')[-1].replace('.md', '')}]]: {r.text[:300]}"
                for r in related if page_slug not in r.page_path
            )

            if page_type == "entity":
                body = self.llm.generate_entity_page(
                    entity=info,
                    source_summary=source_summary,
                    source_path=source_rel,
                    existing_page_content=existing_content,
                    related_context=related_context,
                    known_pages=known_pages,
                )
            else:
                body = self.llm.generate_concept_page(
                    concept=info,
                    source_summary=source_summary,
                    source_path=source_rel,
                    existing_page_content=existing_content,
                    related_context=related_context,
                    known_pages=known_pages,
                )

            tags = analysis.get("tags", [])
            fm = existing_fm.copy()
            fm.update({
                "title": info.get("name", page_slug),
                "type": page_type,
                "tags": tags,
                "updated": date.today().isoformat(),
                "sources": list(set(fm.get("sources", []) + [source_rel])),
            })
            if "created" not in fm:
                fm["created"] = date.today().isoformat()

            write_page(page_path, fm, body)

            if existing_content:
                result.pages_updated.append(page_slug)
            else:
                result.pages_created.append(page_slug)

        result.entities_found = [e.get("name", "") for e in analysis.get("entities", [])]
        result.concepts_found = [c.get("name", "") for c in analysis.get("concepts", [])]
        result.key_facts = analysis.get("key_facts", [])

        # ── Phase 5: Post-write lint repair ──────────────────────────
        touched_pages = result.pages_created + result.pages_updated
        self._repair_dangling_links(touched_pages, planned_slugs)

        # ── Phase 6: Index, navigate, log ────────────────────────────
        for page_slug in touched_pages:
            for subdir in ["entities", "concepts"]:
                p = self.config.wiki_path / subdir / f"{page_slug}.md"
                if p.exists():
                    n = self.search.index_page(p)
                    result.chunks_indexed += n
                    break

        source_chunks = self.search.index_source(source_path)
        result.chunks_indexed += source_chunks

        for page_slug in result.pages_created:
            for subdir, ptype in [("entities", "entity"), ("concepts", "concept")]:
                p = self.config.wiki_path / subdir / f"{page_slug}.md"
                if p.exists():
                    fm, _ = read_page(p)
                    self.indexer.add_to_index(page_slug, fm.get("title", page_slug), ptype, fm.get("tags", []))
                    break

        log_details = []
        if result.pages_created:
            log_details.append(f"Created: {', '.join(result.pages_created)}")
        if result.pages_updated:
            log_details.append(f"Updated: {', '.join(result.pages_updated)}")
        log_details.append(f"Indexed {result.chunks_indexed} chunks")
        self.indexer.append_log("ingest", analysis.get("title", name), log_details)

        return result

    def _repair_dangling_links(self, page_slugs: list[str], known_slugs: set[str]) -> int:
        """Post-write repair: downgrade wikilinks to unknown pages to plain text.

        Returns count of links repaired.
        """
        repaired = 0
        for page_slug in page_slugs:
            for subdir in ["entities", "concepts", "comparisons", "queries"]:
                page_path = self.config.wiki_path / subdir / f"{page_slug}.md"
                if not page_path.exists():
                    continue

                text = page_path.read_text(encoding="utf-8")
                original = text

                def replace_unknown(match):
                    nonlocal repaired
                    raw = match.group(1)
                    target = raw.split("|")[0].strip()
                    display = raw.split("|")[1].strip() if "|" in raw else target
                    target_slug = slugify(target)
                    if target_slug in known_slugs:
                        return match.group(0)
                    repaired += 1
                    return display

                text = re.sub(r"\[\[([^\]]+)\]\]", replace_unknown, text)

                if text != original:
                    page_path.write_text(text, encoding="utf-8")
                break

        if repaired:
            logger.info("Repaired %d dangling wikilinks", repaired)
        return repaired

    def ingest_file(self, file_path: str | Path, dry_run: bool = False) -> IngestResult:
        """Ingest a file from the filesystem."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Source file not found: {path}")

        text = path.read_text(encoding="utf-8")
        source_type = "articles"
        if path.suffix.lower() == ".pdf":
            source_type = "papers"
        elif any(kw in path.stem.lower() for kw in ["transcript", "meeting", "interview"]):
            source_type = "transcripts"

        return self.ingest_text(text, path.stem, source_type=source_type, dry_run=dry_run)

    def query(self, question: str, file_result: bool = False,
              log_query: bool = True) -> QueryResult:
        """Answer a question using the wiki's compiled knowledge.

        Args:
            file_result: If True, file worthy answers as wiki pages.
                         Default False — caller must opt in.
            log_query: If False, skip appending to log.md (for read-only use).
        """
        search_results = self.search.search(question, limit=8, exclude_sources=False)

        if not search_results:
            return QueryResult(
                question=question,
                answer="No relevant content found in the wiki. Try ingesting some sources first.",
                confidence="low",
            )

        unique_pages = {}
        for r in search_results:
            page_file = r.page_path.split("/")[-1].replace(".md", "")
            if page_file not in unique_pages:
                unique_pages[page_file] = []
            unique_pages[page_file].append(r.text)

        wiki_context_parts = []
        for page_slug, chunks in list(unique_pages.items())[:5]:
            for subdir in ["entities", "concepts", "comparisons", "queries"]:
                page_path = self.config.wiki_path / subdir / f"{page_slug}.md"
                if page_path.exists():
                    fm, body = read_page(page_path)
                    wiki_context_parts.append(
                        f"## [[{page_slug}]] ({fm.get('type', 'unknown')})\n{body[:2000]}"
                    )
                    break
            else:
                combined = "\n\n".join(chunks[:2])
                wiki_context_parts.append(f"## {page_slug}\n{combined}")

        wiki_context = "\n\n---\n\n".join(wiki_context_parts)
        search_text = "\n\n".join(
            f"[{r.title}] (score: {r.score:.3f}): {r.text[:300]}"
            for r in search_results[:5]
        )

        known_pages = self.indexer.list_all_slugs()
        response = self.llm.answer_query(question, wiki_context, search_text, known_pages)

        result = QueryResult(
            question=question,
            answer=response.get("answer", response.get("raw_response", "No answer generated.")),
            sources_consulted=response.get("sources_consulted", []),
            confidence=response.get("confidence", "medium"),
        )

        if file_result and response.get("worth_filing") and response.get("file_slug"):
            file_slug = self._safe_slug(response.get("file_slug"))
            if not file_slug:
                logger.warning("Skipping filed query with unsafe slug: %r", response.get("file_slug"))
                return result
            file_title = response.get("file_title", question[:80])
            query_path = self._safe_child_path(self.config.queries_dir, file_slug)
            if query_path is None:
                logger.warning("Skipping filed query with unsafe slug: %r", file_slug)
                return result

            fm = {
                "title": file_title,
                "created": date.today().isoformat(),
                "updated": date.today().isoformat(),
                "type": "query",
                "tags": [],
                "sources": result.sources_consulted,
            }
            write_page(query_path, fm, result.answer)
            self.search.index_page(query_path)
            self.indexer.add_to_index(file_slug, file_title, "query")
            result.filed_as = file_slug

        if log_query:
            self.indexer.append_log(
                "query",
                question[:80],
                [f"Confidence: {result.confidence}"]
                + ([f"Filed as: {result.filed_as}"] if result.filed_as else []),
            )

        return result

    _GENERIC_PROVENANCE = re.compile(
        r"\^\[raw/(?:articles|papers|transcripts)/"
        r"(?:source(?:-(?:file|name))?|TODO|example)\.md\]"
    )

    def lint(self, write_log: bool = True) -> LintReport:
        """Health-check the wiki.

        Args:
            write_log: If False, skip appending to log.md (for tests/CI).
        """
        report = LintReport()
        all_pages = self.indexer.get_all_pages()
        all_slugs = set()
        slug_to_path: dict[str, Path] = {}
        inbound_links: dict[str, set[str]] = {}

        known_source_paths = set()
        for source in self.indexer.list_sources():
            known_source_paths.add(source["path"])
        report.total_sources = len(known_source_paths)

        for page_type, entries in all_pages.items():
            for entry in entries:
                slug = entry["slug"]
                all_slugs.add(slug)
                page_path = self.config.wiki_path / entry["path"]
                slug_to_path[slug] = page_path
                inbound_links.setdefault(slug, set())
                report.total_pages += 1

        # ── Wikilink analysis ────────────────────────────────────────
        for slug, page_path in slug_to_path.items():
            if not page_path.exists():
                continue
            _, body = read_page(page_path)
            links = extract_wikilinks(body)
            report.total_links += len(links)

            for link in links:
                link_slug = slugify(link)
                if link_slug in all_slugs:
                    inbound_links[link_slug].add(slug)
                else:
                    report.broken_links += 1
                    report.issues.append(LintIssue(
                        severity=IssueSeverity.ERROR,
                        category="broken_link",
                        message=f"Broken wikilink [[{link}]]",
                        file_path=str(page_path.relative_to(self.config.wiki_path)),
                        suggestion=f"Create page '{link}' or remove the link",
                    ))

        # ── Orphan detection ─────────────────────────────────────────
        for slug, sources in inbound_links.items():
            if not sources:
                report.orphan_pages += 1
                report.issues.append(LintIssue(
                    severity=IssueSeverity.WARNING,
                    category="orphan_page",
                    message=f"Page [[{slug}]] has no inbound links",
                    file_path=slug_to_path[slug].name if slug in slug_to_path else None,
                    suggestion="Add [[wikilinks]] from related pages",
                ))

        # ── Index completeness ───────────────────────────────────────
        index_content = self.indexer.read_index()
        for slug in all_slugs:
            if f"[[{slug}]]" not in index_content:
                report.issues.append(LintIssue(
                    severity=IssueSeverity.WARNING,
                    category="missing_from_index",
                    message=f"Page [[{slug}]] not listed in index.md",
                    suggestion="Run index rebuild",
                ))

        # ── Per-page checks ──────────────────────────────────────────
        for slug, page_path in slug_to_path.items():
            if not page_path.exists():
                continue
            fm, body = read_page(page_path)
            rel_path = str(page_path.relative_to(self.config.wiki_path))

            # Required frontmatter fields
            for field in ["title", "type", "created", "updated"]:
                if field not in fm:
                    report.issues.append(LintIssue(
                        severity=IssueSeverity.WARNING,
                        category="missing_frontmatter",
                        message=f"Missing '{field}' in frontmatter",
                        file_path=rel_path,
                    ))

            # Page size
            line_count = len(body.split("\n"))
            if line_count > self.config.page_split_threshold:
                report.issues.append(LintIssue(
                    severity=IssueSeverity.INFO,
                    category="large_page",
                    message=f"Page has {line_count} lines",
                    file_path=rel_path,
                    suggestion="Consider splitting into sub-pages",
                ))

            # Contested flag
            if fm.get("contested"):
                report.issues.append(LintIssue(
                    severity=IssueSeverity.WARNING,
                    category="contested",
                    message="Page marked as contested",
                    file_path=rel_path,
                ))

            # Generic/fake provenance markers
            fake_prov = self._GENERIC_PROVENANCE.findall(body)
            if fake_prov:
                report.issues.append(LintIssue(
                    severity=IssueSeverity.ERROR,
                    category="fake_provenance",
                    message=f"{len(fake_prov)} generic provenance markers",
                    file_path=rel_path,
                    suggestion="Replace with real source paths",
                ))

            # Provenance paths that don't exist on disk (skip already-flagged generics)
            all_prov = re.findall(r"\^\[(raw/[^\]]+\.md)\]", body)
            for prov_path in all_prov:
                if self._GENERIC_PROVENANCE.search(f"^[{prov_path}]"):
                    continue
                full_path = self.config.wiki_path / prov_path
                if not full_path.exists() and prov_path not in known_source_paths:
                    report.issues.append(LintIssue(
                        severity=IssueSeverity.ERROR,
                        category="missing_source",
                        message=f"Provenance marker references non-existent source: {prov_path}",
                        file_path=rel_path,
                    ))

            # Frontmatter sources that don't exist
            fm_sources = fm.get("sources", [])
            if isinstance(fm_sources, list):
                for src in fm_sources:
                    src_path = self.config.wiki_path / src
                    if not src_path.exists():
                        report.issues.append(LintIssue(
                            severity=IssueSeverity.WARNING,
                            category="missing_fm_source",
                            message=f"Frontmatter source not found: {src}",
                            file_path=rel_path,
                        ))

            # Body provenance not in frontmatter sources
            if fm_sources and all_prov:
                fm_source_set = set(fm_sources) if isinstance(fm_sources, list) else set()
                for prov_path in set(all_prov):
                    if prov_path not in fm_source_set and not self._GENERIC_PROVENANCE.match(f"^[{prov_path}]"):
                        report.issues.append(LintIssue(
                            severity=IssueSeverity.INFO,
                            category="provenance_mismatch",
                            message=f"Body cites {prov_path} but it's not in frontmatter sources",
                            file_path=rel_path,
                        ))

        # ── Source integrity ─────────────────────────────────────────
        for source_path_str in known_source_paths:
            source_path = self.config.wiki_path / source_path_str
            if source_path.exists():
                fm, body = read_page(source_path)
                stored_hash = fm.get("sha256")
                if stored_hash:
                    actual_hash = hashlib.sha256(body.encode()).hexdigest()
                    if actual_hash != stored_hash:
                        report.issues.append(LintIssue(
                            severity=IssueSeverity.WARNING,
                            category="source_drift",
                            message="Source content hash mismatch",
                            file_path=source_path_str,
                            suggestion="Raw sources should be immutable",
                        ))

        # ── Vector collection health ─────────────────────────────────
        vector_stats = self.search.collection_stats()
        if report.total_pages > 0 and vector_stats.get("points", 0) == 0:
            report.issues.append(LintIssue(
                severity=IssueSeverity.ERROR,
                category="empty_vector_index",
                message=f"Wiki has {report.total_pages} pages but vector index has 0 chunks",
                suggestion="Run reindex to rebuild the vector collection",
            ))

        if write_log:
            self.indexer.append_log(
                "lint",
                f"{len(report.issues)} issues found",
                [f"Errors: {report.error_count}, Warnings: {report.warning_count}"],
            )

        return report

    def status(self) -> dict:
        pages = self.indexer.get_all_pages()
        sources = self.indexer.list_sources()
        vector_stats = self.search.collection_stats()
        recent_log = self.indexer.recent_log_entries(10)

        return {
            "wiki_path": str(self.config.wiki_path),
            "total_pages": sum(len(v) for v in pages.values()),
            "pages_by_type": {k: len(v) for k, v in pages.items()},
            "total_sources": len(sources),
            "vector_collection": vector_stats.get("collection", ""),
            "indexed_chunks": vector_stats.get("points", 0),
            "recent_activity": recent_log,
            "initialized": self.config.schema_path.exists(),
        }

    def browse(self, page_type: str | None = None, tag: str | None = None,
               limit: int = 20) -> list[dict]:
        pages = self.indexer.get_all_pages()
        results = []
        for ptype, entries in pages.items():
            if page_type and ptype != page_type + "s" and ptype != page_type:
                continue
            for entry in entries:
                if tag and tag not in entry.get("tags", []):
                    continue
                entry["page_type"] = ptype
                results.append(entry)

        results.sort(key=lambda x: x.get("updated", ""), reverse=True)
        return results[:limit]

    def reindex(self) -> dict:
        counts = self.search.reindex_all()
        self.indexer.rebuild_index()
        self.indexer.append_log(
            "reindex", "Full reindex",
            [f"Pages: {counts['pages']}, Sources: {counts['sources']}, Chunks: {counts['chunks']}"],
        )
        return counts

    def close(self) -> None:
        self.search.close()
