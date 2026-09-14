"""Walking a declared source, extracting it, chunking it and writing the index.

Ingestion is the moment a customer's documents become something a model can quote, so it is
governed like one. Every run is bracketed by a write-ahead audit record under the
``knowledge.indexed`` kind: intent before anything is written, commit after, failure if it
raises. That is the same invariant that guards materialising an agent, applied for the same
reason — the model-visible surface must never move without a record that it did.

Incremental by content hash. A document whose text is byte-identical to the indexed version
is skipped, so re-running an ingest over a 2,000-file corpus after editing one policy
re-chunks one file. A document that has disappeared from disk is dropped from the index in
the same pass, because a corpus that keeps serving deleted documents is a disclosure that
nobody authorised.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol, Sequence

from nova.audit.log import AuditLog, new_correlation_id
from nova.knowledge.chunk import (
    DEFAULT_MAX_CHARS,
    DEFAULT_OVERLAP_CHARS,
    chunk_document,
    document_digest,
)
from nova.knowledge.index import DocumentRecord, KnowledgeIndex
from nova.knowledge.sources import KnowledgeCatalog, KnowledgeSource, iter_documents


class TextExtractor(Protocol):
    """Anything that can turn a file into text.

    :class:`nova.runtime.base.AgentRuntime` satisfies this, which is the point: extraction
    is a runtime capability, and ingestion consumes it through a one-method protocol rather
    than depending on a runtime package.
    """

    def extract_text(self, path: Path) -> Any: ...


class PlainTextExtractor:
    """The fallback when no runtime is supplied: UTF-8 text, and an honest refusal otherwise."""

    def extract_text(self, path: Path) -> Any:
        from nova.runtime.base import ExtractedDocument

        try:
            return ExtractedDocument(
                path=str(path), text=path.read_text(encoding="utf-8"),
                extracted=True, method="native",
            )
        except (OSError, UnicodeDecodeError) as exc:
            return ExtractedDocument(
                path=str(path), text="", extracted=False, method="native",
                detail=f"not readable as UTF-8 text and no runtime extractor was supplied: {exc}",
            )


@dataclass
class SourceReport:
    """What one ingest run did to one corpus."""

    source_id: str
    indexed: int = 0
    unchanged: int = 0
    chunks: int = 0
    removed: list[str] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)

    @property
    def considered(self) -> int:
        return self.indexed + self.unchanged

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "indexed": self.indexed,
            "unchanged": self.unchanged,
            "chunks": self.chunks,
            "removed": list(self.removed),
            "skipped": [{"path": path, "reason": reason} for path, reason in self.skipped],
        }


@dataclass
class IngestReport:
    """What one ingest run did, across every corpus it touched."""

    index_path: Path
    sources: list[SourceReport] = field(default_factory=list)
    dry_run: bool = False

    @property
    def indexed(self) -> int:
        return sum(report.indexed for report in self.sources)

    @property
    def chunks(self) -> int:
        return sum(report.chunks for report in self.sources)

    @property
    def skipped(self) -> int:
        return sum(len(report.skipped) for report in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_path": str(self.index_path),
            "dry_run": self.dry_run,
            "indexed": self.indexed,
            "chunks": self.chunks,
            "skipped": self.skipped,
            "sources": [report.to_dict() for report in self.sources],
        }

    def summary(self) -> str:
        if not self.sources:
            return "no knowledge sources declared"
        verb = "would index" if self.dry_run else "indexed"
        parts = [
            f"{report.source_id}: {verb} {report.indexed}, unchanged {report.unchanged}, "
            f"{report.chunks} chunks"
            + (f", removed {len(report.removed)}" if report.removed else "")
            + (f", skipped {len(report.skipped)}" if report.skipped else "")
            for report in self.sources
        ]
        return "\n".join(parts)


def ingest(
    catalog: KnowledgeCatalog,
    index_path: Path | str,
    *,
    extractor: Optional[TextExtractor] = None,
    source_ids: Optional[Sequence[str]] = None,
    audit: Optional[AuditLog] = None,
    correlation_id: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_chars: int = DEFAULT_OVERLAP_CHARS,
) -> IngestReport:
    """Index every declared source, or only those named in *source_ids*.

    A dry run walks and chunks exactly as a real one does and writes nothing, so the report
    it returns is the report a real run would produce rather than an estimate of it.
    """
    index_path = Path(index_path).expanduser()
    extractor = extractor or PlainTextExtractor()
    correlation_id = correlation_id or new_correlation_id()

    selected = (
        catalog.subset(source_ids) if source_ids is not None else catalog.sources
    )
    if source_ids is not None:
        unknown = sorted(set(source_ids) - catalog.ids)
        if unknown:
            catalog.get(unknown[0])  # raises SpecError naming the declared sources

    report = IngestReport(index_path=index_path, dry_run=dry_run)
    if not selected:
        return report

    if dry_run:
        for spec in selected:
            report.sources.append(
                _ingest_source(
                    spec, None, extractor,
                    force=force, max_chars=max_chars, overlap_chars=overlap_chars,
                )
            )
        return report

    with KnowledgeIndex.open(index_path) as index:
        for spec in selected:
            detail = {
                "source_id": spec.id,
                "root": str(spec.root),
                "classification": spec.classification,
                "index": str(index_path),
            }
            if audit is None:
                report.sources.append(
                    _ingest_source(
                        spec, index, extractor,
                        force=force, max_chars=max_chars, overlap_chars=overlap_chars,
                    )
                )
                continue
            with audit.model_visible_change(
                "knowledge.indexed",
                correlation_id=correlation_id,
                subject=spec.id,
                detail=detail,
            ) as committed:
                source_report = _ingest_source(
                    spec, index, extractor,
                    force=force, max_chars=max_chars, overlap_chars=overlap_chars,
                )
                report.sources.append(source_report)
                committed.update(
                    {
                        "indexed": source_report.indexed,
                        "unchanged": source_report.unchanged,
                        "chunks": source_report.chunks,
                        "removed": source_report.removed,
                        "skipped": len(source_report.skipped),
                    }
                )
        index.optimize()

    return report


def _ingest_source(
    spec: KnowledgeSource,
    index: Optional[KnowledgeIndex],
    extractor: TextExtractor,
    *,
    force: bool,
    max_chars: int,
    overlap_chars: int,
) -> SourceReport:
    """One corpus. *index* is None on a dry run, which otherwise does identical work."""
    report = SourceReport(source_id=spec.id)
    walk = iter_documents(spec)
    report.skipped.extend(
        (_relative(path, spec.root), reason) for path, reason in
        ((entry.path, entry.reason) for entry in walk.skipped)
    )

    seen: list[str] = []
    for path in walk.files:
        doc_path = _relative(path, spec.root)
        document = extractor.extract_text(path)
        if not getattr(document, "extracted", False) or not (document.text or "").strip():
            report.skipped.append(
                (doc_path, getattr(document, "detail", "") or "no text could be extracted")
            )
            continue

        seen.append(doc_path)
        digest = document_digest(document.text)
        if index is not None and not force and index.known_digest(spec.id, doc_path) == digest:
            report.unchanged += 1
            index.touch_document(
                DocumentRecord(source_id=spec.id, doc_path=doc_path, digest=digest)
            )
            continue

        chunks = chunk_document(
            document.text,
            source_id=spec.id,
            doc_path=doc_path,
            doc_title=_title(document.text, doc_path),
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        )
        if not chunks:
            report.skipped.append((doc_path, "produced no chunks"))
            seen.pop()
            continue

        record = DocumentRecord(
            source_id=spec.id,
            doc_path=doc_path,
            digest=digest,
            doc_title=_title(document.text, doc_path),
            byte_size=len(document.text.encode("utf-8")),
            method=getattr(document, "method", "") or "",
        )
        if index is not None:
            index.replace_document(record, chunks)
        report.indexed += 1
        report.chunks += len(chunks)

    if index is not None:
        report.removed = index.prune(spec.id, seen)
    return report


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _title(text: str, doc_path: str) -> str:
    """The document's own first heading, falling back to its filename.

    Titles matter more than they look: they are what a citation shows a human reading the
    dashboard, and "Refund Policy (2024)" is a very different answer from "doc_final_v3.md".
    """
    for line in text.split("\n", 40)[:40]:
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            if title:
                return title[:200]
        elif stripped:
            break
    return Path(doc_path).stem.replace("_", " ").replace("-", " ").strip()[:200]
