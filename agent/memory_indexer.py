from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from agent.memory_documents import MemoryDocument, MemorySourceRef
from agent.memory_embeddings import MemoryEmbedder
from agent.pinecone_memory import PineconeMemoryClient
from hermes_cli.config import load_config
from hermes_state import SessionDB

logger = logging.getLogger(__name__)

_DEFAULT_TOPIC_FILES = (
    "platform-state.md",
    "linear-labels.md",
    "projects.md",
    "decisions.md",
)

_LINK_RE = re.compile(r"https?://\S+|\b[A-Z]{2,10}-\d+\b")
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_./:-]{2,}")


@dataclass(frozen=True)
class IndexResult:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    upserted: int = 0


class MemoryIndexer:
    """Indexes curated topic files and distilled session summaries into Pinecone."""

    def __init__(
        self,
        *,
        pinecone: PineconeMemoryClient,
        embedder: MemoryEmbedder,
        session_db: SessionDB,
        config: dict[str, Any] | None = None,
        topic_files: Sequence[str | Path] | None = None,
    ) -> None:
        self.pinecone = pinecone
        self.embedder = embedder
        self.session_db = session_db
        self.config = config if config is not None else load_config()
        self.memory_config = self.config.get("memory", {}) if isinstance(self.config, dict) else {}
        self.topic_files = tuple(Path(p) for p in topic_files) if topic_files is not None else self._discover_topic_files()

    def index_topic_files(self, paths: Sequence[str | Path] | None = None) -> IndexResult:
        if not self.memory_config.get("pinecone_ingest_topic_files", True):
            return IndexResult()
        selected = [Path(p) for p in (paths if paths is not None else self.topic_files)]
        result = IndexResult()
        for path in selected:
            result = self._accumulate(result, self._index_topic_file(path))
        return result

    def index_session_summaries(self, *, limit: int = 100, session_ids: Sequence[str] | None = None) -> IndexResult:
        if not self.memory_config.get("pinecone_ingest_session_summaries", True):
            return IndexResult()
        summaries = self.session_db.export_session_summaries(limit=limit)
        if session_ids is not None:
            allowed = set(session_ids)
            summaries = [item for item in summaries if item.get("session_id") in allowed]
        result = IndexResult(scanned=len(summaries))
        for summary in summaries:
            if not summary.get("text"):
                result = self._accumulate(result, IndexResult(scanned=0, skipped=1))
                continue
            source_id = str(summary["session_id"])
            source_key = self._source_key("session_summary", source_id)
            content_hash = self._content_hash(str(summary["text"]))
            if self._get_stored_hash(source_key) == content_hash:
                result = self._accumulate(result, IndexResult(scanned=0, skipped=1))
                continue
            document = MemoryDocument(
                memory_type="session_summary",
                scope=str(summary.get("scope") or "global"),
                source=MemorySourceRef(
                    source_kind="session_summary",
                    source_id=source_id,
                    source_path=str(summary.get("source_path") or source_id),
                ),
                title=str(summary.get("title") or source_id),
                text=str(summary["text"]),
                created_at=summary.get("created_at"),
                updated_at=summary.get("updated_at"),
                freshness_hint="weekly",
                confidence=0.62,
                canonical=False,
                tags=tuple(summary.get("tags") or ("session_search",)),
            )
            upserted = self._upsert_document(document)
            self._set_stored_hash(source_key, content_hash)
            result = self._accumulate(result, IndexResult(indexed=1, upserted=upserted))
        return result

    def reindex_all(self, *, limit: int = 100) -> dict[str, IndexResult]:
        return {
            "topic_files": self.index_topic_files(),
            "session_summaries": self.index_session_summaries(limit=limit),
        }

    def _index_topic_file(self, path: Path) -> IndexResult:
        if not path.exists() or not path.is_file():
            return IndexResult(scanned=1, skipped=1)
        text = path.read_text(encoding="utf-8")
        source_key = self._source_key("file", str(path.resolve()))
        content_hash = self._content_hash(text)
        if self._get_stored_hash(source_key) == content_hash:
            return IndexResult(scanned=1, skipped=1)
        stat = path.stat()
        document = MemoryDocument(
            memory_type="project_context",
            scope="global",
            source=MemorySourceRef(
                source_kind="file",
                source_id=str(path.resolve()),
                source_path=str(path.resolve()),
            ),
            title=path.name,
            text=text,
            created_at=self._iso_from_timestamp(stat.st_ctime),
            updated_at=self._iso_from_timestamp(stat.st_mtime),
            freshness_hint="monthly",
            confidence=0.96,
            canonical=True,
            tags=self._file_tags(path),
        )
        upserted = self._upsert_document(document)
        self._set_stored_hash(source_key, content_hash)
        return IndexResult(scanned=1, indexed=1, upserted=upserted)

    def _upsert_document(self, document: MemoryDocument) -> int:
        chunks = document.chunk()
        if not chunks:
            return 0
        vectors = self.embedder.embed_texts([chunk.text for chunk in chunks])
        records = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            metadata = dict(chunk.metadata)
            metadata["text"] = chunk.text
            metadata["title"] = document.title
            metadata["content_hash"] = self._content_hash(chunk.text)
            records.append({"id": chunk.id, "values": vector, "metadata": metadata})
        return self.pinecone.upsert_records(records)

    def _discover_topic_files(self) -> tuple[Path, ...]:
        discovered: list[Path] = []
        roots = [Path.cwd(), Path.home() / ".hermes"]
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for name in _DEFAULT_TOPIC_FILES:
                for candidate in root.rglob(name):
                    if ".plans" in candidate.parts:
                        continue
                    resolved = candidate.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        discovered.append(resolved)
        return tuple(discovered)

    def _source_key(self, source_kind: str, source_id: str) -> str:
        return f"pinecone_index_hash:{source_kind}:{source_id}"

    def _get_stored_hash(self, source_key: str) -> str | None:
        return self.session_db.get_meta(source_key)

    def _set_stored_hash(self, source_key: str, value: str) -> None:
        self.session_db.set_meta(source_key, value)

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _iso_from_timestamp(value: float) -> str:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    @staticmethod
    def _file_tags(path: Path) -> tuple[str, ...]:
        tags = {"topic-file", path.stem.lower()}
        if any(part.startswith(".") for part in path.parts):
            tags.add("hidden")
        return tuple(sorted(tags))

    @staticmethod
    def _accumulate(left: IndexResult, right: IndexResult) -> IndexResult:
        return IndexResult(
            scanned=left.scanned + right.scanned,
            indexed=left.indexed + right.indexed,
            skipped=left.skipped + right.skipped,
            upserted=left.upserted + right.upserted,
        )


def build_session_summary_text(*, title: str, tools: Iterable[str], links_and_ids: Iterable[str], end_reason: str | None) -> str:
    tool_list = sorted({tool.strip() for tool in tools if tool and tool.strip()})
    linked = sorted({item.strip() for item in links_and_ids if item and item.strip()})
    topics = _extract_topics_from_title(title)
    lines = [f"Session title: {title.strip() or 'Untitled session'}"]
    if topics:
        lines.append(f"Key topics: {', '.join(topics)}")
    if linked:
        lines.append(f"Linked IDs/URLs: {', '.join(linked[:8])}")
    if tool_list:
        lines.append(f"Tools used: {', '.join(tool_list[:8])}")
    if end_reason:
        lines.append(f"Final resolution: session ended with reason `{end_reason}`")
    lines.append("Transcript body intentionally omitted; this summary only stores distilled metadata.")
    return "\n".join(lines)


def extract_links_and_ids(text: str) -> list[str]:
    return sorted({match.group(0).rstrip(').,') for match in _LINK_RE.finditer(text or "")})


def _extract_topics_from_title(title: str) -> list[str]:
    raw_tokens = [token.lower() for token in _TOKEN_RE.findall(title or "")]
    stopwords = {"the", "and", "for", "with", "from", "into", "this", "that", "ticket", "issue", "auto"}
    topics: list[str] = []
    seen: set[str] = set()
    for token in raw_tokens:
        if token in stopwords or token.isdigit() or len(token) < 3:
            continue
        if token not in seen:
            seen.add(token)
            topics.append(token)
        if len(topics) >= 8:
            break
    return topics


__all__ = [
    "IndexResult",
    "MemoryIndexer",
    "build_session_summary_text",
    "extract_links_and_ids",
]
