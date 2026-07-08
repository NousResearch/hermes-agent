"""Compiler orchestration service."""

from __future__ import annotations

from pathlib import Path

from llmwiki_hermes.compiler.classify import classify_text
from llmwiki_hermes.compiler.episodic import create_or_append_episodic_note
from llmwiki_hermes.compiler.semantic import upsert_semantic_note
from llmwiki_hermes.compiler.source import derive_title, render_source_note
from llmwiki_hermes.errors import IngestInputError
from llmwiki_hermes.schemas.notes import SourceNoteFrontmatter
from llmwiki_hermes.storage.frontmatter import load_note, write_note
from llmwiki_hermes.storage.sqlite_index import IndexService
from llmwiki_hermes.storage.vault import VaultService


def normalize_text(raw: str) -> str:
    """Normalize ingest input into plain text."""

    normalized = raw.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = "".join(
        char for char in normalized if char == "\n" or char == "\t" or ord(char) >= 32
    )
    return cleaned.strip()


class CompilerService:
    """Compile normalized input into persisted notes and index updates."""

    def __init__(self, vault_service: VaultService, index_service: IndexService) -> None:
        self.vault_service = vault_service
        self.index_service = index_service

    def ingest_input(
        self,
        *,
        input_path: Path | None,
        raw_content: str,
        source_type: str,
        tags: list[str],
        dry_run: bool,
    ) -> list[str]:
        """Compile a single input payload into source, semantic, and episodic notes."""

        normalized = normalize_text(raw_content)
        if not normalized:
            raise IngestInputError("Ingest input is empty after normalization.")

        source_frontmatter, source_body, filename = render_source_note(
            content=normalized,
            path=input_path,
            source_type=source_type,
            tags=tags,
        )
        existing_source = self.find_source_by_hash(source_frontmatter.content_hash)
        source_path = existing_source or self.vault_service.sources_dir / filename

        created_or_updated: list[str] = []
        source_note = self._persist_source_note(
            source_path=source_path,
            existing_source=existing_source,
            source_frontmatter=source_frontmatter,
            source_body=source_body,
            dry_run=dry_run,
        )
        if existing_source is None:
            created_or_updated.append(str(source_path))

        title = derive_title(input_path, normalized)
        classes = classify_text(normalized, source_type=source_type)
        if classes["semantic"]:
            semantic_path = upsert_semantic_note(
                vault_service=self.vault_service,
                title=title,
                content=normalized,
                source_note=source_note,
                dry_run=dry_run,
            )
            if semantic_path is not None:
                created_or_updated.append(semantic_path)
        if classes["episodic"]:
            episodic_path = create_or_append_episodic_note(
                vault_service=self.vault_service,
                title=title,
                content=normalized,
                source_note=source_note,
                dry_run=dry_run,
            )
            if episodic_path is not None:
                created_or_updated.append(episodic_path)
        return created_or_updated

    def reindex(self) -> None:
        """Rebuild the SQLite index after successful compilation."""

        self.index_service.reindex()

    def find_source_by_hash(self, content_hash: str) -> Path | None:
        """Return an existing source note path for a known content hash."""

        for path in self.vault_service.sources_dir.glob("*.md"):
            document = load_note(path)
            if document.frontmatter.get("content_hash") == content_hash:
                return path
        return None

    def _persist_source_note(
        self,
        *,
        source_path: Path,
        existing_source: Path | None,
        source_frontmatter: SourceNoteFrontmatter,
        source_body: str,
        dry_run: bool,
    ) -> SourceNoteFrontmatter:
        if not dry_run and existing_source is None:
            write_note(source_path, source_frontmatter.model_dump(mode="json"), source_body)
            return source_frontmatter
        if existing_source is not None:
            return SourceNoteFrontmatter.model_validate(load_note(existing_source).frontmatter)
        return source_frontmatter
