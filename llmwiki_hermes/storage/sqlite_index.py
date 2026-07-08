"""SQLite FTS5 index management."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from llmwiki_hermes.errors import IndexCorruptionError
from llmwiki_hermes.schemas.cli import CommandOutput
from llmwiki_hermes.schemas.diagnostics import DiagnosticIssue, DiagnosticSeverity
from llmwiki_hermes.schemas.notes import validate_note_frontmatter
from llmwiki_hermes.storage.frontmatter import load_note
from llmwiki_hermes.storage.links import extract_links
from llmwiki_hermes.storage.vault import VaultService

logger = logging.getLogger(__name__)


def chunk_markdown(body: str) -> list[dict[str, str]]:
    """Split a Markdown body by headings into simple searchable chunks."""

    chunks: list[dict[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    for raw_line in body.splitlines():
        if raw_line.startswith("#"):
            if current_lines:
                chunks.append(
                    {"heading": current_heading, "text": "\n".join(current_lines).strip()}
                )
                current_lines = []
            current_heading = raw_line.lstrip("#").strip()
            continue
        current_lines.append(raw_line)
    if current_lines:
        chunks.append({"heading": current_heading, "text": "\n".join(current_lines).strip()})
    return [chunk for chunk in chunks if chunk["text"]]


def build_fts_match_query(query: str) -> str:
    """Quote non-word tokens so SQLite FTS5 does not misparse them."""

    terms: list[str] = []
    for token in query.split():
        escaped = token.replace('"', '""')
        if all(character == "_" or character.isalnum() for character in token):
            terms.append(token)
        else:
            terms.append(f'"{escaped}"')
    return " ".join(terms)


class IndexService:
    """Manage the sidecar SQLite database."""

    def __init__(self, vault_service: VaultService) -> None:
        self.vault_service = vault_service

    def connect(self) -> sqlite3.Connection:
        """Open the SQLite database."""

        self.vault_service.hidden_dir.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.vault_service.index_db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def create_schema(self) -> None:
        """Create the required SQLite tables."""

        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS notes (
                  id TEXT PRIMARY KEY,
                  schema_version INTEGER,
                  kind TEXT NOT NULL,
                  title TEXT NOT NULL,
                  path TEXT NOT NULL UNIQUE,
                  updated_at TEXT,
                  aliases_json TEXT,
                  entity_refs_json TEXT,
                  confidence TEXT,
                  date TEXT,
                  project TEXT,
                  tags_json TEXT,
                  source_refs_json TEXT
                );

                CREATE TABLE IF NOT EXISTS chunks (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  note_id TEXT NOT NULL,
                  ord INTEGER NOT NULL,
                  heading TEXT,
                  text TEXT NOT NULL,
                  FOREIGN KEY(note_id) REFERENCES notes(id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                  text,
                  content='chunks',
                  content_rowid='id'
                );

                CREATE TABLE IF NOT EXISTS links (
                  src_note_id TEXT NOT NULL,
                  dst_note_id TEXT NOT NULL,
                  relation TEXT DEFAULT 'wikilink'
                );
                """
            )

    def reset_schema(self) -> None:
        """Drop and recreate the SQLite schema."""

        with self.connect() as connection:
            connection.executescript(
                """
                DROP TABLE IF EXISTS links;
                DROP TABLE IF EXISTS chunks_fts;
                DROP TABLE IF EXISTS chunks;
                DROP TABLE IF EXISTS notes;
                """
            )
        self.create_schema()

    def reindex(self) -> CommandOutput:
        """Rebuild all index tables from the Markdown file tree."""

        self.vault_service.ensure_initialized()
        self.reset_schema()
        note_paths = list(self.vault_service.iter_note_paths())
        failed_notes: list[DiagnosticIssue] = []
        indexed = 0
        chunk_count = 0
        link_count = 0
        logger.info(
            "Reindexing %s note file(s) into %s.",
            len(note_paths),
            self.vault_service.index_db_path,
        )
        with self.connect() as connection:
            for path in note_paths:
                try:
                    document = load_note(path)
                    frontmatter = validate_note_frontmatter(document.frontmatter)
                    raw_schema_version = document.frontmatter.get("schema_version")
                    schema_version = (
                        int(raw_schema_version)
                        if raw_schema_version is not None and str(raw_schema_version).strip()
                        else None
                    )
                    note_id = str(frontmatter.id)
                    updated_at = getattr(frontmatter, "updated_at", None) or getattr(
                        frontmatter, "date", ""
                    )
                    connection.execute(
                        """
                        INSERT INTO notes (
                          id,
                          schema_version,
                          kind,
                          title,
                          path,
                          updated_at,
                          aliases_json,
                          entity_refs_json,
                          confidence,
                          date,
                          project,
                          tags_json,
                          source_refs_json
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            note_id,
                            schema_version,
                            frontmatter.kind.value,
                            frontmatter.title,
                            str(path),
                            str(updated_at),
                            json.dumps(
                                getattr(frontmatter, "aliases", []),
                                ensure_ascii=False,
                            ),
                            json.dumps(
                                getattr(frontmatter, "entity_refs", []),
                                ensure_ascii=False,
                            ),
                            getattr(frontmatter, "confidence", None),
                            str(getattr(frontmatter, "date", "") or ""),
                            getattr(frontmatter, "project", None),
                            json.dumps(frontmatter.tags, ensure_ascii=False),
                            json.dumps(frontmatter.source_refs, ensure_ascii=False),
                        ),
                    )
                    for order, chunk in enumerate(chunk_markdown(document.body), start=1):
                        fts_text = (
                            f"{chunk['heading']}\n{chunk['text']}".strip()
                            if chunk["heading"]
                            else chunk["text"]
                        )
                        cursor = connection.execute(
                            """
                            INSERT INTO chunks (note_id, ord, heading, text)
                            VALUES (?, ?, ?, ?)
                            """,
                            (note_id, order, chunk["heading"], chunk["text"]),
                        )
                        connection.execute(
                            "INSERT INTO chunks_fts(rowid, text) VALUES (?, ?)",
                            (cursor.lastrowid, fts_text),
                        )
                        chunk_count += 1
                    for link in extract_links(document.body):
                        connection.execute(
                            (
                                "INSERT INTO links "
                                "(src_note_id, dst_note_id, relation) VALUES (?, ?, ?)"
                            ),
                            (note_id, link, "wikilink"),
                        )
                        link_count += 1
                    indexed += 1
                except Exception as exc:
                    logger.warning("Skipping invalid note during reindex: %s (%s)", path, exc)
                    failed_notes.append(
                        DiagnosticIssue(
                            code="reindex_skipped_note",
                            severity=DiagnosticSeverity.WARNING,
                            path=str(path),
                            message=str(exc),
                        )
                    )
                    continue
        skipped_notes = len(failed_notes)
        logger.info(
            "Reindex finished for %s: %s indexed, %s skipped, %s chunks, %s links.",
            self.vault_service.index_db_path,
            indexed,
            skipped_notes,
            chunk_count,
            link_count,
        )
        message = (
            (f"Reindexed {indexed} note(s), built {chunk_count} chunk(s) and {link_count} link(s).")
            if not failed_notes
            else (
                f"Reindexed {indexed} of {len(note_paths)} note(s), built {chunk_count} chunk(s), "
                f"{link_count} link(s), skipped {skipped_notes} note(s)."
            )
        )
        return CommandOutput(
            message=message,
            data={
                "note_files": len(note_paths),
                "indexed_notes": indexed,
                "skipped_notes": skipped_notes,
                "failed_notes": [issue.model_dump(mode="json") for issue in failed_notes],
                "link_count": link_count,
                "chunk_count": chunk_count,
                "index_path": str(self.vault_service.index_db_path),
            },
        )

    def search(
        self,
        query: str,
        kind: str | None = None,
        top_k: int = 8,
    ) -> list[dict[str, Any]]:
        """Search indexed note chunks using FTS5."""

        if not self.vault_service.index_db_path.exists():
            raise IndexCorruptionError("SQLite index does not exist. Run reindex first.")
        fts_query = build_fts_match_query(query)
        sql = """
            SELECT
              notes.id,
              notes.schema_version,
              notes.kind,
              notes.title,
              notes.path,
              notes.updated_at,
              notes.aliases_json,
              notes.entity_refs_json,
              notes.confidence,
              notes.date,
              notes.project,
              notes.tags_json,
              notes.source_refs_json,
              chunks.heading,
              snippet(chunks_fts, 0, '[', ']', '...', 18) AS snippet,
              bm25(chunks_fts) AS fts_score
            FROM chunks_fts
            JOIN chunks ON chunks_fts.rowid = chunks.id
            JOIN notes ON notes.id = chunks.note_id
            WHERE chunks_fts MATCH ?
        """
        params: list[Any] = [fts_query]
        if kind:
            sql += " AND notes.kind = ?"
            params.append(kind)
        sql += " ORDER BY bm25(chunks_fts) LIMIT ?"
        params.append(top_k * 3)
        metadata_sql = """
            SELECT
              notes.id,
              notes.schema_version,
              notes.kind,
              notes.title,
              notes.path,
              notes.updated_at,
              notes.aliases_json,
              notes.entity_refs_json,
              notes.confidence,
              notes.date,
              notes.project,
              notes.tags_json,
              notes.source_refs_json,
              '' AS heading,
              notes.title AS snippet,
              0.0 AS fts_score
            FROM notes
            WHERE (
              lower(notes.title) LIKE ?
              OR lower(notes.aliases_json) LIKE ?
              OR lower(COALESCE(notes.project, '')) LIKE ?
              OR COALESCE(notes.date, '') = ?
              OR lower(notes.source_refs_json) LIKE ?
            )
        """
        metadata_query = f"%{query.lower()}%"
        metadata_params: list[Any] = [
            metadata_query,
            metadata_query,
            metadata_query,
            query,
            metadata_query,
        ]
        if kind:
            metadata_sql += " AND notes.kind = ?"
            metadata_params.append(kind)
        metadata_sql += " LIMIT ?"
        metadata_params.append(top_k * 3)
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
            metadata_rows = connection.execute(metadata_sql, metadata_params).fetchall()
        return [dict(row) for row in [*rows, *metadata_rows]]

    def note_count(self) -> int:
        """Count indexed notes."""

        if not self.vault_service.index_db_path.exists():
            return 0
        with self.connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM notes").fetchone()
        return int(row["count"] if row else 0)

    def note_rows(self) -> list[dict[str, Any]]:
        """Return indexed note ids and paths for diagnostics."""

        if not self.vault_service.index_db_path.exists():
            return []
        with self.connect() as connection:
            rows = connection.execute("SELECT id, path FROM notes").fetchall()
        return [dict(row) for row in rows]

    def validate(self) -> None:
        """Raise when the index is unavailable."""

        if not Path(self.vault_service.index_db_path).exists():
            raise IndexCorruptionError("SQLite index does not exist.")
