"""W5.A / F-012: Obsidian-sync persistence methods, extracted from
`hermes_state.py` as the first piece of the incremental SessionDB split.

The mixin pattern keeps `SessionDB`'s public API identical: callers
still call `db.upsert_obsidian_managed_file(...)` etc. through the
final composed class. All methods rely on `self._execute_write` and
`self._lock`/`self._conn` provided by the base class, so this file
must NOT be instantiated alone — only mixed into `SessionDB`.

Self-contained because the obsidian methods touch a dedicated cluster
of tables (obsidian_managed_files, obsidian_sync_events,
obsidian_file_revisions, obsidian_attachment_refs, obsidian_canvas_refs,
obsidian_import_candidates, obsidian_conflicts, obsidian_sync_checkpoint)
and have no cross-references to session/message/routing methods.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional


class _ObsidianMixin:
    """Mixin providing all `obsidian_*` persistence methods on SessionDB."""

    def upsert_obsidian_managed_file(
        self,
        vault_relative_path: str,
        managed_relative_path: str,
        uuid: str = None,
        entity_type: str = None,
        wiki_page_type: str = None,
        file_ext: str = None,
        content_hash: str = None,
        last_vault_mtime: float = None,
        last_vault_size: int = None,
        last_db_revision_id: int = None,
        last_sync_direction: str = "vault_scan",
        sync_status: str = "synced",
        conflict_state: str = "none",
        source_origin: str = "managed",
        tombstoned: bool = False,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Insert or update metadata for a managed Obsidian file."""
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO obsidian_managed_files (
                    uuid, vault_relative_path, managed_relative_path, entity_type, wiki_page_type,
                    file_ext, content_hash, last_vault_mtime, last_vault_size, last_db_revision_id,
                    last_sync_direction, sync_status, conflict_state, source_origin, tombstoned,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vault_relative_path) DO UPDATE SET
                    uuid = excluded.uuid,
                    managed_relative_path = excluded.managed_relative_path,
                    entity_type = excluded.entity_type,
                    wiki_page_type = excluded.wiki_page_type,
                    file_ext = excluded.file_ext,
                    content_hash = excluded.content_hash,
                    last_vault_mtime = excluded.last_vault_mtime,
                    last_vault_size = excluded.last_vault_size,
                    last_db_revision_id = COALESCE(excluded.last_db_revision_id, obsidian_managed_files.last_db_revision_id),
                    last_sync_direction = excluded.last_sync_direction,
                    sync_status = excluded.sync_status,
                    conflict_state = excluded.conflict_state,
                    source_origin = excluded.source_origin,
                    tombstoned = excluded.tombstoned,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    uuid,
                    vault_relative_path,
                    managed_relative_path,
                    entity_type,
                    wiki_page_type,
                    file_ext,
                    content_hash,
                    last_vault_mtime,
                    last_vault_size,
                    last_db_revision_id,
                    last_sync_direction,
                    sync_status,
                    conflict_state,
                    source_origin,
                    1 if tombstoned else 0,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM obsidian_managed_files WHERE vault_relative_path = ?",
                (vault_relative_path,),
            ).fetchone()
            return row[0] if row else 0

        return self._execute_write(_do)

    def record_obsidian_sync_event(
        self,
        event_type: str,
        path: str = None,
        direction: str = None,
        status: str = "ok",
        detail: str = None,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Append a sync event for the Obsidian integration."""

        def _do(conn):
            cursor = conn.execute(
                """
                INSERT INTO obsidian_sync_events (event_type, path, direction, status, detail, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (event_type, path, direction, status, detail, json.dumps(metadata or {}), time.time()),
            )
            return cursor.lastrowid

        return self._execute_write(_do)

    def record_obsidian_file_revision(
        self,
        vault_relative_path: str,
        content_hash: str,
        content_text: Optional[str],
        source: str,
        actor: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Persist a canonical content snapshot for a managed Obsidian file."""

        def _do(conn):
            cursor = conn.execute(
                """
                INSERT INTO obsidian_file_revisions (
                    vault_relative_path, content_hash, content_text, source, actor, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vault_relative_path,
                    content_hash,
                    content_text,
                    source,
                    actor,
                    json.dumps(metadata or {}),
                    time.time(),
                ),
            )
            return cursor.lastrowid

        return self._execute_write(_do)

    def get_obsidian_managed_file(self, vault_relative_path: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest managed-file metadata row for a vault-relative path."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    id,
                    uuid,
                    vault_relative_path,
                    managed_relative_path,
                    entity_type,
                    wiki_page_type,
                    file_ext,
                    content_hash,
                    last_vault_mtime,
                    last_vault_size,
                    last_db_revision_id,
                    last_sync_direction,
                    sync_status,
                    conflict_state,
                    source_origin,
                    tombstoned,
                    metadata_json,
                    created_at,
                    updated_at
                FROM obsidian_managed_files
                WHERE vault_relative_path = ?
                """,
                (vault_relative_path,),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            data["tombstoned"] = bool(data.get("tombstoned"))
            metadata_json = data.pop("metadata_json", None)
            data["metadata"] = json.loads(metadata_json) if metadata_json else {}
            return data

    def list_obsidian_managed_files(self, include_tombstoned: bool = False) -> List[Dict[str, Any]]:
        """List managed-file metadata rows for the Obsidian integration."""
        with self._lock:
            query = """
                SELECT
                    id,
                    uuid,
                    vault_relative_path,
                    managed_relative_path,
                    entity_type,
                    wiki_page_type,
                    file_ext,
                    content_hash,
                    last_vault_mtime,
                    last_vault_size,
                    last_db_revision_id,
                    last_sync_direction,
                    sync_status,
                    conflict_state,
                    source_origin,
                    tombstoned,
                    metadata_json,
                    created_at,
                    updated_at
                FROM obsidian_managed_files
            """
            if not include_tombstoned:
                query += " WHERE tombstoned = 0"
            query += " ORDER BY vault_relative_path ASC"
            rows = self._conn.execute(query).fetchall()
            items: List[Dict[str, Any]] = []
            for row in rows:
                data = dict(row)
                data["tombstoned"] = bool(data.get("tombstoned"))
                metadata_json = data.pop("metadata_json", None)
                data["metadata"] = json.loads(metadata_json) if metadata_json else {}
                items.append(data)
            return items

    def get_latest_obsidian_file_revision(self, vault_relative_path: str) -> Optional[Dict[str, Any]]:
        """Fetch the newest stored revision snapshot for a managed file."""
        with self._lock:
            row = self._conn.execute(
                """
                SELECT
                    id,
                    vault_relative_path,
                    content_hash,
                    content_text,
                    source,
                    actor,
                    metadata_json,
                    created_at
                FROM obsidian_file_revisions
                WHERE vault_relative_path = ?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (vault_relative_path,),
            ).fetchone()
            if not row:
                return None
            data = dict(row)
            metadata_json = data.pop("metadata_json", None)
            data["metadata"] = json.loads(metadata_json) if metadata_json else {}
            return data

    def list_open_obsidian_conflicts(self, vault_relative_path: Optional[str] = None) -> List[Dict[str, Any]]:
        """List open conflicts, optionally scoped to a single path."""
        with self._lock:
            if vault_relative_path:
                rows = self._conn.execute(
                    """
                    SELECT *
                    FROM obsidian_conflicts
                    WHERE status = 'open' AND vault_relative_path = ?
                    ORDER BY created_at DESC
                    """,
                    (vault_relative_path,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT *
                    FROM obsidian_conflicts
                    WHERE status = 'open'
                    ORDER BY created_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]

    def replace_obsidian_attachment_refs(self, owner_path: str, refs: List[Dict[str, Any]]) -> None:
        """Replace attachment/embed references for a managed markdown file."""
        now = time.time()

        def _do(conn):
            conn.execute("DELETE FROM obsidian_attachment_index WHERE owner_path = ?", (owner_path,))
            for ref in refs:
                conn.execute(
                    """
                    INSERT INTO obsidian_attachment_index (
                        owner_path, target_path, target_type, exists_flag, mime_type, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        owner_path,
                        ref.get("target_path"),
                        ref.get("target_type", "embed"),
                        1 if ref.get("exists") else 0,
                        ref.get("mime_type"),
                        now,
                        now,
                    ),
                )

        self._execute_write(_do)

    def replace_obsidian_canvas_refs(self, canvas_path: str, refs: List[Dict[str, Any]]) -> None:
        """Replace indexed references for an Obsidian canvas file."""
        now = time.time()

        def _do(conn):
            conn.execute("DELETE FROM obsidian_canvas_index WHERE canvas_path = ?", (canvas_path,))
            for ref in refs:
                conn.execute(
                    """
                    INSERT INTO obsidian_canvas_index (
                        canvas_path, node_id, node_type, target_path, broken, metadata_json, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        canvas_path,
                        ref.get("node_id") or str(random.random()),
                        ref.get("node_type", "unknown"),
                        ref.get("target_path"),
                        1 if ref.get("broken") else 0,
                        json.dumps(ref.get("metadata") or {}),
                        now,
                    ),
                )

        self._execute_write(_do)

    def upsert_obsidian_import_candidate(
        self,
        vault_relative_path: str,
        title: str,
        file_uuid: str = None,
        content_hash: str = None,
        last_vault_mtime: float = None,
        imported: bool = False,
        imported_managed_path: str = None,
        metadata: Dict[str, Any] = None,
    ) -> int:
        """Insert or update an unmanaged Obsidian note as an import candidate."""
        now = time.time()

        def _do(conn):
            conn.execute(
                """
                INSERT INTO obsidian_import_candidates (
                    vault_relative_path, title, file_uuid, content_hash, last_vault_mtime, imported,
                    imported_managed_path, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(vault_relative_path) DO UPDATE SET
                    title = excluded.title,
                    file_uuid = excluded.file_uuid,
                    content_hash = excluded.content_hash,
                    last_vault_mtime = excluded.last_vault_mtime,
                    imported = excluded.imported,
                    imported_managed_path = COALESCE(excluded.imported_managed_path, obsidian_import_candidates.imported_managed_path),
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    vault_relative_path,
                    title,
                    file_uuid,
                    content_hash,
                    last_vault_mtime,
                    1 if imported else 0,
                    imported_managed_path,
                    json.dumps(metadata or {}),
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT id FROM obsidian_import_candidates WHERE vault_relative_path = ?",
                (vault_relative_path,),
            ).fetchone()
            return row[0] if row else 0

        return self._execute_write(_do)

    def record_obsidian_conflict(
        self,
        vault_relative_path: str,
        conflict_type: str,
        summary: str,
        uuid: str = None,
        entity_type: str = None,
        db_snapshot: str = None,
        vault_snapshot: str = None,
    ) -> int:
        """Persist an Obsidian sync conflict."""
        now = time.time()

        def _do(conn):
            cursor = conn.execute(
                """
                INSERT INTO obsidian_conflicts (
                    vault_relative_path, uuid, entity_type, conflict_type, status,
                    summary, db_snapshot, vault_snapshot, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'open', ?, ?, ?, ?, ?)
                """,
                (
                    vault_relative_path,
                    uuid,
                    entity_type,
                    conflict_type,
                    summary,
                    db_snapshot,
                    vault_snapshot,
                    now,
                    now,
                ),
            )
            return cursor.lastrowid

        return self._execute_write(_do)

    def upsert_obsidian_sync_checkpoint(
        self,
        scope: str,
        last_started_at: float = None,
        last_completed_at: float = None,
        last_status: str = None,
        last_error: str = None,
        last_scan_count: int = 0,
        last_change_count: int = 0,
    ) -> None:
        """Persist last sync-run checkpoint state for a scope."""

        def _do(conn):
            conn.execute(
                """
                INSERT INTO obsidian_sync_checkpoints (
                    scope, last_started_at, last_completed_at, last_status, last_error,
                    last_scan_count, last_change_count, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope) DO UPDATE SET
                    last_started_at = COALESCE(excluded.last_started_at, obsidian_sync_checkpoints.last_started_at),
                    last_completed_at = COALESCE(excluded.last_completed_at, obsidian_sync_checkpoints.last_completed_at),
                    last_status = COALESCE(excluded.last_status, obsidian_sync_checkpoints.last_status),
                    last_error = excluded.last_error,
                    last_scan_count = excluded.last_scan_count,
                    last_change_count = excluded.last_change_count,
                    updated_at = excluded.updated_at
                """,
                (
                    scope,
                    last_started_at,
                    last_completed_at,
                    last_status,
                    last_error,
                    last_scan_count,
                    last_change_count,
                    time.time(),
                ),
            )

        self._execute_write(_do)
