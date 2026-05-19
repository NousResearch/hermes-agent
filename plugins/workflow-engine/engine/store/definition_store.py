"""
DefinitionStore — CRUD for workflow_definitions table.

Wraps raw sqlite3.Connection. All methods are synchronous (SQLite is sync).
"""
from __future__ import annotations

import hashlib
import json
import time
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from engine.schemas.workflow import WorkflowDefinition, WorkflowSource
from engine.discovery.validator import validate_workflow_yaml


def _sha256(text: str) -> str:
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _now_ms() -> int:
    return int(time.time() * 1000)


def _row_to_def(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(row)


class DefinitionStore:
    """CRUD operations over workflow_definitions."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # ------------------------------------------------------------------
    # list
    # ------------------------------------------------------------------

    def list_definitions(
        self,
        *,
        source: Optional[str] = None,
        kind: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if source:
            clauses.append("source = ?")
            params.append(source)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = self._conn.execute(
            f"SELECT * FROM workflow_definitions {where} ORDER BY name LIMIT ?",
            params,
        ).fetchall()
        return [_row_to_def(r) for r in rows]

    # ------------------------------------------------------------------
    # get
    # ------------------------------------------------------------------

    def get_definition(self, definition_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM workflow_definitions WHERE id = ?",
            (definition_id,),
        ).fetchone()
        return _row_to_def(row) if row else None

    # ------------------------------------------------------------------
    # upsert
    # ------------------------------------------------------------------

    def upsert_definition(
        self,
        *,
        yaml_text: str,
        source_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse yaml_text, validate, upsert, return the row dict."""
        workflow, error = validate_workflow_yaml(yaml_text, source_path or "<inline>")
        if error or workflow is None:
            raise ValueError(f"Invalid workflow YAML: {error.error if error else 'unknown'}")

        if workflow.id is None:
            stem = Path(source_path).stem if source_path else "unnamed"
            object.__setattr__(workflow, "id", stem.lower().replace(" ", "-"))

        checksum = _sha256(yaml_text)
        now = _now_ms()
        source: WorkflowSource = "user"

        existing = self._conn.execute(
            "SELECT checksum FROM workflow_definitions WHERE id = ?",
            (workflow.id,),
        ).fetchone()

        if existing is not None:
            if existing["checksum"] == checksum:
                return self.get_definition(workflow.id)  # type: ignore[return-value]
            self._conn.execute(
                """
                UPDATE workflow_definitions
                   SET name=?, description=?, source=?, scope_path=?, yaml=?,
                       checksum=?, updated_at=?, kind=?
                 WHERE id=?
                """,
                (
                    workflow.name,
                    workflow.description,
                    source,
                    source_path,
                    yaml_text,
                    checksum,
                    now,
                    workflow.kind or "workflow",
                    workflow.id,
                ),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO workflow_definitions
                  (id, name, description, source, scope_path, yaml, checksum,
                   created_at, updated_at, kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow.id,
                    workflow.name,
                    workflow.description,
                    source,
                    source_path,
                    yaml_text,
                    checksum,
                    now,
                    now,
                    workflow.kind or "workflow",
                ),
            )
        self._conn.commit()
        return self.get_definition(workflow.id)  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # delete
    # ------------------------------------------------------------------

    def delete_definition(self, definition_id: str) -> int:
        """Delete a non-bundled definition. Returns rows deleted (0 or 1)."""
        result = self._conn.execute(
            "DELETE FROM workflow_definitions WHERE id = ? AND source != 'bundled'",
            (definition_id,),
        )
        self._conn.commit()
        return result.rowcount

    # ------------------------------------------------------------------
    # seed bundled
    # ------------------------------------------------------------------

    def seed_bundled(self, bundled_dir: Path) -> Dict[str, int]:
        """Upsert all *.yaml files from bundled_dir. Returns {inserted, skipped, errors}."""
        inserted = skipped = errors = 0
        if not bundled_dir.exists():
            return {"inserted": 0, "skipped": 0, "errors": 0}
        for yaml_file in sorted(bundled_dir.glob("*.yaml")):
            try:
                content = yaml_file.read_text(encoding="utf-8")
                workflow, error = validate_workflow_yaml(content, yaml_file.name)
                if error or not workflow:
                    errors += 1
                    continue
                if workflow.id is None:
                    object.__setattr__(workflow, "id", yaml_file.stem.lower().replace(" ", "-"))
                checksum = _sha256(content)
                existing = self._conn.execute(
                    "SELECT checksum FROM workflow_definitions WHERE id = ?",
                    (workflow.id,),
                ).fetchone()
                if existing and existing["checksum"] == checksum:
                    skipped += 1
                    continue
                now = _now_ms()
                if existing:
                    self._conn.execute(
                        """UPDATE workflow_definitions
                              SET name=?, description=?, source=?, yaml=?, checksum=?, updated_at=?, kind=?
                            WHERE id=?""",
                        (workflow.name, workflow.description, "bundled", content, checksum, now, workflow.kind or "workflow", workflow.id),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO workflow_definitions
                             (id, name, description, source, yaml, checksum, created_at, updated_at, kind)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (workflow.id, workflow.name, workflow.description, "bundled", content, checksum, now, now, workflow.kind or "workflow"),
                    )
                inserted += 1
            except Exception:
                errors += 1
        self._conn.commit()
        return {"inserted": inserted, "skipped": skipped, "errors": errors}
