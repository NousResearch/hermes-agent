#!/usr/bin/env python3
"""Typed artifact ledger for Hermes Code Mode."""

from __future__ import annotations

import time
import uuid
from typing import Any

from hermes_cli.code.event_bus import get_code_event_bus
from hermes_state import SessionDB


ARTIFACT_TYPES: tuple[str, ...] = (
    "task_intake",
    "prd_lite",
    "acceptance_criteria",
    "architecture_note",
    "adr",
    "implementation_plan",
    "command_log",
    "diff_summary",
    "test_report",
    "review_report",
    "deploy_plan",
    "deploy_report",
    "memory_update",
)


class ArtifactLedger:
    def __init__(self, db_path=None):
        self._db_path = db_path

    def _db(self) -> SessionDB:
        return SessionDB(db_path=self._db_path) if self._db_path else SessionDB()

    def _emit(self, event_type: str, artifact: dict[str, Any]) -> None:
        bus = get_code_event_bus(self._db_path)
        payload = {
            "artifact_id": artifact.get("id"),
            "artifact_type": artifact.get("artifact_type"),
            "title": artifact.get("title"),
        }
        bus.publish(
            event_type,
            payload=payload,
            workspace_id=artifact.get("workspace_id"),
            code_session_id=artifact.get("code_session_id"),
            orchestrated_run_id=artifact.get("orchestrated_run_id"),
            metadata={"source": "artifact_ledger"},
            source="artifact_ledger",
        )

    def create_artifact(
        self,
        artifact_type: str,
        content: str,
        *,
        title: str | None = None,
        content_type: str = "markdown",
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        command_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if artifact_type not in ARTIFACT_TYPES:
            raise ValueError(f"Unknown artifact_type: {artifact_type}")
        artifact = {
            "id": str(uuid.uuid4()),
            "artifact_type": artifact_type,
            "title": title or artifact_type.replace("_", " ").title(),
            "content": content or "",
            "content_type": content_type or "markdown",
            "workspace_id": workspace_id,
            "code_session_id": code_session_id,
            "orchestrated_run_id": orchestrated_run_id,
            "command_id": command_id,
            "metadata": metadata or {},
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        db = self._db()
        try:
            db.create_code_artifact(artifact)
            created = artifact
        finally:
            db.close()
        self._emit("code.artifact.created", created)
        self._emit("artifact.created", created)
        return created

    def list_artifacts(
        self,
        *,
        workspace_id: str | None = None,
        code_session_id: str | None = None,
        orchestrated_run_id: str | None = None,
        artifact_type: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        db = self._db()
        try:
            return db.list_code_artifacts(
                workspace_id=workspace_id,
                code_session_id=code_session_id,
                orchestrated_run_id=orchestrated_run_id,
                artifact_type=artifact_type,
                limit=limit,
                offset=offset,
            )
        finally:
            db.close()

    def list_session_artifacts(
        self,
        code_session_id: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        return self.list_artifacts(code_session_id=code_session_id, limit=limit, offset=offset)
