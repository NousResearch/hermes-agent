#!/usr/bin/env python3
"""
CodeSessionService — create and manage code sessions.

A CodeSession tracks a unit of coding work within a workspace:
  workspace + hermes_session + task + provider/model + status + timeline.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CodeSessionService:
    """Business logic for code sessions.

    Delegates persistence to CodeSessionDB and WorkspaceDB (hermes_state).
    Does not execute commands or modify workspace files.

    Pass db_path to override the default DB location (useful in tests).
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path

    def _code_session_db(self):
        from hermes_state import CodeSessionDB
        return CodeSessionDB(db_path=self._db_path)

    def _workspace_db(self):
        from hermes_state import WorkspaceDB
        return WorkspaceDB(db_path=self._db_path)

    def create_session(
        self,
        workspace_id: str,
        hermes_session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        title: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> dict:
        """Create a CodeSession for an existing workspace.

        Copies the workspace's current branch into the session.
        Raises ValueError if workspace_id not found.
        """
        wdb = self._workspace_db()
        try:
            workspace = wdb.get_workspace(workspace_id)
        finally:
            wdb.close()

        if not workspace:
            raise ValueError(f"Workspace not found: {workspace_id}")

        branch = workspace.get("branch")

        db = self._code_session_db()
        try:
            session = db.create_session(
                workspace_id=workspace_id,
                hermes_session_id=hermes_session_id,
                task_id=task_id,
                title=title,
                provider=provider,
                model=model,
                branch=branch,
                metadata=metadata,
            )
            db.add_event(
                session["id"],
                "code_session.created",
                message=f"Session created for workspace {workspace_id}",
                payload={"workspace_id": workspace_id, "title": title},
            )
        finally:
            db.close()

        return session

    def list_sessions(
        self,
        workspace_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        db = self._code_session_db()
        try:
            return db.list_sessions(
                workspace_id=workspace_id,
                status=status,
                limit=limit,
                offset=offset,
            )
        finally:
            db.close()

    def get_session(self, code_session_id: str) -> Optional[dict]:
        db = self._code_session_db()
        try:
            return db.get_session(code_session_id)
        finally:
            db.close()

    def update_session(self, code_session_id: str, **updates) -> dict:
        """Update allowed fields. Emits status_changed event when status changes."""
        db = self._code_session_db()
        try:
            existing = db.get_session(code_session_id)
            if not existing:
                raise ValueError(f"CodeSession not found: {code_session_id}")

            old_status = existing.get("status")
            updated = db.update_session(code_session_id, updates)
            if updated is None:
                raise ValueError(f"CodeSession not found: {code_session_id}")

            new_status = updated.get("status")
            if new_status and new_status != old_status:
                db.add_event(
                    code_session_id,
                    "code_session.status_changed",
                    message=f"Status changed: {old_status} -> {new_status}",
                    payload={"old_status": old_status, "new_status": new_status},
                )
            else:
                db.add_event(
                    code_session_id,
                    "code_session.updated",
                    message="Session updated",
                    payload={"fields": list(updates.keys())},
                )
        finally:
            db.close()

        return updated

    def update_status(
        self, code_session_id: str, status: str, message: Optional[str] = None
    ) -> dict:
        return self.update_session(code_session_id, status=status)

    def cancel_session(
        self, code_session_id: str, reason: Optional[str] = None
    ) -> dict:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db = self._code_session_db()
        try:
            existing = db.get_session(code_session_id)
            if not existing:
                raise ValueError(f"CodeSession not found: {code_session_id}")

            updated = db.update_session(
                code_session_id,
                {"status": "cancelled", "completed_at": now},
            )
            db.add_event(
                code_session_id,
                "code_session.cancelled",
                message=reason or "Session cancelled",
                payload={"reason": reason},
            )
        finally:
            db.close()

        return updated

    def complete_session(
        self, code_session_id: str, summary: Optional[str] = None
    ) -> dict:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        db = self._code_session_db()
        try:
            existing = db.get_session(code_session_id)
            if not existing:
                raise ValueError(f"CodeSession not found: {code_session_id}")

            upd: dict = {"status": "done", "completed_at": now}
            if summary is not None:
                upd["summary"] = summary
            updated = db.update_session(code_session_id, upd)
            db.add_event(
                code_session_id,
                "code_session.completed",
                message=summary or "Session completed",
                payload={"summary": summary},
            )
        finally:
            db.close()

        return updated

    def add_event(
        self,
        code_session_id: str,
        event_type: str,
        message: Optional[str] = None,
        payload: Optional[dict] = None,
    ) -> dict:
        db = self._code_session_db()
        try:
            if not db.get_session(code_session_id):
                raise ValueError(f"CodeSession not found: {code_session_id}")
            return db.add_event(code_session_id, event_type, message, payload)
        finally:
            db.close()

    def list_events(self, code_session_id: str) -> list[dict]:
        db = self._code_session_db()
        try:
            return db.list_events(code_session_id)
        finally:
            db.close()

    def list_artifacts(self, code_session_id: str) -> list[dict]:
        db = self._code_session_db()
        try:
            session = db.get_session(code_session_id)
            if not session:
                raise ValueError(f"CodeSession not found: {code_session_id}")
            hermes_session_id = session.get("hermes_session_id")
            return db.list_artifacts_for_code_session(code_session_id, hermes_session_id)
        finally:
            db.close()

    def link_artifact(self, code_session_id: str, artifact_id: str) -> dict:
        db = self._code_session_db()
        try:
            if not db.get_session(code_session_id):
                raise ValueError(f"CodeSession not found: {code_session_id}")
            artifact = db.link_artifact_to_session(artifact_id, code_session_id)
            if not artifact:
                raise ValueError(f"Artifact not found: {artifact_id}")
            db.add_event(
                code_session_id,
                "artifact.linked",
                message=f"Artifact {artifact_id} linked",
                payload={"artifact_id": artifact_id, "path": artifact.get("path")},
            )
        finally:
            db.close()

        return artifact
