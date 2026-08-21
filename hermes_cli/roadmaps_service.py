"""Read-only Roadmaps service backed by an existing profile ``projects.db``."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote


class RoadmapsUnavailable(RuntimeError):
    """The profile store exists but cannot be read safely."""


class RoadmapsService:
    MAX_IDENTIFIER_LENGTH = 128
    _PUBLIC_COLUMNS = {
        "roadmaps": (
            "profile_id", "project_id", "roadmap_id", "title", "purpose",
            "lifecycle_state", "active_version", "created_by", "updated_by",
            "created_at", "updated_at",
        ),
        "roadmap_versions": (
            "profile_id", "project_id", "roadmap_id", "version", "state",
            "source", "reason", "created_by", "created_at", "content_hash",
        ),
        "roadmap_nodes": (
            "profile_id", "project_id", "roadmap_id", "version", "node_id",
            "parent_node_id", "kind", "title", "description", "state",
            "progress", "owner_agent", "block_reason", "created_at", "updated_at",
        ),
        "roadmap_relations": (
            "profile_id", "project_id", "roadmap_id", "version", "relation_id",
            "from_node_id", "to_node_id", "kind", "state", "reason",
        ),
        "roadmap_todos": (
            "profile_id", "project_id", "roadmap_id", "version", "todo_id",
            "node_id", "title", "state", "position", "created_at", "updated_at",
        ),
        "roadmap_sessions": (
            "profile_id", "project_id", "roadmap_id", "stored_session_id",
            "kind", "node_id", "plan_version", "state", "actor", "created_at",
            "updated_at",
        ),
    }

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    @staticmethod
    def _required(value: str, name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        value = value.strip()
        if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError(f"{name} required")
        if len(value) > RoadmapsService.MAX_IDENTIFIER_LENGTH:
            raise ValueError(f"{name} must be at most {RoadmapsService.MAX_IDENTIFIER_LENGTH} characters")
        return value

    @classmethod
    def _optional(cls, value: str | None, name: str) -> str | None:
        return None if value is None else cls._required(value, name)

    def _connection(self) -> sqlite3.Connection | None:
        path = Path(self.db_path) if self.db_path is not None else None
        if path is None or not path.is_file():
            return None
        uri_path = quote(str(path.resolve()), safe="/")
        uri = f"file:{uri_path}?mode=ro"
        conn = None
        handed_off = False
        try:
            conn = sqlite3.connect(uri, uri=True)
            conn.row_factory = sqlite3.Row
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )}
            required = {"roadmaps", "roadmap_versions", "roadmap_nodes",
                        "roadmap_relations", "roadmap_todos", "roadmap_sessions"}
            if not required.issubset(tables):
                raise RoadmapsUnavailable
            handed_off = True
            return conn
        except RoadmapsUnavailable:
            raise
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            if conn is not None and not handed_off:
                conn.close()

    @classmethod
    def _select(cls, table: str) -> str:
        return f"SELECT {', '.join(cls._PUBLIC_COLUMNS[table])} FROM {table}"

    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict[str, Any] | None:
        return dict(row) if row is not None else None

    def list(self, profile_id: str, project_id: str | None = None) -> dict[str, Any]:
        profile_id = self._required(profile_id, "profile_id")
        project_id = self._optional(project_id, "project_id")
        query = (
            f"SELECT r.{', r.'.join(self._PUBLIC_COLUMNS['roadmaps'])}, p.name AS project_name "
            "FROM roadmaps r LEFT JOIN projects p ON p.id = r.project_id "
            "WHERE r.profile_id = ?"
        )
        args: list[Any] = [profile_id]
        if project_id is not None:
            query += " AND r.project_id = ?"
            args.append(project_id)
        query += " ORDER BY r.project_id, r.roadmap_id"
        scope = {"profile_id": profile_id, **({"project_id": project_id} if project_id is not None else {})}
        conn = self._connection()
        if conn is None:
            return {"roadmaps": [], "scope": scope}
        try:
            rows = [self._row(row) for row in conn.execute(query, args)]
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            conn.close()
        return {"roadmaps": rows, "scope": scope}

    def get(self, profile_id: str, project_id: str, roadmap_id: str) -> dict[str, Any]:
        profile_id = self._required(profile_id, "profile_id")
        project_id = self._required(project_id, "project_id")
        roadmap_id = self._required(roadmap_id, "roadmap_id")
        scope = {"profile_id": profile_id, "project_id": project_id, "roadmap_id": roadmap_id}
        conn = self._connection()
        if conn is None:
            return {"found": False, "scope": scope, "roadmap": None}
        try:
            roadmap = conn.execute(
                f"{self._select('roadmaps')} WHERE profile_id=? AND project_id=? AND roadmap_id=?",
                (profile_id, project_id, roadmap_id),
            ).fetchone()
            if roadmap is None:
                return {"found": False, "scope": scope, "roadmap": None}
            payload: dict[str, Any] = self._row(roadmap) or {}
            versions = []
            for version in conn.execute(
                f"{self._select('roadmap_versions')} WHERE profile_id=? AND project_id=? AND roadmap_id=? ORDER BY version",
                (profile_id, project_id, roadmap_id),
            ):
                version_payload: dict[str, Any] = self._row(version) or {}
                key = (profile_id, project_id, roadmap_id, version["version"])
                version_payload["nodes"] = [self._row(row) for row in conn.execute(f"{self._select('roadmap_nodes')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY node_id", key)]
                version_payload["relations"] = [self._row(row) for row in conn.execute(f"{self._select('roadmap_relations')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY relation_id", key)]
                version_payload["todos"] = [self._row(row) for row in conn.execute(f"{self._select('roadmap_todos')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY position, todo_id", key)]
                versions.append(version_payload)
            payload["versions"] = versions
            return {"found": True, "scope": scope, "roadmap": payload}
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            conn.close()

    def snapshot(self, profile_id: str, project_id: str, roadmap_id: str) -> dict[str, Any]:
        return self.get(profile_id, project_id, roadmap_id)

    def get_snapshot(self, profile_id: str, project_id: str, roadmap_id: str) -> dict[str, Any]:
        return self.snapshot(profile_id, project_id, roadmap_id)

    def list_plans(self, profile_id: str, project_id: str, roadmap_id: str) -> dict[str, Any]:
        """List the roadmap's plan versions, newest first (version DESC)."""
        profile_id = self._required(profile_id, "profile_id")
        project_id = self._required(project_id, "project_id")
        roadmap_id = self._required(roadmap_id, "roadmap_id")
        scope = {"profile_id": profile_id, "project_id": project_id, "roadmap_id": roadmap_id}
        conn = self._connection()
        if conn is None:
            return {"plans": [], "scope": scope}
        try:
            plans = [
                self._row(row)
                for row in conn.execute(
                    f"{self._select('roadmap_versions')} "
                    "WHERE profile_id=? AND project_id=? AND roadmap_id=? "
                    "ORDER BY version DESC",
                    (profile_id, project_id, roadmap_id),
                )
            ]
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            conn.close()
        return {"plans": plans, "scope": scope}

    def get_plan(self, profile_id: str, project_id: str, roadmap_id: str, version: int) -> dict[str, Any]:
        """Fetch one complete plan version (nodes + relations + todos)."""
        profile_id = self._required(profile_id, "profile_id")
        project_id = self._required(project_id, "project_id")
        roadmap_id = self._required(roadmap_id, "roadmap_id")
        if isinstance(version, bool) or not isinstance(version, int):
            raise ValueError("version must be an integer")
        if version < 1:
            raise ValueError("version must be at least 1")
        scope = {"profile_id": profile_id, "project_id": project_id, "roadmap_id": roadmap_id}
        conn = self._connection()
        if conn is None:
            return {"found": False, "scope": scope, "plan": None}
        try:
            row = conn.execute(
                f"{self._select('roadmap_versions')} "
                "WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=?",
                (profile_id, project_id, roadmap_id, version),
            ).fetchone()
            if row is None:
                return {"found": False, "scope": scope, "plan": None}
            plan: dict[str, Any] = self._row(row) or {}
            key = (profile_id, project_id, roadmap_id, version)
            plan["nodes"] = [
                self._row(node) for node in conn.execute(
                    f"{self._select('roadmap_nodes')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY node_id",
                    key,
                )
            ]
            plan["relations"] = [
                self._row(relation) for relation in conn.execute(
                    f"{self._select('roadmap_relations')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY relation_id",
                    key,
                )
            ]
            plan["todos"] = [
                self._row(todo) for todo in conn.execute(
                    f"{self._select('roadmap_todos')} WHERE profile_id=? AND project_id=? AND roadmap_id=? AND version=? ORDER BY position, todo_id",
                    key,
                )
            ]
            return {"found": True, "scope": scope, "plan": plan}
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            conn.close()

    def list_sessions(
        self, profile_id: str, project_id: str, roadmap_id: str
    ) -> dict[str, Any]:
        """List durable session links for exactly one qualified roadmap."""
        profile_id = self._required(profile_id, "profile_id")
        project_id = self._required(project_id, "project_id")
        roadmap_id = self._required(roadmap_id, "roadmap_id")
        scope = {
            "profile_id": profile_id,
            "project_id": project_id,
            "roadmap_id": roadmap_id,
        }
        conn = self._connection()
        if conn is None:
            return {"sessions": [], "scope": scope}
        try:
            sessions = [
                self._row(row)
                for row in conn.execute(
                    f"{self._select('roadmap_sessions')} "
                    "WHERE profile_id=? AND project_id=? AND roadmap_id=? "
                    "ORDER BY CASE state WHEN 'active' THEN 0 ELSE 1 END, "
                    "updated_at DESC, stored_session_id",
                    (profile_id, project_id, roadmap_id),
                )
            ]
        except sqlite3.Error as exc:
            raise RoadmapsUnavailable from exc
        finally:
            conn.close()
        return {"sessions": sessions, "scope": scope}
