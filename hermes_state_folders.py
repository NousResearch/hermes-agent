"""Session folder operations mixed into :class:`hermes_state.SessionDB`."""

import secrets
import sqlite3
import time


class SessionFolderMixin:
    def list_folders(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT f.*, COUNT(m.session_id) AS session_count,
                          COALESCE(GROUP_CONCAT(m.session_id), '') AS session_ids
                   FROM session_folders f
                   LEFT JOIN session_folder_members m ON m.folder_id = f.id
                   GROUP BY f.id
                   ORDER BY f.sort_order ASC, f.created_at ASC"""
            ).fetchall()

        folders = []
        for row in rows:
            folder = dict(row)
            session_ids = folder.pop("session_ids", "")
            folder["session_ids"] = session_ids.split(",") if session_ids else []
            folders.append(folder)
        return folders

    def create_folder(self, *, name: str) -> dict:
        name = (name or "").strip()
        if not name:
            raise ValueError("folder name must not be empty")

        folder_id = "sf_" + secrets.token_hex(4)
        created_at = time.time()

        def create(conn):
            sort_order = conn.execute(
                "SELECT COALESCE(MAX(sort_order), -1) + 1 FROM session_folders"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO session_folders (id, name, sort_order, created_at) "
                "VALUES (?, ?, ?, ?)",
                (folder_id, name, sort_order, created_at),
            )
            row = conn.execute(
                "SELECT f.*, 0 AS session_count FROM session_folders f WHERE f.id = ?",
                (folder_id,),
            ).fetchone()
            result = dict(row) if row else {}
            result["session_ids"] = []
            return result

        return self._execute_write(create)

    def update_folder(self, folder_id: str, *, name: str) -> bool:
        name = (name or "").strip()
        if not name:
            raise ValueError("folder name must not be empty")
        return bool(
            self._execute_write(
                lambda conn: conn.execute(
                    "UPDATE session_folders SET name = ? WHERE id = ?", (name, folder_id)
                ).rowcount
            )
        )

    def delete_folder(self, folder_id: str) -> bool:
        return bool(
            self._execute_write(
                lambda conn: conn.execute(
                    "DELETE FROM session_folders WHERE id = ?", (folder_id,)
                ).rowcount
            )
        )

    def add_sessions_to_folder(self, folder_id: str, session_ids: list[str]) -> int:
        now = time.time()

        def add(conn):
            unique_ids = list(dict.fromkeys(sid for sid in session_ids if sid))
            if not unique_ids:
                return 0
            if not conn.execute(
                "SELECT 1 FROM session_folders WHERE id = ?", (folder_id,)
            ).fetchone():
                raise sqlite3.IntegrityError("Folder not found")

            placeholders = ",".join("?" * len(unique_ids))
            existing = {
                row[0]
                for row in conn.execute(
                    f"SELECT id FROM sessions WHERE id IN ({placeholders})", unique_ids
                ).fetchall()
            }
            missing = [sid for sid in unique_ids if sid not in existing]
            if missing:
                raise ValueError(f"session not found: {missing[0]}")

            return sum(
                1
                for session_id in unique_ids
                if conn.execute(
                    "INSERT OR IGNORE INTO session_folder_members "
                    "(folder_id, session_id, added_at) VALUES (?, ?, ?)",
                    (folder_id, session_id, now),
                ).rowcount
            )

        return self._execute_write(add)

    def remove_sessions_from_folder(self, folder_id: str, session_ids: list[str]) -> int:
        if not session_ids:
            return 0
        placeholders = ",".join("?" * len(session_ids))
        return self._execute_write(
            lambda conn: conn.execute(
                "DELETE FROM session_folder_members "
                f"WHERE folder_id = ? AND session_id IN ({placeholders})",
                [folder_id, *session_ids],
            ).rowcount
        )

    def get_session_folder_map(self, session_ids: list[str]) -> dict[str, list[str]]:
        if not session_ids:
            return {}
        placeholders = ",".join("?" * len(session_ids))
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id, folder_id FROM session_folder_members "
                f"WHERE session_id IN ({placeholders})",
                session_ids,
            ).fetchall()

        result: dict[str, list[str]] = {}
        for row in rows:
            result.setdefault(row["session_id"], []).append(row["folder_id"])
        return result
