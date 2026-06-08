#!/usr/bin/env python3
"""
Memory Location Store for Hermes Agent.

Provides CRUD operations and search for memory locations (markers) tied to
session messages or persistent across sessions. Supports lineage-aware anchor
resolution for cases where messages have been compressed or parent sessions
split.
"""

import json
import logging
import sqlite3
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemoryLocationStore:
    """Manages memory locations (markers) in the Hermes session database."""

    def __init__(self, session_db):
        """
        Initialize the MemoryLocationStore.
        :param session_db: Instance of SessionDB from hermes_state
        """
        if session_db is None:
            raise ValueError("session_db must be provided to MemoryLocationStore")
        self.db = session_db

    @property
    def _conn(self) -> sqlite3.Connection:
        return self.db._conn

    def create(
        self,
        session_id: str,
        label: str,
        time_type: str = "point",
        anchor_guid: Optional[str] = None,
        anchor_end_guid: Optional[str] = None,
        color: Optional[str] = None,
        tags: Optional[List[str]] = None,
        sort_order: int = 0,
        is_persistent: bool = False,
        origin: str = "manual",
        recall_state: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Create a new memory location."""
        created_at = self._conn.execute("SELECT strftime('%s', 'now')").fetchone()[0]
        tags_json = json.dumps(tags) if tags else None
        recall_state_json = json.dumps(recall_state) if recall_state else None

        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT INTO memory_locations (
                session_id, label, time_type, anchor_guid, anchor_end_guid,
                color, tags, sort_order, is_persistent, origin, recall_state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                label,
                time_type,
                anchor_guid,
                anchor_end_guid,
                color,
                tags_json,
                sort_order,
                1 if is_persistent else 0,
                origin,
                recall_state_json,
                created_at,
            ),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get(self, location_id: int) -> Optional[Dict[str, Any]]:
        """Get a single memory location by ID."""
        cursor = self._conn.cursor()
        row = cursor.execute(
            "SELECT * FROM memory_locations WHERE id = ?", (location_id,)
        ).fetchone()
        if not row:
            return None

        location = dict(row)
        if location.get("tags"):
            try:
                location["tags"] = json.loads(location["tags"])
            except json.JSONDecodeError:
                location["tags"] = []
        if location.get("recall_state"):
            try:
                location["recall_state"] = json.loads(location["recall_state"])
            except json.JSONDecodeError:
                location["recall_state"] = None
        return location

    def list(
        self,
        session_id: Optional[str] = None,
        persistent_only: bool = False,
        origin: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List memory locations with optional filters."""
        cursor = self._conn.cursor()

        query = "SELECT * FROM memory_locations WHERE 1=1"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if persistent_only:
            query += " AND is_persistent = 1"
        if origin:
            query += " AND origin = ?"
            params.append(origin)

        query += " ORDER BY sort_order ASC, created_at ASC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        locations = []
        for row in rows:
            loc = dict(row)
            if loc.get("tags"):
                try:
                    loc["tags"] = json.loads(loc["tags"])
                except json.JSONDecodeError:
                    loc["tags"] = []
            if loc.get("recall_state"):
                try:
                    loc["recall_state"] = json.loads(loc["recall_state"])
                except json.JSONDecodeError:
                    loc["recall_state"] = None
            locations.append(loc)

        return locations

    def update(self, location_id: int, **fields) -> bool:
        """Update fields of a memory location."""
        allowed_fields = {
            "label",
            "time_type",
            "anchor_guid",
            "anchor_end_guid",
            "color",
            "tags",
            "sort_order",
            "is_persistent",
            "origin",
            "recall_state",
        }
        filtered_fields = {k: v for k, v in fields.items() if k in allowed_fields}
        if not filtered_fields:
            return False

        cursor = self._conn.cursor()

        if "tags" in filtered_fields:
            filtered_fields["tags"] = (
                json.dumps(filtered_fields["tags"]) if filtered_fields["tags"] else None
            )
        if "recall_state" in filtered_fields:
            filtered_fields["recall_state"] = (
                json.dumps(filtered_fields["recall_state"])
                if filtered_fields["recall_state"]
                else None
            )
        if "is_persistent" in filtered_fields:
            filtered_fields["is_persistent"] = (
                1 if filtered_fields["is_persistent"] else 0
            )

        set_clause = ", ".join(f"{k} = ?" for k in filtered_fields.keys())
        params = list(filtered_fields.values()) + [location_id]

        cursor.execute(
            f"UPDATE memory_locations SET {set_clause} WHERE id = ?", params
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def delete(self, location_id: int) -> bool:
        """Delete a memory location by ID."""
        cursor = self._conn.cursor()
        cursor.execute("DELETE FROM memory_locations WHERE id = ?", (location_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    def reorder(self, location_id: int, new_sort_order: int) -> bool:
        """Update the sort order of a memory location."""
        return self.update(location_id, sort_order=new_sort_order)

    def resolve_anchor(self, anchor_guid: str, session_id: str) -> Optional[int]:
        """
        Resolve an anchor GUID to a message ID, walking the parent_session_id
        chain if necessary (lineage-aware resolution).
        """
        cursor = self._conn.cursor()

        # 1. Try current session
        row = cursor.execute(
            "SELECT id FROM messages WHERE session_id = ? AND message_guid = ?",
            (session_id, anchor_guid),
        ).fetchone()
        if row:
            return row[0] if isinstance(row, tuple) else row["id"]

        # 2. Walk parent_session_id chain
        current = session_id
        while current:
            parent_row = cursor.execute(
                "SELECT parent_session_id FROM sessions WHERE id = ?", (current,)
            ).fetchone()
            if not parent_row:
                break
            parent = parent_row[0] if isinstance(parent_row, tuple) else parent_row["parent_session_id"]
            if not parent:
                break
            current = parent
            row = cursor.execute(
                "SELECT id FROM messages WHERE session_id = ? AND message_guid = ?",
                (current, anchor_guid),
            ).fetchone()
            if row:
                return row[0] if isinstance(row, tuple) else row["id"]

        return None

    def search_fts(self, query: str, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search memory locations using FTS5.
        """
        cursor = self._conn.cursor()

        base_query = """
            SELECT ml.*, snippet(memory_locations_fts, 0, '[', ']', '...', 10) as label_snippet
            FROM memory_locations ml
            JOIN memory_locations_fts fts ON ml.id = fts.rowid
            WHERE memory_locations_fts MATCH ?
        """
        params = [query]

        if session_id:
            base_query += " AND ml.session_id = ?"
            params.append(session_id)

        base_query += " ORDER BY ml.sort_order ASC, ml.created_at ASC"

        cursor.execute(base_query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            loc = dict(row)
            if loc.get("tags"):
                try:
                    loc["tags"] = json.loads(loc["tags"])
                except json.JSONDecodeError:
                    loc["tags"] = []
            if loc.get("recall_state"):
                try:
                    loc["recall_state"] = json.loads(loc["recall_state"])
                except json.JSONDecodeError:
                    loc["recall_state"] = None
            results.append(loc)

        return results
