"""SQLite registry: the source of truth for who exists and how to reach them.

Maps the single Mattermost bot's inbound `user_id` -> internal employee/tenant.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from .models import Employee


class Registry:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS employees (
                id           TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                mm_user_id   TEXT NOT NULL UNIQUE,
                created_at   REAL NOT NULL
            )
            """
        )
        self._db.commit()

    def add(self, employee_id: str, display_name: str, mm_user_id: str) -> Employee:
        if not Employee.valid_id(employee_id):
            raise ValueError(
                f"invalid employee id {employee_id!r} (need ^[a-z0-9][a-z0-9_-]{{0,63}})"
            )
        emp = Employee(employee_id, display_name, mm_user_id, time.time())
        try:
            self._db.execute(
                "INSERT INTO employees VALUES (?,?,?,?)",
                (emp.id, emp.display_name, emp.mm_user_id, emp.created_at),
            )
            self._db.commit()
        except sqlite3.IntegrityError as e:
            raise ValueError(f"employee id or mm_user_id already registered: {e}") from e
        return emp

    def _row_to_emp(self, row: sqlite3.Row) -> Employee:
        return Employee(row["id"], row["display_name"], row["mm_user_id"], row["created_at"])

    def get(self, employee_id: str) -> Employee | None:
        row = self._db.execute(
            "SELECT * FROM employees WHERE id=?", (employee_id,)
        ).fetchone()
        return self._row_to_emp(row) if row else None

    def by_mm_user(self, mm_user_id: str) -> Employee | None:
        row = self._db.execute(
            "SELECT * FROM employees WHERE mm_user_id=?", (mm_user_id,)
        ).fetchone()
        return self._row_to_emp(row) if row else None

    def all(self) -> list[Employee]:
        return [self._row_to_emp(r) for r in self._db.execute(
            "SELECT * FROM employees ORDER BY created_at"
        ).fetchall()]

    def remove(self, employee_id: str) -> bool:
        cur = self._db.execute("DELETE FROM employees WHERE id=?", (employee_id,))
        self._db.commit()
        return cur.rowcount > 0

    def close(self) -> None:
        self._db.close()
