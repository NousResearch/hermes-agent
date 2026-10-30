"""Durable scheduler state: ``task_state`` (registration spec + breaker) and ``task_runs``.

Both tables live in the per-profile ``state.db``. Writes are best-effort observers: the
scheduler logs and never raises persistence failures into a turn.

Design: website/docs/developer-guide/scheduled-tasks.md.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("hermes_state")


class SessionTasksMixin:
    def get_task_state(self, plugin_id: str, task_name: str) -> Optional[dict[str, Any]]:
        row = self._read_one(
            "SELECT * FROM task_state WHERE plugin_id = ? AND task_name = ?",
            (plugin_id, task_name),
        )
        return dict(row) if row is not None else None

    def save_task_state(
        self, plugin_id: str, task_name: str, *, period: str, every: int, phase: str,
        spec_hash: str, next_due_hint: Optional[int] = None, enabled: bool = True,
        disabled_by: Optional[str] = None, reset: bool = False, last_status: Optional[str] = None,
    ) -> None:
        """Upsert a task's registration state.

        ``reset`` re-bases the cadence and clears a breaker disable (an admin disable survives);
        it never clears counters on an unchanged spec.
        """
        now = time.time()

        def _do(conn):
            existing = conn.execute(
                "SELECT error_streak, run_count, error_count, disabled_by FROM task_state"
                " WHERE plugin_id = ? AND task_name = ?",
                (plugin_id, task_name),
            ).fetchone()
            if existing is None:
                conn.execute(
                    "INSERT INTO task_state (plugin_id, task_name, period, every, phase, spec_hash,"
                    " enabled, disabled_by, next_due_hint, updated_at, last_status)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (plugin_id, task_name, period, int(every), phase, spec_hash,
                     1 if enabled else 0, disabled_by, next_due_hint, now, last_status),
                )
                return
            if reset:
                keep_disabled = existing["disabled_by"] == "admin"
                enabled_value = False if keep_disabled else enabled
                disabled_value = "admin" if keep_disabled else disabled_by
                conn.execute(
                    "UPDATE task_state SET period = ?, every = ?, phase = ?, spec_hash = ?,"
                    " enabled = ?, disabled_by = ?, error_streak = 0, next_due_hint = ?,"
                    " updated_at = ?, last_status = ? WHERE plugin_id = ? AND task_name = ?",
                    (period, int(every), phase, spec_hash, 1 if enabled_value else 0, disabled_value,
                     next_due_hint, now, last_status, plugin_id, task_name),
                )
                return
            conn.execute(
                "UPDATE task_state SET period = ?, every = ?, phase = ?, spec_hash = ?,"
                " updated_at = ? WHERE plugin_id = ? AND task_name = ?",
                (period, int(every), phase, spec_hash, now, plugin_id, task_name),
            )

        self._execute_write(_do)

    def set_task_enabled(
        self, plugin_id: str, task_name: str, *, enabled: bool, disabled_by: Optional[str] = None,
    ) -> None:
        self._execute_write(lambda conn: conn.execute(
            "UPDATE task_state SET enabled = ?, disabled_by = ?, updated_at = ?"
            " WHERE plugin_id = ? AND task_name = ?",
            (1 if enabled else 0, None if enabled else (disabled_by or "admin"), time.time(),
             plugin_id, task_name),
        ))

    def record_task_run(
        self, *, plugin_id: str, task_name: str, period: str, phase: str, status: str,
        owner_pid: str = "", session_id: Optional[str] = None, turn_id: Optional[str] = None,
        turn_seq: Optional[int] = None, step_seq: Optional[int] = None,
        step_index: Optional[int] = None, profile_name: str = "", agent_name: str = "",
        started_at: Optional[float] = None, ended_at: Optional[float] = None,
        duration_ms: int = 0, error: Optional[str] = None, patience_s: float = 0.5,
    ) -> None:
        end = time.time() if ended_at is None else float(ended_at)
        start = (end - max(0, int(duration_ms)) / 1000.0) if started_at is None else float(started_at)

        def _do(conn):
            conn.execute(
                "INSERT INTO task_runs (plugin_id, task_name, period, phase, owner_pid, session_id,"
                " turn_id, turn_seq, step_seq, step_index, profile_name, agent_name, started_at,"
                " ended_at, duration_ms, status, error)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (plugin_id, task_name, period, phase, owner_pid or "", session_id, turn_id,
                 turn_seq, step_seq, step_index, profile_name or "", agent_name or "",
                 start, end, max(0, int(duration_ms)), status, error),
            )
            if status not in ("rescheduled", "disabled"):
                conn.execute(
                    "UPDATE task_state SET run_count = run_count + 1,"
                    " error_count = error_count + (CASE WHEN ? IN ('error','timeout') THEN 1 ELSE 0 END),"
                    " last_fired_at = ?, last_status = ?, updated_at = ?"
                    " WHERE plugin_id = ? AND task_name = ?",
                    (status, end, status, end, plugin_id, task_name),
                )

        self._execute_write(_do, patience_s=patience_s)

    def task_runs(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._read_all(
            "SELECT * FROM task_runs ORDER BY id DESC LIMIT ?", (max(0, int(limit)),),
        )
        return [dict(r) for r in rows]
