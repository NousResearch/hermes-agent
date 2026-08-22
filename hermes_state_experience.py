"""``ExperienceStoreMixin`` — durable Level 2 experience records on ``SessionDB``.

The ``experiences`` table is declared in :data:`hermes_state_common.SCHEMA_SQL`,
so it is created by the ordinary ``executescript`` on every DB open — fresh and
pre-existing databases alike, with no version-gated migration. See
``hermes_state_schema._reconcile_columns`` for the same declarative contract
applied to columns.

Retrieval deliberately uses a bounded prefilter + Python scoring rather than a
new FTS5 index:

* the store is pruned to a hard row cap, so the candidate window is small;
* ranking must blend lexical overlap with recency, confidence and correction
  count, which bm25 alone cannot express — an FTS hit list would be
  re-scored in Python anyway;
* it works identically for CJK and Vietnamese, where the default unicode61
  tokenizer does not.

``ponytail``: O(n) scan over the prefilter window. If the store ever needs to
grow past ~10k live rows, add an FTS5 shadow table via
``_ensure_fts_schema`` and use it as the prefilter — the scoring pass above it
does not change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Hard cap on live rows. Prune drops the least useful rows past this — oldest
# first among low-confidence, low-observation records.
MAX_LIVE_EXPERIENCES = 2000

# Candidate window pulled for scoring. Kept well under the row cap so a full
# store still costs one small indexed scan per turn.
RETRIEVAL_WINDOW = 400


_SELECT_COLS = (
    "id, session_id, turn_id, created_at, updated_at, task, task_norm, "
    "task_hash, strategy, tools, outcome, exit_reason, failure_reason, "
    "recovery, user_correction, metrics, confidence, observations, "
    "success_count, failure_count, correction_count, model, cwd, workspace, "
    "verification, superseded"
)


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, sqlite3.Row):
        return {k: row[k] for k in row.keys()}
    return dict(row)


def _confidence(success: int, failure: int, corrections: int) -> float:
    """Laplace-smoothed success rate, discounted by user corrections.

    Smoothing keeps a single observation from claiming certainty (1 success →
    0.67, not 1.0), which is what stops one lucky turn from outranking a
    well-evidenced record.
    """
    base = (success + 1.0) / (success + failure + 2.0)
    return round(max(0.05, base * (1.0 / (1.0 + 0.5 * max(0, corrections)))), 4)


class ExperienceStoreMixin:
    """Experience read/write surface. Mixed into ``SessionDB``."""

    # -- Write ---------------------------------------------------------------

    def record_experience(self, exp_row: Dict[str, Any]) -> Optional[str]:
        """Insert *exp_row*, or merge it into the matching prior observation.

        Dedup key is ``(task_hash, cwd)``: the same request in the same
        working directory is the same task. A merge bumps the observation and
        outcome counters, refreshes the freshest strategy/outcome, recomputes
        confidence, and clears ``superseded`` — a task attempted again is live
        again.

        Returns the row id (new or merged), or ``None`` if the write failed.
        """
        required = ("id", "task", "task_hash", "outcome")
        if not all(exp_row.get(k) for k in required):
            return None
        if self.read_only:
            return None

        now = time.time()
        outcome = str(exp_row["outcome"])
        verification = str(exp_row.get("verification") or "")
        # A ``partial`` turn hit tool errors but still completed. Two things
        # redeem it, and either one means the agent reached a working state:
        # a recovery (it found the path itself), or build/test evidence that
        # passed. An unrecovered, unverified partial counts against the
        # approach.
        #
        # ``verification == "passed"`` deliberately does NOT rescue an outright
        # ``failure``: the turn did not complete, and stale evidence from
        # earlier in the session is not proof that this attempt worked.
        redeemed = (
            bool(str(exp_row.get("recovery") or "").strip())
            or verification == "passed"
        )
        is_success = 1 if (outcome == "success" or (outcome == "partial" and redeemed)) else 0
        is_failure = 1 if (outcome == "failure" or (outcome == "partial" and not redeemed)) else 0
        cwd = str(exp_row.get("cwd") or "")
        workspace = str(exp_row.get("workspace") or "") or cwd

        def _do(conn: sqlite3.Connection) -> Optional[str]:
            existing = conn.execute(
                "SELECT id, success_count, failure_count, correction_count "
                "FROM experiences WHERE task_hash = ? AND workspace = ?",
                (exp_row["task_hash"], workspace),
            ).fetchone()

            if existing is None:
                conn.execute(
                    """INSERT INTO experiences (
                        id, session_id, turn_id, created_at, updated_at, task,
                        task_norm, task_hash, strategy, tools, outcome,
                        exit_reason, failure_reason, recovery, user_correction,
                        metrics, confidence, observations, success_count,
                        failure_count, correction_count, model, cwd, workspace,
                        verification, superseded, schema_rev
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        exp_row["id"],
                        exp_row.get("session_id") or "",
                        exp_row.get("turn_id") or "",
                        float(exp_row.get("created_at") or now),
                        now,
                        exp_row["task"],
                        exp_row.get("task_norm") or "",
                        exp_row["task_hash"],
                        exp_row.get("strategy") or "",
                        exp_row.get("tools") or "[]",
                        outcome,
                        exp_row.get("exit_reason") or "",
                        exp_row.get("failure_reason") or "",
                        exp_row.get("recovery") or "",
                        exp_row.get("user_correction") or "",
                        exp_row.get("metrics") or "{}",
                        _confidence(is_success, is_failure, 0),
                        1,
                        is_success,
                        is_failure,
                        0,
                        exp_row.get("model") or "",
                        cwd,
                        workspace,
                        verification,
                        0,
                        1,
                    ),
                )
                return str(exp_row["id"])

            row = _row_to_dict(existing)
            success = int(row["success_count"]) + is_success
            failure = int(row["failure_count"]) + is_failure
            conn.execute(
                """UPDATE experiences SET
                       updated_at = ?, session_id = ?, turn_id = ?,
                       task = ?, task_norm = ?, strategy = ?, tools = ?,
                       outcome = ?, exit_reason = ?, failure_reason = ?,
                       recovery = ?, metrics = ?, model = ?, cwd = ?,
                       verification = ?,
                       observations = observations + 1,
                       success_count = ?, failure_count = ?,
                       confidence = ?, superseded = 0
                   WHERE id = ?""",
                (
                    now,
                    exp_row.get("session_id") or "",
                    exp_row.get("turn_id") or "",
                    exp_row["task"],
                    exp_row.get("task_norm") or "",
                    exp_row.get("strategy") or "",
                    exp_row.get("tools") or "[]",
                    outcome,
                    exp_row.get("exit_reason") or "",
                    exp_row.get("failure_reason") or "",
                    exp_row.get("recovery") or "",
                    exp_row.get("metrics") or "{}",
                    exp_row.get("model") or "",
                    cwd,
                    verification,
                    success,
                    failure,
                    _confidence(success, failure, int(row["correction_count"])),
                    row["id"],
                ),
            )
            return str(row["id"])

        try:
            return self._execute_write(_do)
        except Exception:
            logger.warning("record_experience failed", exc_info=True)
            return None

    def record_experience_correction(
        self, experience_id: str, correction_text: str
    ) -> bool:
        """Attach a user correction to a stored experience.

        Corrections are the strongest negative signal available: they lower
        confidence directly, and a *corrected success* is marked ``superseded``
        so it stops being retrieved at all. A corrected failure stays live —
        the fact that the path failed is still true and still worth surfacing.
        """
        if not experience_id or self.read_only:
            return False

        def _do(conn: sqlite3.Connection) -> bool:
            row = conn.execute(
                "SELECT success_count, failure_count, correction_count, outcome "
                "FROM experiences WHERE id = ?",
                (experience_id,),
            ).fetchone()
            if row is None:
                return False
            data = _row_to_dict(row)
            corrections = int(data["correction_count"]) + 1
            conn.execute(
                """UPDATE experiences SET
                       correction_count = ?,
                       user_correction = ?,
                       confidence = ?,
                       superseded = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    corrections,
                    (correction_text or "")[:240],
                    _confidence(
                        int(data["success_count"]),
                        int(data["failure_count"]),
                        corrections,
                    ),
                    1 if str(data["outcome"]) == "success" else 0,
                    time.time(),
                    experience_id,
                ),
            )
            return True

        try:
            return bool(self._execute_write(_do))
        except Exception:
            logger.warning("record_experience_correction failed", exc_info=True)
            return False

    # -- Read ----------------------------------------------------------------

    def fetch_experience_candidates(
        self,
        *,
        workspace: str = "",
        limit: int = RETRIEVAL_WINDOW,
        max_age_days: float = 90.0,
    ) -> List[Dict[str, Any]]:
        """Live rows for the scoring pass, newest first.

        When *workspace* is given, same-project rows are preferred but not
        required: cross-project experience about a tool or an error class is
        still useful, so the filter is a sort key, never a WHERE clause.
        """
        cutoff = time.time() - max(1.0, float(max_age_days)) * 86400.0
        try:
            with self._read_ctx() as conn:
                rows = conn.execute(
                    f"""SELECT {_SELECT_COLS} FROM experiences
                        WHERE superseded = 0 AND updated_at >= ?
                        ORDER BY (workspace = ?) DESC, updated_at DESC
                        LIMIT ?""",
                    (cutoff, str(workspace or ""), int(limit)),
                ).fetchall()
        except Exception:
            logger.debug("fetch_experience_candidates failed", exc_info=True)
            return []
        return [_row_to_dict(r) for r in rows]

    def get_experience(self, experience_id: str) -> Optional[Dict[str, Any]]:
        if not experience_id:
            return None
        try:
            with self._read_ctx() as conn:
                row = conn.execute(
                    f"SELECT {_SELECT_COLS} FROM experiences WHERE id = ?",
                    (experience_id,),
                ).fetchone()
        except Exception:
            return None
        return _row_to_dict(row) if row is not None else None

    def latest_experience_for_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The most recently written experience for *session_id*.

        Used by the correction hook: when the next user message reads as a
        correction, the experience it corrects is the one this session just
        wrote. Looked up from the DB rather than held on the agent because the
        gateway can rebuild (or evict) the ``AIAgent`` between turns.
        """
        if not session_id:
            return None
        try:
            with self._read_ctx() as conn:
                row = conn.execute(
                    f"""SELECT {_SELECT_COLS} FROM experiences
                        WHERE session_id = ?
                        ORDER BY updated_at DESC LIMIT 1""",
                    (str(session_id),),
                ).fetchone()
        except Exception:
            return None
        return _row_to_dict(row) if row is not None else None

    def experience_stats(self) -> Dict[str, Any]:
        """Aggregate counters — the measurable half of "did it learn?"."""
        empty = {
            "total": 0, "live": 0, "success": 0, "partial": 0, "failure": 0,
            "interrupted": 0, "corrected": 0, "recovered": 0,
            "verified_pass": 0, "verified_fail": 0, "unverified": 0,
            "observations": 0, "avg_confidence": 0.0,
        }
        try:
            with self._read_ctx() as conn:
                row = conn.execute(
                    """SELECT
                        COUNT(*) AS total,
                        SUM(superseded = 0) AS live,
                        SUM(outcome = 'success') AS success,
                        SUM(outcome = 'partial') AS partial,
                        SUM(outcome = 'failure') AS failure,
                        SUM(outcome = 'interrupted') AS interrupted,
                        SUM(correction_count > 0) AS corrected,
                        SUM(recovery IS NOT NULL AND recovery != '') AS recovered,
                        SUM(verification = 'passed') AS verified_pass,
                        SUM(verification = 'failed') AS verified_fail,
                        SUM(COALESCE(verification, '') NOT IN
                            ('passed', 'failed')) AS unverified,
                        SUM(observations) AS observations,
                        AVG(confidence) AS avg_confidence
                       FROM experiences"""
                ).fetchone()
        except Exception:
            return empty
        if row is None:
            return empty
        data = _row_to_dict(row)
        out = {k: int(data.get(k) or 0) for k in empty if k != "avg_confidence"}
        out["avg_confidence"] = round(float(data.get("avg_confidence") or 0.0), 4)
        return out

    # -- Maintenance ---------------------------------------------------------

    def prune_experiences(
        self, *, max_rows: int = MAX_LIVE_EXPERIENCES, max_age_days: float = 365.0
    ) -> int:
        """Drop expired and surplus rows. Returns the number deleted.

        Retention order keeps the evidence that matters: rows are ranked by
        observation count, then confidence distance from 0.5 (a confident
        success *or* a confident failure is informative; a coin-flip row is
        not), then recency.
        """
        if self.read_only:
            return 0
        cutoff = time.time() - max(1.0, float(max_age_days)) * 86400.0

        def _do(conn: sqlite3.Connection) -> int:
            deleted = conn.execute(
                "DELETE FROM experiences WHERE updated_at < ?", (cutoff,)
            ).rowcount or 0
            surplus = conn.execute(
                """DELETE FROM experiences WHERE id IN (
                       SELECT id FROM experiences
                       ORDER BY observations DESC,
                                ABS(confidence - 0.5) DESC,
                                updated_at DESC
                       LIMIT -1 OFFSET ?
                   )""",
                (int(max_rows),),
            ).rowcount or 0
            return deleted + surplus

        try:
            return int(self._execute_write(_do) or 0)
        except Exception:
            logger.warning("prune_experiences failed", exc_info=True)
            return 0

    def delete_experience(self, experience_id: str) -> bool:
        """Delete one experience outright. Returns whether a row went away.

        Distinct from ``superseded``, which only stops a row being retrieved
        while keeping it as evidence. This is the "forget it entirely" path a
        user needs when a stored task should never have been recorded — the
        row is gone, not hidden.
        """
        if not experience_id or self.read_only:
            return False

        def _do(conn: sqlite3.Connection) -> int:
            return conn.execute(
                "DELETE FROM experiences WHERE id = ?", (str(experience_id),)
            ).rowcount or 0

        try:
            return bool(self._execute_write(_do))
        except Exception:
            logger.warning("delete_experience failed", exc_info=True)
            return False

    def clear_experiences(self) -> int:
        """Delete every experience. Used by ``/forget``-style flows and tests."""
        if self.read_only:
            return 0

        def _do(conn: sqlite3.Connection) -> int:
            return conn.execute("DELETE FROM experiences").rowcount or 0

        try:
            return int(self._execute_write(_do) or 0)
        except Exception:
            return 0

    def export_experiences(self) -> List[Dict[str, Any]]:
        """All rows, decoded — for inspection, backup and benchmarking."""
        try:
            with self._read_ctx() as conn:
                rows = conn.execute(
                    f"SELECT {_SELECT_COLS} FROM experiences ORDER BY updated_at DESC"
                ).fetchall()
        except Exception:
            return []
        out = []
        for r in rows:
            d = _row_to_dict(r)
            for key in ("tools", "metrics"):
                try:
                    d[key] = json.loads(d.get(key) or ("[]" if key == "tools" else "{}"))
                except Exception:
                    d[key] = [] if key == "tools" else {}
            out.append(d)
        return out
