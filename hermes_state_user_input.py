"""Durable, session-scoped asynchronous user-input requests.

This module is deliberately a small SessionDB mixin. Requests are durable state,
not transcript messages: an answer is delivered to the active agent through its
role-safe steering API, so no out-of-band writer can break provider turn order.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any, Dict, List, Optional


_USER_INPUT_STATUSES = frozenset({"pending", "answered", "expired", "cancelled"})


def _json_load(raw: Any, default: Any = None) -> Any:
    if not isinstance(raw, str) or not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


def _json_dump(value: Any) -> str:
    # ensure_ascii keeps lone surrogate input bindable on every supported SQLite build.
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


def _clean_text(value: Any, *, max_chars: int, default: str = "") -> str:
    text = str(value if value is not None else default).strip()
    return text[:max_chars]


def _question_default(question: Any, index: int) -> Any:
    if not isinstance(question, dict):
        return ""
    return question.get("default", "")


def _default_answers(questions: Any) -> Dict[str, Any]:
    if not isinstance(questions, list):
        return {}
    answers: Dict[str, Any] = {}
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        question_id = _clean_text(question.get("id") or f"q{index}", max_chars=128)
        if question_id:
            answers[question_id] = _question_default(question, index)
    return answers


class SessionUserInputMixin:
    """Pending structured user-input records stored in the canonical state DB."""

    _USER_INPUT_MAX_REQUEST_ID = 256
    _USER_INPUT_MAX_SESSION_ID = 256
    _USER_INPUT_MAX_TURN_ID = 256
    _USER_INPUT_MAX_CONTEXT = 4096

    @staticmethod
    def _user_input_record(row) -> Dict[str, Any]:
        """Decode one row into the stable public record shape."""
        questions = _json_load(row["questions"], [])
        answer = _json_load(row["answer"], None)
        return {
            "request_id": str(row["request_id"] or ""),
            "session_id": str(row["session_id"] or ""),
            "turn_id": str(row["turn_id"] or ""),
            "questions": questions if isinstance(questions, list) else [],
            "context": str(row["context"] or ""),
            "status": str(row["status"] or "pending"),
            "answer": answer,
            "created_at": float(row["created_at"] or 0),
            "expires_at": float(row["expires_at"] or 0),
            "answered_at": float(row["answered_at"]) if row["answered_at"] is not None else None,
        }

    @staticmethod
    def _user_input_status_record(row, *, accepted: Optional[bool] = None) -> Dict[str, Any]:
        record = SessionUserInputMixin._user_input_record(row)
        if accepted is not None:
            record["accepted"] = bool(accepted)
        return record

    @staticmethod
    def _settle_expired_row(conn, row, *, now: float):
        """Settle a pending expired row inside the caller's write transaction."""
        if row is None or str(row["status"] or "") != "pending":
            return row
        expires_at = float(row["expires_at"] or 0)
        if expires_at <= 0 or expires_at > now:
            return row
        defaults = _default_answers(_json_load(row["questions"], []))
        conn.execute(
            """UPDATE pending_user_inputs
               SET status = 'expired', answer = ?, answered_at = ?
             WHERE request_id = ? AND session_id = ? AND status = 'pending'""",
            (_json_dump(defaults), now, row["request_id"], row["session_id"]),
        )
        return conn.execute(
            "SELECT * FROM pending_user_inputs WHERE request_id = ? AND session_id = ?",
            (row["request_id"], row["session_id"]),
        ).fetchone()

    def create_pending_user_input(
        self,
        request_id: str,
        session_id: str,
        questions: List[Dict[str, Any]],
        context: str = "",
        expires_at: Optional[float] = None,
        *,
        turn_id: str = "",
        session_key: str = "",
        source: str = "",
        user_id: str = "",
        chat_id: str = "",
        thread_id: str = "",
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Insert one pending request and return its canonical record.

        Reusing an existing request id is idempotent only for the owning session;
        a collision from another session raises instead of exposing that record.
        """
        request_id = _clean_text(request_id, max_chars=self._USER_INPUT_MAX_REQUEST_ID)
        session_id = _clean_text(session_id, max_chars=self._USER_INPUT_MAX_SESSION_ID)
        turn_id = _clean_text(turn_id, max_chars=self._USER_INPUT_MAX_TURN_ID)
        if not request_id or not session_id:
            raise ValueError("request_id and session_id are required")
        if not isinstance(questions, list) or not questions:
            raise ValueError("questions must be a non-empty list")
        now_value = time.time() if now is None else float(now)
        expiry = now_value + 3600.0 if expires_at is None else float(expires_at)
        if not math.isfinite(expiry):
            raise ValueError("expires_at must be finite")
        context = _clean_text(context, max_chars=self._USER_INPUT_MAX_CONTEXT)
        params = (
            request_id, session_id, turn_id,
            _clean_text(session_key, max_chars=512), _clean_text(source, max_chars=64),
            _clean_text(user_id, max_chars=256), _clean_text(chat_id, max_chars=256),
            _clean_text(thread_id, max_chars=256), _json_dump(questions), context,
            "pending", None, now_value, expiry, None,
        )

        def _do(conn):
            conn.execute(
                """INSERT OR IGNORE INTO pending_user_inputs (
                    request_id, session_id, turn_id, session_key, source, user_id,
                    chat_id, thread_id, questions, context, status, answer,
                    created_at, expires_at, answered_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                params,
            )
            row = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE request_id = ?", (request_id,)
            ).fetchone()
            if row is None:
                raise RuntimeError("pending user-input request was not persisted")
            if str(row["session_id"] or "") != session_id:
                raise ValueError("request_id is already owned by another session")
            return self._user_input_record(row)

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def get_pending_user_input(
        self, request_id: str, *, session_id: str, now: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Read one request only when it belongs to *session_id*.

        Expiry is a durable state transition, so an expired row is never merely
        reported as pending because a read happened before a later answer.
        """
        request_id = _clean_text(request_id, max_chars=self._USER_INPUT_MAX_REQUEST_ID)
        session_id = _clean_text(session_id, max_chars=self._USER_INPUT_MAX_SESSION_ID)
        if not request_id or not session_id:
            return None
        now_value = time.time() if now is None else float(now)

        def _settle(conn):
            row = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE request_id = ? AND session_id = ?",
                (request_id, session_id),
            ).fetchone()
            return self._settle_expired_row(conn, row, now=now_value)

        row = self._execute_write(_settle, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)
        return self._user_input_record(row) if row is not None else None

    def list_pending_user_inputs(
        self, session_id: str, *, now: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Return pending requests for exactly one session, oldest first."""
        session_id = _clean_text(session_id, max_chars=self._USER_INPUT_MAX_SESSION_ID)
        if not session_id:
            return []
        now_value = time.time() if now is None else float(now)

        def _settle(conn):
            rows = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE session_id = ? AND status = 'pending'",
                (session_id,),
            ).fetchall()
            for row in rows:
                self._settle_expired_row(conn, row, now=now_value)

        self._execute_write(_settle, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)
        rows = self._read_all(
            """SELECT * FROM pending_user_inputs
               WHERE session_id = ? AND status = 'pending'
               ORDER BY created_at, request_id""",
            (session_id,),
        )
        return [self._user_input_record(row) for row in rows]

    def answer_pending_user_input(
        self,
        request_id: str,
        answer: Dict[str, Any],
        *,
        session_id: str,
        turn_id: Optional[str] = None,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        """CAS-set one answer; exactly one concurrent caller receives ``accepted``.

        A request from another session is indistinguishable from a missing request.
        This is intentional: callers must not be able to probe request ids across
        sessions or use an answer to steer another live agent.
        """
        request_id = _clean_text(request_id, max_chars=self._USER_INPUT_MAX_REQUEST_ID)
        session_id = _clean_text(session_id, max_chars=self._USER_INPUT_MAX_SESSION_ID)
        if not request_id or not session_id:
            return {"status": "not_found", "accepted": False}
        if not isinstance(answer, dict):
            return {"status": "invalid", "accepted": False, "error": "answer must be an object"}
        now_value = time.time() if now is None else float(now)
        expected_turn_id = _clean_text(turn_id, max_chars=self._USER_INPUT_MAX_TURN_ID) if turn_id is not None else None

        def _do(conn):
            row = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE request_id = ? AND session_id = ?",
                (request_id, session_id),
            ).fetchone()
            if row is None:
                return {"status": "not_found", "accepted": False}
            stored_turn_id = str(row["turn_id"] or "")
            if expected_turn_id is not None and stored_turn_id and expected_turn_id != stored_turn_id:
                return {"status": "not_found", "accepted": False}
            row = self._settle_expired_row(conn, row, now=now_value)
            if row is None:
                return {"status": "not_found", "accepted": False}
            if str(row["status"] or "") != "pending":
                return self._user_input_status_record(row, accepted=False)
            cur = conn.execute(
                """UPDATE pending_user_inputs
                   SET status = 'answered', answer = ?, answered_at = ?
                 WHERE request_id = ? AND session_id = ? AND status = 'pending'""",
                (_json_dump(answer), now_value, request_id, session_id),
            )
            accepted = int(cur.rowcount or 0) == 1
            final_row = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE request_id = ? AND session_id = ?",
                (request_id, session_id),
            ).fetchone()
            return self._user_input_status_record(final_row, accepted=accepted)

        return self._execute_write(_do, patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S)

    def expire_pending_user_input(
        self, request_id: str, *, session_id: str, now: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Explicitly settle one request as expired, preserving question defaults."""
        request_id = _clean_text(request_id, max_chars=self._USER_INPUT_MAX_REQUEST_ID)
        session_id = _clean_text(session_id, max_chars=self._USER_INPUT_MAX_SESSION_ID)
        if not request_id or not session_id:
            return None
        now_value = time.time() if now is None else float(now)

        def _do(conn):
            row = conn.execute(
                "SELECT * FROM pending_user_inputs WHERE request_id = ? AND session_id = ?",
                (request_id, session_id),
            ).fetchone()
            row = self._settle_expired_row(conn, row, now=now_value)
            return self._user_input_record(row) if row is not None else None

        return self._execute_write(_do, patience_s=self._ACTIVITY_WRITE_PATIENCE_S)


__all__ = ["SessionUserInputMixin"]
