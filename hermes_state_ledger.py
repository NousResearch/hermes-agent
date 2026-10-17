"""Bounded decision events; text is evidence, never an executable approval gate."""

import logging
import time

logger = logging.getLogger(__name__)

MAX_LEDGER_ENTRIES = 50
MAX_LEDGER_TEXT_BYTES = 2048
MAX_LEDGER_TURN_BYTES = 128
LEDGER_PRIORITY = {"denial": 0, "correction": 1, "approval": 2, "preference": 3}


def omitted_decision(kind: str, reason: str) -> str:
    return f"[Text omitted: {reason}; {kind} remains recorded. Do not infer permission; ask the user to restate scope.]"


def bounded_decision_text(kind: str, text: str, *, limit: int = MAX_LEDGER_TEXT_BYTES) -> str:
    # Reject the WHOLE text rather than cut away a trailing negation or scope.
    # Check before redaction to bound the redactor's work on arbitrary input.
    if len(text) > limit or len(text.encode("utf-8")) > limit:
        return omitted_decision(kind, "size limit")
    from agent.redact import redact_for_egress, redact_sensitive_text
    try:
        text = redact_sensitive_text(text, force=True, redact_url_credentials=True)
        text = redact_for_egress(text)
    except Exception:
        logger.debug("Decision ledger redaction unavailable", exc_info=True)
        return omitted_decision(kind, "redaction unavailable")
    if len(text.encode("utf-8")) > limit:
        return omitted_decision(kind, "size limit after redaction")
    return text


def bounded_ledger_turn_id(turn_id: str) -> str:
    text = bounded_decision_text("decision", turn_id or "", limit=MAX_LEDGER_TURN_BYTES)
    return text if len(text.encode("utf-8")) <= MAX_LEDGER_TURN_BYTES else "[turn id omitted]"


def _sanitize_ledger(conn, session_id: str) -> None:
    rows = conn.execute("SELECT id, kind, text, turn_id FROM decision_ledger WHERE session_id = ?", (session_id,)).fetchall()
    for row in rows:
        text = bounded_decision_text(row["kind"], row["text"])
        turn_id = bounded_ledger_turn_id(row["turn_id"])
        if text != row["text"] or turn_id != row["turn_id"]:
            conn.execute("UPDATE decision_ledger SET text = ?, turn_id = ? WHERE id = ?", (text, turn_id, row["id"]))
    # Within a kind keep the newest window; restore chronological order on read.
    conn.execute(
        """DELETE FROM decision_ledger WHERE session_id = ? AND id NOT IN (
            SELECT id FROM decision_ledger WHERE session_id = ?
            ORDER BY CASE kind WHEN 'denial' THEN 0 WHEN 'correction' THEN 1
                WHEN 'approval' THEN 2 ELSE 3 END, created_at DESC, turn_id DESC, text DESC, id DESC LIMIT ?)""",
        (session_id, session_id, MAX_LEDGER_ENTRIES),
    )


class SessionDecisionLedgerMixin:
    """Session-scoped capture, legacy sanitization and idempotent child copying."""

    def append_decision_ledger_entry(self, session_id: str, kind: str, text: str, *, turn_id: str = "") -> None:
        if kind not in LEDGER_PRIORITY:
            raise ValueError(f"Unsupported decision ledger kind: {kind!r}")
        if not isinstance(text, str) or not text:
            return
        text = bounded_decision_text(kind, text)
        turn_id = bounded_ledger_turn_id(turn_id)

        def _append(conn):
            conn.execute(
                "INSERT INTO decision_ledger (session_id, turn_id, kind, text, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, turn_id, kind, text, time.time()),
            )
            _sanitize_ledger(conn, session_id)
        self._execute_write(_append)

    def get_decision_ledger_entries(self, session_id: str) -> list[dict[str, str]]:
        def _read(conn):
            _sanitize_ledger(conn, session_id)
            rows = conn.execute(
                "SELECT turn_id, kind, text FROM decision_ledger WHERE session_id = ? ORDER BY created_at ASC, id ASC", (session_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        return self._execute_write(_read)

    def copy_decision_ledger_entries(self, parent_session_id: str, child_session_id: str) -> None:
        def _copy(conn):
            _sanitize_ledger(conn, parent_session_id)
            _sanitize_ledger(conn, child_session_id)
            conn.execute(
                """INSERT INTO decision_ledger (session_id, turn_id, kind, text, created_at)
                   SELECT ?, parent.turn_id, parent.kind, parent.text, parent.created_at
                   FROM decision_ledger AS parent
                   WHERE parent.session_id = ?
                     AND NOT EXISTS (
                         SELECT 1 FROM decision_ledger AS child
                         WHERE child.session_id = ?
                           AND child.turn_id = parent.turn_id
                           AND child.kind = parent.kind
                           AND child.text = parent.text
                           AND child.created_at = parent.created_at
                     )
                   ORDER BY parent.id ASC""",
                (child_session_id, parent_session_id, child_session_id),
            )
            _sanitize_ledger(conn, child_session_id)
        self._execute_write(_copy)
