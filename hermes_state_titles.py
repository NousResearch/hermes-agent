"""Session title mixin for SessionDB: sanitizing, auto/user provenance ranking, and
lineage-aware lookups."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from agent.message_sanitization import _sanitize_surrogates
from hermes_state_common import (
    _COMPRESSION_CHILD_SQL, _RECOVERABLE_END_REASONS, _RECOVERABLE_END_REASONS_SQL,
    escape_like as _escape_like,
)

# Title-as-identity lookup: live rows, plus recoverable auto-archives that
# unarchive_recoverable_session still has to find (#92687). Deliberate archives
# (archived=1 and end_reason not in that set, including NULL) are skipped so a
# retired hidden Bot Chat cannot occupy the name or reopen on click (#139779).
_TITLE_LOOKUP_VISIBLE_SQL = (
    f"(COALESCE({{alias}}.archived, 0) = 0 "
    f"OR {{alias}}.end_reason IN ({_RECOVERABLE_END_REASONS_SQL}))"
)

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

# ASCII controls (keeping \t \n \r for the whitespace collapse), then zero-width,
# bidi override, object-replacement and interlinear-annotation code points.
_TITLE_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_TITLE_INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]')
_NUMBERED_TITLE_RE = re.compile(r'^(.*?) #(\d+)$')


def next_title_in_lineage(conn, base_title: str) -> str:
    """Next title in a lineage ("my session" -> "my session #2") as seen by *conn*: strip any
    " #N" suffix, then increment the highest existing number."""
    match = _NUMBERED_TITLE_RE.match(base_title)
    base = match.group(1) if match else base_title
    rows = conn.execute(
        "SELECT title FROM sessions WHERE title = ? OR title LIKE ? ESCAPE '\\'",
        (base, f"{_escape_like(base)} #%")).fetchall()
    if not rows:
        return base
    # The unnumbered original counts as #1.
    numbers = [int(m.group(2)) for m in (_NUMBERED_TITLE_RE.match(row[0]) for row in rows) if m]
    return f"{base} #{max([1, *numbers]) + 1}"


class SessionTitlesMixin:
    """Sanitizing, ranking auto/user titles, lineage-aware lookups."""

    @classmethod
    def _title_rank(cls, source: Optional[str]) -> int:
        """Rank a stored title_source. NULL (pre-provenance rows) is indistinguishable from a
        manual ``/title`` of that era, so it ranks as ``user``."""
        rank = cls._TITLE_SOURCE_RANK
        return rank[cls.TITLE_SOURCE_USER] if source is None else rank.get(str(source), 0)

    @staticmethod
    def sanitize_title(title: Optional[str]) -> Optional[str]:
        """Strip control/zero-width/bidi chars (and lone surrogates sqlite3 cannot bind),
        collapse whitespace, normalize empty to None. ValueError past MAX_TITLE_LENGTH."""
        from hermes_state import SessionDB
        if not title:
            return None
        cleaned = _TITLE_INVISIBLE_RE.sub('', _TITLE_CONTROL_RE.sub('', _sanitize_surrogates(title)))
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        if not cleaned:
            return None
        if len(cleaned) > SessionDB.MAX_TITLE_LENGTH:
            raise ValueError(f"Title too long ({len(cleaned)} chars, max {SessionDB.MAX_TITLE_LENGTH})")
        return cleaned

    def _is_compression_ancestor(self, conn, *, ancestor_id: str, descendant_id: str) -> bool:
        """True if *ancestor_id* is a compression predecessor of *descendant_id*, via the
        canonical continuation edge ``_COMPRESSION_CHILD_SQL`` (excludes delegate/branch
        children that also carry ``parent_session_id``)."""
        if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
            return False
        edge = _COMPRESSION_CHILD_SQL.format(a="child")
        return conn.execute(f"""
            WITH RECURSIVE ancestors(id) AS (
                SELECT ?
                UNION
                SELECT parent.id
                FROM ancestors a
                JOIN sessions child ON child.id = a.id
                JOIN sessions parent ON parent.id = child.parent_session_id
                WHERE {edge}
            )
            SELECT 1 FROM ancestors WHERE id = ? AND id != ? LIMIT 1
            """, (descendant_id, ancestor_id, descendant_id)).fetchone() is not None

    def _set_session_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Write a title, enforcing provenance precedence. A ``user`` write always lands;
        ``derived``/``llm`` land only when the row is untitled or holds strictly lower
        authority (derived upgrades to llm exactly once, nothing overwrites a user name,
        re-running the titler on an llm row is a no-op). No writer may move a hidden
        canonical Bot Chat off its title. Read and write are one compare-and-swap
        transaction, so a manual ``/title`` racing an in-flight generation is not clobbered."""
        return bool(self._execute_write(
            lambda conn: self._set_session_title_in_transaction(conn, session_id, title, source=source)
        ))

    def _set_session_title_in_transaction(self, conn, session_id: str, title: str, *, source: str):
        """Reuse title rules within an authority-owned transaction; never commit here."""
        title = self.sanitize_title(title)
        is_user = source == self.TITLE_SOURCE_USER
        new_rank = self._title_rank(source) if not is_user else None
        affected = []

        current = conn.execute(
            "SELECT title, title_source, hidden FROM sessions WHERE id = ?", (session_id,),
        ).fetchone()
        if current is None:
            return []
        # The canonical Bot Chat's NAME is its identity (Bot Mode resolves it by
        # exact-title lookup on every open), so a rename orphans the conversation. Hidden
        # is the discriminator: canonical chats are born hidden; a visible session merely
        # named "Bot Chat" stays renameable. Provenance-blind.
        if ((current["title"] or "") == self.CANONICAL_BOT_CHAT_TITLE and bool(current["hidden"])
                and title != self.CANONICAL_BOT_CHAT_TITLE):
            if is_user:
                raise ValueError("This is the bot's canonical Bot Chat — its name is its "
                                 "identity, and renaming it would orphan the conversation. "
                                 "To start fresh, create a new bot instead.")
            return []
        if not is_user and current["title"] is not None and self._title_rank(current["title_source"]) >= new_rank:
            return []
        if title:
            cleared = self._resolve_title_conflict(conn, session_id, title)
            if cleared:
                affected.append(cleared)
        # CAS on the values just read (``IS`` is NULL-safe): a concurrent write between
        # the SELECT and here loses instead of being overwritten.
        changed = conn.execute(
            "UPDATE sessions SET title = ?, title_source = ? WHERE id = ? AND title IS ? AND title_source IS ?",
            (title, source if title else None, session_id, current["title"], current["title_source"]),
        ).rowcount
        return [*affected, session_id] if changed else affected

    def _resolve_title_conflict(self, conn, session_id: str, title: str) -> Optional[str]:
        """Free ``title`` for ``session_id`` inside the caller's write transaction, or raise
        ValueError when another session legitimately holds it. The one place the uniqueness
        rule lives, shared by renames and the API server's create-with-title. Returns the id
        whose title was cleared (an affected row for mutation receipts), else None."""
        conflict = conn.execute(
            f"SELECT id, archived, hidden, {self._EMPTY_SESSION_WHERE} AS ghost "
            "FROM sessions WHERE title = ? AND id != ?", (title, session_id),
        ).fetchone()
        if not conflict:
            return None
        conflict_id = conflict["id"]
        # A hidden compressed ancestor holding the title cannot be freed by the
        # user, so transfer it onto the tip (uniqueness + lineage kept).
        if self._is_compression_ancestor(conn, ancestor_id=conflict_id, descendant_id=session_id):
            conn.execute("UPDATE sessions SET title = NULL WHERE id = ?", (conflict_id,))
            return conflict_id
        # A deliberately archived hidden Bot Chat is a retired registry
        # entry, not a live identity. Retire its name in the same title
        # transaction so a replacement can become the sole canonical row;
        # the old session remains archived and otherwise untouched.
        # An ended, empty, visible ghost (abandoned new chat that listings filter
        # out, so the user cannot find it to free the name) yields too (#81888).
        # The full index stays: a ``message_count > 0`` partial index would make
        # the ghost's first append_message fail with IntegrityError.
        if (title == self.CANONICAL_BOT_CHAT_TITLE and bool(conflict["archived"])
              and bool(conflict["hidden"])) or (conflict["ghost"] and not conflict["hidden"]):
            conn.execute(
                "UPDATE sessions SET title = NULL, title_source = NULL WHERE id = ?",
                (conflict_id,),
            )
            return conflict_id
        raise ValueError(f"Title '{title}' is already in use by session {conflict_id}")

    def set_session_title(self, session_id: str, title: str) -> bool:
        """Set a title on the user's behalf (``user`` provenance). Empty clears it. Raises
        ValueError on conflict or validation failure."""
        return self._set_session_title(session_id, title, source=self.TITLE_SOURCE_USER)

    def set_auto_title(self, session_id: str, title: str, *, source: str) -> bool:
        """Set an automatic title; False (untouched) when a higher-authority title holds the row."""
        if source not in (self.TITLE_SOURCE_DERIVED, self.TITLE_SOURCE_LLM):
            raise ValueError(f"invalid automatic title source: {source!r}")
        return self._set_session_title(session_id, title, source=source)

    def get_session_title(self, session_id: str) -> Optional[str]:
        """Get the title for a session, or None."""
        row = self._read_one("SELECT title FROM sessions WHERE id = ?", (session_id,))
        return row["title"] if row else None

    def get_session_title_source(self, session_id: str) -> Optional[str]:
        """Get the provenance of a session's title, or None when untitled."""
        row = self._read_one("SELECT title, title_source FROM sessions WHERE id = ?", (session_id,))
        return row["title_source"] if row and row["title"] is not None else None

    def set_session_title_source(self, session_id: str, source: str) -> bool:
        """Overwrite a title's provenance without touching the text (a title copied across a
        compression rotation keeps the original's authority)."""
        if source not in self._TITLE_SOURCE_RANK:
            raise ValueError(f"invalid title source: {source!r}")
        return self._write_rowcount(
            "UPDATE sessions SET title_source = ? WHERE id = ? AND title IS NOT NULL", (source, session_id)
        ) > 0

    def get_session_by_title(self, title: str) -> Optional[dict[str, Any]]:
        """Look up a session by exact title. Returns session dict or None.

        Deliberately archived rows are skipped. Recoverable auto-archives stay
        visible so REST / Bot Mode resurrection can still find them (#92687).
        Recoverability is judged on the compression tip (same as
        ``unarchive_recoverable_session``): a compressed Bot Chat keeps
        ``title`` on the root (``end_reason='compression'``) while the reaper
        stamp lives on the tip.
        """
        row = self._read_one(
            "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
            "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
            "WHERE s.title = ?", (title,))
        if not row:
            return None
        session = self._session_row_dict(row)
        if not session.get("archived"):
            return session
        tip = session
        try:
            tip_id = self.get_compression_tip(session["id"]) or session["id"]
            if tip_id != session["id"]:
                tip = self.get_session(tip_id) or session
        except Exception:
            tip = session
        if (tip.get("end_reason") or "") in _RECOVERABLE_END_REASONS:
            return session
        return None

    def resolve_session_by_title(self, title: str) -> Optional[str]:
        """Resolve a title to a session ID, preferring the latest "title #N" continuation."""
        exact = self.get_session_by_title(title)
        # Exception to the "#N continuation" preference: the canonical Bot Chat's identity
        # IS its exact title (Bot Mode re-resolves it by name on every open, no id pointer).
        # A "<title> #N" sibling — a Desktop branch or a client-minted numbered row — is NOT
        # a Bot Mode session: it is visible, unmanaged, and the message_agent gate is off in
        # it. Every DM transport (``hermes -p <bot> chat --in ~ -c "Bot Chat"``: message_agent,
        # bot_relay, cron delivery) resolves through here, so letting the numbered row win
        # silently routes teammates' messages into a chat whose bot cannot answer back.
        # A retired (deliberately archived) Bot Chat must not fall through to "#N" either:
        # lookup returning None is what lets Desktop mint a fresh canonical chat (#139779).
        if title == self.CANONICAL_BOT_CHAT_TITLE:
            return exact["id"] if exact is not None else None
        # Escape LIKE wildcards so "%"/"_" in titles cannot false-match.
        # Same deliberate-archive skip as get_session_by_title (#139779).
        numbered = self._read_all(
            "SELECT id, title, started_at FROM sessions "
            "WHERE title LIKE ? ESCAPE '\\' "
            f"AND {_TITLE_LOOKUP_VISIBLE_SQL.format(alias='sessions')} "
            "ORDER BY started_at DESC",
            (f"{_escape_like(title)} #%",))
        return numbered[0]["id"] if numbered else (exact["id"] if exact else None)

    def get_next_title_in_lineage(self, base_title: str) -> str:
        """Next title in a lineage ("my session" -> "my session #2")."""
        return self._read_retrying_ioerr(lambda conn: next_title_in_lineage(conn, base_title))


def lineage_title_on_conn(conn, base_title: str) -> str:
    """``get_next_title_in_lineage`` inside an authority-owned transaction (same connection, so a
    sibling branch committed in this transaction is counted)."""
    return next_title_in_lineage(conn, base_title)
