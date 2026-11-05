"""Session titles: set with conflict resolution, auto titles, title reads and lineage numbering."""


from __future__ import annotations
import re
from typing import Any, Optional
from persistence.sqlite._schema import (
    TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM, TITLE_SOURCE_USER, _TITLE_SOURCE_RANK, _title_rank,
    next_title_in_lineage,
)
from persistence.sqlite._holder import StoreHolder
from persistence.sqlite._schema_ddl import _COMPRESSION_CHILD_SQL
from persistence.sqlite.conversation._store.rules import _sanitize_surrogates
from persistence.sqlite.conversation._store.sql_common import _non_continuation_child_sql, escape_like as _escape_like


_TITLE_CONTROL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


_TITLE_INVISIBLE_RE = re.compile(r'[\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff\ufffc\ufff9-\ufffb]')


def sanitize_title(title: Optional[str]) -> Optional[str]:
    """Strip control/zero-width/bidi chars (and lone surrogates sqlite3 cannot bind),
    collapse whitespace, normalize empty to None. ValueError past MAX_TITLE_LENGTH."""
    if not title:
        return None
    cleaned = _TITLE_INVISIBLE_RE.sub('', _TITLE_CONTROL_RE.sub('', _sanitize_surrogates(title)))
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_TITLE_LENGTH:
        raise ValueError(f"Title too long ({len(cleaned)} chars, max {MAX_TITLE_LENGTH})")
    return cleaned


def _is_compression_ancestor(holder: StoreHolder, conn, *, ancestor_id: str, descendant_id: str) -> bool:
    """True if *ancestor_id* is a compression predecessor of *descendant_id*, via the
    canonical continuation edge ``_COMPRESSION_CHILD_SQL`` (excludes delegate/branch
    children that also carry ``parent_session_id``, even under a compressed parent)."""
    if not ancestor_id or not descendant_id or ancestor_id == descendant_id:
        return False
    edge = _COMPRESSION_CHILD_SQL.format(a="child") + _non_continuation_child_sql("child.", "parent.id")
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


def _set_session_title(holder: StoreHolder, session_id: str, title: str, *, source: str) -> bool:
    """Write a title, enforcing provenance precedence. A ``user`` write always lands;
    ``derived``/``llm`` land only when the row is untitled or holds strictly lower
    authority (derived upgrades to llm exactly once, nothing overwrites a user name,
    re-running the titler on an llm row is a no-op). No writer may move a hidden
    canonical Bot Chat off its title. Read and write are one compare-and-swap
    transaction, so a manual ``/title`` racing an in-flight generation is not clobbered."""
    return bool(holder._execute_write(
        lambda conn: _set_session_title_in_transaction(holder, conn, session_id, title, source=source)
    ))


def _set_session_title_in_transaction(holder: StoreHolder, conn, session_id: str, title: str, *, source: str):
    """Reuse title rules within an authority-owned transaction; never commit here."""
    title = sanitize_title(title)
    is_user = source == TITLE_SOURCE_USER
    new_rank = _title_rank(source) if not is_user else None
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
    if ((current["title"] or "") == CANONICAL_BOT_CHAT_TITLE and bool(current["hidden"])
            and title != CANONICAL_BOT_CHAT_TITLE):
        if is_user:
            raise ValueError("This is the bot's canonical Bot Chat — its name is its "
                             "identity, and renaming it would orphan the conversation. "
                             "To start fresh, create a new bot instead.")
        return []
    if not is_user and current["title"] is not None and _title_rank(current["title_source"]) >= new_rank:
        return []
    if title:
        cleared = _resolve_title_conflict(holder, conn, session_id, title)
        if cleared:
            affected.append(cleared)
    # CAS on the values just read (``IS`` is NULL-safe): a concurrent write between
    # the SELECT and here loses instead of being overwritten.
    changed = conn.execute(
        "UPDATE sessions SET title = ?, title_source = ? WHERE id = ? AND title IS ? AND title_source IS ?",
        (title, source if title else None, session_id, current["title"], current["title_source"]),
    ).rowcount
    return [*affected, session_id] if changed else affected


def _resolve_title_conflict(holder: StoreHolder, conn, session_id: str, title: str) -> Optional[str]:
    """Free ``title`` for ``session_id`` inside the caller's write transaction, or raise
    ValueError when another session legitimately holds it. The one place the uniqueness
    rule lives, shared by renames and the API server's create-with-title. Returns the id
    whose title was cleared (an affected row for mutation receipts), else None."""
    conflict = conn.execute(
        f"SELECT id, archived, hidden, {_EMPTY_SESSION_WHERE} AS ghost "
        "FROM sessions WHERE title = ? AND id != ?", (title, session_id),
    ).fetchone()
    if not conflict:
        return None
    conflict_id = conflict["id"]
    # A hidden compressed ancestor holding the title cannot be freed by the
    # user, so transfer it onto the tip (uniqueness + lineage kept).
    if _is_compression_ancestor(holder, conn, ancestor_id=conflict_id, descendant_id=session_id):
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
    if (title == CANONICAL_BOT_CHAT_TITLE and bool(conflict["archived"])
          and bool(conflict["hidden"])) or (conflict["ghost"] and not conflict["hidden"]):
        conn.execute(
            "UPDATE sessions SET title = NULL, title_source = NULL WHERE id = ?",
            (conflict_id,),
        )
        return conflict_id
    raise ValueError(f"Title '{title}' is already in use by session {conflict_id}")


def set_session_title(holder: StoreHolder, session_id: str, title: str) -> bool:
    """Set a title on the user's behalf (``user`` provenance). Empty clears it. Raises
    ValueError on conflict or validation failure."""
    return _set_session_title(holder, session_id, title, source=TITLE_SOURCE_USER)


def set_auto_title(holder: StoreHolder, session_id: str, title: str, *, source: str) -> bool:
    """Set an automatic title; False (untouched) when a higher-authority title holds the row."""
    if source not in (TITLE_SOURCE_DERIVED, TITLE_SOURCE_LLM):
        raise ValueError(f"invalid automatic title source: {source!r}")
    return _set_session_title(holder, session_id, title, source=source)


def get_session_title(holder: StoreHolder, session_id: str) -> Optional[str]:
    """Get the title for a session, or None."""
    row = holder._read_one("SELECT title FROM sessions WHERE id = ?", (session_id,))
    return row["title"] if row else None


def get_session_title_source(holder: StoreHolder, session_id: str) -> Optional[str]:
    """Get the provenance of a session's title, or None when untitled."""
    row = holder._read_one("SELECT title, title_source FROM sessions WHERE id = ?", (session_id,))
    return row["title_source"] if row and row["title"] is not None else None


def set_session_title_source(holder: StoreHolder, session_id: str, source: str) -> bool:
    """Overwrite a title's provenance without touching the text (a title copied across a
    compression rotation keeps the original's authority)."""
    if source not in _TITLE_SOURCE_RANK:
        raise ValueError(f"invalid title source: {source!r}")
    return holder._write_rowcount(
        "UPDATE sessions SET title_source = ? WHERE id = ? AND title IS NOT NULL", (source, session_id)
    ) > 0


def get_session_by_title(holder: StoreHolder, title: str) -> Optional[dict[str, Any]]:
    """Look up a session by exact title. Returns session dict or None."""
    row = holder._read_one(
        "SELECT s.*, COALESCE(sp.prompt, s.system_prompt) AS _system_prompt_resolved "
        "FROM sessions s LEFT JOIN system_prompts sp ON sp.hash = s.system_prompt_hash "
        "WHERE s.title = ?", (title,))
    return _session_row_dict(row) if row else None


def resolve_session_by_title(holder: StoreHolder, title: str) -> Optional[str]:
    """Resolve a title to a session ID, preferring the latest "title #N" continuation."""
    exact = get_session_by_title(holder, title)
    # Exception to the "#N continuation" preference: the canonical Bot Chat's identity
    # IS its exact title (Bot Mode re-resolves it by name on every open, no id pointer).
    # A "<title> #N" sibling — a Desktop branch or a client-minted numbered row — is NOT
    # a Bot Mode session: it is visible, unmanaged, and the message_agent gate is off in
    # it. Every DM transport (``hermes -p <bot> chat --in ~ -c "Bot Chat"``: message_agent,
    # bot_relay, cron delivery) resolves through here, so letting the numbered row win
    # silently routes teammates' messages into a chat whose bot cannot answer back.
    if exact is not None and title == CANONICAL_BOT_CHAT_TITLE:
        return exact["id"]
    # Escape LIKE wildcards so "%"/"_" in titles cannot false-match.
    numbered = holder._read_all(
        "SELECT id, title, started_at FROM sessions "
        "WHERE title LIKE ? ESCAPE '\\' ORDER BY started_at DESC",
        (f"{_escape_like(title)} #%",))
    return numbered[0]["id"] if numbered else (exact["id"] if exact else None)


def get_next_title_in_lineage(holder: StoreHolder, base_title: str) -> str:
    """Next title in a lineage ("my session" -> "my session #2")."""
    return holder._read_retrying_ioerr(lambda conn: next_title_in_lineage(conn, base_title))


# Sibling store modules import this module's copies in turn, so their copies are imported after these.
from persistence.sqlite.conversation._store.core import CANONICAL_BOT_CHAT_TITLE, MAX_TITLE_LENGTH, _session_row_dict
from persistence.sqlite.conversation._store.sessions import _EMPTY_SESSION_WHERE
