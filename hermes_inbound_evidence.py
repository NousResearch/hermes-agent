"""Authenticated ingress -> committed transcript binding (local contract v1).

Trusted adapters/core and the profile SQLite file are the trust boundary, not
hostile Python code or an OS user able to edit that file. This attests authorship
of exact text, never its truth, intent, or permission to mutate a memory.
"""
from contextvars import ContextVar
from contextlib import closing, contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sqlite3

_SEAL = object()
_current = ContextVar("authenticated_inbound", default=None)


def _identity(event):
    s = event.source
    return (s.platform.value, s.chat_id, s.user_id, event.message_id)


@dataclass(frozen=True)
class _TransportReceipt:
    adapter: object
    identity: tuple
    text: str
    seal: object


@dataclass(frozen=True)
class _Admission:
    transport: _TransportReceipt
    home: str
    session_key: str
    session_id: str = ""


def mark_authenticated_human(event, adapter):
    """Adapter-only: call after transport authentication and positive human typing.

    The adapter must supply unmodified authored text, not extracted media, quoted
    context, batched text, or an application-generated prompt. Never deserialize
    this receipt from wire metadata. Unsupported adapters simply do not call it.
    """
    if (event.internal or event.source.is_bot or event.source.chat_type != "dm"
            or not isinstance(event.text, str) or not event.text.strip()
            or not all(isinstance(v, str) and v for v in _identity(event))):
        return
    event._human_transport_receipt = _TransportReceipt(adapter, _identity(event), event.text, _SEAL)


def reset_admission():
    """Each gateway ingress, including internal events, starts without authority."""
    _current.set(None)


def admit_authenticated_human(event, adapter, home, session_key):
    """Core-only: after user authorization, validate the adapter's exact snapshot."""
    _current.set(None)
    r = getattr(event, "_human_transport_receipt", None)
    if (type(r) is _TransportReceipt and r.seal is _SEAL and r.adapter is adapter
            and adapter is not None and r.identity == _identity(event) and r.text == event.text
            and not event.internal and event.source.is_bot is False
            and event.source.chat_type == "dm" and session_key):
        _current.set(_Admission(r, str(Path(home).resolve()), session_key))


def current_row_admission():
    """Private in-process carrier; never put it in API messages or persisted JSON."""
    return _current.get()


@contextmanager
def authenticated_turn(session_id, session_key):
    """Pin admission to the actual runner's session; restore across nested turns."""
    admission = _current.get()
    bound = (replace(admission, session_id=session_id)
             if type(admission) is _Admission and admission.session_key == session_key
             and admission.session_id in ("", session_id) else None)
    token = _current.set(bound)
    try:
        yield
    finally:
        _current.reset(token)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode()).hexdigest()


def bind_committed_row(conn, db_path, session_id, row_id, admission):
    """Called ONLY by append_messages_batch, in the message's write transaction."""
    if type(admission) is not _Admission or admission.transport.seal is not _SEAL:
        return
    if admission.session_id != session_id or str(Path(db_path).resolve().parent) != admission.home:
        return
    r = admission.transport
    row = conn.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (row_id, session_id)).fetchone()
    session = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    if (row is None or session is None or session["session_key"] != admission.session_key
            or (session["source"], session["chat_id"], session["user_id"]) != r.identity[:3]
            or session["chat_type"] != "dm" or row["role"] != "user" or row["active"] != 1
            or row["_compressed_summary"] or row["display_kind"] not in (None, "")
            or row["platform_message_id"] != r.identity[3] or row["content"] != r.text):
        return
    conn.execute("""CREATE TABLE IF NOT EXISTS authenticated_inbound_v1 (
        event_key TEXT PRIMARY KEY, message_row_id INTEGER NOT NULL UNIQUE REFERENCES messages(id) ON DELETE CASCADE,
        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE, session_key TEXT NOT NULL, source TEXT NOT NULL,
        chat_id TEXT NOT NULL, user_id TEXT NOT NULL, platform_message_id TEXT NOT NULL,
        content_digest TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0)""")
    # Namespace by receiving conversation, never by a user-supplied role/row ID.
    event_key = _hash([r.identity[0], r.identity[1], r.identity[3]])
    existing = conn.execute("SELECT message_row_id FROM authenticated_inbound_v1 WHERE event_key=?", (event_key,)).fetchone()
    if existing:
        if existing["message_row_id"] != row_id:
            conn.execute("UPDATE authenticated_inbound_v1 SET revoked=1 WHERE event_key=?", (event_key,))
        return
    conn.execute("INSERT INTO authenticated_inbound_v1 VALUES (?,?,?,?,?,?,?,?,?,0)",
                 (event_key, row_id, session_id, admission.session_key, *r.identity, _hash(r.text)))


def read_authenticated_inbound(profile_home, session_id, row_id):
    """Read-only plugin API. None means unavailable/unbound/revoked; fail closed.

    Re-read before acting. Consumers must separately authorize the returned actor
    for their memory bucket and verify semantics/target/version/rollback policy.
    Missing table on an older core/profile is ordinary unsupported evidence.
    """
    if type(row_id) is not int or row_id <= 0 or not isinstance(session_id, str):
        return None
    home = Path(profile_home).resolve()
    path = home / "state.db"
    if path.resolve().parent != home:
        return None
    try:
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=1)) as conn:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA query_only=ON")
            conn.execute("BEGIN")
            proof = conn.execute("SELECT * FROM authenticated_inbound_v1 WHERE message_row_id=? AND session_id=? AND revoked=0",
                                 (row_id, session_id)).fetchone()
            if proof is None:
                return None
            row = conn.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (row_id, session_id)).fetchone()
            s = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
            if (row is None or s is None or row["role"] != "user" or row["active"] != 1
                    or row["_compressed_summary"] or row["display_kind"] not in (None, "")
                    or s["chat_type"] != "dm" or s["session_key"] != proof["session_key"]
                    or any(s[k] != proof[k] for k in ("source", "chat_id", "user_id"))
                    or row["platform_message_id"] != proof["platform_message_id"]
                    or _hash(row["content"]) != proof["content_digest"]):
                return None
            return {"schema": "authenticated-inbound-v1", "profile_home": str(home),
                    **dict(proof), "content": row["content"]}
    except (sqlite3.Error, OSError, ValueError, OverflowError):
        return None


def current_authenticated_inbound(profile_home, session_id=None):
    """Return this admitted turn's COMMITTED row, without positional/text guessing.

    Only the process-local adapter/core admission can select the event key. A
    caller-supplied role, message ID or context dictionary cannot select a row.
    Background workers inheriting the turn context still re-read durable proof.
    """
    admission = _current.get()
    home = Path(profile_home).resolve()
    if (type(admission) is not _Admission or admission.transport.seal is not _SEAL
            or not admission.session_id or str(home) != admission.home
            or session_id is not None and session_id != admission.session_id):
        return None
    path = home / 'state.db'
    if path.resolve().parent != home:
        return None
    identity = admission.transport.identity
    key = _hash([identity[0], identity[1], identity[3]])
    try:
        with closing(sqlite3.connect(path.as_uri() + '?mode=ro', uri=True, timeout=1)) as conn:
            row = conn.execute('SELECT message_row_id FROM authenticated_inbound_v1 '
                'WHERE event_key=? AND session_id=? AND revoked=0', (key, admission.session_id)).fetchone()
        if row is None:
            return None
        proof = read_authenticated_inbound(home, admission.session_id, row[0])
        if proof and proof['content'] == admission.transport.text:
            return proof
    except (sqlite3.Error, OSError, ValueError, OverflowError):
        return None
    return None


@contextmanager
def authenticated_inbound_guard(profile_home, session_id, row_id):
    """Serialize an evidence-dependent operation with transcript/revocation writes.

    Acquiring the SQLite writer reservation is the admission linearization point:
    earlier revocations are observed, later writers wait until the operation exits.
    This writes no data and never creates a database. It is NOT a transaction with
    the consumer's remote stores; those still need CAS, journaling and recovery.
    Consumers must bound their remote calls because profile writes wait here.
    """
    if type(row_id) is not int or row_id <= 0 or not isinstance(session_id, str):
        raise ValueError('Exact source row required')
    home = Path(profile_home).resolve()
    path = home / 'state.db'
    if path.resolve().parent != home:
        raise ValueError('Source database outside profile')
    with closing(sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=1)) as conn:
        conn.execute('BEGIN IMMEDIATE')
        try:
            conn.execute('PRAGMA query_only=ON')
            proof = read_authenticated_inbound(home, session_id, row_id)
            if proof is None:
                raise ValueError('Authenticated source unavailable or revoked')
            yield proof
        finally:
            conn.rollback()
