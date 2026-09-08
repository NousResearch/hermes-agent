"""Durable WhatsApp notification-address and owner-mailbox routing.

A transport quote is an address, never authority to approve a tool. Only normal
authenticated ingress may deposit a reply for an existing session owner.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from pathlib import Path
import re
import sqlite3
import time
from uuid import uuid4

from gateway.session_context import get_session_env
from gateway.whatsapp_identity import canonical_whatsapp_identifier
from hermes_constants import get_hermes_home


_REPLY_TTL = 7 * 24 * 60 * 60
_OWNER_SURFACES = frozenset({"desktop", "tui"})
_APPROVAL = re.compile(
    r"^/?(?:yes|no|y|n|ok(?:ay)?|approve(?:d)?|deny|confirm|continue|proceed|go ahead|do it|sure|sounds good)\b",
    re.I,
)


@contextmanager
def connect(home):
    db = sqlite3.connect(Path(home) / "notification-replies.db", timeout=10)
    try:
        db.row_factory = sqlite3.Row
        db.execute(
            """CREATE TABLE IF NOT EXISTS notification_routes (
            message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            chat_id TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', reply_id TEXT, reply_text TEXT)"""
        )
        db.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS notification_reply_once "
            "ON notification_routes(chat_id, reply_id) WHERE reply_id IS NOT NULL"
        )
        db.execute(
            """CREATE TABLE IF NOT EXISTS notification_reply_queue (
            reply_id TEXT PRIMARY KEY, message_id TEXT NOT NULL,
            session_id TEXT NOT NULL, source TEXT NOT NULL,
            reply_text TEXT NOT NULL, created REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued')"""
        )
        db.execute(
            "CREATE INDEX IF NOT EXISTS notification_reply_owner_queue "
            "ON notification_reply_queue(session_id, status, created)"
        )
        with db:
            yield db
    finally:
        db.close()


def capture_origin(platform):
    if platform != "whatsapp":
        return None
    session_id = get_session_env("HERMES_SESSION_ID")
    home = get_hermes_home()
    if not session_id or not (home / "state.db").exists():
        return None
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        row = db.execute("SELECT source FROM sessions WHERE id=?", (session_id,)).fetchone()
    if not row or row[0] == "whatsapp":
        return None
    return {"home": home, "session_id": session_id, "source": row[0]}


def prepare_send(origin, chat_id, message, media_files, max_length):
    """Persist intent before transport dispatch without restricting its native payload."""
    if not origin:
        return
    origin["intent_id"] = "sending:" + uuid4().hex
    with connect(origin["home"]) as db:
        db.execute(
            "INSERT INTO notification_routes(message_id, session_id, chat_id, source, created, status) "
            "VALUES (?, ?, ?, ?, ?, 'sending')",
            (
                origin["intent_id"],
                origin["session_id"],
                canonical_whatsapp_identifier(chat_id),
                origin["source"],
                time.time(),
            ),
        )


def _result_message_ids(result) -> list[str]:
    values = result.get("message_ids") or result.get("partial_message_ids") or []
    if not values and isinstance(result.get("raw_response"), dict):
        values = result["raw_response"].get("message_ids") or []
    if isinstance(values, str):
        values = [values]
    ids = [str(value) for value in values if value]
    last = result.get("message_id")
    if last:
        ids.append(str(last))
    return list(dict.fromkeys(ids))


def record_sent(origin, chat_id, result):
    """Replace the pre-send intent with every transport-confirmed outbound ID."""
    if not origin:
        return
    message_ids = _result_message_ids(result)
    try:
        with connect(origin["home"]) as db:
            row = db.execute(
                "SELECT session_id, chat_id, source, created FROM notification_routes WHERE message_id=?",
                (origin["intent_id"],),
            ).fetchone()
            if row is None:
                raise sqlite3.IntegrityError("send intent disappeared")
            if not message_ids:
                db.execute(
                    "UPDATE notification_routes SET status='uncertain' WHERE message_id=?",
                    (origin["intent_id"],),
                )
            else:
                db.execute("DELETE FROM notification_routes WHERE message_id=?", (origin["intent_id"],))
                db.executemany(
                    "INSERT INTO notification_routes(message_id, session_id, chat_id, source, created, status) "
                    "VALUES (?, ?, ?, ?, ?, 'pending')",
                    [(message_id, row["session_id"], row["chat_id"], row["source"], row["created"])
                     for message_id in message_ids],
                )
    except sqlite3.Error:
        # Delivery already happened. Returning a send failure invites a duplicate
        # send, so report uncertain correlation separately.
        message_ids = []
        try:
            with connect(origin["home"]) as db:
                db.execute(
                    "UPDATE notification_routes SET status='uncertain' WHERE message_id=?",
                    (origin["intent_id"],),
                )
        except sqlite3.Error:
            pass
    if not message_ids:
        result["routing_warning"] = (
            "Notification delivery/correlation is uncertain. Do not resend automatically; "
            "reply in the originating session."
        )
        return
    result["message_ids"] = message_ids
    result["reply_routing"] = {
        "session_id": origin["session_id"],
        "status": "partial" if result.get("error") else "pending",
        "message_ids": message_ids,
    }
    if result.get("error"):
        result["routing_warning"] = (
            "Some notification parts were delivered and are reply-addressable. Do not resend automatically."
        )


def resolve_origin_session(home, session_id, source):
    """Return the native compression tip, or the existing explicit-closed owner.

    Explicitly closed owners remain addressable so normal ``session.resume`` can
    recreate them; missing rows and source changes fail closed.
    """
    if not (Path(home) / "state.db").exists():
        return None
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(home) / "state.db")
    try:
        row = db.get_session(session_id)
        if not row or row.get("source") != source:
            return None
        tip_id = db.get_compression_tip(session_id) or session_id
        tip = db.get_session(tip_id)
        return tip_id if tip and tip.get("source") == source else None
    finally:
        db.close()


def owner_session_lineage(home, session_id, source) -> tuple[str, ...]:
    """Return compression ancestors addressable by one currently live owner."""
    if not (Path(home) / "state.db").exists():
        return ()
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(home) / "state.db")
    try:
        current = db.get_session(session_id)
        if not current or current.get("source") != source:
            return ()
        lineage = db.get_compression_lineage(session_id) or [session_id]
        return tuple(
            candidate_id for candidate_id in lineage
            if (candidate := db.get_session(candidate_id)) and candidate.get("source") == source
        )
    finally:
        db.close()


def _live_cutoff() -> float:
    return time.time() - _REPLY_TTL


def _uncertain_quote(db, event):
    return bool(
        event.reply_to_message_id
        and event.reply_to_is_own_message
        and db.execute(
            "SELECT 1 FROM notification_routes WHERE chat_id=? AND created>=? "
            "AND status IN ('sending', 'uncertain') LIMIT 1",
            (canonical_whatsapp_identifier(event.source.chat_id), _live_cutoff()),
        ).fetchone()
    )


def _routing_shaped(event) -> bool:
    return bool(
        (event.reply_to_message_id and event.reply_to_is_own_message)
        or _APPROVAL.match(event.text.strip())
    )


def is_reply_candidate(event):
    """Read-only adapter bypass hint; authorization still happens in the runner."""
    if event.source.platform.value != "whatsapp" or event.internal:
        return False
    home = get_hermes_home()
    if not (home / "notification-replies.db").exists():
        return False
    try:
        with connect(home) as db:
            if event.reply_to_message_id:
                return bool(
                    db.execute(
                        "SELECT 1 FROM notification_routes WHERE message_id=?",
                        (event.reply_to_message_id,),
                    ).fetchone()
                    or _uncertain_quote(db, event)
                    or (event.reply_to_is_own_message and _APPROVAL.match(event.text.strip()))
                )
            return bool(
                _APPROVAL.match(event.text.strip())
                and db.execute(
                    "SELECT 1 FROM notification_routes WHERE chat_id=? AND created>=? "
                    "AND status IN ('pending', 'queued', 'dispatching', 'dispatched', 'sending', 'uncertain') LIMIT 1",
                    (canonical_whatsapp_identifier(event.source.chat_id), _live_cutoff()),
                ).fetchone()
            )
    except sqlite3.Error:
        # Do not block unrelated conversation merely because this optional store broke.
        return _routing_shaped(event)


def accept_reply(event):
    """Called AFTER gateway authentication, BEFORE current-chat controls. None = ordinary chat."""
    if event.source.platform.value != "whatsapp" or event.internal:
        return None
    home = get_hermes_home()
    if not (home / "notification-replies.db").exists():
        return None
    try:
        return _accept_stored_reply(home, event)
    except (sqlite3.Error, OSError):
        if _routing_shaped(event):
            return (
                "Notification routing is unavailable. Nothing was started; "
                "please reply in the originating session."
            )
        return None


def _accept_stored_reply(home, event):
    with connect(home) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM notification_routes WHERE message_id=?",
            (event.reply_to_message_id,),
        ).fetchone()
        if row is None:
            if _uncertain_quote(db, event):
                return (
                    "Cannot correlate this quote while notification delivery is uncertain. "
                    "Please reply in the originating session."
                )
            pending = db.execute(
                "SELECT 1 FROM notification_routes WHERE chat_id=? AND created>=? "
                "AND status IN ('pending', 'queued', 'dispatching', 'dispatched', 'sending', 'uncertain') LIMIT 1",
                (canonical_whatsapp_identifier(event.source.chat_id), _live_cutoff()),
            ).fetchone()
            if pending and not event.reply_to_message_id and _APPROVAL.match(event.text.strip()):
                return (
                    "Please quote the specific notification you are answering, or reply in its "
                    "originating session. No approval was applied."
                )
            if (
                event.reply_to_message_id
                and event.reply_to_is_own_message
                and _APPROVAL.match(event.text.strip())
            ):
                return (
                    "Cannot correlate this historical quote. No approval was applied; "
                    "confirm in the originating session."
                )
            return None
        source = event.source
        route_identity = canonical_whatsapp_identifier(row["chat_id"])
        if (
            source.chat_type != "dm"
            or canonical_whatsapp_identifier(source.chat_id) != route_identity
            or canonical_whatsapp_identifier(source.user_id or "") != route_identity
        ):
            return (
                "This notification reply cannot be verified for this sender and chat. "
                "Reply in the originating session."
            )
        if row["source"] not in _OWNER_SURFACES:
            return (
                "Please reply in the originating session; this surface does not support "
                "remote continuation yet."
            )
        owner_session_id = resolve_origin_session(home, row["session_id"], row["source"])
        if not owner_session_id:
            return (
                "The originating session is unavailable. Nothing was started; "
                "check that workstream manually."
            )
        if row["created"] < _live_cutoff():
            return "This notification has expired. Please confirm in the originating session."
        if not event.message_id or not event.text.strip() or event.media_urls:
            return "Please send a text-only reply to this notification; no action was taken."
        if row["status"] in {"sending", "uncertain", "expired"}:
            return (
                "This notification cannot currently accept a routed reply. "
                "Please confirm in the originating session."
            )
        try:
            db.execute(
                "INSERT INTO notification_reply_queue(reply_id, message_id, session_id, source, reply_text, created) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (event.message_id, row["message_id"], owner_session_id, row["source"], event.text, time.time()),
            )
        except sqlite3.IntegrityError:
            return (
                "This reply was already received; it will not be run again. "
                "Check the originating session."
            )
        db.execute(
            "UPDATE notification_routes SET status='queued', reply_id=?, reply_text=? WHERE message_id=?",
            (event.message_id, event.text, row["message_id"]),
        )
    return (
        "Reply queued for the originating session. Open or resume that session if it is not running; "
        "no task was started in WhatsApp."
    )
