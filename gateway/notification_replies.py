"""Profile-local notification correlation, separate from chat transcripts.

A transport quote is an address, never authority to approve a tool. Only the
normal authenticated ingress may deposit a reply for an existing owner.
"""
from uuid import uuid4
from contextlib import closing, contextmanager
from pathlib import Path
import sqlite3
import re
import time

from hermes_constants import get_hermes_home
from gateway.session_context import get_session_env
from gateway.whatsapp_identity import to_whatsapp_jid


@contextmanager
def connect(home):
    db = sqlite3.connect(Path(home) / 'notification-replies.db', timeout=10)
    try:
        db.row_factory = sqlite3.Row
        db.execute('''CREATE TABLE IF NOT EXISTS notification_routes (
            message_id TEXT PRIMARY KEY, session_id TEXT NOT NULL,
            chat_id TEXT NOT NULL, source TEXT NOT NULL, created REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending', reply_id TEXT, reply_text TEXT)''')
        db.execute('CREATE UNIQUE INDEX IF NOT EXISTS notification_reply_once '
                   'ON notification_routes(chat_id, reply_id) WHERE reply_id IS NOT NULL')
        with db:
            yield db
    finally:
        db.close()


def capture_origin(platform):
    if platform != 'whatsapp':
        return None
    session_id = get_session_env('HERMES_SESSION_ID')
    home = get_hermes_home()
    if not session_id or not (home / 'state.db').exists():
        return None
    with closing(sqlite3.connect(f'file:{home / "state.db"}?mode=ro', uri=True)) as db:
        row = db.execute('SELECT source FROM sessions WHERE id=?', (session_id,)).fetchone()
    if not row or row[0] == 'whatsapp':
        return None
    return dict(home=home, session_id=session_id, source=row[0])


def prepare_send(origin, chat_id, message, media_files, max_length):
    if not origin:
        return
    # Until the sender returns every partial-success ID, never commission an
    # approval notification whose first chunk/attachment cannot be correlated.
    if media_files or not message.strip() or len(message) > min(max_length or 4000, 4000):
        raise ValueError('Cross-session WhatsApp notifications currently require one text-only message (at most 4000 characters). Send attachments in the originating session.')
    origin['intent_id'] = 'sending:' + uuid4().hex
    with connect(origin['home']) as db:
        db.execute('INSERT INTO notification_routes(message_id, session_id, chat_id, source, created, status) '
                   "VALUES (?, ?, ?, ?, ?, 'sending')",
                   (origin['intent_id'], origin['session_id'], to_whatsapp_jid(chat_id), origin['source'], time.time()))


def record_sent(origin, chat_id, result):
    if not origin:
        return
    message_id = result.get('message_id') if result.get('success') else None
    try:
        with connect(origin['home']) as db:
            db.execute('UPDATE notification_routes SET message_id=?, status=? WHERE message_id=?',
                       (message_id or origin['intent_id'], 'pending' if message_id else 'uncertain', origin['intent_id']))
    except sqlite3.Error:
        # Delivery already happened. Returning a send failure invites a duplicate
        # send; keep the pre-send intent and report routing failure separately.
        message_id = None
    if not message_id:
        result['routing_warning'] = 'Notification delivery/correlation is uncertain. Do not resend automatically; reply in the originating session.'
    else:
        result['reply_routing'] = {'session_id': origin['session_id'], 'status': 'pending'}


_REPLY_TTL = 7 * 24 * 60 * 60
_OWNER_SURFACES = frozenset({'desktop', 'tui'})
_APPROVAL = re.compile(r'^/?(?:yes|no|y|n|ok(?:ay)?|approve(?:d)?|deny|confirm|continue|proceed|go ahead|do it|sure|sounds good)\b', re.I)


def origin_exists(home, session_id, source):
    if not (Path(home) / 'state.db').exists():
        return False
    db = sqlite3.connect(f'file:{Path(home) / "state.db"}?mode=ro', uri=True)
    try:
        row = db.execute('SELECT source FROM sessions WHERE id=?', (session_id,)).fetchone()
        return bool(row and row[0] == source)
    finally:
        db.close()


def _uncertain_quote(db, event):
    return bool(event.reply_to_message_id and event.reply_to_is_own_message and db.execute(
        "SELECT 1 FROM notification_routes WHERE chat_id=? AND status IN ('sending', 'uncertain') LIMIT 1",
        (to_whatsapp_jid(event.source.chat_id),)).fetchone())


def is_reply_candidate(event):
    """Read-only adapter bypass hint; authorization still happens in the runner."""
    if event.source.platform.value != 'whatsapp' or event.internal:
        return False
    home = get_hermes_home()
    if not (home / 'notification-replies.db').exists():
        return False
    try:
        with connect(home) as db:
            if event.reply_to_message_id:
                return db.execute('SELECT 1 FROM notification_routes WHERE message_id=?',
                                  (event.reply_to_message_id,)).fetchone() is not None or _uncertain_quote(db, event)
            return bool(_APPROVAL.match(event.text.strip()) and db.execute(
                "SELECT 1 FROM notification_routes WHERE chat_id=? AND status IN ('pending', 'sending', 'uncertain') LIMIT 1",
                (to_whatsapp_jid(event.source.chat_id),)).fetchone())
    except sqlite3.Error:
        return True  # a broken routing store must not steer the current workstream


def accept_reply(event):
    """Called AFTER gateway authentication, BEFORE current-chat controls. None = ordinary chat."""
    if event.source.platform.value != 'whatsapp' or event.internal:
        return None
    home = get_hermes_home()
    if not (home / 'notification-replies.db').exists():
        return None
    try:
        return _accept_stored_reply(home, event)
    except (sqlite3.Error, OSError):
        return 'Notification routing is unavailable. Nothing was started; please reply in the originating session.'


def _accept_stored_reply(home, event):
    with connect(home) as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM notification_routes WHERE message_id=?',
                         (event.reply_to_message_id,)).fetchone()
        if row is None:
            if _uncertain_quote(db, event):
                return 'Cannot correlate this quote while notification delivery is uncertain. Please reply in the originating session.'
            pending = db.execute("SELECT 1 FROM notification_routes WHERE chat_id=? AND status IN ('pending', 'sending', 'uncertain') LIMIT 1",
                                 (to_whatsapp_jid(event.source.chat_id),)).fetchone()
            if pending and not event.reply_to_message_id and _APPROVAL.match(event.text.strip()):
                return 'Please quote the specific notification you are answering, or reply in its originating session. No approval was applied.'
            return None
        source = event.source
        if (source.chat_type != 'dm' or to_whatsapp_jid(source.chat_id) != row['chat_id']
                or to_whatsapp_jid(source.user_id or '') != row['chat_id']):
            return 'This notification reply cannot be verified for this sender and chat. Reply in the originating session.'
        if row['source'] not in _OWNER_SURFACES:
            return 'Please reply in the originating session; this surface does not support remote continuation yet.'
        if not origin_exists(home, row['session_id'], row['source']):
            return 'The originating session is unavailable. Nothing was started; check that workstream manually.'
        if time.time() - row['created'] > _REPLY_TTL:
            return 'This notification has expired. Please confirm in the originating session.'
        if not event.message_id or not event.text.strip() or event.media_urls:
            return 'Please send a text-only reply to this notification; no action was taken.'
        if row['status'] != 'pending':
            return 'This notification already has a reply; it will not be run again. Check the originating session.'
        db.execute("UPDATE notification_routes SET status='queued', reply_id=?, reply_text=? "
                   "WHERE message_id=? AND status='pending'", (event.message_id, event.text, row['message_id']))
    return 'Reply queued for the originating session. Open or resume that session if it is not running; no task was started in WhatsApp.'
