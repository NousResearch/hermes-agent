"""Explicit, single-bot Desktop mirror into a persisted Slack thread.

No delivery retry: a timeout after Slack accepts chat.postMessage is ambiguous.
Claims precede network I/O and survive restarts, at the cost of possibly missing
one mirror on a transport failure. This is deliberately at-most-once, not a queue.
"""
from __future__ import annotations

import asyncio
from contextlib import ExitStack, contextmanager
import errno
import hashlib
import json
import logging
import os
from pathlib import Path
import time
from typing import Callable

logger = logging.getLogger(__name__)


@contextmanager
def _session_delivery_lock(db, session_id):
    """Cross-process per-session lease, distinct from state.db's writer lock.

    Keep it through the transport call: a successful opt-out must not be
    followed by a post that was already in flight. The HTTP timeout is 20s;
    callers cannot report successful revocation while a hung custom transport
    still owns the lease. Never delete/recreate this file (inode identity).
    """
    digest = hashlib.sha256(session_id.encode('utf-8')).hexdigest()
    path = Path(db.db_path).resolve().with_name(f'.slack-desktop-{digest}.lock')
    deadline = time.monotonic() + 25
    with open(path, 'a+b') as handle:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                handle.write(b'\0')
                handle.flush()
            def lock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            def unlock():
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            def lock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            def unlock():
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        while True:
            try:
                lock()
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError('Slack mirror delivery still in flight; opt-out cannot be confirmed') from None
                time.sleep(0.05)
        try:
            yield
        finally:
            unlock()


def _candidate(db, row):
    if not row or not row.get('session_key'):
        return None
    home = Path(db.db_path).resolve().parent
    own_profile = db._own_profile_name()
    if own_profile and row.get('profile_name') not in (None, own_profile):
        return None
    namespace = 'main' if own_profile == 'default' else 'main~' if own_profile == 'main' else own_profile
    key = row['session_key']
    standalone = bool(own_profile and own_profile != 'default'
                      and str(key).startswith('agent:main:slack:'))
    if own_profile and not (str(key).startswith(f"agent:{namespace}:slack:") or standalone):
        return None
    try:
        from agent.secret_scope import build_profile_secret_scope, load_env_file, _is_process_home

        secrets = build_profile_secret_scope(home)
        raw = secrets.get('SLACK_BOT_TOKEN', '')
        file_raw = load_env_file(home / '.env').get('SLACK_BOT_TOKEN', '')
        # External secret sources override .env at runtime; count both rather
        # than letting that precedence silently hide a second configured bot.
        raw = ','.join(filter(None, (raw, file_raw)))
        # A launch-profile process credential may override .env in the gateway.
        # Count BOTH sources; a disagreement is multiple bots, not a fallback.
        if _is_process_home(home):
            import os
            ambient = os.environ.get('SLACK_BOT_TOKEN', '')
            raw = ','.join(filter(None, (raw, ambient)))
        saved = home / 'slack_tokens.json'
        if saved.exists():
            saved_tokens = json.loads(saved.read_text(encoding='utf-8-sig'))
            if not isinstance(saved_tokens, dict):
                return None
            if any(not isinstance(entry, dict) or not isinstance(entry.get('token'), str)
                   for entry in saved_tokens.values()):
                return None
            extra = [entry['token'] for entry in saved_tokens.values()]
        else:
            extra = []
        tokens = {token.strip() for token in str(raw).split(',') if token.strip()}
        tokens.update(token.strip() for token in extra if isinstance(token, str) and token.strip())
        if len(tokens) != 1:
            return None
        token = next(iter(tokens))
        if not token.startswith('xoxb-'):
            return None
        routing_db = db
        if own_profile and own_profile != 'default' and not standalone:
            # Multiplex persists one routing index at the root, while each
            # named profile keeps its transcript in its own state.db. A
            # standalone named profile instead uses agent:main and its local
            # state.db for both route and transcript. Never
            # authorize delivery using a stale profile-local copy of a route.
            from hermes_constants import get_default_hermes_root
            from hermes_state import SessionDB
            root = Path(get_default_hermes_root()).resolve()
            if home != root / 'profiles' / own_profile:
                return None
            routing_db = SessionDB(db_path=root / 'state.db', read_only=True)
        try:
            routes = routing_db._read_all('SELECT entry_json FROM gateway_routing WHERE session_key = ?', (key,))
            if len(routes) != 1:
                return None
            entry = json.loads(routes[0]['entry_json'])
            receiving_bot_user_id = entry.get('receiving_bot_user_id')
            if not isinstance(receiving_bot_user_id, str) or not receiving_bot_user_id:
                return None  # A current token cannot attest who received a historical route.
            routed_id = entry.get('session_id')
            if not isinstance(routed_id, str) or not routed_id:
                return None
            # Conflicting routes for the same session id make provenance ambiguous.
            if sum(1 for route in routing_db._read_all('SELECT entry_json FROM gateway_routing')
                   if json.loads(route['entry_json']).get('session_id') == routed_id) != 1:
                return None
        finally:
            if routing_db is not db:
                routing_db.close()
        # A compression continuation can keep the original routing entry and
        # take the Desktop source. Only the live tip of that exact Slack-origin
        # lineage is authorized; an explicit branch/reset is not its tip.
        lineage = db.get_compression_lineage(row['id'])
        if (not lineage or lineage[-1] != row['id'] or routed_id not in lineage
                or db.get_compression_tip(routed_id) != row['id']):
            return None
        ancestor = db.get_session(lineage[0])
        if (not ancestor or ancestor.get('source') != 'slack'
                or ancestor.get('session_key') != key
                or ancestor.get('origin_json') != row.get('origin_json')):
            return None
        origin = entry.get('origin')
        stored = json.loads(row.get('origin_json') or '{}')
        if (entry.get('session_key') != key
                or not isinstance(origin, dict) or origin != stored
                or origin.get('platform') != 'slack'):
            return None
        team, channel, thread = (origin.get(k) for k in ('scope_id', 'chat_id', 'thread_id'))
        if not all(isinstance(v, str) and v for v in (team, channel, thread)):
            return None  # DM without an explicit anchor is not a known original thread.
        if (row.get('chat_id') != channel or row.get('thread_id') != thread
                or f':slack:' not in key or f':{team}:{channel}:{thread}' not in key):
            return None
        # A known different receiving bot is not licensed by this profile's token.
        owner = row.get('transport_profile') or entry.get('transport_profile')
        if owner and owner != (row.get('profile_name') or db._own_profile_name() or 'default'):
            return None
        return token, channel, thread, team, receiving_bot_user_id
    except Exception:
        logger.debug('Slack mirror provenance/credential unavailable')
        return None


def availability(db, row) -> bool:
    """Local eligibility only; auth.test checks team identity at opt-in and delivery."""
    return _candidate(db, row) is not None


def _slack_call(token: str, method: str, payload: dict) -> dict:
    from plugins.platforms.slack.adapter import _slack_json_post, _standalone_proxy_kwargs
    import aiohttp

    async def call():
        sess_kw, req_kw = _standalone_proxy_kwargs()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20), **sess_kw) as session:
            return await _slack_json_post(session, token, method, payload, req_kw)
    return asyncio.run(call())


def _auth_test(token: str) -> dict:
    return _slack_call(token, 'auth.test', {})


def _send(token: str, channel: str, thread: str, text: str) -> dict:
    return _slack_call(token, 'chat.postMessage', {
        'channel': channel, 'thread_ts': thread, 'text': text,
        'unfurl_links': False, 'unfurl_media': False,
    })


def _verified(db, row, auth_test):
    candidate = _candidate(db, row)
    if candidate is None:
        return None
    token, channel, thread, team, receiving_bot_user_id = candidate
    try:
        response = auth_test(token)
        if (response.get('ok') is True and response.get('team_id') == team
                and response.get('user_id') == receiving_bot_user_id
                and isinstance(response.get('bot_id'), str) and response['bot_id']):
            return token, channel, thread
    except Exception:
        logger.warning('Slack mirror auth.test failed; refusing delivery')
    return None


def set_opt_in(db, session_id: str, enabled: bool, *, auth_test: Callable | None = None) -> bool:
    row = db.get_session(session_id)
    if not row:
        raise ValueError('Session not found')
    # Snapshot BEFORE remote auth: a later opt-out invalidates this request,
    # even when the initial and final booleans happen to be identical.
    version = row['slack_sync_version']
    if enabled and not _verified(db, row, auth_test or _auth_test):
        raise ValueError('Slack mirror unavailable: original thread, unique bot token, workspace or receiving bot identity not verified')
    revoke_ids: tuple[str, ...] = ()
    if not enabled:
        # A stale Desktop menu can still name a compression-ended ancestor.
        # Resolve and mark the *entire* continuation while holding the DB write
        # lock, so publication cannot insert a new opted-in child between the
        # lineage lookup and revocation. Explicit branches/resets are separate.
        def revoke_lineage(conn):
            ancestors = db.get_compression_lineage(session_id)
            if not ancestors or session_id not in ancestors:
                raise ValueError('Session no longer exists')
            preferred = db.get_compression_chain(ancestors[0])
            ids = tuple(sorted(set(ancestors + preferred)))
            marks = ','.join('?' for _ in ids)
            conn.execute(
                'UPDATE sessions SET slack_sync = 0, slack_sync_revoking = 1, '
                f'slack_sync_version = slack_sync_version + 1 WHERE id IN ({marks})', ids)
            return ids

        # Persist intent before waiting on any in-flight post. Every member's
        # send gate now sees disabled consent, including the live child.
        revoke_ids = db._execute_write(revoke_lineage)
    with ExitStack() as locks:
        for locked_id in (revoke_ids if not enabled else (session_id,)):
            locks.enter_context(_session_delivery_lock(db, locked_id))
        if enabled:
            changed = db._write_rowcount(
                'UPDATE sessions SET slack_sync = 1, slack_sync_version = slack_sync_version + 1 '
                'WHERE id = ? AND slack_sync_version = ? AND slack_sync_revoking = 0',
                (session_id, version))
            if changed != 1:
                raise ValueError('Slack mirror consent changed while identity was being verified')
        else:
            # An opt-in that raced ahead of this waiting revoker is cancelled;
            # a repeated opt-out also invalidates an older auth.test. A child
            # published while the leases were settling inherited the revoking
            # guard; include it before clearing that guard. The write lock
            # prevents another publication from slipping through this step.
            def finish_revocation(conn):
                ancestors = db.get_compression_lineage(session_id)
                preferred = db.get_compression_chain(ancestors[0]) if ancestors else []
                ids = tuple(sorted(set(revoke_ids + tuple(ancestors + preferred))))
                marks = ','.join('?' for _ in ids)
                conn.execute('UPDATE sessions SET slack_sync = 0, slack_sync_revoking = 0, '
                             f'slack_sync_version = slack_sync_version + 1 WHERE id IN ({marks})', ids)

            db._execute_write(finish_revocation)
    return bool(enabled)


def mirror_row(db, session_id: str, row_id: int, role: str, *,
               send: Callable | None = None, auth_test: Callable | None = None) -> bool:
    """Claim a persisted visible row before attempting ONE authenticated send."""
    if role not in ('user', 'assistant') or type(row_id) is not int or row_id <= 0:
        return False
    row = db.get_session(session_id)
    if not row or not row.get('slack_sync'):
        return False
    target = _verified(db, row, auth_test or _auth_test)
    if not target:
        return False

    def claim(conn):
        message = conn.execute(
            "SELECT content FROM messages WHERE id=? AND session_id=? AND role=? "
            "AND active=1 AND COALESCE(display_kind,'') != 'hidden' AND content IS NOT NULL",
            (row_id, session_id, role)).fetchone()
        if not message or not message['content'].strip():
            return None
        # Recheck opt-in inside the write transaction; a concurrent toggle cannot
        # authorize a post based on an earlier stale read.
        active = conn.execute('SELECT slack_sync, slack_sync_revoking FROM sessions WHERE id=?', (session_id,)).fetchone()
        if not active or not active[0] or active[1]:
            return None
        result = conn.execute('INSERT OR IGNORE INTO slack_desktop_mirror_claims (message_id) VALUES (?)', (row_id,))
        return message['content'] if result.rowcount == 1 else None

    text = db._execute_write(claim)
    if text is None:
        return False
    if role == 'user':
        text = f'[Desktop] {text}'

    try:
        with _session_delivery_lock(db, session_id):
            # No state.db write transaction spans network I/O. This read must
            # happen AFTER acquiring the per-session lease: completed opt-out
            # then necessarily precedes the send or waits for it to finish.
            active = db._read_one('SELECT slack_sync, slack_sync_revoking FROM sessions WHERE id=?', (session_id,))
            if not active or not active[0] or active[1]:
                return False
            try:
                response = (send or _send)(*target, text)
                return response.get('ok') is True
            except Exception:
                logger.warning('Slack mirror send outcome unknown; durable claim prevents retry')
                return False
    except Exception:
        logger.warning('Slack mirror delivery lease failed; durable claim prevents retry')
        return False
