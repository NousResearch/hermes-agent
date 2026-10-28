"""Temporary chats on the canonical gateway (``session.create {temporary: true}``).

The row is stamped temporary + hidden right after its creation commit (``model_config.temporary``
owned by this gateway process: ``hermes_state_temporary``), and every agent built for it runs with
``temporary=True`` (no memory, recall or title). It is deleted:

- when its last viewer detaches with a close reason (TUI/CLI exit, ``/new``) — not on a socket drop
  or another backend-side reclaim, where the client may come straight back;
- by the owner's sweep once nobody has viewed it for ``TEMPORARY_UNVIEWED_TTL_SECONDS`` (a client
  that crashed or quit without detaching: Desktop keeps a chat across navigation, so it detaches
  only by deleting it);
- at startup, when the gateway process that owned it is gone (crash, restart, update).
"""
import asyncio
import contextlib
import inspect
import logging
import time
from pathlib import Path

from hermes_state_runtime import RuntimeStoreError

logger = logging.getLogger(__name__)

# Close reasons a client sends when the chat is really over; anything else keeps the row.
TEMPORARY_CLOSE_REASONS = frozenset({'tui_close', 'tui_shutdown', 'cli_close', 'cli_exit', 'new_session',
                                     'desktop_close'})
TEMPORARY_UNVIEWED_TTL_SECONDS = 30 * 60
TEMPORARY_SWEEP_INTERVAL_SECONDS = 60


def pop_temporary_flag(params, *, titled=False):
    """``temporary`` off the create params (it is not launch policy); a bool or invalid_params."""
    temporary = params.pop('temporary', False)
    if type(temporary) is not bool:
        raise RuntimeStoreError('invalid_params', field='temporary', detail='must be a boolean')
    if temporary and titled:
        # A titled create may resolve to an EXISTING saved conversation: never adopt one as temporary.
        raise RuntimeStoreError('invalid_params', field='temporary', detail='cannot be combined with title')
    return temporary


def mark_temporary(db, session_id):
    """Stamp a just-created row temporary + hidden, owned by this gateway process."""
    row = db.get_session(session_id)
    if row is None:
        raise RuntimeStoreError('not_found')
    if row.get('message_count') and not db.is_session_temporary(session_id):
        raise RuntimeStoreError('invalid_params', field='temporary', detail='the session already holds a saved conversation')
    db.mark_session_temporary(session_id)


def refuse_temporary_branch(db, session_id):
    """A branch would copy the temporary transcript into a normal, saved conversation."""
    if session_is_temporary(db, session_id):
        raise RuntimeStoreError('temporary_chat_cannot_branch')


def session_is_temporary(db, session_id):
    """Sync 'is this session's row temporary'. Reads THROUGH the async door
    (``AsyncSessionDB._db``, same unwrap as ``session_execution._session_is_temporary``):
    an un-awaited door call yields a truthy coroutine, never a yes/no, so an unresolvable
    answer reads as a normal chat (fail-open for logging; callers protecting a transcript
    pass through a real store)."""
    with contextlib.suppress(Exception):
        if not session_id:
            return False
        db = getattr(db, "_db", db)
        result = db.is_session_temporary(session_id)
        if inspect.iscoroutine(result):
            result.close()
            return False
        return bool(result)
    return False


def inbound_log_previews(db, source, event):
    """``(text_preview, reply_preview)`` for the inbound message line: a temporary chat's words
    never reach the logs. CLI/TUI/local chat ids ARE session ids, so the temporary marker is
    decidable here, before session resolution; other platforms' chat ids never name a session
    (the lookup falls through), and an unresolvable store answer logs normally.
    """
    text = (event.text or "")[:80].replace("\n", " ")
    reply = (getattr(event, "reply_to_text", None) or "")[:80].replace("\n", " ")
    if session_is_temporary(db, getattr(source, "chat_id", None)):
        return "<temporary chat>", "<temporary chat>"
    return text, reply


def _sessions_dir():
    from hermes_constants import get_hermes_home
    return Path(get_hermes_home()) / 'sessions'


def _viewed(authority, session_id):
    live = authority.sessions.get(session_id)
    return live is not None and bool(live.subscribers)


def delete_unviewed_temporary(authority, session_id, *, viewed_ok=False):
    """Off the loop: delete *session_id* when it is a temporary chat nobody views (any viewer, with
    *viewed_ok*: shutdown) and no turn is writing (an active turn lease refuses; the sweep
    retries). Returns the deleted ids."""
    from hermes_state_errors import SessionActiveWriteGuardError
    db = authority.db
    if (not viewed_ok and _viewed(authority, session_id)) or not session_is_temporary(db, session_id):
        return []
    ids = [session_id]
    with contextlib.suppress(Exception):
        from hermes_state_local import local_receipt
        ids.append(local_receipt(db, session_id)['entry']['session_id'])
    # The scrubbed delete (FTS5 secure-delete + WAL reset) when the store has it; a managed worker's
    # proxy store falls back to the plain chain delete (its backing writer still zero-fills).
    scrubbed = getattr(db, "delete_temporary_session", None)
    try:
        if scrubbed is not None:
            deleted = scrubbed(session_id, sessions_dir=_sessions_dir(), exclude_active_write_guards=True)
        else:
            deleted = db.delete_session(session_id, sessions_dir=_sessions_dir(), exclude_active_write_guards=True,
                                        include_compression_chain=True)
    except SessionActiveWriteGuardError:
        return []
    return sorted(set(ids)) if deleted else []


async def retire_deleted(authority, deleted_ids):
    """On the owner loop: drop the deleted chat's live session, route and cached agent."""
    if deleted_ids:
        from gateway.session_mutations import retire_live_sessions
        retire_live_sessions(authority, deleted_ids)
    return bool(deleted_ids)


async def close_if_temporary(authority, session_id, end_reason):
    """``session.detach`` from the last viewer. True when this was a temporary chat (deleted, or
    kept for the sweep): the caller then skips the normal end-of-session stamp."""
    if not await asyncio.to_thread(session_is_temporary, authority.db, session_id):
        return False
    if end_reason in TEMPORARY_CLOSE_REASONS:
        from functools import partial
        from gateway.session_runtime_workers import tracked_write
        await tracked_write(authority, partial(delete_unviewed_temporary, authority, session_id),
                            then=partial(retire_deleted, authority))
    return True


def _owned_here(owner):
    """True when *owner* (a row's ``model_config.temporary`` record) names this process."""
    from hermes_state_temporary import temporary_owner_record
    if not isinstance(owner, dict):
        return False
    mine = temporary_owner_record()
    if any(owner.get(k) != mine[k] for k in ('pid', 'host', 'pidns')):
        return False
    stamped, started = owner.get('create_time'), mine['create_time']
    return stamped is None or started is None or abs(float(stamped) - float(started)) < 2.0


def _temporary_ids(db):
    """Temporary chats this gateway may delete: its own, and those whose owner provably exited.
    A chat another live process owns (a terminal ``hermes --temporary``, a second gateway on the
    profile) is never this process's to remove, viewed here or not."""
    from hermes_state_sessions import _parse_model_config
    from hermes_state_temporary import TEMPORARY_MODEL_CONFIG_KEY, owner_is_gone
    ids = []
    for row in db._temporary_session_rows():
        owner = _parse_model_config(row.get('model_config')).get(TEMPORARY_MODEL_CONFIG_KEY)
        if _owned_here(owner) or owner_is_gone(owner) is True:
            ids.append(row['id'])
    return ids


async def sweep_unviewed(authority, unviewed_since, now=None):
    """One pass: start the clock for each unviewed temporary chat, delete those past the TTL."""
    now = time.monotonic() if now is None else now
    ids = await asyncio.to_thread(_temporary_ids, authority.db)
    for stale in set(unviewed_since) - set(ids):
        unviewed_since.pop(stale, None)
    for sid in ids:
        if _viewed(authority, sid):
            unviewed_since.pop(sid, None)
            continue
        since = unviewed_since.setdefault(sid, now)
        if now - since >= TEMPORARY_UNVIEWED_TTL_SECONDS:
            deleted = await asyncio.to_thread(delete_unviewed_temporary, authority, sid)
            if await retire_deleted(authority, deleted):
                unviewed_since.pop(sid, None)
                logger.info('Removed temporary chat %s (unviewed for %ds)', sid, TEMPORARY_UNVIEWED_TTL_SECONDS)


async def _sweeper(authority):
    # Startup: chats a previous (now dead) gateway owned are orphans; this process owns no chat yet.
    with contextlib.suppress(Exception):
        await asyncio.to_thread(authority.db.sweep_orphaned_temporary_sessions, sessions_dir=_sessions_dir())
    unviewed_since = {}
    while not getattr(authority, 'retired', False):
        await asyncio.sleep(TEMPORARY_SWEEP_INTERVAL_SECONDS)
        try:
            await sweep_unviewed(authority, unviewed_since)
        except Exception:  # health: allow BLE001 -- a failed pass is retried next interval; never kills the owner
            logger.warning('temporary-chat sweep failed', exc_info=True)


def start_temporary_sweeper(authority):
    """Owner bring-up: the startup orphan sweep, then the unviewed-chat sweep on this loop."""
    task = asyncio.get_running_loop().create_task(_sweeper(authority))
    authority._temporary_sweeper = task
    return task


def delete_all_unviewed(authority):
    """Gateway shutdown: nothing this process owns may outlive it on disk (other owners' chats stay)."""
    deleted = []
    for sid in _temporary_ids(authority.db):
        with contextlib.suppress(Exception):
            deleted += delete_unviewed_temporary(authority, sid, viewed_ok=True)
    return deleted
