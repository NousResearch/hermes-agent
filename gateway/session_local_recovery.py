"""Restore only private, profile-bound local policy into the existing authority."""
import asyncio
import contextlib
import json
import logging
from pathlib import Path

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_contract import SessionRef
from hermes_state_keys import local_identity
from hermes_state_local import POLICY_PREFIX, local_receipt
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def local_adapter_map(authority):
    """The adapter map a LOCAL session of *authority* lives in: ``runner.adapters`` for the launch
    profile, ``runner._profile_adapters[name]`` for a served secondary (its routed home, so
    ``_adapter_for_source`` / ``_adapters_for_profile`` resolve the right transport)."""
    runner = authority.runner
    registry = getattr(runner, 'session_authorities', None)
    name = registry.profile_name(authority) if registry is not None else None
    if name is None:
        return runner.adapters
    profile_adapters = getattr(runner, '_profile_adapters', None)
    if profile_adapters is None:
        profile_adapters = runner._profile_adapters = {}
    return profile_adapters.setdefault(name, {})


def local_source(authority, chat_id, user_id):
    """LOCAL SessionSource for a session owned by *authority*. A served secondary's local sessions
    carry its profile so ``build_session_key`` namespaces them (``agent:<name>:local:...``) and the
    SessionStore writes their rows to that profile's ``state.db``, not the launch store."""
    source = SessionSource(platform=Platform.LOCAL, chat_id=chat_id, user_id=user_id, chat_type='dm')
    registry = getattr(authority.runner, 'session_authorities', None)
    name = registry.profile_name(authority) if registry is not None else None
    if name is not None:
        source.profile = name
    return source


def _refused_check(exc):
    """``<reason or error> at <function>:<line>``: where a receipt check raised, and why, followed
    through every chained ``__cause__`` (``restore_policy`` and ``local_receipt`` wrap the check
    that failed into a bare ``storage_unavailable``)."""
    import traceback
    frames = traceback.extract_tb(exc.__traceback__)
    cause = exc.reason if isinstance(exc, RuntimeStoreError) else f'{type(exc).__name__}: {exc}'
    where = f' at {frames[-1].name}:{frames[-1].lineno}' if frames else ''
    chained = f', caused by {_refused_check(exc.__cause__)}' if exc.__cause__ is not None else ''
    return f'{cause}{where}{chained}'


class _ReceiptRefused(RuntimeStoreError):
    """A refused receipt ``restore_local_session`` already logged with its failing check."""


class _CatchUpPending(RuntimeStoreError):
    """The receipt's compression catch-up is committing in a worker; this restore is refused."""


def _with_untracked_compression(authority, sid, receipt):
    """*receipt*, after recording compression steps an older build published past its tip (#140060).

    The catch-up writes, so it never runs on the owner loop: a contended writer there would stall every
    profile the loop serves. The async restore paths commit it in a worker first
    (``session_local_adoption``); a sync restore on the loop that still finds it hands it to a tracked
    worker, which publishes the session once it commits, and refuses this read.
    """
    from hermes_state_local_lineage import adopt_untracked_compression, adoptable_compression
    with authority.db._read_ctx() as conn:
        if adoptable_compression(conn, receipt) is None:
            return receipt
    from gateway.session_local_adoption import on_owner_loop, schedule_adoption
    if on_owner_loop():
        schedule_adoption(authority, sid, receipt)
        raise _CatchUpPending('storage_unavailable')
    adopt_untracked_compression(authority.db, epoch=authority.epoch, session_id=sid, expected=receipt)
    return local_receipt(authority.db, sid)


def _checked_entry(authority, receipt, source, route):
    """*receipt*'s entry, once its route, origin and stored identity are this authority's."""
    if receipt['route'] != route:
        raise ValueError('route mismatch')
    entry = SessionEntry.from_dict(receipt['entry'])
    # Older local receipts omitted the served profile from the origin. The receipt's
    # home and route already matched this authority; fill only that absent field,
    # never accept an explicitly different profile or change the durable route.
    if 'profile' not in receipt['entry']['origin']:
        entry.origin.profile = source.profile
    if (entry.session_key != route
            or entry.origin.to_dict() != source.to_dict()):
        raise ValueError('entry mismatch')
    row = authority.db.get_session(entry.session_id)
    if row is None or (row['session_key'], row['chat_id'], row['user_id']) != (route, source.chat_id, source.user_id):
        raise ValueError('stored identity mismatch')
    return entry


def owned_receipt(authority, sid):
    """*sid*'s stored receipt with its LOCAL source, route and entry, once all are this owner's."""
    receipt = local_receipt(authority.db, sid)
    if receipt['profile_id'] != authority.profile_id:
        raise RuntimeStoreError('profile_mismatch')
    chat_id = (receipt['legacy_session_id'] if 'legacy_session_id' in receipt else
               local_identity(receipt['profile_id'], receipt['principal_id'], receipt['request_id']))
    if receipt['session_id'] != sid or sid != chat_id:
        raise ValueError('identity mismatch')
    source = local_source(authority, chat_id, receipt['principal_id'])
    route = authority.runner.session_store._generate_session_key(source)
    return receipt, source, route, _checked_entry(authority, receipt, source, route)


def unused_session(conn, session_id):
    """*session_id* holds nothing: no transcript, ledger rows or children (what the SessionStore's
    stale-route heal, #54878, mints as a route's replacement before any turn runs on it)."""
    return conn.execute(
        'SELECT 1 FROM messages WHERE session_id=? UNION ALL SELECT 1 FROM session_admissions '
        'WHERE target_session_id=? UNION ALL SELECT 1 FROM worker_executions WHERE session_id=? '
        'UNION ALL SELECT 1 FROM sessions WHERE parent_session_id=? LIMIT 1',
        (session_id, session_id, session_id, session_id)).fetchone() is None


def _stray_route_target(authority, receipt, session_id):
    """*session_id* is a replacement the SessionStore's stale-route self-heal minted on this local
    route before #140642: outside the receipt's lineage, never admitted, no transcript. The receipt
    is the route's truth, so restoring it drops that entry instead of refusing the conversation
    forever (``admission_conflict`` on every resume). Anything with history or ledger rows still
    refuses: it is not provably a stray."""
    if not session_id or session_id in receipt.get('lineage', [receipt['session_id']]):
        return False
    with authority.db._read_ctx() as conn:
        if not unused_session(conn, session_id):
            return False
    logging.getLogger(__name__).warning(
        'Local route for %s named an empty replacement %s outside its lineage; restoring the receipt target',
        receipt['session_id'], session_id)
    return True


def restore_local_session(authority, sid):
    from gateway.session_authority import LiveSession
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import restore_policy
    try:
        # The catch-up writes, so the receipt as stored must already be this owner's (#140060).
        receipt, source, route, entry = owned_receipt(authority, sid)
        chat_id, store = source.chat_id, authority.runner.session_store
        adopted = _with_untracked_compression(authority, sid, receipt)
        policy = restore_policy(adopted['policy'])
        if adopted is not receipt:  # re-read after a catch-up: check the copy that will be served
            receipt, entry = adopted, _checked_entry(authority, adopted, source, route)
        target = entry.session_id
        from hermes_state_local_lineage import validate_local_lineage
        with authority.db._read_ctx() as conn:
            validate_local_lineage(conn, receipt)
    except _CatchUpPending:
        raise
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        # The client only sees the bounded code: gateway.log names the check that refused (#140099).
        logging.getLogger(__name__).warning('Local session %s receipt refused: %s', sid, _refused_check(exc))
        if isinstance(exc, RuntimeStoreError):
            raise _ReceiptRefused(exc.reason, field=exc.field, detail=exc.detail) from exc
        raise _ReceiptRefused('storage_unavailable') from exc
    adapters = local_adapter_map(authority)
    adapter = adapters.get(Platform.LOCAL)
    if adapter is None:
        adapter = LocalSessionAdapter(authority)
        adapters[Platform.LOCAL] = adapter
    if not isinstance(adapter, LocalSessionAdapter) or adapter.authority is not authority:
        raise RuntimeStoreError('runtime_draining')
    live = authority.sessions.get(sid)
    if live is None:
        with store._lock:
            store._ensure_loaded_locked()
            current = store._entries.get(route)
            if current is not None and current.session_id != target:
                if not _stray_route_target(authority, receipt, current.session_id):
                    raise RuntimeStoreError('admission_conflict')
                current = None
            # The receipt is the creation copy; a persisted entry for the same target keeps the
            # route's explicit ``/yolo`` toggle (an OFF over a ``--yolo`` launch must survive a
            # cold restore). An untoggled entry stays None and keeps following the launch policy.
            if current is not None:
                entry.yolo = current.yolo
            entry.origin = source
            store._entries[route] = entry
        adapter.policies[chat_id] = policy
        authority.sessions[sid] = LiveSession(source, route)
        adapter.register_source(source)
    elif (not adapter.authorize_source(live.source)
          or live.source.to_dict() != source.to_dict()
          or adapter.policies.get(chat_id) != policy):
        raise RuntimeStoreError('storage_unavailable')
    elif _route_already_restored(store, route, target, receipt.get('lineage', [sid])):
        pass  # every claim of a live session lands here: no write, so no wait on the store lock
    else:
        with store._lock:
            current = store._entry_locked(route)
            if current is None:
                # A route the loaded store cannot see (an older release's scope, a pruned row) is the
                # receipt's to rebuild, as on a cold restore: refusing wedges every claim until a restart.
                entry.origin = source
                store._entries[route] = entry
            elif (current.session_id not in receipt.get('lineage', [sid])
                    and not _stray_route_target(authority, receipt, current.session_id)):
                raise RuntimeStoreError('admission_conflict')
            else:
                current.session_id = target
    return SessionRef(authority.profile_id, sid)


def _route_already_restored(store, route, target, lineage):
    """The loaded route already names *target*, a lineage member: the locked branch would only
    re-assign the same id. Read without ``store._lock`` (one dict lookup, atomic under the GIL),
    because worker threads hold that lock across state.db writes and the restore runs on the loop.
    Any other shape takes the lock and is re-checked there."""
    current = store._entries.get(route)
    return current is not None and current.session_id == target and target in lineage


def transcript_target(authority, ref):
    """The current physical transcript of logical *ref* (a local receipt's entry, else the tip)."""
    live = authority.sessions[ref.session_id]
    if live.source is not None and live.source.platform == Platform.LOCAL:
        restore_local_session(authority, ref.session_id)
        return local_receipt(authority.db, ref.session_id)['entry']['session_id']
    return authority.physical_target(ref)


# A resume reply's rows reach Ink twice (``messages`` and ``display_messages``). Archived rows are display-only
# and fill only what the live rows leave of this budget, so they keep the reply well under the gateway clients'
# 8 MiB WebSocket frame cap (``hermes_cli/gateway_client.py``) and never push a reply main could send past it.
# Live rows are never dropped.
RESUME_HISTORY_BYTES = 3 * 1024 * 1024


def display_rows(db, target):
    """``(live row ids, display rows, truncated)`` of physical *target*, the live ids read first
    (``local_history``'s fence). The display rows are the compression lineage as main's resume and the
    REST transcript show it: every live row, and the newest compaction-archived rows that fit what the
    live rows leave of ``RESUME_HISTORY_BYTES`` (``truncated`` when older ones were left out). A lineage
    over the resume cap shows only the live rows."""
    from hermes_state import SessionResumeTooLargeError
    live_rows = db.get_messages_as_conversation(target, include_row_ids=True)
    live = {row['_row_id'] for row in live_rows}
    try:
        db.assert_resume_safe(target)
    except SessionResumeTooLargeError:
        return live, [], False
    kept, truncated = _fit_archive(live_rows, db.get_messages_as_conversation(
        target, include_ancestors=True, include_compacted=True, include_row_ids=True))
    return live, kept, truncated


def _fit_archive(live_rows, display):
    """``(display rows kept, truncated)``: every live row of *display*, and its newest archived rows that fit
    what *live_rows* leave of ``RESUME_HISTORY_BYTES``."""
    live = {row['_row_id'] for row in live_rows}
    kept, used, truncated = [], sum(len(json.dumps(row, default=str)) for row in live_rows), False
    for row in reversed(display):
        if row['_row_id'] not in live:
            used += len(json.dumps(row, default=str))
            truncated = truncated or used > RESUME_HISTORY_BYTES
            if truncated:
                continue
        kept.append(row)
    return kept[::-1], truncated


async def prepare_history(authority, ref):
    """``local_history``'s lineage read, off the owner loop; the locked part re-reads only the live tip."""
    target = transcript_target(authority, ref)
    return target, await asyncio.to_thread(display_rows, authority.db, target)


def omitted_history_state(db, target):
    """``(todo_state, message_count)`` of a resume that omits its transcript: the physical *target*'s stored
    count and its todo plan. Skips the display-row lineage read; the todo lookup still walks the lineage's
    todo rows (removed ones included, so an undone plan stays gone)."""
    from tui_gateway.tool_progress import _todo_state_from_db
    row = db.get_session(target)
    if row is None:  # deleted since the attach resolved it
        raise RuntimeStoreError('not_found')
    return _todo_state_from_db(db, target), int(row.get('message_count') or 0)


def _merge_live_in_stored_order(display, live):
    """The live rows in stored (``row_id``) order, each archived display row placed before the live
    row it preceded in display order.

    Display deduplication keeps a carried message at its FIRST copy's position, so after a rotated
    compression the copied tail sits before the new summary. Clients take the live rows of a resume
    as the transcript (the CLI's ``conversation_history``, ``/undo``'s prefix, rollback, ACP replay),
    so the live subsequence must be the stored one; archived rows only anchor between live rows."""
    current = {row['_row_id'] for row in live}
    groups, pending, shown = [], [], 0
    for row in display:
        if row['_row_id'] in current:
            groups.append(pending)
            pending, shown = [], shown + 1
        else:
            pending.append({**row, 'archived': True})
    shown_ids = {row['_row_id'] for row in display} & current
    rows, seen = [], 0
    for row in sorted(live, key=lambda r: r['_row_id']):
        if row['_row_id'] in shown_ids:
            rows.extend(groups[seen])
            seen += 1
        rows.append(row)
    return rows + pending


def local_history(authority, ref, prepared=None):
    """Display history of logical *ref*: every stored row the user saw, in display order.

    A row outside the current physical transcript's active set (summarized away by a compression)
    carries ``archived: true``: it is shown, never live, so no undo, retry or rewind may target it.
    The rows without it are exactly the live transcript's, so a client choosing among live rows sees
    what it saw before. That includes a micro-compaction merge (``display_metadata.model_only``): its
    originals are the archived rows renderers show, so renderers skip the merge itself.

    Each row carries its durable ``messages.id`` as ``row_id`` (the ``TranscriptMessage`` field
    Desktop hydration and the legacy ``session.history`` already use): a ``session.mutate`` rewind
    addresses its ``target_message_id`` by it, so a snapshot without it leaves no way to undo/retry.
    *prepared* (``prepare_history``) is kept while every row it read as live still is; a rewind or
    compaction since then re-reads the lineage here. When the byte budget left older archived rows out
    (``display_rows``), the first row carries ``archived_truncated: true``.
    """
    db = authority.db
    target = transcript_target(authority, ref)
    read_live, display, truncated = (prepared[1] if prepared is not None and prepared[0] == target
                                     else display_rows(db, target))
    live = db.get_messages_as_conversation(target, include_row_ids=True)
    live_ids = {row['_row_id'] for row in live}
    if not read_live <= live_ids:
        read_live, display, truncated = display_rows(db, target)
        live = db.get_messages_as_conversation(target, include_row_ids=True)
    elif live_ids - read_live:
        # Rows appended since the prepare count against the archive budget too (no lineage re-read).
        display, cut = _fit_archive(live, display)
        truncated = truncated or cut
    rows = [{**{k: v for k, v in row.items() if k != '_row_id'}, 'row_id': row['_row_id']}
            for row in _merge_live_in_stored_order(display, live)]
    if truncated and rows:
        rows[0]['archived_truncated'] = True  # earlier archived history was left out (the byte budget)
    return rows


def reopen_local_session(authority, ref):
    """The first admitted turn makes a finalized local session live again (#85303).

    Mounting (``session.resume``) is a read and leaves ``ended_at``/``end_reason`` alone; the drain
    calls this after the turn's local preflight, before execution routes it. A TUI shutdown / WS
    disconnect / idle eviction stamps ``end_reason`` on the physical row. ``SessionStore`` reads any
    stamped row as a stale route (#54878) and, since those reasons are not recoverable, answers the
    turn with a FRESH session — the client keeps waiting on an id that never emits again. The classic
    surfaces clear the stamp on the first real turn (tui_gateway ``_reopen_if_finalized``); the
    authority must too. Only the lineage tip is reopened: a
    compression parent or reset predecessor is never the physical target.
    """
    live = authority.sessions.get(ref.session_id)
    if live is None or live.source is None or live.source.platform != Platform.LOCAL:
        return
    target = local_receipt(authority.db, ref.session_id)['entry']['session_id']
    row = authority.db.get_session(target)
    if row is not None and row.get('end_reason') is not None:
        authority.db.reopen_session(target)


def reset_local_session(store, old_entry, session_id, now, display_name):
    from hermes_state_local_lineage import reset_local_target
    entry = SessionEntry(old_entry.session_key, session_id, now, now,
        origin=old_entry.origin, platform=old_entry.platform, chat_type=old_entry.chat_type,
        display_name=display_name if display_name is not None else old_entry.display_name,
        is_fresh_reset=True)
    db = store._db_for_key(old_entry.session_key)
    epochs = getattr(store, '_local_authority_epochs', None) or {}
    epoch = epochs.get(Path(db.db_path).resolve(), store._local_authority_epoch) if db is not None else store._local_authority_epoch
    reset_local_target(db, epoch=epoch,
                       parent_session_id=old_entry.session_id, entry=entry.to_dict())
    store._entries[old_entry.session_key] = entry
    return entry


# The receipts boot restores: an open row (or one compressed away, for its catch-up), unsettled input
# or a stored close. An ended session keeps its receipt and stays resumable; ``authorize`` restores it
# on first touch, so the live registry no longer grows with every session ever opened (#143547 LC-36).
_BOOT_RECEIPTS_SQL = (
    "SELECT m.key FROM state_meta m LEFT JOIN sessions s "
    "ON s.id = CASE WHEN json_valid(m.value) THEN json_extract(m.value, '$.entry.session_id') END "
    "WHERE m.key LIKE ? AND (s.ended_at IS NULL OR s.end_reason = 'compression' "
    "OR EXISTS (SELECT 1 FROM session_admissions a WHERE a.target_session_id = substr(m.key, ?) "
    "AND a.status IN ('queued','started','unknown')) "
    "OR EXISTS (SELECT 1 FROM state_meta c WHERE c.key = ? || substr(m.key, ?)))")


def restore_completion_route(runner, session_key):
    """Background work that outlived its local session reports after a restart: boot left that ended
    session unrestored, so restore it first, as ``authorize`` does on a viewer's first touch, and its
    completion is noted on it as before. Anything else is left to the delivery's own checks."""
    from gateway.run import _parse_session_key
    from gateway.session_authorities import active_authority
    authority, parsed = active_authority(runner), _parse_session_key(session_key) or {}
    if authority is None or parsed.get('platform') != Platform.LOCAL.value or parsed['chat_id'] in authority.sessions:
        return
    with contextlib.suppress(RuntimeStoreError):
        restore_local_session(authority, parsed['chat_id'])


def release_handed_off_receipts(db, *, epoch):
    """Boot heal for #143286: an older build's ``/handoff`` (or a platform ``/resume``) moved a local
    session onto a messaging lane but kept its local receipt, so every open, pin or rename refused it
    (``stored identity mismatch``) and its title never resolved. Retire each such receipt, so the row
    restores from its route like any other platform session, and move the rest of its conversation
    there with it (``hand_off_conversations``). Only receipts whose CURRENT target a messaging lane
    now holds: any other refused receipt stays as it is, logged when it is touched."""
    from gateway.session_route_recovery import _messaging_platform
    from hermes_state_local import hand_off_conversations, moved_receipts
    from hermes_state_runtime import _epoch

    def write(conn):
        _epoch(conn, epoch)
        moved = {owner: (target, source) for owner, (target, source) in moved_receipts(conn).items()
                 if _messaging_platform(source) is not None}
        return hand_off_conversations(conn, moved)
    for owner, (target, source) in db._execute_write(write).items():
        logging.getLogger(__name__).info(
            'Local session %s was handed off to %s (row %s); released its local receipt', owner, source, target)


def local_receipt_ids(authority):
    from gateway.session_viewer_close import CLOSE_PREFIX
    start = len(POLICY_PREFIX) + 1
    with authority.db._read_ctx() as conn:
        return [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            _BOOT_RECEIPTS_SQL, (POLICY_PREFIX + '%', start, CLOSE_PREFIX, start))]


def recover_local_sessions(authority, *, schedule=False):
    orphans = []
    for sid in local_receipt_ids(authority):
        try:
            ref = restore_local_session(authority, sid)
            pending = list_session_admissions(authority.db, session_id=sid)
            if any(row['status'] == 'unknown' for row in pending):
                continue  # the drain pauses on it until its user's Discard
            if schedule and pending:
                authority._schedule(ref)
            orphans.append(sid)
        except (_ReceiptRefused, _CatchUpPending):
            continue  # logged: the check that refused it, or the catch-up handed to a worker
        except RuntimeStoreError as exc:
            logging.getLogger(__name__).warning('Local session %s paused: %s', sid, exc.reason)
    if schedule:
        # A last-viewer close the previous owner stopped before completing (#134889 LC-6b).
        from gateway.session_viewer_close import rearm_persisted_closes, settle_viewer_close
        from gateway.session_runtime_workers import track_mutation
        for sid in rearm_persisted_closes(authority):
            authority._schedule(SessionRef(authority.profile_id, sid))
            if any(row['status'] == 'unknown' for row in list_session_admissions(authority.db, session_id=sid)):
                # A hard-killed owner's turn: the drain pauses on it, so the close resolves it.
                track_mutation(authority, settle_viewer_close(authority, sid))
        from gateway.session_viewer_close import arm_startup_orphan_closes
        arm_startup_orphan_closes(authority, orphans)
