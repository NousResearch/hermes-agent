"""Restore only private, profile-bound local policy into the existing authority."""
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
                raise RuntimeStoreError('admission_conflict')
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
    else:
        with store._lock:
            current = store._entry_locked(route)
            if current is None or current.session_id not in receipt.get('lineage', [sid]):
                raise RuntimeStoreError('admission_conflict')
            current.session_id = target
    return SessionRef(authority.profile_id, sid)


def transcript_target(authority, ref):
    """The current physical transcript of logical *ref* (a local receipt's entry, else the tip)."""
    live = authority.sessions[ref.session_id]
    if live.source is not None and live.source.platform == Platform.LOCAL:
        restore_local_session(authority, ref.session_id)
        return local_receipt(authority.db, ref.session_id)['entry']['session_id']
    return authority.physical_target(ref)


def local_history(authority, ref):
    """Display history from the current physical transcript; *ref* stays the logical root.

    Each row carries its durable ``messages.id`` as ``row_id`` (the ``TranscriptMessage`` field
    Desktop hydration and the legacy ``session.history`` already use): a ``session.mutate`` rewind
    addresses its ``target_message_id`` by it, so a snapshot without it leaves no way to undo/retry.
    """
    rows = authority.db.get_messages_as_conversation(transcript_target(authority, ref), include_row_ids=True)
    return [{**{k: v for k, v in row.items() if k != '_row_id'}, 'row_id': row['_row_id']}
            if '_row_id' in row else row for row in rows]


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


def local_receipt_ids(authority):
    with authority.db._read_ctx() as conn:
        return [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            'SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',))]


def recover_local_sessions(authority, *, schedule=False):
    for sid in local_receipt_ids(authority):
        try:
            ref = restore_local_session(authority, sid)
            pending = list_session_admissions(authority.db, session_id=sid)
            if schedule and pending and not any(row['status'] == 'unknown' for row in pending):
                authority._schedule(ref)
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
