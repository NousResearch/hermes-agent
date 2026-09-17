"""Restore only private, profile-bound local policy into the existing authority."""
import hashlib
import json
import logging
from pathlib import Path

from gateway.config import Platform
from gateway.session import SessionEntry, SessionSource
from gateway.session_contract import SessionRef
from hermes_state_local import POLICY_PREFIX, local_receipt
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def local_identity(profile_id, principal_id, request_id):
    identity = json.dumps([profile_id, principal_id, request_id], separators=(',', ':'))
    return 'local-' + hashlib.sha256(identity.encode()).hexdigest()


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


def restore_local_session(authority, sid):
    from gateway.session_authority import LiveSession
    from gateway.session_local import LocalSessionAdapter
    from gateway.session_policy import restore_policy
    receipt = local_receipt(authority.db, sid)
    try:
        if receipt['profile_id'] != authority.profile_id:
            raise RuntimeStoreError('profile_mismatch')
        chat_id = (receipt['legacy_session_id'] if 'legacy_session_id' in receipt else
                   local_identity(receipt['profile_id'], receipt['principal_id'], receipt['request_id']))
        if receipt['session_id'] != sid or sid != chat_id:
            raise ValueError('identity mismatch')
        source = local_source(authority, chat_id, receipt['principal_id'])
        store = authority.runner.session_store
        route = store._generate_session_key(source)
        if receipt['route'] != route:
            raise ValueError('route mismatch')
        policy = restore_policy(receipt['policy'])
        entry = SessionEntry.from_dict(receipt['entry'])
        target = entry.session_id
        if (entry.session_key != route
                or entry.origin.to_dict() != source.to_dict()):
            raise ValueError('entry mismatch')
        row = authority.db.get_session(target)
        if row is None or (row['session_key'], row['chat_id'], row['user_id']) != (route, chat_id, source.user_id):
            raise ValueError('stored identity mismatch')
        from hermes_state_local_lineage import validate_local_lineage
        with authority.db._read_ctx() as conn:
            validate_local_lineage(conn, receipt)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        if isinstance(exc, RuntimeStoreError):
            raise
        raise RuntimeStoreError('storage_unavailable') from exc
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


def local_history(authority, ref):
    """Display history from the current physical transcript; *ref* stays the logical root."""
    live = authority.sessions[ref.session_id]
    if live.source is not None and live.source.platform == Platform.LOCAL:
        restore_local_session(authority, ref.session_id)
        target = local_receipt(authority.db, ref.session_id)['entry']['session_id']
    else:
        target = authority.physical_target(ref)
    return authority.db.get_messages_as_conversation(target)


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


def recover_local_sessions(authority, *, schedule=False):
    with authority.db._read_ctx() as conn:
        ids = [row[0][len(POLICY_PREFIX):] for row in conn.execute(
            'SELECT key FROM state_meta WHERE key LIKE ?', (POLICY_PREFIX + '%',))]
    for sid in ids:
        try:
            ref = restore_local_session(authority, sid)
            pending = list_session_admissions(authority.db, session_id=sid)
            if schedule and pending and not any(row['status'] == 'unknown' for row in pending):
                authority._schedule(ref)
        except RuntimeStoreError as exc:
            logging.getLogger(__name__).warning('Local session %s paused: %s', sid, exc.reason)
