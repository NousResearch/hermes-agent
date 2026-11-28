import json
import os

import pytest

from agent.runtime_session_store import RuntimeSessionStore, WorkerPersistenceError


def test_outbox_freezes_failed_payload_and_reopens_exclusively(tmp_path):
    scope = {'session_id': 's', 'execution_id': 'w', 'epoch': 1}
    rows = [{'role': 'user', 'content': 'frozen'}]
    def unavailable(method, **params):
        rows[0]['content'] = 'caller changed'
        raise TimeoutError('offline')
    store = RuntimeSessionStore(unavailable, scope, tmp_path / 'private')
    try:
        with pytest.raises(TimeoutError):
            store.append_messages_batch('s', rows)
        disk = json.loads(store.path.read_text())
        assert disk['pending'][0]['payload']['messages'][0]['content'] == 'frozen'
        assert store.journal['pending'] == disk['pending']
        with pytest.raises(WorkerPersistenceError, match='outbox_in_use'):
            RuntimeSessionStore(unavailable, scope, tmp_path / 'private')
        with pytest.raises(WorkerPersistenceError, match='offline'):
            store.queue_token_counts('s', input_tokens=1)
    finally:
        with pytest.raises(WorkerPersistenceError, match='offline'):
            store.close()
    seen = []
    def commit(method, **params):
        seen.append(params)
        return {'count': 1, 'annotations': [{'_row_id': 42}]}
    restored = RuntimeSessionStore(commit, scope, tmp_path / 'private')
    try:
        assert restored.retry_pending() == []  # reopening already reconciled the acknowledged journal
        assert seen[0]['payload']['messages'][0]['content'] == 'frozen'
        assert json.loads(restored.path.read_text())['pending'] == []
    finally:
        restored.close()


@pytest.mark.platforms('posix')
def test_outbox_has_private_posix_mode(tmp_path):
    store = RuntimeSessionStore(lambda *a, **k: {}, {'session_id': 's'}, tmp_path / 'private')
    try:
        assert store.path.stat().st_mode & 0o077 == 0
    finally:
        store.close()


def _recording_rpc(seen, durable=None):
    """An owner stand-in that enforces the owner's sequence rule.

    The owner accepts a receipt only when its sequence is exactly one past the
    last one it acknowledged, and it returns the stored result for an exact
    retry. The store's sequence counter advances only over entries the journal
    admitted, so a refused row never gets a sequence — the owner cannot see
    that gap. ``durable`` records the transcript rows the owner accepted.
    """
    state = {'last': 0}

    def rpc(method, **params):
        if method != 'worker.persist':
            return {'value': True}
        seen.append(params)
        sequence = params['sequence']
        if sequence != state['last'] + 1:
            raise RuntimeError(f'invalid_params: sequence={sequence} last={state["last"]}')
        state['last'] = sequence
        operation = params.get('operation')
        payload = params.get('payload') or {}
        if operation == 'transcript.append':
            messages = payload.get('messages') or []
            if durable is not None:
                durable.extend(m.get('content') for m in messages)
            return {'count': len(messages), 'annotations': [None] * len(messages)}
        # Every other operation answers with the value its caller reads.
        return {'value': True, 'count': 0, 'annotations': []}

    return rpc


def test_outbox_capacity_refusal_releases_only_a_releasable_archive(tmp_path):
    """An oversized ``compression.archive`` must not latch the whole store.

    The archive is the one operation whose own caller keeps the original
    transcript when the write is refused (context_compressor_prune: "keeping
    the original transcript"), so refusing it loses nothing durable — but the
    latch it used to set also vetoed ``turn.renew``, the lease whose expiry
    hard-interrupted the whole turn.
    """
    seen = []
    durable = []
    rpc = _recording_rpc(seen, durable)
    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private', max_bytes=4096)
    try:
        before = store.path.read_bytes()
        sequence_before = store.journal['next_sequence']
        with pytest.raises(WorkerPersistenceError, match='outbox_full'):
            store._apply('compression.archive', {'messages': [{'role': 'user', 'content': 'x' * 8000}]})
        # The refusal left the journal untouched, so it advanced nothing and wrote nothing.
        assert store.path.read_bytes() == before
        assert store.journal['next_sequence'] == sequence_before
        assert store.journal['pending'] == []
        # ... and it did not latch the store.
        assert store.failure is None
        # The turn lease stays renewable, which is what the old latch broke.
        assert store.refresh_session_turn_lease('s', 'holder', ttl_seconds=300.0) is True
        assert [params['operation'] for params in seen] == ['turn.renew']
    finally:
        store.close()


def test_outbox_capacity_refusal_of_a_transcript_append_stays_sticky(tmp_path):
    """A refused ``transcript.append`` must keep failing loudly.

    The refused row never received a sequence, so the next append would take
    that same sequence and the owner would accept it: the row would be missing
    from the durable transcript with no error anywhere. The owner cannot
    detect that gap, so the store has to refuse to continue instead.
    """
    seen = []
    durable = []
    rpc = _recording_rpc(seen, durable)
    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private', max_bytes=4096)
    try:
        with pytest.raises(WorkerPersistenceError, match='outbox_full'):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'x' * 8000}])
        assert store.failure == 'outbox_full'
        # The store refuses to continue, so no later append can slip past the gap.
        with pytest.raises(WorkerPersistenceError, match='outbox_full'):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'small'}])
        assert durable == []
        assert not any(params.get('operation') == 'transcript.append' for params in seen)
        # The lease is lost with the transcript write: the session cannot persist
        # its own conversation, so ending the turn is the correct outcome.
        with pytest.raises(WorkerPersistenceError, match='outbox_full'):
            store.refresh_session_turn_lease('s', 'holder', ttl_seconds=300.0)
    finally:
        # close() flushes the usage counts, and the store is failed, so the flush
        # refuses — the same failure the caller sees.
        with pytest.raises(WorkerPersistenceError, match='outbox_full'):
            store.close()


def test_outbox_capacity_refusal_after_an_owner_acknowledgement_stays_sticky(tmp_path):
    """A failure while the receipt is journalled stays sticky.

    The owner already acknowledged the entry when ``retry_pending`` saves the
    receipt, so the entry is still pending on disk. Releasing the latch there
    would only turn the refusal into a ``pending_receipt`` on the next
    operation and hide the real cause.
    """
    def rpc(method, **params):
        if method != 'worker.persist':
            return {'value': True}
        # The owner answers, then the receipt cannot be journalled: the store is
        # out of room for the pending entry it still holds.
        raise RuntimeError('receipt could not be journalled')

    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private', max_bytes=4096)
    try:
        with pytest.raises(RuntimeError, match='receipt could not be journalled'):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'small'}])
        # The failure latched the store, so nothing else runs behind it.
        assert store.failure == 'receipt could not be journalled'
        with pytest.raises(WorkerPersistenceError, match='receipt could not be journalled'):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'small'}])
    finally:
        with pytest.raises(WorkerPersistenceError, match='receipt could not be journalled'):
            store.close()


def test_outbox_capacity_refusal_does_not_clear_an_earlier_failure(tmp_path):
    """An archive refusal must not wash out an earlier operation's failure."""
    def rpc(method, **params):
        raise TimeoutError('offline')

    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private', max_bytes=4096)
    try:
        with pytest.raises(TimeoutError):
            store.append_messages_batch('s', [{'role': 'user', 'content': 'small'}])
        assert store.failure == 'offline'
        with pytest.raises(WorkerPersistenceError, match='offline'):
            store._apply('compression.archive', {'messages': [{'role': 'user', 'content': 'x' * 8000}]})
        assert store.failure == 'offline'
    finally:
        with pytest.raises(WorkerPersistenceError, match='offline'):
            store.close()


def test_worker_turn_wait_callbacks_remain_local(tmp_path):
    attempts, notices = [], []
    def rpc(method, **params):
        attempts.append(params)
        return {'value': len(attempts) > 1}
    store = RuntimeSessionStore(rpc, {'session_id': 's'}, tmp_path / 'private')
    try:
        assert store.acquire_session_turn_lease('s', 'holder', wait_seconds=3,
            poll_interval_seconds=.01, on_wait=notices.append)
        assert notices and len(attempts) == 2
        assert all(set(row['payload']) == {'holder', 'ttl_seconds'} for row in attempts)
        assert not store.acquire_session_turn_lease('s', 'holder', should_abort=lambda: True)
    finally:
        store.close()


def test_outbox_filesystem_failure_is_sticky(tmp_path, monkeypatch):
    store = RuntimeSessionStore(lambda *a, **k: {}, {'session_id': 's'}, tmp_path / 'private')
    def fail_replace(*args):
        raise OSError('disk unavailable')
    monkeypatch.setattr('agent.runtime_session_store.os.replace', fail_replace)
    try:
        with pytest.raises(OSError, match='disk unavailable'):
            store.queue_token_counts('s', input_tokens=1)
        assert store.failure == 'disk unavailable'
    finally:
        store._outbox_owner.close()


def test_outbox_journal_survives_a_cross_device_rename(tmp_path, monkeypatch):
    """The journal publishes through utils.atomic_replace: a bind-mounted outbox whose rename
    fails EXDEV falls back to copy+fsync instead of failing every worker write, and the
    journal stays owner-only (POSIX mode bits; Windows has none to check)."""
    import errno
    import stat
    real_replace = os.replace
    def exdev_replace(src, dst):
        if str(dst).endswith('pending.json'):
            raise OSError(errno.EXDEV, 'cross-device link')
        return real_replace(src, dst)
    monkeypatch.setattr(os, 'replace', exdev_replace)
    store = RuntimeSessionStore(lambda *a, **k: {}, {'session_id': 's'}, tmp_path / 'private')
    try:
        journal = store.path
        assert json.loads(journal.read_text())['scope'] == {'session_id': 's'}
        if os.name == 'posix':
            assert stat.S_IMODE(journal.stat().st_mode) == 0o600
        assert [p.name for p in journal.parent.iterdir() if p.name.startswith('.pending-')] == []
    finally:
        store._outbox_owner.close()


def test_worker_store_persists_and_restores_the_tools_prefix_pin(tmp_path):
    """persist_agent_tool_names writes the pin through the worker facade and a rebuilt worker
    agent reads it back from the same row (F26). Without the facade method the pin stays NULL
    and a config flip between turns silently forks the cached tools[] prefix."""
    from types import SimpleNamespace
    from persistence_assembly import open_persistence, profile_scope
    from tests.fixtures.storage_read import session_row
    from tests.fixtures.storage_seed import seed_session
    from tools.mcp_tool_agent import persist_agent_tool_names
    db = open_persistence(profile_scope(home=tmp_path))
    try:
        seed_session(db, 'owned', 'cli', system_prompt='prefix')
        epoch = db.runtime.begin_epoch(instance_id='fixture')
        scope = dict(epoch=epoch, execution_id='worker', session_id='owned', generation=0)
        db.worker_executions.register(**scope, kind='compute', adoption_secret='test-secret')
        store = RuntimeSessionStore(lambda method, **p: db.worker_executions.persist_operation(**p), scope,
                                    tmp_path / 'outbox')
        try:
            tools = [{'type': 'function', 'function': {'name': n, 'parameters': {}}}
                     for n in ('terminal', 'tool_search', 'tool_describe', 'tool_call')]
            persist_agent_tool_names(SimpleNamespace(_session_db=store, session_id='owned', tools=tools))
            seed_session(db, 'direct', 'cli', system_prompt='prefix')
            with open_persistence(profile_scope(home=tmp_path)) as persistence:
                persist_agent_tool_names(SimpleNamespace(_session_db=db, persistence=persistence,
                                                         session_id='direct', tools=tools))
            # The same pin the direct store writes (the RPC canonicalizes key order for its receipt
            # digest): a worker-pinned session rebuilt in-process reads back version + tools intact.
            assert (json.loads(session_row(db, 'owned')['tool_names'])
                    == json.loads(session_row(db, 'direct')['tool_names']))
            assert json.loads(store.get_session('owned')['tool_names'])['tools'] == tools
            with pytest.raises(WorkerPersistenceError, match='permission_denied'):
                store.update_session_tool_names('foreign', ['terminal'])
            with pytest.raises(Exception, match='invalid_params'):
                store.update_session_tool_names('owned', [{'not': 'a name'}])
        finally:
            store._outbox_owner.close()
    finally:
        db.close()
