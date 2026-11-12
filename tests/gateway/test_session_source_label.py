"""A creator's ``source_label`` is the sessions-row ``source`` for the row's whole life (#142895).

``hermes -z ... --source heartbeat`` runs as a ``oneshot`` session; the row is labelled ``heartbeat`` at
creation and keeps it through every per-turn peer refresh, while ``created_source`` keeps the surface.
"""
import pytest

from tests.gateway.test_session_client_env import _home, _owner


def _create(owner, actor, home, request_id, **extra):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    ref = create_local_session(owner, actor, {'request_id': request_id, 'source': 'oneshot', 'cwd': str(home),
                                              'toolsets': [], **extra})
    return ref, restore_policy(local_receipt(owner.db, ref.session_id)['policy'])


def _refresh(owner, sid):
    """The per-turn routing-peer refresh every gateway turn performs."""
    live = owner.sessions[sid]
    owner.runner.session_store._record_gateway_session_peer(sid, live.route, live.source)


@pytest.mark.asyncio
async def test_a_source_label_is_stored_at_creation_and_survives_every_turn_refresh(tmp_path, monkeypatch):
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection.actor, home, 'wake', source_label='heartbeat')
        assert policy.source == 'oneshot' and policy.platform == 'cli'
        row = owner.db.get_session(ref.session_id)
        assert (row['source'], row['created_source']) == ('heartbeat', 'oneshot')
        for _ in range(3):
            _refresh(owner, ref.session_id)
            assert owner.db.get_session(ref.session_id)['source'] == 'heartbeat'
        # Without a label the same refresh keeps the plain ``oneshot`` label, as before.
        plain, _ = _create(owner, connection.actor, home, 'plain')
        _refresh(owner, plain.session_id)
        assert owner.db.get_session(plain.session_id)['source'] == 'oneshot'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('label', ['cli', 'gui', 'local', 'cron', 'kanban', 'a2a', 'telegram', 'webhook',
                                   'api_server', 'Heartbeat', '', 'x' * 65, 7])
async def test_a_label_another_reader_gives_meaning_to_is_refused(tmp_path, monkeypatch, label):
    from hermes_state_runtime import RuntimeStoreError
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        with pytest.raises(RuntimeStoreError) as refused:
            _create(owner, connection.actor, home, 'bad', source_label=label)
        assert refused.value.args[0] == 'invalid_params'
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_runtime_describe_advertises_source_label(tmp_path, monkeypatch):
    home = _home(tmp_path)
    owner, connection = await _owner(home, monkeypatch)
    try:
        described = await connection.dispatch({'id': 1, 'method': 'runtime.describe', 'params': {}})
        assert 'source_label' in described['result']['session_create']['parameters']
        from tui_gateway.contracts.registry import CANONICAL_METHODS
        CANONICAL_METHODS['session.create'].params.model_validate(
            {'request_id': 'r', 'source': 'oneshot', 'source_label': 'heartbeat'})
    finally:
        await connection.close()
        owner.db.close()


def test_a_labelled_oneshot_row_may_still_register_compute_delegations(tmp_path):
    """Background ``delegate_task`` keys on the row: a ``heartbeat`` row created as ``oneshot`` is a CLI run,
    a messaging row is not. A mismatched route is refused by the NEXT check, so ``permission_denied`` means
    the producer gate let it through."""
    from gateway.session_worker_delegation import _dispatch
    from hermes_state import SessionDB
    from hermes_state_runtime import RuntimeStoreError
    db = SessionDB(tmp_path / 'state.db')
    try:
        db._execute_write(lambda conn: conn.execute(
            "INSERT INTO sessions(id,source,created_source,session_key,started_at) "
            "VALUES ('wake','heartbeat','oneshot','route',0),('bot','telegram','telegram','route',0)"))
        payload = {'delegation_id': 'd', 'task': {}, 'dispatched_at': 0, 'parent_session_id': 'other',
                   'session_key': 'route', 'origin_ui_session_id': '', 'origin_session_id': ''}

        def refusal(sid):
            def write(conn):
                with pytest.raises(RuntimeStoreError) as refused:
                    _dispatch(db, conn, sid, dict(payload), execution_id='e', worker_pid=0, worker_birth=0)
                return refused.value.args[0]
            return db._execute_write(write)
        assert refusal('wake') == 'permission_denied'
        assert refusal('bot') == 'unsupported_producer'
    finally:
        db.close()
