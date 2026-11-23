"""A selected-message branch copies exactly the requested prefix (helix4u #5).

Desktop branches from a chosen message; the canonical branch used to clone every active parent
row, so later instructions and answers the user meant to leave behind rode into the child. The
boundary is a physical row id (``through_message_id``), validated against the parent's active
transcript inside the receipt transaction and part of the retry identity.
"""
from types import SimpleNamespace

import pytest

from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
from tests.fixtures.storage_read import conversation, session_row
from tests.fixtures.storage_seed import seed_messages
from tests.gateway.fixtures.storage_handles import close_storage_handles, storage_handle


@pytest.mark.asyncio
async def test_branch_through_a_middle_message_excludes_later_and_concurrent_rows(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setattr(run, '_load_gateway_config', dict)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store.persistence
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', persistence=storage_handle(db.scope.profile_dir),
                                 epoch=db.runtime.begin_epoch(instance_id='owner'))
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='branch-boundary', source='cli',
                                                              cwd=str(tmp_path), model='m', toolsets=[]))
    sid = ref.session_id
    seed_messages(db, sid, {'role': 'user', 'content': 'KEEP question'})
    call = [{'id': 'c1', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{}'}}]
    boundary, = seed_messages(db, sid, {'role': 'assistant', 'content': 'KEEP reading', 'tool_calls': call})
    seed_messages(db, sid, {'role': 'tool', 'content': 'KEEP tool result', 'tool_call_id': 'c1'})
    seed_messages(db, sid, {'role': 'user', 'content': 'DROP later instruction'})
    seed_messages(db, sid, {'role': 'assistant', 'content': 'DROP later answer'})
    session = session_row(db, sid)

    async def mutate(request_id, payload):
        return await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': sid, 'request_id': request_id, 'expected_revision': session['runtime_revision'],
            'expected_generation': session['runtime_generation'], 'operation': 'branch', 'payload': payload}})

    try:
        # A message that lands between the click and the commit (a concurrent turn) is later too.
        seed_messages(db, sid, {'role': 'user', 'content': 'DROP concurrent message'})
        reply = await mutate('pick', {'through_message_id': boundary})
        receipt = reply['result']
        child = receipt['branched_session_id']
        # The assistant boundary keeps the tool result that answers its call, nothing after it.
        assert [m['content'] for m in conversation(db, child)] == [
            'KEEP question', 'KEEP reading', 'KEEP tool result']
        assert receipt['copied_messages'] == 3
        # Exact retry: the same child, even though the parent kept growing.
        seed_messages(db, sid, {'role': 'assistant', 'content': 'DROP after commit'})
        assert (await mutate('pick', {'through_message_id': boundary}))['result'] == receipt
        # The boundary is retry identity: the same request id with another boundary is a conflict.
        conflict = await mutate('pick', {'through_message_id': boundary + 1})
        assert conflict['error']['message'] == 'admission_conflict'
        # A row that is not in this session's active transcript is refused, not silently widened.
        foreign, = seed_messages(db, child, {'role': 'user', 'content': 'child row'})
        session = session_row(db, sid)
        refused = await mutate('foreign', {'through_message_id': foreign})
        assert refused['error']['message'] == 'invalid_params', refused
    finally:
        await owner.close()
