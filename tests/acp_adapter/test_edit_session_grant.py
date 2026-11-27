"""Exercise interactive edit grants through the real ACP async bridge."""
import asyncio
from types import SimpleNamespace

from acp.schema import AllowedOutcome, RequestPermissionResponse
from acp_adapter.edit_approval import EditProposal, make_acp_edit_approval_requester


def test_interactive_session_choice_is_offered_and_reused(tmp_path):
    async def run():
        calls = []

        async def request_permission(**kwargs):
            calls.append(kwargs)
            options = {option.option_id for option in kwargs['options']}
            assert 'allow_session' in options
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id='allow_session'))

        requester = make_acp_edit_approval_requester(request_permission, asyncio.get_running_loop(), 'one', timeout=2)
        first = EditProposal('write_file', str(tmp_path / 'one.txt'), None, 'one', {})
        second = EditProposal('write_file', str(tmp_path / 'two.txt'), None, 'two', {})
        assert await asyncio.to_thread(requester, first), 'session option must authorize the edit'
        assert await asyncio.to_thread(requester, second)
        assert len(calls) == 1

    asyncio.run(run())


def test_server_grant_survives_turn_callbacks_but_not_other_sessions_or_mode_reset(tmp_path):
    from unittest.mock import AsyncMock, Mock
    from acp_adapter.server import HermesACPAgent
    from acp_adapter.session import SessionState

    async def run():
        state = SessionState('one', SimpleNamespace(), cwd=str(tmp_path))
        other = SessionState('two', SimpleNamespace(), cwd=str(tmp_path))
        manager = Mock()
        manager.get_session.side_effect = lambda sid: {'one': state, 'two': other}[sid]
        server = HermesACPAgent(session_manager=manager)
        conn = SimpleNamespace(request_permission=AsyncMock(return_value=RequestPermissionResponse(
            outcome=AllowedOutcome(outcome='selected', option_id='allow_session'))), session_update=AsyncMock())
        proposal = EditProposal('write_file', str(tmp_path / 'x.txt'), None, 'x', {})
        loop = asyncio.get_running_loop()
        async def edit(s):
            callbacks = server._wire_turn_callbacks(s, s.session_id, conn, loop)
            assert callbacks.edit_approval_requester is not None
            assert await asyncio.to_thread(callbacks.edit_approval_requester, proposal)
        await edit(state)
        await edit(state)  # new callback, same session
        assert conn.request_permission.await_count == 1
        await edit(other)
        assert conn.request_permission.await_count == 2
        await server.set_session_mode('default', 'one')
        await edit(state)
        assert conn.request_permission.await_count == 3
        await server.set_config_option(server._EDIT_APPROVAL_POLICY_CONFIG_ID, 'one', 'ask')
        await edit(state)
        assert conn.request_permission.await_count == 4
    asyncio.run(run())


def test_sensitive_multifile_grant_is_not_offered_or_accepted(tmp_path):
    from acp_adapter.edit_approval import EditApprovalState
    async def run():
        calls = []
        async def permission(**kwargs):
            calls.append(kwargs)
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id='allow_session'))
        state = EditApprovalState()
        requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'one', timeout=2, session_state=state)
        paths = (str(tmp_path / 'a.py'), str(tmp_path / '.env'))
        proposal = EditProposal('patch', ', '.join(paths), None, 'patch', {}, paths)
        assert not await asyncio.to_thread(requester, proposal)
        assert not state.allow_for_session
        assert 'allow_session' not in {x.option_id for x in calls[-1]['options']}
        state.allow_for_session = True
        assert not await asyncio.to_thread(requester, proposal)
        assert len(calls) == 2
    asyncio.run(run())


def test_once_denial_unknown_and_exception_never_enable_grant(tmp_path):
    from acp_adapter.edit_approval import EditApprovalState
    async def run():
        for choice in ('allow_once', 'deny', 'allow_always', 'exception'):
            state = EditApprovalState()
            calls = []
            async def permission(**kwargs):
                calls.append(kwargs)
                if choice == 'exception':
                    raise RuntimeError('offline fixture')
                return RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id=choice))
            requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'one', timeout=2, session_state=state)
            proposal = EditProposal('write_file', str(tmp_path / 'x.py'), None, 'x', {})
            assert await asyncio.to_thread(requester, proposal) == (choice == 'allow_once')
            assert await asyncio.to_thread(requester, proposal) == (choice == 'allow_once')
            assert len(calls) == 2
            assert not state.allow_for_session
    asyncio.run(run())


def test_inflight_mode_reset_rejects_stale_session_consent(tmp_path):
    from acp_adapter.edit_approval import EditApprovalState
    async def run():
        state = EditApprovalState()
        async def permission(**kwargs):
            state.revoke()
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id='allow_session'))
        requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'one', timeout=2, session_state=state)
        proposal = EditProposal('write_file', str(tmp_path / 'x.py'), None, 'x', {})
        assert not await asyncio.to_thread(requester, proposal)
        assert not state.allow_for_session
    asyncio.run(run())


def test_real_file_dispatch_reuses_consent(tmp_path):
    from acp_adapter.edit_approval import set_edit_approval_requester, reset_edit_approval_requester
    from model_tools import handle_function_call
    import json

    async def run():
        calls = []
        async def permission(**kwargs):
            calls.append(kwargs)
            return RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id='allow_session'))
        requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'file-session', timeout=2)
        def dispatch(name, args):
            token = set_edit_approval_requester(requester)
            try:
                return json.loads(handle_function_call(name, args, task_id='file-session'))
            finally:
                reset_edit_approval_requester(token)
        target = tmp_path / 'document.txt'
        result = await asyncio.to_thread(dispatch, 'write_file', {'path': str(target), 'content': 'before'})
        assert 'error' not in result, result
        result = await asyncio.to_thread(dispatch, 'patch', {'path': str(target), 'old_string': 'before', 'new_string': 'after'})
        assert 'error' not in result, result
        assert target.read_text() == 'after'
        assert len(calls) == 1
    asyncio.run(run())


def test_restored_session_does_not_inherit_runtime_consent(tmp_path):
    from acp_adapter.session import SessionManager
    from hermes_state import SessionDB
    db = SessionDB(tmp_path / 'sessions.db')
    try:
        factory = lambda: SimpleNamespace(model='fixture', provider=None, base_url=None, api_mode=None)
        first = SessionManager(agent_factory=factory, db=db)
        state = first.create_session(cwd=str(tmp_path))
        assert state.edit_approval_state.grant(state.edit_approval_state.snapshot()[1])
        state.history.append({'role': 'user', 'content': 'fixture'})
        first.save_session(state.session_id)
        restored = SessionManager(agent_factory=factory, db=db).get_session(state.session_id)
        assert restored is not None
        assert not restored.edit_approval_state.snapshot()[0]
    finally:
        db.close()


def test_consent_generation_is_checked_under_the_same_lock_as_revocation():
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from acp_adapter.edit_approval import EditApprovalState
    state = EditApprovalState()
    generation = state.snapshot()[1]
    started = threading.Event()
    def grant():
        started.set()
        return state.grant(generation)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with state._lock:
            pending = pool.submit(grant)
            assert started.wait(2)
            # Reproduce a revocation ordered before a waiting grant.
            state.generation += 1
        assert not pending.result(timeout=2)
    assert not state.snapshot()[0]


def test_existing_policy_and_failing_policy_preserve_prompt_behavior(tmp_path):
    from unittest.mock import AsyncMock
    async def run():
        permission = AsyncMock(return_value=RequestPermissionResponse(outcome=AllowedOutcome(outcome='selected', option_id='allow_once')))
        proposal = EditProposal('write_file', str(tmp_path / 'x.py'), None, 'x', {})
        requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'one', timeout=2,
            auto_approve_getter=lambda: ('workspace_session', str(tmp_path)))
        assert await asyncio.to_thread(requester, proposal)
        permission.assert_not_awaited()
        def broken():
            raise RuntimeError('policy unavailable')
        requester = make_acp_edit_approval_requester(permission, asyncio.get_running_loop(), 'one', timeout=2,
            auto_approve_getter=broken)
        assert await asyncio.to_thread(requester, proposal)
        assert permission.await_count == 1
    asyncio.run(run())
