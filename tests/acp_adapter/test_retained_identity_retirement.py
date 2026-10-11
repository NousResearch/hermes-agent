"""Class: a retained submission identity (an un-acked prompt's input_id, a prepared control's
request_id) is only for the retry that is the session's very next verb. Every native surface that
retains one must retire it when any other verb goes to that session first; otherwise a later
deliberate repeat replays the stale receipt (or its stale fence) instead of doing new work.

One row per sweep site (SWEEP_retained-identity-retirement.md), each over a real authority + SQLite.
"""
import asyncio
from types import SimpleNamespace

import acp
import pytest
import pytest_asyncio

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_state_runtime import list_session_admissions

LOST = 'Gateway disconnected; turn outcome is unknown.'


class Wire:
    """The surfaces' gateway client over a real AuthorityConnection. ``lose`` drops one matching
    call: ``after`` = the owner committed but the reply is lost; ``before`` = it never arrived."""

    def __init__(self, loop):
        self.loop, self.events, self.serial, self.viewer, self.lose = loop, asyncio.Queue(), 0, None, None

    def write(self, frame):
        self.loop.call_soon_threadsafe(self.events.put_nowait, frame)
        return True

    async def rpc(self, method, _timeout=None, **params):
        lose, self.serial = self.lose, self.serial + 1
        hit = lose is not None and lose[0] == method and lose[1](params)
        if hit:
            self.lose = None
            if lose[2] == 'before':
                raise GatewayClientError(LOST)
        reply = await self.viewer.dispatch({'id': self.serial, 'method': method, 'params': params})
        if 'error' in reply:
            raise GatewayRPCError(reply['error'].get('data', {}).get('reason', 'request_failed'))
        if hit:
            raise GatewayClientError(LOST)
        return reply['result']


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    from gateway.session_local import create_local_session
    from hermes_cli import model_cost_guard, model_switch
    monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {'platform_toolsets': {'acp': []}})
    monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
        success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
    monkeypatch.setattr(model_cost_guard, 'expensive_model_warning', lambda model, **k: None)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    executed = []

    async def answer(event):
        from gateway.session_results import execution_result
        executed.append(event.text)
        execution_result.get()['result'] = {'final_response': 'ACK_' + event.text, 'completed': True, 'messages': []}
        return 'ACK_' + event.text

    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False, adapters={},
                             _handle_message=answer,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _resolve_session_agent_runtime=lambda **kwargs: ('frozen', {}))
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    wire = Wire(asyncio.get_running_loop())
    wire.viewer = AuthorityConnection(authority, wire, {'user_id': 'editor', 'profile_id': 'p'})
    sid = create_local_session(authority, wire.viewer.actor, {'request_id': 'editor', 'source': 'acp',
                                                             'cwd': str(tmp_path), 'model': 'frozen',
                                                             'toolsets': []}).session_id
    yield SimpleNamespace(authority=authority, wire=wire, sid=sid, executed=executed, store=store)
    await wire.viewer.close()
    store.close_all_db_handles()


async def _acp(owner):
    from acp_adapter.gateway_server import GatewayACPAgent
    agent = GatewayACPAgent()
    agent._gateway = owner.wire
    agent._conn = SimpleNamespace(session_update=lambda **kwargs: asyncio.sleep(0))
    agent._snapshots[owner.sid] = await owner.wire.rpc('session.resume', session_id=owner.sid)
    agent._event_task = asyncio.create_task(agent._events())
    return agent


async def _settled(owner, count):
    async with asyncio.timeout(5):
        while sum(row['status'] == 'terminal' for row in list_session_admissions(
                owner.authority.db, session_id=owner.sid, pending_only=False)) < count:
            await asyncio.sleep(0.01)


def _surface(name, owner):
    """(model control, prompt) verbs for one surface; each awaits its own outcome."""
    async def acp_model(agent, model):
        await asyncio.wait_for(agent.set_session_model(model_id=model, session_id=owner.sid), 5)

    async def acp_slash(agent, model):
        await asyncio.wait_for(agent.prompt([acp.text_block(f'/model {model}')], owner.sid), 5)

    async def acp_prompt(agent, text):
        await asyncio.wait_for(agent.prompt([acp.text_block(text)], owner.sid), 5)

    async def cli_model(view, model):
        await view.command(f'/model {model}')

    async def cli_prompt(view, text):
        before = len(owner.executed)
        await view.submit(text)
        await _settled(owner, before + 1)

    return {'acp-set-model': (acp_model, acp_prompt), 'acp-slash-model': (acp_slash, acp_prompt),
            'cli-model': (cli_model, cli_prompt)}[name]


def _model(params):
    return params.get('operation') == 'model'


@pytest.mark.asyncio
@pytest.mark.parametrize('surface', ['acp-set-model', 'acp-slash-model', 'cli-model'])
@pytest.mark.parametrize('between', ['other-control', 'prompt'])
async def test_a_repeated_control_after_other_work_is_new_work(owner, surface, between, capsys):
    """/model A (reply lost) -> other verb -> /model A again must leave the session on A."""
    control, prompt = _surface(surface, owner)
    if surface.startswith('acp'):
        handle = await _acp(owner)
    else:
        from hermes_cli.gateway_chat_view import GatewayChatView
        handle = GatewayChatView(owner.wire, {'stored_session_id': owner.sid})
    try:
        # other-control: A committed, its reply was lost; then B lands -> the repeat must re-apply A.
        # prompt: A never reached the owner; a turn moves the generation -> the repeat must not
        # re-send A's stale fence (stale_generation) but prepare A afresh.
        owner.wire.lose = ('session.mutate', _model, 'after' if between == 'other-control' else 'before')
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await control(handle, 'model-a')
        if between == 'other-control':
            await control(handle, 'model-b')
            assert owner.authority.db.get_session(owner.sid)['model'] == 'model-b'
        else:
            await prompt(handle, 'between')
        await control(handle, 'model-a')
        assert owner.authority.db.get_session(owner.sid)['model'] == 'model-a'
    finally:
        if surface.startswith('acp'):
            handle._event_task.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize('between', ['set-model', 'fork'])
async def test_a_repeated_prompt_after_a_control_is_new_work(owner, tmp_path, between):
    """An un-acked prompt's input_id is retired by an ACP control (set_session_model, fork)."""
    agent = await _acp(owner)
    try:
        owner.wire.lose = ('prompt.submit', lambda params: True, 'after')
        with pytest.raises(GatewayClientError, match='outcome is unknown'):
            await agent.prompt([acp.text_block('yes')], owner.sid)
        await _settled(owner, 1)
        if between == 'set-model':
            await asyncio.wait_for(agent.set_session_model(model_id='model-b', session_id=owner.sid), 5)
        else:
            await asyncio.wait_for(agent.fork_session(cwd=str(tmp_path), session_id=owner.sid), 5)
        response = await asyncio.wait_for(agent.prompt([acp.text_block('yes')], owner.sid), 5)
        assert response.stop_reason == 'end_turn'
        assert owner.executed == ['yes', 'yes']
        assert len(list_session_admissions(owner.authority.db, session_id=owner.sid, pending_only=False)) == 2
    finally:
        agent._event_task.cancel()
