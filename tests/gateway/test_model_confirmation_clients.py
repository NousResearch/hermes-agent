"""Every native surface collects the owner's guarded-model confirmation (same as main's CLI).

The real session authority answers a flagged ``session.mutate operation=model`` with
``confirmation_required`` and a one-time token; the classic CLI chat view and the ACP adapter must
ask, re-send once with the token on yes, and leave the session untouched on no.
"""
from types import SimpleNamespace

import pytest

from hermes_cli.gateway_client import GatewayClientError


class _Owner:
    """A real SessionAuthority behind the client ``rpc`` surface the viewers use."""

    def __init__(self, tmp_path, monkeypatch):
        from gateway import run
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from gateway.session_authority import SessionAuthority
        from gateway.session_controls import AuthorityConnection
        from gateway.session_local import create_local_session
        from hermes_cli import model_cost_guard, model_switch
        import hermes_state_runtime as rt
        monkeypatch.setattr(run, '_load_gateway_config', lambda *a: {})
        monkeypatch.setattr(model_switch, 'switch_model', lambda **k: model_switch.ModelSwitchResult(
            success=True, new_model=k['raw_input'], target_provider='custom', base_url='http://127.0.0.1:9/v1'))
        monkeypatch.setattr(model_cost_guard, 'expensive_model_warning', lambda model, **k: (
            model_cost_guard.ExpensiveModelWarning(model, 'custom', None, None, 'test', f'{model} IS EXPENSIVE')
            if model.startswith('pricey') else None))
        self.store = SessionStore(tmp_path / 'sessions', GatewayConfig())
        self.db = self.store._db
        runner = SimpleNamespace(session_store=self.store, _session_db=self.db, adapters={}, _draining=False,
                                 _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                                 _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
        runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
        authority = SessionAuthority(runner, profile_id='owned', instance_id='owner', db=self.db,
                                     epoch=rt.begin_runtime_epoch(self.db, instance_id='owner'))
        self.connection = AuthorityConnection(authority, object(), {'user_id': 'human'})
        self.sid = create_local_session(authority, self.connection.actor, dict(
            request_id='m', source='cli', cwd=str(tmp_path), model='frozen', toolsets=[])).session_id
        self.mutations = []

    async def rpc(self, method, **params):
        if method == 'session.mutate':
            self.mutations.append(params['payload'])
        response = await self.connection.dispatch({'id': 1, 'method': method, 'params': params})
        if 'error' in response:
            raise GatewayClientError((response['error'].get('data') or {}).get('reason') or response['error']['message'])
        return response['result']

    def state(self):
        row = self.db.get_session(self.sid)
        return row['model'], row['runtime_revision'], row['runtime_generation']

    async def close(self):
        await self.connection.close()
        self.store.close_all_db_handles()


@pytest.mark.asyncio
async def test_cli_chat_view_asks_and_applies_a_guarded_model_once_with_the_token(tmp_path, monkeypatch, capsys):
    from hermes_cli.gateway_chat_view import GatewayChatView
    owner = _Owner(tmp_path, monkeypatch)

    class Composer:
        def __init__(self, answer):
            self.answer, self.asked = answer, 0

        async def prompt_async(self, symbol):
            self.asked += 1
            return self.answer
    try:
        before = owner.state()
        view = GatewayChatView(owner, {'stored_session_id': owner.sid})
        view._composer = Composer('2')  # cancel
        assert await view.command('/model pricey') is True
        assert view._composer.asked == 1 and owner.state() == before, 'declined switch changed the session'
        assert 'pricey IS EXPENSIVE' in capsys.readouterr().out
        assert [p.get('confirm') for p in owner.mutations] == [None], 'a declined switch was re-sent'

        owner.mutations.clear()
        view._composer = Composer('y')
        assert await view.command('/model pricey') is True
        assert view._composer.asked == 1
        token = owner.mutations[1].get('confirm')
        assert token and [p.get('confirm') for p in owner.mutations] == [None, token]
        assert owner.state() == ('pricey', before[1] + 1, before[2] + 1), 'confirmed switch not applied once'
        assert view.model == 'pricey'

        # Non-interactive (`-q`, no TTY): no prompt, a refusal naming the message; nothing written.
        owner.mutations.clear()
        view._composer = None
        with pytest.raises(GatewayClientError, match='pricey-xl IS EXPENSIVE'):
            await view.command('/model pricey-xl')
        assert len(owner.mutations) == 1 and owner.state()[0] == 'pricey'
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_acp_set_session_model_asks_through_request_permission(tmp_path, monkeypatch):
    from acp.schema import AllowedOutcome, DeniedOutcome
    from acp_adapter.gateway_server import GatewayACPAgent
    owner = _Owner(tmp_path, monkeypatch)
    asked = []

    class Editor:
        def __init__(self, outcome):
            self.outcome = outcome

        async def request_permission(self, session_id, tool_call, options):
            asked.append((session_id, [o.option_id for o in options], tool_call))
            return SimpleNamespace(outcome=self.outcome)

    agent = GatewayACPAgent()
    agent._gateway = owner
    agent._client = lambda: _ready(owner)
    try:
        before = owner.state()
        agent._conn = Editor(DeniedOutcome(outcome='cancelled'))
        with pytest.raises(GatewayClientError, match='model_switch_cancelled'):
            await agent.set_session_model(model_id='pricey', session_id=owner.sid)
        assert owner.state() == before and [p.get('confirm') for p in owner.mutations] == [None]
        assert asked[0][0] == owner.sid and asked[0][1] == ['allow_once', 'deny']

        owner.mutations.clear()
        agent._conn = Editor(AllowedOutcome(outcome='selected', option_id='allow_once'))
        await agent.set_session_model(model_id='pricey', session_id=owner.sid)
        token = owner.mutations[1].get('confirm')
        assert token and [p.get('confirm') for p in owner.mutations] == [None, token]
        assert owner.state() == ('pricey', before[1] + 1, before[2] + 1)
    finally:
        await owner.close()


async def _ready(value):
    return value
