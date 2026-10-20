"""A frontend names itself on the gateway's ``session.create`` (``client``): the id and capabilities are frozen
with the launch request, survive a restart, and reach plugin system-prompt sections as ``client_id`` /
``client_capabilities``. Real SessionStore / SessionAuthority / AuthorityConnection."""
import json
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_follow_profile_config import _owner

TERN = {'id': 'hermes-tern', 'capabilities': ['math', 'mermaid', 'svg']}


def _gateway(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(run, '_hermes_home', home)
    (home / 'config.yaml').write_text(json.dumps({'model': {'provider': 'custom', 'default': 'm',
                                                            'base_url': 'http://127.0.0.1:1/v1'}}))
    store, _runner, authority, connection = _owner(home)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)

    async def create(request_id, **extra):
        return await connection.dispatch({'id': request_id, 'method': 'session.create', 'params': dict(
            request_id=request_id, source='tui', cwd=str(home), toolsets=[], **extra)})
    return store, authority, connection, create


def _section_client(authority, policy, monkeypatch):
    """``(client_id, client_capabilities)`` a plugin section sees for the agent an in-process turn builds."""
    from agent.system_prompt import _plugin_session_info
    from gateway import session_policy
    from gateway.session_execution import GatewaySessionAgentMixin
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = SimpleNamespace(session_authority=authority, _prefill_messages=None, _service_tier=None,
                                   _session_db=None, _refresh_fallback_model=lambda: None)
    source = SimpleNamespace(chat_id='c', user_id='u', user_id_alt=None, user_name=None, chat_name=None,
                             chat_type='dm', thread_id=None)
    turn._ctx = SimpleNamespace(source=source, user_config={}, enabled_toolsets=[], disabled_toolsets=None,
                                session_id='s', session_key='k', AIAgent=lambda **kw: SimpleNamespace(**kw))
    info = _plugin_session_info(turn._build_fresh_agent({'model': 'm', 'runtime': {}}, 'tui', None, 1, None, {}, False))
    return info['client_id'], info['client_capabilities']


@pytest.mark.asyncio
async def test_a_created_sessions_client_reaches_plugin_prompt_sections(monkeypatch):
    from gateway.config import Platform
    from gateway.session_local_recovery import local_adapter_map
    store, authority, connection, create = _gateway(monkeypatch)
    try:
        reply = await create('tern', client=TERN)
        assert 'result' in reply, reply
        chat_id = authority.sessions[reply['result']['session_id']].source.chat_id
        policy = local_adapter_map(authority)[Platform.LOCAL].policies[chat_id]
        assert _section_client(authority, policy, monkeypatch) == ('hermes-tern', 'math,mermaid,svg')

        refused = await create('bad', client={'id': 'Tern Pane'})
        assert refused['error']['data']['fields'] == ['client.id']
    finally:
        await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_the_client_survives_a_restart_and_is_empty_when_none_was_sent(monkeypatch):
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    store, authority, connection, create = _gateway(monkeypatch)
    try:
        tern, plain = (await create('tern', client=TERN))['result'], (await create('plain'))['result']
        # Cold recovery and a managed worker both run the policy restored from the stored receipt.
        for created, seen in ((tern, ('hermes-tern', 'math,mermaid,svg')), (plain, ('', ''))):
            restored = restore_policy(local_receipt(authority.db, created['session_id'])['policy'])
            assert _section_client(authority, restored, monkeypatch) == seen
    finally:
        await connection.close()
        store.close_all_db_handles()
