"""B22 / C-hx-3: a cold Desktop resume settles the reasoning pill from the canonical snapshot.

Desktop marks a resumed slice's effort pending and clears it only on a string
``info.reasoning_effort`` (main's session info always carried it with
``reasoning_effort_wire`` / ``service_tier`` / ``fast``); a lazy canonical session must report the
level its frozen policy will run, or the pill spins forever."""
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_lazy_session_info_reports_frozen_reasoning_and_tier(tmp_path, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: gpt-5.4, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
        'agent: {reasoning_effort: high, service_tier: fast}\nplatform_toolsets: {cli: []}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']

    created = await call('session.create', request_id='cold', source='gui', cwd=str(tmp_path))
    sid = created['session_id']
    for info in (created['info'], await call('session.info', session_id=sid),
                 (await call('session.resume', session_id=sid))['info']):
        assert info['lazy'] is True
        assert (info['reasoning_effort'], info['reasoning_effort_wire']) == ('high', 'high'), info
        # A priority tier never reaches a custom endpoint, so the session must not claim Fast there.
        assert (info['service_tier'], info['fast']) == ('priority', False), info
    off = await call('session.create', request_id='off', source='gui', cwd=str(tmp_path), reasoning='none')
    assert (off['info']['reasoning_effort'], off['info']['reasoning_effort_wire']) == ('none', '')
