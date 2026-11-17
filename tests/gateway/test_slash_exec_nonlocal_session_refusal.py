"""``slash.exec`` on a session a messaging platform owns, forwarded by a local viewer.

``execute_command`` runs a registry command against the session's own local policy. A session
created on Telegram (and later attached from the classic CLI with ``hermes --resume``) has none
here: the owner refused every forwarded command with the bare code ``permission_denied``, so the
CLI printed ``Error: permission_denied`` with nothing about what was refused or where it can run —
and it refused read-only reports too, ahead of the ``_INFO_READS`` dispatch that the gateway
command-parity contract promises (#153927).

The refusal now names the surface that owns the session, through the ``unsupported_command``
reason the gateway already uses for a command that is not available on this session (its
``data.detail`` carries the hint). A ``False`` from ``authorize_local_source`` — a LOCAL source
naming a profile this runtime does not serve, i.e. a forged one — stays ``permission_denied``.
"""
from types import SimpleNamespace

import pytest
import pytest_asyncio


@pytest_asyncio.fixture
async def owner(tmp_path, monkeypatch):
    """The in-process owner: one authority, one connection, a disposable home.

    A copy of ``tests/gateway/test_tui_session_reads.py``'s fixture (a fixture cannot be imported
    from another module without tripping the health gate's F811 rule).
    """
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection

    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'model: {default: fake-model, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n')
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_hermes_home', home)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    runner._adapters_for_profile = lambda profile: runner.adapters
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id='first')
    connection = AuthorityConnection(authority, object(), {'user_id': 'human'})

    async def call(method, **params):
        reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
        assert 'result' in reply, reply
        return reply['result']
    yield SimpleNamespace(call=call, home=home, authority=authority, workdir=tmp_path,
                          connection=connection)


def _owned_by(owner, sid, platform):
    """*sid* as the platform on its stored source owns it (what a resumed messaging chat is)."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    live = owner.authority.sessions[sid]
    live.source = SessionSource(platform=Platform(platform), chat_id=f'{platform}:42', user_id='owner')
    return live


@pytest.mark.asyncio
async def test_a_forwarded_command_on_a_messaging_session_names_the_owning_platform(owner):
    sid = (await owner.call('session.create', request_id='local', source='cli',
                            cwd=str(owner.workdir)))['session_id']
    _owned_by(owner, sid, 'telegram')

    reply = await owner.connection.dispatch({'id': 7, 'method': 'slash.exec',
                                             'params': {'session_id': sid,
                                                        'command': 'review the PR as a whole'}})
    error = reply.get('error') or {}
    assert error.get('message') == 'unsupported_command', reply
    detail = str((error.get('data') or {}).get('detail') or '')
    assert 'telegram' in detail and '/review' in detail, error


@pytest.mark.asyncio
async def test_a_read_only_report_is_refused_with_the_same_hint(owner):
    """``/toolsets`` is an ``_INFO_READS`` command: the refusal is about WHERE it can run, and it
    must say so instead of a bare code (the parity table's reviewed reads never see a messaging
    session's live policy)."""
    sid = (await owner.call('session.create', request_id='local', source='cli',
                            cwd=str(owner.workdir)))['session_id']
    _owned_by(owner, sid, 'slack')

    reply = await owner.connection.dispatch({'id': 8, 'method': 'slash.exec',
                                             'params': {'session_id': sid, 'command': 'toolsets'}})
    error = reply.get('error') or {}
    assert error.get('message') == 'unsupported_command', reply
    assert 'slack' in str((error.get('data') or {}).get('detail') or ''), error


@pytest.mark.asyncio
async def test_a_local_session_still_runs_the_command(owner):
    """The control: an untouched local session resolves commands on the owner as before."""
    sid = (await owner.call('session.create', request_id='local', source='cli',
                            cwd=str(owner.workdir)))['session_id']
    reply = await owner.connection.dispatch({'id': 9, 'method': 'slash.exec',
                                             'params': {'session_id': sid, 'command': 'toolsets'}})
    assert 'result' in reply and reply['result'].get('output'), reply
