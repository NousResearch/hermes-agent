"""LC-4: the gateway runs every local session and never loads ``./.hermes/plugins``. A session whose
client opted in (``HERMES_ENABLE_PROJECT_PLUGINS``, forwarded in ``client_env``) from a folder that has
project plugins gets a gateway.log line and a client-visible ``config_warning`` saying so."""
import logging
from types import SimpleNamespace

import pytest


async def _owner(home, monkeypatch):
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('HERMES_HOME', str(home))
    store = SessionStore(home / 'sessions', GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _adapter_for_source=lambda source: None,
                             _evict_cached_agent=lambda route: None)
    owner = await initialize_session_authority(runner, profile_id=str(home), instance_id='fixture')
    # This fixture's runner has no adapter table: hand session.info the session's frozen policy.
    from gateway import session_policy
    from hermes_state_local import local_receipt
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: session_policy.restore_policy(
        local_receipt(owner.db, source.chat_id)['policy']))
    return owner, AuthorityConnection(owner, object(), {'user_id': 'owner'})


@pytest.mark.asyncio
async def test_an_opted_in_session_is_told_project_plugins_do_not_load(tmp_path, monkeypatch, caplog):
    from gateway.session_local import create_local_session, local_session_info
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n')
    repo, plain = tmp_path / 'repo', tmp_path / 'plain'
    (repo / '.hermes' / 'plugins' / 'mine').mkdir(parents=True)
    (plain / '.hermes' / 'plugins').mkdir(parents=True)  # opted in, but no plugin in it
    monkeypatch.delenv('HERMES_ENABLE_PROJECT_PLUGINS', raising=False)
    owner, connection = await _owner(home, monkeypatch)

    def create(request_id, cwd, **extra):
        ref = create_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
                                                             'cwd': str(cwd), 'toolsets': [], **extra},
                                   client_local=True)  # the client's folder is on this host
        return local_session_info(owner, ref).get('config_warning')
    try:
        with caplog.at_level(logging.WARNING):
            warning = create('opted', repo, client_env={'HERMES_ENABLE_PROJECT_PLUGINS': 'true'})
        assert warning and 'not loaded' in warning and str(repo / '.hermes' / 'plugins') in warning
        assert any('project plugins' in record.getMessage() and str(repo) in record.getMessage()
                   for record in caplog.records)
        assert create('off', repo, client_env={'HERMES_ENABLE_PROJECT_PLUGINS': '0'}) is None
        assert create('unset', repo, client_env={}) is None
        assert create('empty', plain, client_env={'HERMES_ENABLE_PROJECT_PLUGINS': '1'}) is None
        assert create('desktop', repo) is None  # a client that sends no environment
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_session_create_carries_the_notice_through_its_strict_wire_contract(tmp_path, monkeypatch):
    """The notice also rides the snapshot ``session.create`` returns: it must pass that method's declared
    result contract (strict here, as under ``HERMES_TEST_ISOLATION``), for both session-start notices."""
    from tui_gateway.contracts import registry
    monkeypatch.setattr(registry, 'STRICT', True)
    home, repo = tmp_path / 'home', tmp_path / 'repo'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n')
    (repo / '.hermes' / 'plugins' / 'mine').mkdir(parents=True)
    monkeypatch.delenv('HERMES_ENABLE_PROJECT_PLUGINS', raising=False)
    owner, connection = await _owner(home, monkeypatch)

    async def create(request_id, local):
        connection.native_owner = local  # the gateway's own ticketed attach, or a remote client
        reply = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {
            'request_id': request_id, 'source': 'cli', 'cwd': str(repo), 'toolsets': [],
            'client_env': {'HERMES_ENABLE_PROJECT_PLUGINS': '1'}}})
        assert 'result' in reply, reply
        return reply['result']['info'].get('config_warning') or ''
    try:
        assert 'not loaded' in await create('local', True)
        assert 'not attached to this gateway locally' in await create('remote', False)
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_an_unlistable_plugins_folder_never_blocks_the_session(tmp_path, monkeypatch):
    """``is_dir`` can pass while listing fails (no read permission): the optional notice is skipped and
    the session is still created and described."""
    import pathlib
    from gateway.session_local import create_local_session, local_session_info
    home, folder = tmp_path / 'home', tmp_path / 'repo' / '.hermes' / 'plugins'
    home.mkdir()
    (home / 'config.yaml').write_text(
        'platform_toolsets: {cli: []}\nmodel: {provider: custom, default: m, base_url: http://x.invalid/v1}\n')
    (folder / 'mine').mkdir(parents=True)
    owner, connection = await _owner(home, monkeypatch)
    real = pathlib.Path.iterdir

    def iterdir(path):
        if path == folder:
            raise PermissionError(13, 'Permission denied', str(path))
        return real(path)
    monkeypatch.setattr(pathlib.Path, 'iterdir', iterdir)
    try:
        try:
            ref = create_local_session(owner, connection.actor, {
                'request_id': 'locked', 'source': 'cli', 'cwd': str(tmp_path / 'repo'), 'toolsets': [],
                'client_env': {'HERMES_ENABLE_PROJECT_PLUGINS': '1'}}, client_local=True)
            info = local_session_info(owner, ref)
        except OSError as exc:
            info = {'error': repr(exc)}
        assert 'error' not in info and info.get('config_warning') is None, info
    finally:
        await connection.close()
        owner.db.close()


def test_classic_cli_prints_the_sessions_config_warning(capsys):
    from hermes_cli.gateway_session_store import GatewaySessionStore

    class Connection:
        final = False

        def call(self, method, **params):
            if method == 'runtime.describe':
                return {'session_create': {'parameters': []}}
            return {'stored_session_id': 'sid', 'info': {'config_warning': 'project plugins in /r are not loaded'}}

        def state(self, sid):
            return SimpleNamespace(adopt_snapshot=lambda snapshot: None)
    GatewaySessionStore(Connection()).create({})
    assert 'Warning: project plugins in /r are not loaded' in capsys.readouterr().err
