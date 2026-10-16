"""CFG-3: an MCP server added to a profile's config after a local session started reaches that session only
through a gateway-committed policy generation (``reload.mcp`` for the calling idle session, or the CFG-2
refresh when a client attaches), definitions and private values included, so a managed worker connects it.
``reload.mcp`` names a configured server that did not connect instead of answering plain success."""
import asyncio
import json
import os
from types import SimpleNamespace

import pytest

from tests.gateway.test_session_resume_config_refresh import _call, _connect, _owner_authority, _runner
from tests.waiting import async_wait_until

CONFIG = ('model: {default: m1, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
          'platform_toolsets: {cli: [terminal]}\nterminal: {backend: local}\n'
          'mcp_servers: {old: {command: old-mcp}}\n')
ADDED = CONFIG.replace('mcp_servers: {old: {command: old-mcp}}', 'mcp_servers: {old: {command: old-mcp}, '
                       'fixt: {command: fixt-mcp, env: {FIXT_TOKEN: sekret}}, dormant: {command: dormant-mcp, enabled: false}}')
OLD = {'old': {'command': 'old-mcp'}}


def _home(monkeypatch):
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(CONFIG)
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _reloadable(authority, monkeypatch):
    """reload.mcp's owner side, with the profile's servers already reconciled."""
    authority.runner.config = SimpleNamespace(multiplex_profiles=False)
    authority.runner._mcp_reload_refresh_cached_agents = lambda multiplex, profile: None
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', lambda profile_id: 'default')
    monkeypatch.setattr('tools.mcp_tool_discovery.reconcile_mcp_servers_with_config',
                        lambda: {'removed': [], 'added': ['fixt'], 'pending': []})
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_not_connected', list)
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_ready', lambda: ['fixt', 'old'])


def _worker_frame(authority, sid, monkeypatch):
    """The managed-worker bootstrap a turn admitted now would get."""
    from gateway import session_managed_worker as smw
    from gateway.session_contract import SessionRef
    from gateway.session_policy import policy_for_source
    monkeypatch.setattr('gateway.session_worker_construct.construct_inputs', lambda *args: {})
    policy = policy_for_source(authority.runner, authority.sessions[sid].source)
    scope = {'profile_id': authority.profile_id, 'session_id': sid, 'execution_id': 'x', 'generation': 1,
             'pid': os.getpid(), 'birth': 0, 'secret': 's', 'epoch': 1}
    return smw._bootstrap(authority, SessionRef(authority.profile_id, sid), {'payload': {'text': 'hi'}}, policy, scope)


def _worker_servers(frame, monkeypatch):
    """What the worker's discovery connects: the servers of its bound config (``bind_worker_policy`` makes
    it every config read) that its toolsets name, as ``discover_mcp_tools`` filters them."""
    from tools.mcp_tool_config import _load_mcp_config
    config, toolsets = json.loads(frame['policy']['config_json']), set(frame['policy']['toolsets'])
    with monkeypatch.context() as bound:
        bound.setattr('hermes_cli.config.load_config', lambda *args, **kwargs: config)
        servers = _load_mcp_config()
    return {name: cfg for name, cfg in servers.items() if name in toolsets}, toolsets


def _generation(store, sid):
    return store._db.get_session(sid)['runtime_generation']


@pytest.mark.asyncio
async def test_a_new_mcp_server_reaches_a_session_only_through_a_committed_generation(tmp_path, monkeypatch):
    from hermes_state_local import local_receipt
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner, later = _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='a', source='cli', cwd=str(tmp_path)))['session_id']
        other = (await _call(owner, 'session.create', request_id='b', source='cli', cwd=str(tmp_path)))['session_id']
        generation, other_generation = _generation(store, sid), _generation(store, other)
        (home / 'config.yaml').write_text(ADDED)
        # A turn admitted (or queued) now runs the generation it was admitted under: no capability appears
        # without a committed transition.
        assert 'fixt' not in _worker_frame(authority, sid, monkeypatch)['policy']['toolsets']
        result = await _call(owner, 'reload.mcp', session_id=sid, confirm=True)
        assert result['session'] == 'applied' and 'unavailable' not in result, result
        assert _generation(store, sid) == generation + 1
        servers, toolsets = _worker_servers(_worker_frame(authority, sid, monkeypatch), monkeypatch)
        assert {'terminal', 'old', 'fixt'} <= toolsets and 'dormant' not in toolsets
        assert servers['fixt']['command'] == 'fixt-mcp' and servers['fixt']['env'] == {'FIXT_TOKEN': 'sekret'}
        from gateway.session_policy import policy_for_source
        assert 'fixt' in policy_for_source(authority.runner, authority.sessions[sid].source).toolsets  # in process
        assert 'sekret' not in json.dumps(local_receipt(store._db, sid))  # the private value stays private
        # Another session of the profile is not changed implicitly...
        assert 'fixt' not in _worker_frame(authority, other, monkeypatch)['policy']['toolsets']
        assert _generation(store, other) == other_generation
        # ...until a client attaches to it.
        await _call(later, 'session.resume', session_id=other)
        assert _generation(store, other) == other_generation + 1
        servers, toolsets = _worker_servers(_worker_frame(authority, other, monkeypatch), monkeypatch)
        assert set(servers) == {'old', 'fixt'} and servers['fixt']['env'] == {'FIXT_TOKEN': 'sekret'}
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resume_hands_a_worker_the_definition_of_every_server_it_names(tmp_path, monkeypatch):
    """A server added after creation reaches a managed worker with its definition and private values, not
    just its name (discovery filters the bound config's servers by the toolsets)."""
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    owner, later = _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='w', source='cli', cwd=str(tmp_path)))['session_id']
        (home / 'config.yaml').write_text(ADDED)
        await _call(later, 'session.resume', session_id=sid)
        servers, toolsets = _worker_servers(_worker_frame(authority, sid, monkeypatch), monkeypatch)
        assert 'fixt' in toolsets
        assert set(servers) == {'old', 'fixt'}, servers
        assert servers['fixt']['env'] == {'FIXT_TOKEN': 'sekret'}
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_reload_never_changes_a_running_or_queued_turns_servers(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef, Submission
    from gateway.session_policy import policy_for_source
    import hermes_state_runtime as rt
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store = SessionStore(home / 'sessions', GatewayConfig())
    started, release = asyncio.Event(), asyncio.Event()

    async def handle(event):
        started.set()
        await release.wait()
        return 'ok'
    authority = await initialize_session_authority(_runner(store, handle), profile_id=str(home), instance_id='only')
    _reloadable(authority, monkeypatch)
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='busy', source='cli', cwd=str(tmp_path)))['session_id']
        turn = await authority.submit(owner.actor, Submission('turn-1', SessionRef(authority.profile_id, sid),
                                                               {'text': 'one'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        (home / 'config.yaml').write_text(ADDED)
        generation = _generation(store, sid)
        busy = await _call(owner, 'reload.mcp', session_id=sid, confirm=True)
        assert busy['session'] == 'busy' and 'once the turn finishes' in busy['message'], busy
        assert _generation(store, sid) == generation
        assert 'fixt' not in policy_for_source(authority.runner, authority.sessions[sid].source).toolsets
        release.set()
        await async_wait_until(lambda: rt.get_session_admission(store._db, admission_id=turn.admission_id)['status']
                               == 'terminal', what='turn-1 to settle')
        assert (await _call(owner, 'reload.mcp', session_id=sid, confirm=True))['session'] == 'applied'
        assert 'fixt' in policy_for_source(authority.runner, authority.sessions[sid].source).toolsets
    finally:
        release.set()
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_an_admission_queued_during_a_reload_commit_keeps_the_old_servers(tmp_path, monkeypatch):
    """reload.mcp's session commit is fenced inside its write transaction: a turn queued after the refresh
    read the stored policy refuses it (``session: busy``), and that turn runs without the new server."""
    import hermes_state_runtime as rt
    from hermes_state_local import local_receipt
    from tests.gateway.test_session_resume_config_refresh import admit_during_refresh, hold_refresh_commit
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    seen = []

    async def handle(event):
        from gateway.session_policy import policy_for_source
        seen.append('fixt' in policy_for_source(authority.runner, event.source).toolsets)
        return 'ok'
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    authority.runner._handle_message = handle
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='q', source='cli', cwd=str(tmp_path)))['session_id']
        stored = local_receipt(store._db, sid)['policy']
        (home / 'config.yaml').write_text(ADDED)
        barrier = hold_refresh_commit(monkeypatch)
        reload = _call(owner, 'reload.mcp', session_id=sid, confirm=True)
        result, turn = await admit_during_refresh(authority, owner.actor, sid, reload, barrier)
        assert result['session'] == 'busy', result
        assert local_receipt(store._db, sid)['policy'] == stored
        await async_wait_until(lambda: rt.get_session_admission(store._db, admission_id=turn.admission_id)['status']
                               == 'terminal', what='the queued turn to run')
        assert seen == [False]
    finally:
        await owner.close()
        store.close_all_db_handles()


def _state_bytes(store):
    path = store._db.db_path
    return b''.join(p.read_bytes() for p in (path, path.with_name(path.name + '-wal')) if p.exists())


@pytest.mark.asyncio
async def test_a_rotated_mcp_secret_commits_a_generation_with_the_new_value(tmp_path, monkeypatch):
    """An edit that only rotates a server's ``env`` token leaves the redacted config as it was. The refresh
    compares the private values' fingerprints too: ``reload.mcp`` and a resume (also after an owner restart)
    commit it, and the next turn's worker gets the new token, which never reaches state.db."""
    from gateway.session_authority import initialize_session_authority
    from hermes_state_local import local_receipt
    home = _home(monkeypatch)
    (home / 'config.yaml').write_text(ADDED)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner = _connect(authority)
    cold = viewer = None
    try:
        sid = (await _call(owner, 'session.create', request_id='a', source='cli', cwd=str(tmp_path)))['session_id']
        other = (await _call(owner, 'session.create', request_id='b', source='cli', cwd=str(tmp_path)))['session_id']
        generations = {key: _generation(store, key) for key in (sid, other)}
        ref = local_receipt(store._db, sid)['policy']['config_secret_ref']
        (home / 'config.yaml').write_text(ADDED.replace('FIXT_TOKEN: sekret', 'FIXT_TOKEN: rotated-sekret'))
        result = await _call(owner, 'reload.mcp', session_id=sid, confirm=True)
        assert result['session'] == 'applied' and _generation(store, sid) == generations[sid] + 1, result
        assert local_receipt(store._db, sid)['policy']['config_secret_ref'] != ref
        servers, _ = _worker_servers(_worker_frame(authority, sid, monkeypatch), monkeypatch)
        assert servers['fixt']['env'] == {'FIXT_TOKEN': 'rotated-sekret'}
        # Nothing else moved: the next reload commits nothing.
        assert (await _call(owner, 'reload.mcp', session_id=sid, confirm=True))['session'] == 'unchanged'
        await owner.close()
        # An owner restart has no cached values: the stored reference still names the old token.
        cold = await initialize_session_authority(_runner(store), profile_id=str(home), instance_id='restarted')
        viewer = _connect(cold)
        await _call(viewer, 'session.resume', session_id=other)
        assert _generation(store, other) == generations[other] + 1
        servers, _ = _worker_servers(_worker_frame(cold, other, monkeypatch), monkeypatch)
        assert servers['fixt']['env'] == {'FIXT_TOKEN': 'rotated-sekret'}
        assert b'rotated-sekret' not in _state_bytes(store)
    finally:
        await owner.close()
        if viewer is not None:
            await viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_server_that_fails_to_start_is_unavailable_on_every_reload(tmp_path, monkeypatch):
    """A stdio server whose ``command`` is missing parks after its first connect: discovery keeps the parked
    task registered (it self-probes), yet it serves nothing. ``reload.mcp`` names it unavailable, never as
    added or among the chat's ``session_servers``, on the first reload and on the next."""
    from tools import mcp_tool
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import _server_key
    home = _home(monkeypatch)
    (home / 'config.yaml').write_text(CONFIG.replace('mcp_servers: {old: {command: old-mcp}}\n', ''))
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    authority.runner.config = SimpleNamespace(multiplex_profiles=False)
    authority.runner._mcp_reload_refresh_cached_agents = lambda multiplex, profile: None
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', lambda profile_id: 'default')
    owner = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='p', source='cli', cwd=str(tmp_path)))['session_id']
        (home / 'config.yaml').write_text(CONFIG.replace(
            '{old: {command: old-mcp}}', f'{{brokensrv: {{command: {tmp_path / "no-such-mcp"}}}}}'))
        first = await _call(owner, 'reload.mcp', session_id=sid, confirm=True)
        assert _server_key('brokensrv') in mcp_tool._servers  # parked, still registered
        assert first.get('unavailable') == ['brokensrv'] and first['session_servers'] == [], first
        assert 'added: brokensrv' not in first['message'], first
        second = await owner.dispatch({'id': 2, 'method': 'reload.mcp',
                                       'params': {'session_id': sid, 'confirm': True}})
        assert 'result' not in second and 'brokensrv' in json.dumps(second['error']), second
    finally:
        await asyncio.to_thread(shutdown_mcp_servers)
        for state in (mcp_tool._server_connect_errors, mcp_tool._server_connect_failures,
                      mcp_tool._server_connect_retry_after):
            state.pop(_server_key('brokensrv'), None)
        await owner.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_server_disabled_after_the_chat_picked_it_stops_on_resume_and_on_reload(tmp_path, monkeypatch):
    """A chat whose launch pick names a server keeps it only while config enables it: once it is disabled, a
    resume (and, for another chat, ``reload.mcp``) commits a generation without it whose definition is the
    disabled one, so neither the session nor a managed worker starts it, and reload does not list it."""
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner, later = _connect(authority), _connect(authority)
    try:
        resumed, reloaded = [(await _call(owner, 'session.create', request_id=request, source='cli', cwd=str(tmp_path),
                                          toolsets=['terminal', 'old']))['session_id'] for request in ('a', 'b')]
        generations = {sid: _generation(store, sid) for sid in (resumed, reloaded)}
        (home / 'config.yaml').write_text(CONFIG.replace('{command: old-mcp}', '{command: old-mcp, enabled: false}'))
        await _call(later, 'session.resume', session_id=resumed)
        result = await _call(owner, 'reload.mcp', session_id=reloaded, confirm=True)
        assert result['session'] == 'applied' and result['session_servers'] == [], result
        for sid in (resumed, reloaded):
            frame = _worker_frame(authority, sid, monkeypatch)
            servers, toolsets = _worker_servers(frame, monkeypatch)
            assert 'terminal' in toolsets and 'old' not in toolsets and 'old' not in servers, (toolsets, servers)
            assert json.loads(frame['policy']['config_json'])['mcp_servers']['old']['enabled'] is False
            from gateway.session_policy import policy_for_source
            assert 'old' not in policy_for_source(authority.runner, authority.sessions[sid].source).toolsets
            assert _generation(store, sid) == generations[sid] + 1
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


def test_a_lazy_server_whose_first_connect_failed_is_not_ready(tmp_path, monkeypatch):
    """A lazy (schema-cache) registration that never started is ready. Once its first-use connect fails it keeps
    the registration (a later use retries it) but serves nothing: a missing executable parks a registered
    task, and a server that never answers its handshake times out and leaves no task at all."""
    import sys
    from tools import mcp_tool, mcp_tool_discovery as discovery
    from tools.mcp_tool_lifecycle import shutdown_mcp_servers
    from tools.mcp_tool_scope import _server_key
    lazy = {'missing': {'command': str(tmp_path / 'no-such-mcp'), 'lazy': True, 'connect_timeout': 5},
            'silent': {'command': sys.executable, 'args': ['-c', 'import sys; sys.stdin.read()'], 'lazy': True,
                       'connect_timeout': 1}}
    keys = {name: _server_key(name) for name in lazy}
    monkeypatch.setattr(discovery._config, '_load_mcp_config', lambda: lazy)
    for name, config in lazy.items():
        monkeypatch.setitem(mcp_tool._lazy_server_configs, keys[name], config)
    try:
        assert (discovery.mcp_servers_ready(), discovery.mcp_servers_not_connected()) == (['missing', 'silent'], [])
        for name, key in keys.items():
            assert discovery._ensure_lazy_server_connected(name) is False
            assert key in mcp_tool._lazy_server_configs and key in mcp_tool._server_connect_errors, name
        assert keys['silent'] not in mcp_tool._servers
        assert (discovery.mcp_servers_ready(), discovery.mcp_servers_not_connected()) == ([], ['missing', 'silent'])
    finally:
        shutdown_mcp_servers()
        for state in (mcp_tool._server_connect_errors, mcp_tool._server_connect_failures,
                      mcp_tool._server_connect_retry_after):
            for key in keys.values():
                state.pop(key, None)


TWO = CONFIG.replace('{old: {command: old-mcp}}', '{old: {command: old-mcp}, other: {command: other-mcp}}')


@pytest.mark.asyncio
async def test_a_pick_whose_only_server_is_gone_selects_no_server_not_every_enabled_one(tmp_path, monkeypatch):
    """A launch pick that names one server is an allowlist: once config disables or removes that server, a
    resume or ``reload.mcp`` leaves the chat with no MCP server, never with the other enabled ones."""
    from gateway.session_policy import policy_for_source
    home = _home(monkeypatch)
    (home / 'config.yaml').write_text(TWO)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner, later = _connect(authority), _connect(authority)
    try:
        sids = [(await _call(owner, 'session.create', request_id=request, source='cli', cwd=str(tmp_path),
                             toolsets=['terminal', 'old']))['session_id'] for request in 'abcd']
        generations = {sid: _generation(store, sid) for sid in sids}
        assert 'other' not in policy_for_source(authority.runner, authority.sessions[sids[0]].source).toolsets
        for edited, (resumed, reloaded) in (
                (TWO.replace('{command: old-mcp}', '{command: old-mcp, enabled: false}'), sids[:2]),
                (TWO.replace('old: {command: old-mcp}, ', ''), sids[2:])):
            (home / 'config.yaml').write_text(edited)
            await _call(later, 'session.resume', session_id=resumed)
            result = await _call(owner, 'reload.mcp', session_id=reloaded, confirm=True)
            assert result['session'] == 'applied' and result['session_servers'] == [], result
        for sid in sids:
            frame = _worker_frame(authority, sid, monkeypatch)
            servers, toolsets = _worker_servers(frame, monkeypatch)
            assert 'terminal' in toolsets and not {'old', 'other'} & (toolsets | set(servers)), (toolsets, servers)
            assert not {'old', 'other'} & set(policy_for_source(authority.runner, authority.sessions[sid].source).toolsets)
            assert _generation(store, sid) == generations[sid] + 1
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


def test_not_connected_names_a_configured_server_in_cooldown(monkeypatch):
    from tools import mcp_tool, mcp_tool_discovery as discovery
    from tools.mcp_tool_scope import _server_key
    monkeypatch.setattr(discovery._config, '_load_mcp_config', lambda: {**OLD, 'fixt': {'command': 'x'}})
    monkeypatch.setattr(discovery, '_connect_cooldown_active', lambda name: True)
    monkeypatch.setitem(mcp_tool._servers, _server_key('old'), object())
    assert discovery.mcp_servers_not_connected() == ['fixt']
    # The housekeeping retry still waits the cooldown out.
    assert discovery.mcp_servers_awaiting_connect() == []


def test_reload_counts_only_changes_that_took_effect(monkeypatch):
    """reconcile's ``added`` names the servers it ATTEMPTED and ``pending`` unfinished teardowns: neither is an
    applied change. Only a server that registered (live or lazy), a completed removal or a committed session
    generation turns an unavailable server into partial success."""
    from gateway import session_ancillary
    from hermes_state_runtime import RuntimeStoreError
    refreshed = []
    runner = SimpleNamespace(config=SimpleNamespace(multiplex_profiles=False),
                             _mcp_reload_refresh_cached_agents=lambda multiplex, profile: refreshed.append(profile))
    connection = SimpleNamespace(actor=SimpleNamespace(capabilities={'session:control'}),
                                 authority=SimpleNamespace(runner=runner, profile_id='fixture'))
    reconciled, not_connected = {}, ['broken']
    monkeypatch.setattr('tools.mcp_tool_discovery.reconcile_mcp_servers_with_config', lambda: dict(reconciled))
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_not_connected', lambda: list(not_connected))
    monkeypatch.setattr('tools.mcp_tool_discovery.mcp_servers_ready', lambda: ['fixt'])
    monkeypatch.setattr('gateway.session_authorities.served_profile_name', lambda profile_id: 'default')

    def reload(**outcome):
        reconciled.update({'removed': [], 'added': [], 'pending': [], **outcome})
        return asyncio.run(session_ancillary.reload_mcp(connection, None, {'confirm': True}))
    # Every attempted addition failed (and a teardown is still unfinished): nothing applied, the named refusal.
    for outcome in ({}, {'added': ['broken']}, {'added': ['broken'], 'pending': ['slow']}):
        with pytest.raises(RuntimeStoreError) as refused:
            reload(**outcome)
        assert refused.value.reason == 'mcp_servers_unavailable' and 'broken' in refused.value.detail, outcome
    assert refreshed == ['default'] * 3
    # A genuine mix: success that names only the server that did not connect, and reports only real changes.
    result = reload(added=['broken', 'fixt'])
    assert result['status'] == 'reloaded' and result['unavailable'] == ['broken'], result
    assert 'added: fixt' in result['message'] and 'did not connect: broken' in result['message'], result
    # A completed removal is an applied change too.
    result = reload(removed=['gone'], added=['broken'])
    assert result['unavailable'] == ['broken'] and 'removed: gone' in result['message']
    assert 'added' not in result['message'], result
    not_connected.clear()
    result = reload(added=['fixt'])
    assert result['status'] == 'reloaded' and 'unavailable' not in result


def test_messaging_reload_names_a_server_that_did_not_connect():
    from unittest.mock import patch
    from tests.gateway.test_mcp_reload_refreshes_cached_agents import _make_event, _make_runner_with_cached_agents
    runner = _make_runner_with_cached_agents(num_agents=0)
    with (patch('tools.mcp_tool_lifecycle.shutdown_mcp_servers'),
          patch('tools.mcp_tool_discovery.discover_mcp_tools', return_value=[]),
          patch('tools.mcp_tool_discovery.mcp_servers_not_connected', return_value=['fixt']),
          patch.dict('tools.mcp_tool._servers', {}, clear=True)):
        result = asyncio.run(runner._execute_mcp_reload(_make_event()))
    assert 'did not connect: fixt' in result


@pytest.mark.asyncio
async def test_reload_reports_the_servers_the_calling_chat_selects(tmp_path, monkeypatch):
    """A server connected for the profile is not one this chat runs: ``session_servers`` names the calling
    session's committed selection, so a chat created with ``no_mcp`` reports none of them."""
    home = _home(monkeypatch)
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda authority, policy: None)
    store, authority = await _owner_authority(home)
    _reloadable(authority, monkeypatch)
    owner = _connect(authority)
    try:
        plain = (await _call(owner, 'session.create', request_id='a', source='cli', cwd=str(tmp_path)))['session_id']
        bare = (await _call(owner, 'session.create', request_id='b', source='cli', cwd=str(tmp_path),
                            toolsets=['terminal', 'no_mcp']))['session_id']
        (home / 'config.yaml').write_text(ADDED)
        result = await _call(owner, 'reload.mcp', session_id=plain, confirm=True)
        assert result['session'] == 'applied' and result['session_servers'] == ['fixt', 'old'], result
        result = await _call(owner, 'reload.mcp', session_id=bare, confirm=True)
        assert 'unavailable' not in result and result['session_servers'] == [], result
        # No calling session, no selection to report.
        assert 'session_servers' not in await _call(owner, 'reload.mcp', confirm=True)
    finally:
        await owner.close()
        store.close_all_db_handles()
