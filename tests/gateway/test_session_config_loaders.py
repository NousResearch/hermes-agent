"""A gateway-owned session runs with ``provider_routing`` and ``prefill_messages_file`` from its own
profile's config as of its creation, as a new ``hermes chat`` process read them once at startup on
pre-merge main (CFG-6, CFG-7). The daemon-start snapshot on the runner never reaches it, and the
in-process turn and a managed worker get the same values."""
import json
from types import SimpleNamespace

import pytest

LAUNCH = {'model': 'm', 'provider': 'custom', 'base_url': 'http://127.0.0.1:9/v1'}
STALE = [{'role': 'user', 'content': 'DAEMON_START'}]


def _prefill(home, name, mark):
    (home / name).write_text(json.dumps([{'role': 'user', 'content': mark}]))


class _Built(Exception):
    pass


def test_in_process_turn_routes_with_the_sessions_own_provider_routing(tmp_path, monkeypatch):
    """The routing the in-process turn hands its agent is the session's, not the runner's
    daemon-start ``_provider_routing``."""
    from gateway import session_policy
    from gateway.config import Platform
    from gateway.run_turn_runner import TurnRunner
    from gateway.session_policy import build_policy
    policy = build_policy({'cwd': str(tmp_path), **LAUNCH}, {'provider_routing': {'only': ['Fresh'], 'sort': 'price'}})
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    runner = SimpleNamespace(
        _provider_routing={'only': ['DaemonStart']}, _peek_session_state=lambda key: None,
        session_authority=SimpleNamespace(profile_id=str(tmp_path)),
        _resolve_session_agent_runtime=lambda **kw: ('m', {'provider': 'custom'}),
        _resolve_turn_agent_config=lambda message, model, runtime: {'model': model, 'runtime': runtime})
    turn = TurnRunner.__new__(TurnRunner)
    turn._runner = runner
    turn._ctx = SimpleNamespace(source=SimpleNamespace(platform=Platform.LOCAL, chat_id='c'), session_key='k',
                                user_config={}, message='hi', inbound_message_id=None, enabled_toolsets=[])
    turn._combined_ephemeral_prompt = lambda: None
    turn._setup_stream_consumer = lambda platform_key: (None, None, None, False)

    def built(route, platform_key, ephemeral, max_iterations, reasoning, pr):
        raise _Built(pr)
    turn._resolve_turn_agent = built
    with pytest.raises(_Built) as caught:
        turn._run_sync_scoped()
    assert caught.value.args[0] == {'only': ['Fresh'], 'sort': 'price'}


def test_each_new_session_reads_its_profiles_prefill_once_and_workers_match(tmp_path, monkeypatch):
    """The prefill path and file contents a session gets are its own profile's when it was created; a
    later edit reaches the next new session and never an existing one (main: one read per CLI
    process). Bypass sessions never load the profile's file."""
    from gateway.session_local import _bypass_policy
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import construct_inputs
    monkeypatch.delenv('HERMES_PREFILL_MESSAGES_FILE', raising=False)
    _prefill(tmp_path, 'prefill.json', 'FIRST')
    runner = SimpleNamespace(config=None, _prefill_messages=STALE, _refresh_fallback_model=lambda: None,
                             _get_system_prompt_for_channel=lambda *a: None)
    authority = SimpleNamespace(runner=runner, profile_id=str(tmp_path))
    launch = {'cwd': str(tmp_path), **LAUNCH}
    first = build_policy(launch, {'prefill_messages_file': 'prefill.json'})
    assert construct_inputs(authority, first, 'sid-1')['prefill_messages'] == [{'role': 'user', 'content': 'FIRST'}]
    from gateway.session_worker_construct import session_prefill
    _prefill(tmp_path, 'prefill.json', 'EDITED')
    _prefill(tmp_path, 'moved.json', 'MOVED')
    # The open session keeps what it read (no per-turn re-read); the in-process helper agrees.
    assert construct_inputs(authority, first, 'sid-1')['prefill_messages'][0]['content'] == 'FIRST'
    assert session_prefill(authority, first, 'sid-1')[0]['content'] == 'FIRST'
    # New sessions read the file as it is now, at the path their own config names.
    assert session_prefill(authority, first, 'sid-2')[0]['content'] == 'EDITED'
    moved = build_policy(launch, {'agent': {'prefill_messages_file': str(tmp_path / 'moved.json')}})
    assert construct_inputs(authority, moved, 'sid-3')['prefill_messages'][0]['content'] == 'MOVED'
    assert session_prefill(authority, build_policy(launch, {}), 'sid-4') is None
    config_only = _bypass_policy({**launch, 'ignore_user_config': True}, private_secrets={})
    assert session_prefill(authority, config_only, 'sid-5') is None


def _in_process_prefill(authority, policy, monkeypatch, chat_id):
    """The prefill an in-process agent (re)build hands ``AIAgent`` (``_build_fresh_agent``)."""
    from gateway import session_policy
    from gateway.session_execution import GatewaySessionAgentMixin
    monkeypatch.setattr(session_policy, 'policy_for_source', lambda runner, source: policy)
    built = {}
    turn = GatewaySessionAgentMixin.__new__(GatewaySessionAgentMixin)
    turn._runner = SimpleNamespace(session_authority=authority, _prefill_messages=STALE, _service_tier=None,
                                   _session_db=None, _refresh_fallback_model=lambda: None)
    source = SimpleNamespace(chat_id=chat_id, user_id='u', user_id_alt=None, user_name=None, chat_name=None,
                             chat_type='dm', thread_id=None)
    turn._ctx = SimpleNamespace(source=source, user_config={}, enabled_toolsets=[], disabled_toolsets=None,
                                session_id=chat_id, session_key='k', AIAgent=lambda **kw: built.update(kw))
    turn._build_fresh_agent({'model': 'm', 'runtime': {}}, 'local', None, 1, None, {}, False)
    return built['prefill_messages']


def test_an_open_sessions_prefill_survives_cache_pressure_and_file_edits(tmp_path, monkeypatch):
    """However many other sessions read prefill meanwhile, an open session keeps the examples it
    started with on every later construction: a managed worker's and an in-process rebuild's."""
    from gateway.session_policy import build_policy
    from gateway.session_worker_construct import construct_inputs, session_prefill
    monkeypatch.delenv('HERMES_PREFILL_MESSAGES_FILE', raising=False)
    _prefill(tmp_path, 'prefill.json', 'ORIGINAL')
    runner = SimpleNamespace(config=None, _prefill_messages=STALE, _refresh_fallback_model=lambda: None,
                             _get_system_prompt_for_channel=lambda *a: None)
    authority = SimpleNamespace(runner=runner, profile_id=str(tmp_path))
    policy = build_policy({'cwd': str(tmp_path), **LAUNCH}, {'prefill_messages_file': 'prefill.json'})
    assert session_prefill(authority, policy, 'open')[0]['content'] == 'ORIGINAL'
    _prefill(tmp_path, 'prefill.json', 'EDITED')
    for n in range(600):  # past any bounded acceleration cache
        session_prefill(authority, policy, f'other-{n}')
    assert construct_inputs(authority, policy, 'open')['prefill_messages'][0]['content'] == 'ORIGINAL'
    assert _in_process_prefill(authority, policy, monkeypatch, 'open')[0]['content'] == 'ORIGINAL'


@pytest.mark.asyncio
async def test_prefill_is_read_when_the_session_is_created_and_dropped_with_it(tmp_path, monkeypatch):
    """An edit between ``session.create`` and the first turn never reaches the session (main's CLI
    read the file at startup); deleting the session releases its snapshot."""
    from gateway.session_mutations import retire_live_sessions
    from gateway.session_worker_construct import construct_inputs
    from tests.gateway.test_local_create_env_refs import _owner
    monkeypatch.delenv('HERMES_PREFILL_MESSAGES_FILE', raising=False)
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG + 'prefill_messages_file: prefill.json\n')
    _prefill(home, 'prefill.json', 'AT_CREATE')
    owner, connection = await _owner(home, monkeypatch)
    try:
        ref, policy = _create(owner, connection, home, 'prefill')
        _prefill(home, 'prefill.json', 'BEFORE_FIRST_TURN')
        assert construct_inputs(owner, policy, ref.session_id)['prefill_messages'][0]['content'] == 'AT_CREATE'
        assert _in_process_prefill(owner, policy, monkeypatch, ref.session_id)[0]['content'] == 'AT_CREATE'
        retire_live_sessions(owner, [ref.session_id])
        assert ref.session_id not in owner._local_prefill
    finally:
        await connection.close()
        owner.db.close()


def test_bg_side_agent_uses_the_sessions_provider_routing(tmp_path):
    """``/bg`` through its handler: the side agent it constructs routes with the session's
    ``provider_routing``, not the runner's daemon-start copy."""
    import asyncio
    import run_agent
    from gateway import session_tui_side
    from gateway.session_contract import SessionRef
    from tests.gateway.test_local_create_env_refs import _owner
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG + 'provider_routing: {only: [Fresh], sort: price}\n')
    built = {}

    class Agent:
        def __init__(self, **kwargs):
            built.update(kwargs)

        def run_conversation(self, **kwargs):
            return {'final_response': 'done'}

        def close(self):
            pass

    async def probe(monkeypatch):
        monkeypatch.setattr(run_agent, 'AIAgent', Agent)
        owner, connection = await _owner(home, monkeypatch)
        owner.runner._provider_routing = {'only': ['DaemonStart']}
        owner.runner._peek_session_state = lambda key: None
        owner.runner._adapter_for_source = lambda source: owner.runner.adapters.get(source.platform)
        try:
            ref, _policy = _create(owner, connection, home, 'bg')
            connection.subscriptions[ref.session_id] = 'sub'
            reply = await session_tui_side.background(connection, SessionRef(owner.profile_id, ref.session_id),
                                                      {'session_id': ref.session_id, 'text': 'BG'})
            assert reply['task_id'].startswith('bg_')
            for thread in list(session_tui_side._RUNNING):
                thread.join(30)
        finally:
            connection.subscriptions.clear()
            await connection.close()
            owner.db.close()
    with pytest.MonkeyPatch.context() as monkeypatch:
        asyncio.run(probe(monkeypatch))
    assert (built['providers_allowed'], built['provider_sort']) == (['Fresh'], 'price'), built


CONFIG = ('platform_toolsets: {cli: []}\n'
          'model: {provider: custom, default: m, base_url: "http://127.0.0.1:9/v1"}\n')


def _create(owner, connection, home, request_id):
    from gateway.session_local import create_local_session
    from gateway.session_policy import restore_policy
    from hermes_state_local import local_receipt
    ref = create_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
                                                         'cwd': str(home), 'toolsets': []})
    return ref, restore_policy(local_receipt(owner.db, ref.session_id)['policy'])


async def _prefill_owner(tmp_path, monkeypatch):
    from tests.gateway.test_local_create_env_refs import _owner
    monkeypatch.delenv('HERMES_PREFILL_MESSAGES_FILE', raising=False)
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'config.yaml').write_text(CONFIG + 'prefill_messages_file: prefill.json\n')
    _prefill(home, 'prefill.json', 'FIRST')
    owner, connection = await _owner(home, monkeypatch)
    return home, owner, connection


def _stage(owner, connection, cwd, request_id):
    from gateway.session_local import stage_local_session
    return stage_local_session(owner, connection.actor, {'request_id': request_id, 'source': 'cli',
                                                         'cwd': str(cwd), 'toolsets': []})


def _commit(owner, staged):
    """The off-loop commit and publication every async create runs after staging."""
    from gateway.session_local_recovery import restore_local_session
    from hermes_state_local import commit_local_session
    commit_local_session(owner.db, epoch=owner.epoch, receipt=staged)
    return restore_local_session(owner, staged['session_id'])


@pytest.mark.asyncio
async def test_the_prefill_file_is_read_by_the_off_loop_commit_not_while_staging(tmp_path, monkeypatch):
    """``session.create`` stages on the owner loop and commits through ``tracked_write`` (a worker thread):
    the prefill file, possibly on a slow mount, is read by this attempt's commit, never while staging."""
    import threading
    from functools import partial
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_runtime_workers import tracked_write
    from hermes_cli import cli_config_load
    from hermes_state_local import commit_local_session
    home, owner, connection = await _prefill_owner(tmp_path, monkeypatch)
    readers, real = [], cli_config_load._load_prefill_messages
    monkeypatch.setattr(cli_config_load, '_load_prefill_messages',
                        lambda *args: readers.append(threading.get_ident()) or real(*args))
    try:
        staged = _stage(owner, connection, home, 'slow-mount')
        assert readers == []
        await tracked_write(owner, partial(commit_local_session, owner.db, epoch=owner.epoch, receipt=staged))
        assert len(readers) == 1 and readers[0] != threading.get_ident()
        restore_local_session(owner, staged['session_id'])
        assert _seen(owner, monkeypatch, staged['session_id']) == 'FIRST' and len(readers) == 1
    finally:
        await connection.close()
        owner.db.close()


def _seen(owner, monkeypatch, sid):
    """The prefill a managed worker and an in-process build of ``sid`` get (they must agree)."""
    from gateway.session_policy import restore_policy
    from gateway.session_worker_construct import construct_inputs
    from hermes_state_local import local_receipt
    policy = restore_policy(local_receipt(owner.db, sid)['policy'])
    worker = construct_inputs(owner, policy, sid)['prefill_messages'][0]['content']
    assert _in_process_prefill(owner, policy, monkeypatch, sid)[0]['content'] == worker
    return worker


@pytest.mark.asyncio
@pytest.mark.parametrize('winner', [0, 1])
async def test_overlapping_exact_creates_keep_the_committed_attempts_prefill(tmp_path, monkeypatch, winner):
    """Two in-flight creates of one identity (a retry before the first commit lands), the file
    edited between their commits: the session gets the contents the committing attempt read,
    whichever staged last, and only its snapshot is held."""
    home, owner, connection = await _prefill_owner(tmp_path, monkeypatch)
    try:
        staged = [_stage(owner, connection, home, 'overlap'), _stage(owner, connection, home, 'overlap')]
        ordered = staged if winner == 0 else staged[::-1]
        refs = [_commit(owner, ordered[0])]  # both staged before either commits
        _prefill(home, 'prefill.json', 'SECOND')
        refs.append(_commit(owner, ordered[1]))  # the losing retry reads SECOND, never held
        assert refs[0] == refs[1]
        _prefill(home, 'prefill.json', 'LATER')
        assert _seen(owner, monkeypatch, refs[0].session_id) == 'FIRST'
        assert list(owner._local_prefill) == [refs[0].session_id]
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_a_conflicting_losing_create_never_replaces_the_winners_prefill(tmp_path, monkeypatch):
    from hermes_state_runtime import RuntimeStoreError
    home, owner, connection = await _prefill_owner(tmp_path, monkeypatch)
    try:
        winner = _stage(owner, connection, home, 'conflict')
        loser = _stage(owner, connection, tmp_path, 'conflict')  # same identity, different request
        ref = _commit(owner, winner)
        _prefill(home, 'prefill.json', 'LOSER')
        with pytest.raises(RuntimeStoreError) as refused:
            _commit(owner, loser)
        assert refused.value.reason == 'invalid_params'
        assert _seen(owner, monkeypatch, ref.session_id) == 'FIRST'
        assert list(owner._local_prefill) == [ref.session_id]
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_a_create_refused_by_a_retirement_tombstone_holds_no_prefill(tmp_path, monkeypatch):
    """Retrying the create of a deleted session is refused at commit (``not_found``); the attempt
    must not leave a snapshot behind for a session that cannot exist."""
    from gateway.session_mutations import mutate_session
    from hermes_state_runtime import RuntimeStoreError
    home, owner, connection = await _prefill_owner(tmp_path, monkeypatch)
    owner.runner.session_store._db = owner.db  # one routing store: the delete's live guard requires it
    try:
        ref, _policy = _create(owner, connection, home, 'gone')
        await mutate_session(owner, connection.actor, ref, dict(
            session_id=ref.session_id, request_id='delete', expected_revision=0, expected_generation=0,
            operation='delete', payload={}))
        assert ref.session_id not in owner._local_prefill
        with pytest.raises(RuntimeStoreError) as refused:
            _create(owner, connection, home, 'gone')
        assert refused.value.reason == 'not_found'
        assert ref.session_id not in owner._local_prefill
    finally:
        await connection.close()
        owner.db.close()


@pytest.mark.asyncio
async def test_retried_restored_and_branched_sessions_load_prefill_once_and_keep_it(tmp_path, monkeypatch):
    """A sequential exact retry keeps the original snapshot; a session restored by an owner that
    holds nothing for it (a restart) and a new branch read the file on first use, then keep it."""
    from gateway.session_local_recovery import restore_local_session
    from gateway.session_mutations import mutate_session
    home, owner, connection = await _prefill_owner(tmp_path, monkeypatch)
    try:
        ref, _policy = _create(owner, connection, home, 'kept')
        _prefill(home, 'prefill.json', 'RETRY')
        assert _create(owner, connection, home, 'kept')[0] == ref
        assert _seen(owner, monkeypatch, ref.session_id) == 'FIRST'

        owner._local_prefill.clear()  # a restarted owner: nothing held for its restored sessions
        restore_local_session(owner, ref.session_id)
        _prefill(home, 'prefill.json', 'RESTORED')
        assert _seen(owner, monkeypatch, ref.session_id) == 'RESTORED'
        _prefill(home, 'prefill.json', 'AFTER_RESTORE')
        assert _seen(owner, monkeypatch, ref.session_id) == 'RESTORED'

        result = await mutate_session(owner, connection.actor, ref, dict(
            session_id=ref.session_id, request_id='branch', expected_revision=0, expected_generation=0,
            operation='branch', payload={}))
        child = result['branched_session_id']
        _prefill(home, 'prefill.json', 'BRANCHED')
        assert _seen(owner, monkeypatch, child) == 'BRANCHED'
        _prefill(home, 'prefill.json', 'AFTER_BRANCH')
        assert _seen(owner, monkeypatch, child) == 'BRANCHED'
        assert _seen(owner, monkeypatch, ref.session_id) == 'RESTORED'
    finally:
        await connection.close()
        owner.db.close()
