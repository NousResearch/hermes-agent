"""CFG-2: a resumed local session runs the profile's CURRENT config-derived settings.

Main's ``hermes -c`` was a new CLI process: toolsets, ``terminal.backend``, ``max_turns``, checkpoints and
the reasoning default came from the config it read at startup, while the stored session supplied the
model/provider (``cli_model_switch_mixin.py`` restore). The owner freezes a LocalSessionPolicy at creation,
so ``session.resume`` re-derives those fields from the profile as a new policy generation, keeping the
explicit launch flags, the model route and a committed ``/reasoning`` pick, and never touching a turn
that is already running. Real SessionStore/SessionAuthority and a real config.yaml.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from tests.waiting import async_wait_until

BEFORE = ('model: {default: m1, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
          'agent: {reasoning_effort: high, max_turns: 30}\ncheckpoints: {enabled: false}\n'
          'platform_toolsets: {cli: [web, terminal]}\nterminal: {backend: local}\n')
AFTER = ('model: {default: m2, provider: custom, base_url: "http://127.0.0.1:9/v1"}\n'
         'agent: {reasoning_effort: low, max_turns: 7}\ncheckpoints: {enabled: true}\n'
         'platform_toolsets: {cli: [terminal]}\nterminal: {backend: docker}\n')


def _home(monkeypatch):
    """The profile home the store's state.db lives in: creation and the refresh read its config.yaml."""
    from gateway import run
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(BEFORE)
    # gateway.run binds its home at first import; another test module may have imported it elsewhere.
    monkeypatch.setattr(run, '_hermes_home', home)
    return home


def _runner(store, handle=None):
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _cached_agent_for=lambda route: None, _evict_cached_agent=lambda route: None,
                             _set_session_reasoning_override=lambda route, value: None,
                             _resolve_session_agent_runtime=lambda **k: ('m1', {}), _handle_message=handle)
    runner._adapter_for_source = lambda source: runner.adapters.get(source.platform)
    return runner


async def _call(connection, method, **params):
    reply = await connection.dispatch({'id': 1, 'method': method, 'params': params})
    assert 'result' in reply, reply
    return reply['result']


def _policy(authority, sid):
    from gateway.session_policy import policy_for_source
    return policy_for_source(authority.runner, authority.sessions[sid].source)


def _view(policy):
    return {'web': 'web' in policy.toolsets, 'max_turns': policy.max_turns,
            'checkpoints': policy.checkpoints_enabled, 'effort': policy.reasoning_config.get('effort'),
            'backend': json.loads(policy.terminal_json).get('TERMINAL_ENV'), 'model': policy.model}


@pytest.mark.asyncio
async def test_resume_after_restart_rereads_config_derived_fields_and_keeps_picks(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_controls import AuthorityConnection
    home = _home(monkeypatch)
    store = SessionStore(home / 'sessions', GatewayConfig())
    assert store._db.db_path.parent.resolve() == home.resolve()
    first = await initialize_session_authority(_runner(store), profile_id=str(home), instance_id='first')
    owner = AuthorityConnection(first, object(), {'user_id': 'human'})
    plain = (await _call(owner, 'session.create', request_id='plain', source='cli', cwd=str(tmp_path)))['session_id']
    flags = (await _call(owner, 'session.create', request_id='flags', source='cli', cwd=str(tmp_path),
                         toolsets=['web'], max_turns=5, reasoning='minimal', checkpoints=True))['session_id']
    picked = (await _call(owner, 'session.create', request_id='picked', source='cli', cwd=str(tmp_path)))['session_id']
    snap = store._db.get_session(picked)
    mutation = dict(session_id=picked, request_id='r1', operation='reasoning', payload={'effort': 'medium'},
                    expected_revision=snap['runtime_revision'], expected_generation=snap['runtime_generation'])
    applied = await _call(owner, 'session.mutate', **mutation)
    created = _view(_policy(first, plain))
    assert created == {'web': True, 'max_turns': 30, 'checkpoints': False, 'effort': 'high',
                       'backend': 'local', 'model': 'm1'}, created
    await owner.close()

    # `hermes tools disable web`, `hermes config set ...`, then a gateway restart and `hermes -c`.
    (home / 'config.yaml').write_text(AFTER)
    cold = await initialize_session_authority(_runner(store), profile_id=str(home), instance_id='restarted')
    viewer = AuthorityConnection(cold, object(), {'user_id': 'human'})
    try:
        before = store._db.get_session(plain)['runtime_generation']
        resumed = await _call(viewer, 'session.resume', session_id=plain)
        assert _view(_policy(cold, plain)) == {'web': False, 'max_turns': 7, 'checkpoints': True, 'effort': 'low',
                                              'backend': 'docker', 'model': 'm1'}
        # A new policy generation the resume reply already reports, durable in the receipt.
        assert resumed['execution_generation'] == before + 1 == store._db.get_session(plain)['runtime_generation']
        from hermes_state_local import local_receipt
        from gateway.session_policy import restore_policy
        assert restore_policy(local_receipt(store._db, plain)['policy']) == _policy(cold, plain)
        # The launch flags stay the session's own; a config-derived field still follows.
        await _call(viewer, 'session.resume', session_id=flags)
        kept = _view(_policy(cold, flags))
        assert (kept['web'], kept['max_turns'], kept['checkpoints'], kept['effort']) == (True, 5, True, 'minimal'), kept
        assert kept['backend'] == 'docker'
        # A committed /reasoning pick is the session's choice, not a config default.
        await _call(viewer, 'session.resume', session_id=picked)
        assert _view(_policy(cold, picked))['effort'] == 'medium'
        assert _view(_policy(cold, picked))['web'] is False
        # Nothing changed since: a second resume commits no new generation.
        again = await _call(viewer, 'session.resume', session_id=plain)
        assert again['execution_generation'] == before + 1
        # Exact retries committed before the refresh still resolve to their original outcome: the creation
        # request is untouched, and a mutation receipt answers before any revision or generation fence.
        retried = await _call(viewer, 'session.create', request_id='plain', source='cli', cwd=str(tmp_path))
        assert retried['session_id'] == plain
        assert await _call(viewer, 'session.mutate', **mutation) == applied
    finally:
        await viewer.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resume_never_changes_a_running_turns_policy(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import Submission
    from gateway.session_controls import AuthorityConnection
    import hermes_state_runtime as rt
    home = _home(monkeypatch)
    store = SessionStore(home / 'sessions', GatewayConfig())
    assert store._db.db_path.parent.resolve() == home.resolve()
    started, release, seen = asyncio.Event(), asyncio.Event(), []

    async def handle(event):
        from gateway.session_policy import policy_for_source
        seen.append('web' in policy_for_source(authority.runner, event.source).toolsets)
        started.set()
        await release.wait()
        return 'ok'
    authority = await initialize_session_authority(_runner(store, handle), profile_id=str(home), instance_id='only')
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    viewer = AuthorityConnection(authority, object(), {'user_id': 'human'})
    late = AuthorityConnection(authority, object(), {'user_id': 'human'})
    try:
        sid = (await _call(owner, 'session.create', request_id='busy', source='cli', cwd=str(tmp_path)))['session_id']
        ref = authority.sessions[sid]
        from gateway.session_contract import SessionRef
        turn = await authority.submit(owner.actor, Submission('turn-1', SessionRef(authority.profile_id, sid),
                                                               {'text': 'one'}, 'queue'))
        await asyncio.wait_for(started.wait(), 10)
        (home / 'config.yaml').write_text(AFTER)
        generation = store._db.get_session(sid)['runtime_generation']
        await _call(viewer, 'session.resume', session_id=sid)
        assert 'web' in _policy(authority, sid).toolsets  # the running turn's policy is untouched
        assert store._db.get_session(sid)['runtime_generation'] == generation
        release.set()
        await async_wait_until(lambda: rt.get_session_admission(store._db, admission_id=turn.admission_id)['status']
                               == 'terminal', what='turn-1 to settle')
        # A snapshot poll on an attachment is not an attach: like main's running process, it re-reads nothing.
        await _call(viewer, 'session.resume', session_id=sid)
        assert 'web' in _policy(authority, sid).toolsets
        # The next client attaching to the now idle session does.
        await _call(late, 'session.resume', session_id=sid)
        assert 'web' not in _policy(authority, sid).toolsets
        assert seen == [True] and ref is authority.sessions[sid]
    finally:
        release.set()
        for connection in (owner, viewer, late):
            await connection.close()
        store.close_all_db_handles()


class _Frames:
    """A viewer's socket: the event frames the owner fans out to it."""
    def __init__(self):
        self.frames = []

    def write(self, obj):
        self.frames.append(obj)
        return True

    def close(self):
        pass

    def generations(self):
        return [f['params'].get('execution_generation', f['params']['payload'].get('execution_generation'))
                for f in self.frames if f.get('method') == 'event']


async def _owner_authority(home, instance='only'):
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    store = SessionStore(home / 'sessions', GatewayConfig())
    assert store._db.db_path.parent.resolve() == home.resolve()
    return store, await initialize_session_authority(_runner(store), profile_id=str(home), instance_id=instance)


def _connect(authority, transport=None):
    from gateway.session_controls import AuthorityConnection
    return AuthorityConnection(authority, transport or object(), {'user_id': 'human'})


@pytest.mark.asyncio
async def test_a_resume_cancelled_mid_commit_still_publishes_the_committed_policy(tmp_path, monkeypatch):
    """The refresh commit runs off-loop; a caller cancelled while it is in flight must not leave the stored
    policy ahead of the live one (restore_local_session refuses that mismatch, and a later refresh that finds
    the stored policy current returns early without repairing it)."""
    import threading
    import hermes_state_mutation_prepared as prepared
    from gateway.session_policy import restore_policy
    from gateway.session_runtime_workers import mutation_tasks
    from hermes_state_local import local_receipt
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    owner, viewer = _connect(authority), _connect(authority)
    entered, release = threading.Event(), threading.Event()
    commit = prepared.commit_refreshed_policy

    def held_commit(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return commit(*args, **kwargs)
    monkeypatch.setattr(prepared, 'commit_refreshed_policy', held_commit)
    try:
        sid = (await _call(owner, 'session.create', request_id='c', source='cli', cwd=str(tmp_path)))['session_id']
        (home / 'config.yaml').write_text(AFTER)
        resume = asyncio.create_task(viewer.dispatch({'id': 1, 'method': 'session.resume',
                                                      'params': {'session_id': sid}}))
        await async_wait_until(entered.is_set, what='the refresh commit to start')
        resume.cancel()
        with pytest.raises(asyncio.CancelledError):
            await resume
        release.set()
        await asyncio.gather(*mutation_tasks(authority), return_exceptions=True)
        stored = restore_policy(local_receipt(store._db, sid)['policy'])
        assert stored.max_turns == 7  # the commit landed
        assert _policy(authority, sid) == stored  # and its publication with it
    finally:
        release.set()
        for connection in (owner, viewer):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_resume_keeps_a_moved_workspace_on_a_backend_with_its_own_cwd(tmp_path, monkeypatch):
    """Move to project rewrites the policy cwd and TERMINAL_CWD, not the launch request: an ssh session
    created without an explicit cwd keeps the folder it was moved to through a config refresh."""
    home = _home(monkeypatch)
    remote = 'terminal: {backend: ssh, ssh_host: box.invalid, ssh_user: me}\n'
    (home / 'config.yaml').write_text(BEFORE.replace('terminal: {backend: local}\n', remote))
    moved = tmp_path / 'project'
    moved.mkdir()
    store, authority = await _owner_authority(home)
    owner = _connect(authority)
    later = _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='w', source='cli', cwd=str(tmp_path)))['session_id']
        snap = store._db.get_session(sid)
        await _call(owner, 'session.mutate', session_id=sid, request_id='move', operation='workspace',
                    payload={'cwd': str(moved)}, expected_revision=snap['runtime_revision'],
                    expected_generation=snap['runtime_generation'])
        (home / 'config.yaml').write_text(AFTER.replace('terminal: {backend: docker}\n', remote))
        await _call(later, 'session.resume', session_id=sid)
        policy = _policy(authority, sid)
        assert policy.max_turns == 7  # the refresh ran
        terminal = json.loads(policy.terminal_json)
        assert (terminal['TERMINAL_ENV'], policy.cwd, terminal.get('TERMINAL_CWD')) == ('ssh', str(moved), str(moved))
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_refresh_by_one_viewer_reaches_every_attached_viewer(tmp_path, monkeypatch):
    """Viewer B's attach commits generation G+1; viewer A, attached at G, learns it from the fanout and its
    next generation-fenced mutation succeeds without reopening the session."""
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    socket = _Frames()
    first, second = _connect(authority, socket), _connect(authority)
    try:
        sid = (await _call(first, 'session.create', request_id='v', source='cli', cwd=str(tmp_path)))['session_id']
        attached = await _call(first, 'session.resume', session_id=sid)
        (home / 'config.yaml').write_text(AFTER)
        refreshed = await _call(second, 'session.resume', session_id=sid)
        assert refreshed['execution_generation'] == attached['execution_generation'] + 1
        await async_wait_until(lambda: refreshed['execution_generation'] in socket.generations(),
                               what='viewer A to receive the refreshed generation')
        revisions = [f['params']['payload']['revision'] for f in socket.frames
                     if f['params'].get('type') == 'session.info' and 'revision' in f['params']['payload']]
        info = [f['params']['payload'] for f in socket.frames if f['params'].get('type') == 'session.info'
                and 'launch_request' in f['params']['payload']]
        assert info and info[-1]['cwd'] == str(tmp_path)  # the refreshed policy projection
        mutated = await _call(first, 'session.mutate', session_id=sid, request_id='after', operation='reasoning',
                              payload={'effort': 'medium'}, expected_revision=max([attached['revision'], *revisions]),
                              expected_generation=max(g for g in socket.generations() if g is not None))
        assert mutated['execution_generation'] == refreshed['execution_generation'] + 1
    finally:
        for connection in (first, second):
            await connection.close()
        store.close_all_db_handles()


def hold_refresh_commit(monkeypatch):
    """A barrier at the refresh's commit seam (its preflight has read the stored policy and generation, its
    write transaction has not begun): ``(entered, release, done, outcome)``, *outcome* what the commit did."""
    import threading
    import hermes_state_mutation_prepared as prepared
    entered, release, done, outcome = threading.Event(), threading.Event(), threading.Event(), []
    commit = prepared.commit_refreshed_policy

    def held_commit(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        try:
            outcome.append(commit(*args, **kwargs))
        except Exception as exc:
            outcome.append(exc)
            raise
        finally:
            done.set()
    monkeypatch.setattr(prepared, 'commit_refreshed_policy', held_commit)
    return entered, release, done, outcome


async def admit_during_refresh(authority, actor, sid, refresh, barrier):
    """Run *refresh* to its commit seam, commit a queued admission there, then let the write transaction run
    while that admission is still queued: the owner loop is held until it finishes, so the drain (a task the
    admission scheduled) cannot start the turn first. The refused transaction left the stored policy and its
    generation as they were. ``(refresh result, admission)``."""
    import hermes_state_runtime as rt
    from gateway.session_contract import SessionRef, Submission
    from hermes_state_local import local_receipt
    from hermes_state_runtime import RuntimeStoreError
    entered, release, done, outcome = barrier
    task = asyncio.create_task(refresh)
    await async_wait_until(entered.is_set, what='the refresh to reach its commit')
    stored = local_receipt(authority.db, sid)['policy']
    turn = await authority.submit(actor, Submission('queued-1', SessionRef(authority.profile_id, sid),
                                                    {'text': 'one'}, 'queue'))
    assert rt.get_session_admission(authority.db, admission_id=turn.admission_id)['status'] == 'queued'
    generation = authority.db.get_session(sid)['runtime_generation']
    release.set()
    assert done.wait(10)  # blocks the loop on purpose
    assert isinstance(outcome[0], RuntimeStoreError) and outcome[0].reason == 'session_busy', outcome
    assert rt.get_session_admission(authority.db, admission_id=turn.admission_id)['status'] == 'queued'
    assert local_receipt(authority.db, sid)['policy'] == stored
    assert authority.db.get_session(sid)['runtime_generation'] == generation
    return await task, turn


@pytest.mark.asyncio
async def test_an_admission_queued_during_an_attach_refresh_keeps_the_old_policy(tmp_path, monkeypatch):
    """The idle check is inside the refresh's write transaction: a turn queued after the refresh read the
    stored policy refuses the commit, and that turn runs the policy it was admitted under."""
    import hermes_state_runtime as rt
    from hermes_state_local import local_receipt
    home = _home(monkeypatch)
    seen = []

    async def handle(event):
        from gateway.session_policy import policy_for_source
        seen.append('web' in policy_for_source(authority.runner, event.source).toolsets)
        return 'ok'
    store, authority = await _owner_authority(home)
    authority.runner._handle_message = handle
    owner, later = _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='q', source='cli', cwd=str(tmp_path)))['session_id']
        stored = local_receipt(store._db, sid)['policy']
        (home / 'config.yaml').write_text(AFTER)
        barrier = hold_refresh_commit(monkeypatch)
        resume = later.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}})
        reply, turn = await admit_during_refresh(authority, owner.actor, sid, resume, barrier)
        assert 'result' in reply, reply  # the refusal never fails the attach
        assert local_receipt(store._db, sid)['policy'] == stored
        await async_wait_until(lambda: rt.get_session_admission(store._db, admission_id=turn.admission_id)['status']
                               == 'terminal', what='the queued turn to run')
        assert seen == [True] and 'web' in _policy(authority, sid).toolsets
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


async def _mutate(connection, store, sid, request_id, operation, payload):
    snap = store._db.get_session(sid)
    return await _call(connection, 'session.mutate', session_id=sid, request_id=request_id, operation=operation,
                       payload=payload, expected_revision=snap['runtime_revision'],
                       expected_generation=snap['runtime_generation'])


@pytest.mark.asyncio
async def test_a_branch_keeps_the_reasoning_and_workspace_its_parent_picked(tmp_path, monkeypatch):
    """A branch copies its parent's policy, picks included: resuming it after a config change keeps the
    picked reasoning level and, on ssh, the folder the parent was moved to (the receipts are the parent's)."""
    home = _home(monkeypatch)
    remote = 'terminal: {backend: ssh, ssh_host: box.invalid, ssh_user: me}\n'
    (home / 'config.yaml').write_text(BEFORE.replace('terminal: {backend: local}\n', remote))
    moved = tmp_path / 'project'
    moved.mkdir()
    store, authority = await _owner_authority(home)
    owner, later = _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='p', source='cli', cwd=str(tmp_path)))['session_id']
        await _mutate(owner, store, sid, 'move', 'workspace', {'cwd': str(moved)})
        await _mutate(owner, store, sid, 'pick', 'reasoning', {'effort': 'medium'})
        child = (await _mutate(owner, store, sid, 'fork', 'branch', {}))['branched_session_id']
        grandchild = (await _mutate(owner, store, child, 'fork-again', 'branch', {}))['branched_session_id']
        (home / 'config.yaml').write_text(AFTER.replace('terminal: {backend: docker}\n', remote))
        for branch in (child, grandchild):
            await _call(later, 'session.resume', session_id=branch)
            policy = _policy(authority, branch)
            assert policy.max_turns == 7  # the refresh ran
            terminal = json.loads(policy.terminal_json)
            assert (policy.reasoning_config.get('effort'), policy.cwd, terminal.get('TERMINAL_CWD')) == (
                'medium', str(moved), str(moved)), branch
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_branch_stored_before_receipts_carried_picks_keeps_its_lineages_picks(tmp_path, monkeypatch):
    """A branch stored before its receipt recorded ``picks`` takes them from its lineage: the parent's pick
    receipts committed before the branch, and the parent's own inherited picks, walking up. A pick the parent
    committed after the branch is not the branch's."""
    from hermes_state_local import POLICY_PREFIX
    from hermes_state_mutation_prepared import policy_picks
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    owner, later = _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='p', source='cli', cwd=str(tmp_path)))['session_id']
        await _mutate(owner, store, sid, 'pick', 'reasoning', {'effort': 'medium'})
        child = (await _mutate(owner, store, sid, 'fork', 'branch', {}))['branched_session_id']
        grandchild = (await _mutate(owner, store, child, 'fork-again', 'branch', {}))['branched_session_id']
        (tmp_path / 'project').mkdir()
        await _mutate(owner, store, sid, 'move', 'workspace', {'cwd': str(tmp_path / 'project')})
        for branch in (child, grandchild):  # the receipt shape before this field existed
            store._db._execute_write(lambda conn, key=POLICY_PREFIX + branch: conn.execute(
                "UPDATE state_meta SET value=json_remove(value, '$.picks') WHERE key=?", (key,)))
        (home / 'config.yaml').write_text(AFTER)
        for branch in (child, grandchild):
            assert policy_picks(store._db, branch) == {'reasoning'}, branch
            await _call(later, 'session.resume', session_id=branch)
            policy = _policy(authority, branch)
            assert (policy.max_turns, policy.reasoning_config.get('effort')) == (7, 'medium'), branch
    finally:
        for connection in (owner, later):
            await connection.close()
        store.close_all_db_handles()


class _Sandbox:
    cleaned = False

    def cleanup(self):
        self.cleaned = True


@pytest.mark.asyncio
async def test_a_refreshed_terminal_binding_drops_the_sessions_cached_environment(tmp_path, monkeypatch):
    """The terminal tool reuses a cached environment by key, whatever backend made it: a refresh that moves
    the session's terminal binding drops the session's own environment before it is reported, so the next
    command builds one on the new backend. A refresh that leaves the binding alone, and environments
    the profile's other sessions share, keep theirs."""
    from tools import terminal_tool
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    owner, later, last = _connect(authority), _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='cli', cwd=str(tmp_path)))['session_id']
        own, shared = _Sandbox(), _Sandbox()
        key = f'session:{authority.sessions[sid].route}'
        monkeypatch.setitem(terminal_tool._active_environments, key, own)
        monkeypatch.setitem(terminal_tool._active_environments, 'default', shared)
        (home / 'config.yaml').write_text(BEFORE.replace('max_turns: 30', 'max_turns: 9'))
        await _call(later, 'session.resume', session_id=sid)
        assert _policy(authority, sid).max_turns == 9 and terminal_tool._active_environments[key] is own
        (home / 'config.yaml').write_text(AFTER)
        await _call(last, 'session.resume', session_id=sid)
        assert json.loads(_policy(authority, sid).terminal_json)['TERMINAL_ENV'] == 'docker'
        assert key not in terminal_tool._active_environments and own.cleaned
        assert terminal_tool._active_environments['default'] is shared and not shared.cleaned
    finally:
        for connection in (owner, later, last):
            await connection.close()
        store.close_all_db_handles()


class _Backend:
    """An environment the terminal tool built: its backend and the directory each command ran in."""
    def __init__(self, env_type, cwd):
        self.env_type, self.cwd, self.ran = env_type, cwd, []

    def execute(self, command, *, cwd=None, **_):
        self.ran.append(cwd)
        after = command.split(' ', 1)[1] if command.startswith('cd ') else cwd
        return {'output': '', 'returncode': 0, 'cwd': after, 'cwd_observed': True}

    def cleanup(self):
        pass


def _terminal(authority, sid, command):
    """One terminal call as the session's turn makes it: its route's session context, its policy's scope."""
    from gateway.session_context import clear_session_vars, set_session_vars
    from gateway.session_policy import policy_scope
    from tools.terminal_tool import terminal_tool
    tokens = set_session_vars(platform='local', session_key=authority.sessions[sid].route, session_id=sid)
    try:
        with policy_scope(_policy(authority, sid), authority=authority):
            return json.loads(terminal_tool(command, task_id=sid))
    finally:
        clear_session_vars(tokens)


@pytest.mark.asyncio
async def test_the_next_command_after_a_terminal_refresh_runs_on_the_new_backend_in_its_directory(tmp_path, monkeypatch):
    """The directory a session's commands last recorded wins over its binding's default: a refresh that moves
    the binding drops it with the environment, so the next command runs on the new backend in the new
    binding's directory, not the old backend's. A refresh that leaves the binding alone keeps it."""
    from tools import terminal_tool
    built = []
    monkeypatch.setattr(terminal_tool, '_active_environments', {})
    monkeypatch.setattr(terminal_tool, '_session_cwd', {})
    monkeypatch.setattr(terminal_tool, '_pre_exec_block', lambda *a, **k: None)
    monkeypatch.setattr(terminal_tool, '_create_configured_env',
                        lambda config, env_type, **k: built.append(_Backend(env_type, k['cwd'])) or built[-1])
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    owner, later, last = _connect(authority), _connect(authority), _connect(authority)
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='cli', cwd=str(tmp_path)))['session_id']
        assert _terminal(authority, sid, 'cd /srv/old-backend')['exit_code'] == 0
        (home / 'config.yaml').write_text(BEFORE.replace('max_turns: 30', 'max_turns: 9'))
        await _call(later, 'session.resume', session_id=sid)
        _terminal(authority, sid, 'pwd')
        assert (built[-1].env_type, built[-1].ran[-1]) == ('local', '/srv/old-backend')
        (home / 'config.yaml').write_text(AFTER)
        await _call(last, 'session.resume', session_id=sid)
        fresh = (await _call(owner, 'session.create', request_id='f', source='cli', cwd=str(tmp_path)))['session_id']
        _terminal(authority, fresh, 'pwd')
        _terminal(authority, sid, 'pwd')
        # Both on docker (a persistent container is the profile's, one for both), in the new binding's directory.
        ran = [cwd for backend in built if backend.env_type == 'docker' for cwd in backend.ran]
        assert len(ran) == 2 and ran[1] == ran[0] != '/srv/old-backend', [(b.env_type, b.ran) for b in built]
    finally:
        for connection in (owner, later, last):
            await connection.close()
        store.close_all_db_handles()


@pytest.mark.asyncio
async def test_a_second_resume_during_terminal_cleanup_waits_for_the_refresh(tmp_path, monkeypatch, caplog):
    """A refresh that moves the terminal binding commits, then tears the session's sandbox down (10-15 s on a
    remote backend) before it publishes. A second viewer attaching meanwhile waits for that refresh instead of
    failing ``storage_unavailable`` on a stored policy ahead of the live one, and commits nothing itself: one
    generation, each environment torn down once."""
    import logging
    import threading
    from tools import terminal_tool_lifecycle
    home = _home(monkeypatch)
    store, authority = await _owner_authority(home)
    entered, release, cleaned = threading.Event(), threading.Event(), []

    def held_cleanup(key):
        cleaned.append(key)
        entered.set()
        assert release.wait(10)
    monkeypatch.setattr(terminal_tool_lifecycle, 'cleanup_vm', held_cleanup)
    owner, first, second = _connect(authority), _connect(authority), _connect(authority)
    resumes = []
    try:
        sid = (await _call(owner, 'session.create', request_id='t', source='cli', cwd=str(tmp_path)))['session_id']
        generation = store._db.get_session(sid)['runtime_generation']
        (home / 'config.yaml').write_text(AFTER)
        resume = {'id': 1, 'method': 'session.resume', 'params': {'session_id': sid}}
        resumes.append(asyncio.create_task(first.dispatch(resume)))
        await async_wait_until(entered.is_set, what='the first refresh to start its terminal cleanup')
        with caplog.at_level(logging.DEBUG, logger='gateway.session_policy_refresh'):
            resumes.append(asyncio.create_task(second.dispatch(resume)))
            await async_wait_until(lambda: resumes[1].done() or 'waits for its policy refresh' in caplog.text,
                                   what='the second resume to finish or wait')
        assert not resumes[1].done(), resumes[1].result()
        release.set()
        replies = await asyncio.gather(*resumes)
        assert [reply['result']['execution_generation'] for reply in replies] == [generation + 1] * 2, replies
        assert store._db.get_session(sid)['runtime_generation'] == generation + 1
        assert cleaned and len(cleaned) == len(set(cleaned)), cleaned
    finally:
        release.set()
        await asyncio.gather(*resumes, return_exceptions=True)
        for connection in (owner, first, second):
            await connection.close()
        store.close_all_db_handles()


def test_a_skipped_resume_refresh_warns_unless_the_session_is_busy(monkeypatch, caplog):
    """A turn running or queued is the expected skip; a revoked credential or an unreadable profile leaves a
    silent no-refresh, so it is a WARNING."""
    import logging
    from gateway import session_policy_refresh as refresh
    from hermes_state_runtime import RuntimeStoreError
    for failure in (RuntimeStoreError('session_busy'), RuntimeStoreError('launch_credentials_unavailable'),
                    OSError('config.yaml unreadable')):
        async def fail(authority, ref, failure=failure):
            raise failure
        monkeypatch.setattr(refresh, 'commit_refresh', fail)
        with caplog.at_level(logging.INFO, logger=refresh.__name__):
            asyncio.run(refresh.refresh_resumed_policy(None, SimpleNamespace(session_id='s1')))
    levels = [(record.levelno, record.getMessage().rsplit(': ', 1)[1]) for record in caplog.records
              if 'keeps its stored policy' in record.getMessage()]
    assert levels == [(logging.INFO, 'session_busy'), (logging.WARNING, 'launch_credentials_unavailable'),
                      (logging.WARNING, 'OSError')], levels
