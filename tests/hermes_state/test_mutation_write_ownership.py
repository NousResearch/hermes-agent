"""Every row touched by an authority mutation must still belong to this store."""
import asyncio
from pathlib import Path

import pytest

from hermes_state import SessionDB
from hermes_state_lineage import SessionIdOwnedElsewhereError, SessionLineageAmbiguousError
from hermes_state_runtime import begin_runtime_epoch, mutate_runtime_session


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    profile = tmp_path / "profiles" / "science"
    profile.mkdir(parents=True)
    with SessionDB(tmp_path / "state.db") as local, SessionDB(profile / "state.db") as sibling:
        yield local, sibling


def snapshot(db):
    return {table: [tuple(row) for row in db._conn.execute(f"SELECT * FROM {table} ORDER BY 1")]
            for table in ("sessions", "messages", "state_meta", "gateway_routing", "system_prompts",
                          "worker_executions", "worker_receipts")}


def foreign(db, session_id):
    db._conn.execute("INSERT INTO sessions(id,source,started_at,profile_name) VALUES (?,'cli',1,'science')",
                     (session_id,))
    db._conn.commit()


def mutate(db, epoch, session_id, operation, payload, *, generation=None):
    return mutate_runtime_session(db, epoch=epoch, principal_id="owner", session_id=session_id,
                                  request_id="contested", expected_revision=0,
                                  expected_generation=generation, operation=operation, payload=payload)


@pytest.mark.parametrize("operation,payload,generation", [
    ("rename", {"title": "Changed"}, None),
    ("sidebar", {"pinned": True}, None),
    ("archive", {"archived": True}, None),
    ("delete", {}, 0),
])
def test_mutation_refuses_foreign_owned_requested_row(stores, operation, payload, generation):
    local, sibling = stores
    local.create_session("contested", source="cli")
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    foreign(sibling, "contested")
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        mutate(local, epoch, "contested", operation, payload, generation=generation)
    assert (snapshot(local), snapshot(sibling)) == before


def test_import_refuses_foreign_owned_payload_id(stores):
    local, sibling = stores
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    foreign(sibling, "imported")
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        mutate(local, epoch, "imported", "import", {"sessions": [
            {"id": "imported", "source": "cli", "messages": [{"role": "user", "content": "data"}]}
        ]})
    assert (snapshot(local), snapshot(sibling)) == before


def test_sidebar_fences_resolved_compression_ancestor(stores):
    local, sibling = stores
    for sid, parent, reason in (("root", None, "compression"), ("tip", "root", None)):
        local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id,end_reason) "
                            "VALUES (?,'cli',1,?,?)", (sid, parent, reason))
    local._conn.commit()
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    foreign(sibling, "root")
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        mutate(local, epoch, "tip", "sidebar", {"pinned": True})
    assert (snapshot(local), snapshot(sibling)) == before


def test_delete_fences_fk_detached_child(stores):
    local, sibling = stores
    local.create_session("root", source="cli")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id) "
                        "VALUES ('branch','cli',1,'root')")
    local._conn.commit()
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    foreign(sibling, "branch")
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        mutate(local, epoch, "root", "delete", {}, generation=0)
    assert (snapshot(local), snapshot(sibling)) == before


def test_rename_fences_title_conflict_row(stores):
    local, sibling = stores
    local.create_session("target", source="cli")
    local.create_session("ghost", source="cli")
    local._conn.execute("UPDATE sessions SET title='Wanted', ended_at=2 WHERE id='ghost'")
    local._conn.commit()
    epoch = begin_runtime_epoch(local, instance_id="fixture")
    foreign(sibling, "ghost")
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionIdOwnedElsewhereError):
        mutate(local, epoch, "target", "rename", {"title": "Wanted"})
    assert (snapshot(local), snapshot(sibling)) == before


def test_import_fences_existing_skipped_payload_id(stores):
    local, sibling = stores
    local.create_session('existing', source='cli')
    epoch = begin_runtime_epoch(local, instance_id='fixture')
    foreign(sibling, 'existing')
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        mutate(local, epoch, 'new', 'import', {'sessions': [
            {'id': 'new', 'source': 'cli', 'messages': []},
            {'id': 'existing', 'source': 'cli', 'messages': []}]})
    assert (snapshot(local), snapshot(sibling)) == before


def test_sidebar_preserves_explicit_fork_boundary(stores):
    local, sibling = stores
    local._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                        "VALUES ('root','cli',1,2,'compression')")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id,model_config) "
                        "VALUES ('fork','cli',1,'root',?)", ('{"_branched_from":"root"}',))
    local._conn.commit()
    epoch = begin_runtime_epoch(local, instance_id='fixture')
    foreign(sibling, 'root')
    original_root = dict(local.get_session('root'))
    original_sibling = snapshot(sibling)
    result = mutate(local, epoch, 'fork', 'sidebar', {'pinned': True})
    assert result['pinned'] is True
    assert local.get_session('fork')['pinned'] == 1
    assert dict(local.get_session('root')) == original_root
    assert snapshot(sibling) == original_sibling


@pytest.mark.parametrize('operation', [
    'title', 'pin', 'archive', 'hidden', 'read', 'delete', 'bulk_delete',
    'empty_delete', 'discard', 'empty_sweep',
])
def test_direct_session_writers_refuse_live_duplicate(stores, operation):
    local, sibling = stores
    local.create_session('contested', source='cli')
    if operation == 'empty_sweep':
        local._conn.execute("UPDATE sessions SET ended_at=2 WHERE id='contested'")
        local._conn.commit()
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    actions = {
        'title': lambda: local.set_session_title('contested', 'Changed'),
        'pin': lambda: local.set_session_pinned('contested', True),
        'archive': lambda: local.set_session_archived('contested', True),
        'hidden': lambda: local.set_session_hidden('contested', True),
        'read': lambda: local.set_session_read('contested', False),
        'delete': lambda: local.delete_session('contested'),
        'bulk_delete': lambda: local.delete_sessions(['contested']),
        'empty_delete': lambda: local.delete_session_if_empty('contested'),
        'discard': lambda: local.discard_unadmitted_session('contested'),
        'empty_sweep': lambda: local.delete_empty_sessions(),
    }
    expected = SessionIdOwnedElsewhereError if operation == 'empty_sweep' else SessionLineageAmbiguousError
    with pytest.raises(expected):
        actions[operation]()
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.mark.parametrize('shape', ['legacy_reset_child', 'compression_lineage'])
def test_reopen_fences_rows_it_changes_beyond_requested_id(stores, shape):
    local, sibling = stores
    reason = 'session_reset' if shape == 'legacy_reset_child' else 'compression'
    route = 'agent:main:local:dm:root'
    local._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason,session_key) "
                        "VALUES ('root','cli',1,2,?,?)", (reason, route))
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id,archived,auto_archived,session_key) "
                        "VALUES ('child','cli',3,'root',1,1,?)", (route,))
    local._conn.commit()
    foreign(sibling, 'child')
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        local.reopen_session('root')
    assert (snapshot(local), snapshot(sibling)) == before


def test_conversation_config_update_fences_resolved_compression_tip(stores):
    local, sibling = stores
    local._conn.execute("INSERT INTO sessions(id,source,started_at,ended_at,end_reason) "
                        "VALUES ('root','cli',1,2,'compression')")
    local._conn.execute("INSERT INTO sessions(id,source,started_at,parent_session_id) "
                        "VALUES ('tip','cli',3,'root')")
    local._conn.commit()
    foreign(sibling, 'tip')
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        local.update_conversation_model_config_value('root', 'selection', lambda old: 'changed')
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.mark.parametrize('operation', [
    'expiry', 'cwd', 'git_metadata', 'activity', 'clear_activity', 'meta',
    'system_prompt', 'tool_names', 'model', 'config_patch', 'runtime_lock',
    'yolo', 'title_source',
])
def test_direct_session_metadata_writers_refuse_live_duplicate(stores, operation):
    local, sibling = stores
    local.create_session('contested', source='cli')
    if operation == 'git_metadata':
        local._conn.execute("UPDATE sessions SET cwd='/tmp/project',git_metadata_generation=1 WHERE id='contested'")
    if operation == 'clear_activity':
        local._conn.execute("UPDATE sessions SET last_activity_description='working',last_activity_provenance='tool' "
                            "WHERE id='contested'")
    if operation == 'title_source':
        local._conn.execute("UPDATE sessions SET title='Original',title_source='derived' WHERE id='contested'")
    local._conn.commit()
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    actions = {
        'expiry': lambda: local.set_expiry_finalized('contested'),
        'cwd': lambda: local.update_session_cwd('contested', '/tmp/project'),
        'git_metadata': lambda: local.publish_session_git_metadata('contested', '/tmp/project', 1, git_branch='main'),
        'activity': lambda: local.touch_session_activity('contested', ts=9999999999),
        'clear_activity': lambda: local.clear_session_activity_labels('contested'),
        'meta': lambda: local.update_session_meta('contested', '{"changed":true}'),
        'system_prompt': lambda: local.update_system_prompt('contested', 'Changed prompt'),
        'tool_names': lambda: local.update_session_tool_names('contested', ['changed']),
        'model': lambda: local.update_session_model('contested', 'changed'),
        'config_patch': lambda: local.patch_session_model_config('contested', {'changed': True}),
        'runtime_lock': lambda: local.update_session_runtime_lock('contested', model='changed'),
        'yolo': lambda: local.set_session_yolo('contested', True),
        'title_source': lambda: local.set_session_title_source('contested', 'user'),
    }
    with pytest.raises(SessionLineageAmbiguousError):
        actions[operation]()
    assert (snapshot(local), snapshot(sibling)) == before


def test_messaging_pick_receipt_refuses_live_duplicate(stores):
    from hermes_state_mutation_chat_pick import chat_pick_key, commit_chat_pick

    local, sibling = stores
    local.create_session('contested', source='telegram')
    epoch = begin_runtime_epoch(local, instance_id='fixture')
    params = {'request_id': 'pick', 'expected_revision': 0, 'expected_generation': 0,
              'operation': 'reasoning', 'payload': {'effort': 'low'}}
    key, digest = chat_pick_key('owner', 'contested', params)
    foreign(sibling, 'contested')
    before = snapshot(local), snapshot(sibling)
    with pytest.raises(SessionLineageAmbiguousError):
        commit_chat_pick(local, epoch=epoch, key=key, digest=digest, session_id='contested',
                         params=params, result={'session_id': 'contested', 'operation': 'reasoning',
                                                'effort': 'low'})
    assert (snapshot(local), snapshot(sibling)) == before


@pytest.fixture
def local_owner(tmp_path, monkeypatch):
    from gateway import run
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionActor
    from gateway.session_local import create_local_session
    from tests.gateway.test_authority_cold_route_recovery import _Runner
    from tests.gateway.test_handoff_local_receipt import _compress

    home = tmp_path / '.hermes'
    home.mkdir()
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(run, '_load_gateway_config', lambda *_: {
        'platform_toolsets': {'cli': []}, 'model': {'default': 'fixture'}})
    runner = _Runner(home)
    authority = asyncio.run(initialize_session_authority(runner, profile_id='fixture', instance_id='operator'))
    actor = SessionActor('owner', 'fixture', frozenset({'session:create', 'session:read', 'session:control'}), 'socket')
    root = create_local_session(authority, actor, {'request_id': 'root', 'source': 'gui',
        'cwd': str(home), 'model': 'frozen', 'toolsets': []}).session_id
    _compress(authority, root)
    # The gateway test fixture supplies a process home independent of the
    # temporary home used for the runner's configuration.
    managed_home = authority.db.db_path.parent
    monkeypatch.setenv('HERMES_HOME', str(managed_home))
    try:
        yield authority.db, root, managed_home, authority.epoch
    finally:
        authority.db.close()


def _sibling(home):
    profile = home / 'profiles' / 'science'
    profile.mkdir(parents=True, exist_ok=True)
    return SessionDB(profile / 'state.db')


def test_compression_publication_fences_local_receipt_owner(local_owner):
    local, root, home, _ = local_owner
    with _sibling(home) as sibling:
        foreign(sibling, root)
        before = snapshot(local), snapshot(sibling)
        with pytest.raises(SessionIdOwnedElsewhereError):
            local.publish_compression_child(
                parent_session_id=root + '-c2', child_session_id=root + '-c3',
                source='local', messages=[{'role': 'user', 'content': 'continued'}],
                require_compression_lease=False)
        assert (snapshot(local), snapshot(sibling)) == before


def test_worker_compression_receipt_fences_local_root(local_owner, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import mutate_worker_execution, register_worker_execution

    local, root, home, epoch = local_owner
    tip = root + '-c2'
    scope = dict(epoch=epoch, execution_id='worker', session_id=tip, generation=0)
    register_worker_execution(local, **scope, kind='compute', adoption_secret='fixture-only', require_idle=True)
    store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(local, **params),
                                scope, tmp_path / 'outbox')
    try:
        assert store.try_acquire_compression_lock(tip, 'holder')
        with _sibling(home) as sibling:
            foreign(sibling, root)
            before = snapshot(local), snapshot(sibling)
            with pytest.raises(SessionIdOwnedElsewhereError):
                store.publish_compression_child(
                    parent_session_id=tip, child_session_id=root + '-c3', source='local',
                    messages=[{'role': 'user', 'content': 'continued'}],
                    compression_lock_holder='holder')
            assert (snapshot(local), snapshot(sibling)) == before
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


def test_worker_usage_receipt_fences_stamped_ancestor(local_owner, tmp_path):
    from agent.runtime_session_store import RuntimeSessionStore
    from hermes_state_runtime import mutate_worker_execution, register_worker_execution

    local, root, home, epoch = local_owner
    tip = root + '-c2'
    scope = dict(epoch=epoch, execution_id='worker', session_id=tip, generation=0)
    register_worker_execution(local, **scope, kind='compute', adoption_secret='fixture-only', require_idle=True)
    store = RuntimeSessionStore(lambda method, **params: mutate_worker_execution(local, **params),
                                scope, tmp_path / 'outbox')
    try:
        with _sibling(home) as sibling:
            foreign(sibling, root)
            before = snapshot(local), snapshot(sibling)
            with pytest.raises(SessionIdOwnedElsewhereError):
                store.queue_token_counts(tip, source='gui', input_tokens=1)
            assert (snapshot(local), snapshot(sibling)) == before
    finally:
        store.failure = None
        store.journal['pending'] = []
        store.close()


@pytest.mark.parametrize('operation,payload', [
    ('rewind', {'target_message_id': 1}),
    ('reset', {}),
    ('model', {'model': 'frozen'}),
    ('reasoning', {'effort': 'low'}),
    ('workspace', {'cwd': '/tmp/another-project'}),
    ('editor_mcp', {'mcp_servers': {'fingerprint': 'fixture'}}),
    ('edit_policy', {'policy': 'session'}),
])
def test_mutation_fences_resolved_physical_target(local_owner, operation, payload):
    local, root, home, epoch = local_owner
    with _sibling(home) as sibling:
        foreign(sibling, root + '-c2')
        before = snapshot(local), snapshot(sibling)
        expected_revision = local.get_session(root)['runtime_revision']
        expected_generation = local.get_session(root)['runtime_generation']
        if operation == 'rewind':
            payload = {'target_message_id': local.get_messages(root + '-c2')[0]['id']}
        prepared = None
        if operation in {'model', 'reasoning', 'workspace', 'editor_mcp', 'edit_policy'}:
            from hermes_state_mutation_prepared import local_snapshot
            with local._read_ctx() as conn:
                state = local_snapshot(local, conn, root)
            policy = dict(state['receipt']['policy'])
            prepared = {'snapshot': state, 'policy': policy, 'git': {'branch': '', 'root': ''}}
        with pytest.raises(SessionLineageAmbiguousError):
            mutate_runtime_session(local, epoch=epoch,
                principal_id='owner', session_id=root, request_id='target-' + operation,
                expected_revision=expected_revision, expected_generation=expected_generation,
                operation=operation, payload=payload, _prepared=prepared)
        assert (snapshot(local), snapshot(sibling)) == before
