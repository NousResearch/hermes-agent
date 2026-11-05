"""An explicit canonical reset on an ACP session is offered and reopened by its retained owner."""
from dataclasses import asdict

from acp_adapter.catalog import catalog_sessions, logical_session_id
from persistence.authority.runtime import parse_mutation
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_read import session_row
from tests.fixtures.storage_seed import seed_messages, seed_session


def _acp_local_session(db, epoch, cwd):
    from gateway.session_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from persistence.authority.bindings import local_session_id
    from gateway.session_policy import build_policy
    sid = local_session_id('profile', 'human', 'editor')
    source = SessionSource(platform=Platform.LOCAL, chat_id=sid, user_id='human', chat_type='dm')
    entry = SessionEntry('local:' + sid, sid, _now(), _now(), origin=source, platform=Platform.LOCAL)
    policy = build_policy({'source': 'acp', 'cwd': str(cwd), 'model': 'm', 'toolsets': []},
                          {'platform_toolsets': {'acp': []}}, private_secrets={})
    commit_local_session(db, epoch=epoch, receipt={
        'profile_id': 'profile', 'principal_id': 'human', 'request_id': 'editor', 'session_id': sid,
        'route': entry.session_key, 'entry': entry.to_dict(), 'policy': asdict(policy)})
    return sid


def test_reset_child_maps_to_its_owner_while_legacy_reset_histories_stay_independent(tmp_path):
    path = tmp_path / 'state.db'
    with open_persistence(profile_scope(home=tmp_path)) as db:
        epoch = db.runtime.begin_epoch(instance_id='owner')
        sid = _acp_local_session(db, epoch, tmp_path)
        seed_messages(db, sid, {'role': 'user', 'content': 'before reset'})
        row = session_row(db, sid)
        child = db.runtime.mutate(epoch, principal_id='human', session_id=sid,
            request_id='reset', command=parse_mutation('reset', {}), expected_revision=row['runtime_revision'],
            expected_generation=row['runtime_generation'], owner=None).result['target_session_id']
        seed_messages(db, child, {'role': 'user', 'content': 'after reset'})
        # A true legacy reset history (no local receipt) keeps two independent conversations.
        seed_session(db, 'legacy-old', 'acp', cwd=str(tmp_path))
        seed_messages(db, 'legacy-old', {'role': 'user', 'content': 'legacy old'})
        db.sessions.end('legacy-old', 'session_reset')
        seed_session(db, 'legacy-new', 'acp', cwd=str(tmp_path), parent_session_id='legacy-old',
                     model_config={'_reset_from': 'legacy-old'})
        seed_messages(db, 'legacy-new', {'role': 'user', 'content': 'legacy new'})
    offered = [row['session_id'] for row in catalog_sessions(path, str(tmp_path))]
    assert sorted(offered) == sorted([sid, 'legacy-old', 'legacy-new']), offered
    assert logical_session_id(path, child) == sid
    assert logical_session_id(path, sid) == sid
    assert logical_session_id(path, 'legacy-new') == 'legacy-new'
