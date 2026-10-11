"""An explicit canonical reset on an ACP session is offered and reopened by its retained owner."""
from contextlib import closing
from dataclasses import asdict

from acp_adapter.catalog import catalog_sessions, logical_session_id
from hermes_state import SessionDB
import hermes_state_runtime as rt


def _acp_local_session(db, epoch, cwd):
    from hermes_state_local import commit_local_session
    from gateway.config import Platform
    from gateway.session import SessionEntry, SessionSource
    from gateway.session_lifecycle import _now
    from gateway.session_local_recovery import local_identity
    from gateway.session_policy import build_policy
    sid = local_identity('profile', 'human', 'editor')
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
    with closing(SessionDB(path)) as db:
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        sid = _acp_local_session(db, epoch, tmp_path)
        db.append_message(sid, 'user', 'before reset')
        row = db.get_session(sid)
        child = rt.mutate_runtime_session(db, epoch=epoch, principal_id='human', session_id=sid,
            request_id='reset', operation='reset', expected_revision=row['runtime_revision'],
            expected_generation=row['runtime_generation'], payload={})['target_session_id']
        db.append_message(child, 'user', 'after reset')
        # A true legacy reset history (no local receipt) keeps two independent conversations.
        db.create_session('legacy-old', source='acp', cwd=str(tmp_path))
        db.append_message('legacy-old', 'user', 'legacy old')
        db.end_session('legacy-old', 'session_reset')
        db.create_session('legacy-new', source='acp', cwd=str(tmp_path), parent_session_id='legacy-old',
                          model_config={'_reset_from': 'legacy-old'})
        db.append_message('legacy-new', 'user', 'legacy new')
    offered = [row['session_id'] for row in catalog_sessions(path, str(tmp_path))]
    assert sorted(offered) == sorted([sid, 'legacy-old', 'legacy-new']), offered
    assert logical_session_id(path, child) == sid
    assert logical_session_id(path, sid) == sid
    assert logical_session_id(path, 'legacy-new') == 'legacy-new'
