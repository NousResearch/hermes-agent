"""A history binding written by an older home of its profile still names that profile.

Imported/claimed history stores ``[profile_id, subject]`` with the owning home's absolute path, so a
``state.db`` that moved with its profile (another machine, a moved home) refused every bound row:
pin/archive/rename got ``permission_denied`` and the read-only mount ``not_found``.
"""
import json

import hermes_state_runtime as rt
from hermes_state import SessionDB
from hermes_state_mutation_binding import BINDING_PREFIX, bindings_agree, binding_names, import_history_control


def _bound(profile, subject='owner'):
    return json.dumps([str(profile), subject])


def test_binding_follows_its_store_after_the_home_moved(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert binding_names(_bound(departed), 'owner')
    assert bindings_agree(_bound(departed), _bound(home))


def test_binding_still_names_only_its_owner(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert not binding_names(_bound(departed), 'someone-else')
    assert not bindings_agree(_bound(departed), _bound(home, 'someone-else'))
    assert not binding_names('not json', 'owner')
    assert not binding_names(json.dumps([None, 'owner']), 'owner')
    assert not bindings_agree(_bound(departed), 'not json')


def test_import_links_a_parent_its_owner_imported_before_the_home_moved(tmp_path):
    from gateway.session_contract import SessionActor
    with SessionDB(db_path=tmp_path / 'state.db') as db:
        db.create_session('parent', source='cli')
        db.set_meta(BINDING_PREFIX + 'parent', _bound(tmp_path / 'mac' / 'home'))
        actor = SessionActor('owner', str(tmp_path / 'cloud'), frozenset(), 'peer')
        epoch = rt.begin_runtime_epoch(db, instance_id='owner')
        result = rt.mutate_runtime_session(
            db, epoch=epoch, principal_id='owner', session_id='child', request_id='import', expected_revision=0,
            operation='import', payload={'sessions': [
                {'id': 'child', 'source': 'cli', 'parent_session_id': 'parent', 'messages': []}]},
            _authorize_write=lambda conn: import_history_control(conn, actor, ['child']))
        assert result['detached'] == 0
        assert db.get_session('child')['parent_session_id'] == 'parent'
