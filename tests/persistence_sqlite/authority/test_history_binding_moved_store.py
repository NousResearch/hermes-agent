"""A history binding written by an older home of its profile still names that profile.

Imported/claimed history stores ``[profile_id, subject]`` with the owning home's absolute path, so a
``state.db`` that moved with its profile (another machine, a moved home) refused every bound row:
pin/archive/rename got ``permission_denied`` and the read-only mount ``not_found``.
"""
import json

from persistence.authority import runtime as rt
from persistence.sqlite.authority._binding_history import BINDING_PREFIX
from session_rules.legacy_local import bindings_agree, binding_names
from tests.fixtures.storage_seed import seed_session


def _bound(profile, subject='owner'):
    return json.dumps([str(profile), subject], separators=(',', ':'))


def test_binding_follows_its_store_after_the_home_moved(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert binding_names(_bound(departed), 'owner', None)
    assert bindings_agree(_bound(departed), _bound(home), None)


def test_binding_still_names_only_its_owner(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert not binding_names(_bound(departed), 'someone-else', None)
    assert not bindings_agree(_bound(departed), _bound(home, 'someone-else'), None)
    assert not binding_names('not json', 'owner', None)
    assert not binding_names(json.dumps([None, 'owner']), 'owner', None)
    assert not bindings_agree(_bound(departed), 'not json', None)


def test_import_links_a_parent_its_owner_imported_before_the_home_moved(store, tmp_path):
    seed_session(store, 'parent', 'cli')
    store.state_meta.set(BINDING_PREFIX + 'parent', _bound(tmp_path / 'mac' / 'home'))
    actor = rt.HistoryActor(str(tmp_path / 'cloud'), 'owner', None, False)
    epoch = store.runtime.begin_epoch(instance_id='owner')
    outcome = store.runtime.mutate(
        epoch, principal_id='owner', session_id='child', request_id='import', expected_revision=0,
        command=rt.Import(({'id': 'child', 'source': 'cli', 'parent_session_id': 'parent', 'messages': []},)),
        history=rt.ImportControl(actor, ('child',)), owner=None)
    assert outcome.result['detached'] == 0
    assert store.sessions.get('child').parent_session_id == 'parent'
