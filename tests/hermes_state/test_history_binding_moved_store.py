"""A history binding written by an older home of its profile still names that profile.

Imported/claimed history stores ``[profile_id, subject]`` with the owning home's absolute path, so a
``state.db`` that moved with its profile (another machine, a moved home) refused every bound row:
pin/archive/rename got ``permission_denied`` and the read-only mount ``not_found``.
"""
import json

from hermes_state_mutation_binding import _bindings_agree, binding_names


def _bound(profile, subject='owner'):
    return json.dumps([str(profile), subject])


def test_binding_follows_its_store_after_the_home_moved(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert binding_names(_bound(departed), str(home), 'owner')
    assert _bindings_agree(_bound(departed), _bound(home))


def test_binding_still_names_only_its_owner(tmp_path):
    departed, home = tmp_path / 'mac' / 'home', tmp_path / 'cloud'
    assert not binding_names(_bound(departed), str(home), 'someone-else')
    assert not _bindings_agree(_bound(departed), _bound(home, 'someone-else'))
    assert not binding_names('not json', str(home), 'owner')
