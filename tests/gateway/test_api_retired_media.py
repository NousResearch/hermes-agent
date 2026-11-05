"""Canonical API turns retain logical identity, reachable controls and owned image bytes."""
import base64
from pathlib import Path

import pytest

from gateway.session_api_turn import admit_api_turn
from persistence.core.errors import AuthorityRefusal
from session_retention import after_sessions_removed
from tests.fixtures.storage_seed import seed_messages, seed_session

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')


def image(data):
    return [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}]



def test_rejected_api_images_are_collected_and_history_owns_accepted_images(api, owner):
    from gateway.session_ingress_media import _media_root
    kwargs = dict(session_id='s', request_id='one', conversation_history=[])
    _, _, row = admit_api_turn(api, user_message=image(PNG), **kwargs)
    path = Path(row['payload']['api_turn_v1']['media'][0]['path'])
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        admit_api_turn(api, user_message=image(PNG + b'different'), **kwargs)
    assert list(_media_root().glob('*/*')) == [path]
    started = owner.persistence.admissions.claim(epoch=owner.epoch, session_id='s')
    owner.persistence.admissions.settle(epoch=owner.epoch, admission_id=row['admission_id'],
                                        generation=started.generation, outcome='completed')
    seed_session(owner.persistence, 'branch', 'api_server', model_config={'_branched_from': 's'})
    seed_messages(owner.persistence, 'branch', {'role': 'user', 'content': f'[Image attached at: {path}]'})
    store = owner.persistence
    removed = store.sessions.delete('s', media_root=str(_media_root()))
    after_sessions_removed(store, store.scope.profile_dir, removed, reason='delete', primary='s')
    assert path.read_bytes() == PNG, 'branch history still owns its image'
    removed = store.sessions.delete('branch', media_root=str(_media_root()))
    after_sessions_removed(store, store.scope.profile_dir, removed, reason='delete', primary='branch')
    assert not path.exists()
