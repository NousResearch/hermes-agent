"""Canonical API turns retain logical identity, reachable controls and owned image bytes."""



import asyncio



import base64



from pathlib import Path



import pytest



from gateway.session_api import restore_api_session



from gateway.session_api_turn import admit_api_turn



from hermes_state_runtime import RuntimeStoreError, claim_session_input, settle_session_input



PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=')



def image(data):
    return [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(data).decode()}}]



def test_rejected_api_images_are_collected_and_history_owns_accepted_images(api, owner):
    from gateway.session_ingress_media import _media_root
    kwargs = dict(session_id='s', request_id='one', conversation_history=[])
    _, _, row = admit_api_turn(api, user_message=image(PNG), **kwargs)
    path = Path(row['payload']['api_turn_v1']['media'][0]['path'])
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        admit_api_turn(api, user_message=image(PNG + b'different'), **kwargs)
    assert list(_media_root().glob('*/*')) == [path]
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'],
                         generation=started['generation'], outcome='completed')
    owner.db.create_session('branch', source='api_server', model_config={'_branched_from': 's'})
    owner.db.append_message('branch', 'user', '[Image attached at: %s]' % path)
    owner.db.delete_session('s')
    assert path.read_bytes() == PNG, 'branch history still owns its image'
    owner.db.delete_session('branch')
    assert not path.exists()
