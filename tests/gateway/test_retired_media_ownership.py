"""Receipt retries preserve transcript boundaries, successor ownership and media identity."""



from pathlib import Path



from types import SimpleNamespace



import pytest



from gateway.session_authority import LiveSession



from gateway.session_contract import Principal, SessionRef, Submission



from gateway.session_results import admission_result, finish_result



from hermes_state_runtime import (RuntimeStoreError, admit_session_input, begin_runtime_epoch,
    claim_session_input, get_session_admission, recover_session_inputs)



@pytest.mark.asyncio
async def test_attachment_retry_survives_lost_staging_but_rejects_changed_payload(tmp_path, monkeypatch):
    from gateway.platforms.base import cache_image_from_bytes
    from tests.gateway.test_prompt_attachments import _authority, _ONE_PX_PNG
    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    staged = Path(cache_image_from_bytes(_ONE_PX_PNG, '.png'))
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    request = Submission('retry', SessionRef('p', 's'),
        {'text': 'look', 'attachments': [{'path': str(staged), 'mime': 'image/png'}]}, 'queue')
    receipt = await authority.submit(actor, request)
    staged.unlink()
    assert await authority.submit(actor, request) == receipt
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await authority.submit(actor, Submission('retry', request.ref, {**request.payload, 'text': 'changed'}, 'queue'))
    staged.write_bytes(_ONE_PX_PNG + b'changed')
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await authority.submit(actor, request)



def test_terminal_worker_releases_its_linked_admission_media(owner):
    from gateway.session_ingress_media import capture_native_media
    from hermes_state_runtime import register_worker_execution, finish_worker_execution
    owner.db.create_session('s', source='test')
    source = Path(owner.db.db_path).parent / 'input.png'
    source.write_bytes(b'image')
    references = capture_native_media([source])
    admit_session_input(owner.db, epoch=owner.epoch, principal_id='human', session_id='s', request_id='worker',
        payload={'text': 'work', 'attachments_v1': {'media': references, 'media_types': ['image/png']}})
    started = claim_session_input(owner.db, epoch=owner.epoch, session_id='s')
    scope = dict(execution_id='worker', session_id='s', generation=started['generation'])
    register_worker_execution(owner.db, epoch=owner.epoch, **scope, kind='compute', adoption_secret='private')
    finish_worker_execution(owner.db, epoch=owner.epoch, **scope)
    assert not Path(references[0]['path']).exists()
