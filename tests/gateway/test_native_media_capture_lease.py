"""Retained media a capture published or reused is held from capture until its admission commits
or is refused (andrexibiza-7, JoaoMarcos44-R2), and a refused admission leaves no capture behind
(andrexibiza-17). One row per content-addressed capture site whose admission the authority owns."""
import asyncio
from pathlib import Path
import threading

import pytest

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082")


async def _authority(tmp_path, monkeypatch):
    from gateway.session_authority import SessionAuthority
    from tests.gateway.test_prompt_attachments import _authority as store_authority

    async def answer(event):
        return 'ok'
    authority = await store_authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    return authority


def _submit(authority, request_id, staged, text='look'):
    from gateway.session_contract import Principal, SessionRef, Submission
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    return authority.submit(actor, Submission(request_id, SessionRef('p', 's'), {'text': text, 'attachments': [
        {'path': staged, 'mime': 'image/png'}]}, 'queue'))


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup', ['settled', 'cancelled'])
async def test_same_bytes_captured_for_a_new_turn_survive_an_older_turns_cleanup(tmp_path, monkeypatch, cleanup):
    """The same staged image is sent again (a queued prompt edited: cancel + resubmit, or a re-send
    while the previous turn settles): admit_attachments reuses the content-addressed path the older
    row holds, and that row's cleanup runs while the new admission write is still in flight."""
    from gateway import session_authority
    from gateway.session_ingress_media import release_admission_media, restore_attachments
    from hermes_state_runtime import (cancel_session_input, claim_session_input, get_session_admission,
                                      settle_session_input)
    from gateway.platforms.base import cache_image_from_bytes
    authority = await _authority(tmp_path, monkeypatch)
    staged = cache_image_from_bytes(PNG, '.png')
    old = await _submit(authority, 'old', staged)
    if cleanup == 'settled':
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id='s')
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=old.admission_id,
                             generation=claim['generation'], outcome='completed')
    else:
        cancel_session_input(authority.db, epoch=authority.epoch, admission_id=old.admission_id)
    entered, release = threading.Event(), threading.Event()
    real = session_authority.admit_session_input

    def parked(*args, **kwargs):
        entered.set()
        release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(session_authority, 'admit_session_input', parked)
    new = asyncio.create_task(_submit(authority, 'new', staged))
    assert await asyncio.to_thread(entered.wait, 10)
    await asyncio.to_thread(release_admission_media, authority.db, old.admission_id)
    release.set()
    receipt = await asyncio.wait_for(new, 10)
    row = get_session_admission(authority.db, admission_id=receipt.admission_id)
    assert receipt.status == 'queued'
    restore_attachments(row['payload'])  # raises storage_unavailable when the bytes were unlinked


@pytest.mark.asyncio
async def test_refused_native_admission_leaves_no_fresh_capture(tmp_path, monkeypatch):
    """admit_native captured the message's media, then the write refused it (admission_conflict
    on a changed redelivery): the fresh capture no admission owns must not stay retained."""
    from gateway import session_authority
    from gateway.session_ingress_media import _media_root, capture_native_media
    from hermes_state_runtime import RuntimeStoreError
    authority = await _authority(tmp_path, monkeypatch)
    from gateway.platforms.base import cache_image_from_bytes

    async def prepare(runner, event):
        return {'text': 'x', 'native_text_v1': {'source': {}, 'event': {'message_id': 'm'},
                'media': capture_native_media([cache_image_from_bytes(PNG + b'fresh', '.png')])}}
    monkeypatch.setattr('gateway.session_envelope.prepare_native', prepare)
    monkeypatch.setattr('gateway.session_envelope.restore_native',
                        lambda payload: type('E', (), {'source': authority.sessions['s'].source})())

    def refused(*args, **kwargs):
        raise RuntimeStoreError('admission_conflict')
    monkeypatch.setattr(session_authority, 'admit_session_input', refused)
    with pytest.raises(RuntimeStoreError, match='admission_conflict'):
        await authority.admit_native(object())
    assert not [p for p in _media_root().rglob('*') if p.is_file()], 'a refused capture stayed retained'
    assert Path(_media_root()).is_dir()
