"""Retained media a capture published or reused is held from capture until its admission commits
or is refused (andrexibiza-7, JoaoMarcos44-R2), and a refused admission leaves no capture behind
(andrexibiza-17). One row per content-addressed capture site whose admission the authority owns."""
import asyncio
from pathlib import Path
import threading

import pytest

from gateway.session_persistence import authority_persistence

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
    from gateway.session_contract import SessionActor, SessionRef, Submission
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    return authority.submit(actor, Submission(request_id, SessionRef('p', 's'), {'text': text, 'attachments': [
        {'path': staged, 'mime': 'image/png'}]}, 'queue'))


@pytest.mark.asyncio
@pytest.mark.parametrize('cleanup', ['settled', 'cancelled'])
async def test_same_bytes_captured_for_a_new_turn_survive_an_older_turns_cleanup(tmp_path, monkeypatch, cleanup):
    """The same staged image is sent again (a queued prompt edited: cancel + resubmit, or a re-send
    while the previous turn settles): admit_attachments reuses the content-addressed path the older
    row holds, and that row's cleanup runs while the new admission write is still in flight."""
    from gateway.session_ingress_media import release_admission_media, restore_attachments
    from gateway.platforms.base import cache_image_from_bytes
    authority = await _authority(tmp_path, monkeypatch)
    staged = cache_image_from_bytes(PNG, '.png')
    old = await _submit(authority, 'old', staged)
    if cleanup == 'settled':
        claim = authority.persistence.admissions.claim(epoch=authority.epoch, session_id='s')
        authority.persistence.admissions.settle(epoch=authority.epoch, admission_id=old.admission_id,
                                                generation=claim.generation, outcome='completed')
    else:
        authority.persistence.admissions.cancel(epoch=authority.epoch, admission_id=old.admission_id)
    entered, release = threading.Event(), threading.Event()
    store = authority_persistence(authority)
    real = store.admissions.admit

    def parked(*args, **kwargs):
        entered.set()
        release.wait(10)
        return real(*args, **kwargs)
    monkeypatch.setattr(store.admissions, 'admit', parked)
    new = asyncio.create_task(_submit(authority, 'new', staged))
    assert await asyncio.to_thread(entered.wait, 10)
    await asyncio.to_thread(release_admission_media, store, old.admission_id)
    release.set()
    receipt = await asyncio.wait_for(new, 10)
    row = authority.persistence.admissions.get(admission_id=receipt.admission_id)
    assert receipt.status == 'queued'
    restore_attachments(row.payload)  # raises storage_unavailable when the bytes were unlinked


@pytest.mark.asyncio
async def test_same_bytes_admitted_after_cleanup_took_its_snapshot_survive_the_unlink(tmp_path, monkeypatch):
    """An older turn's cleanup has read which references nothing holds; before it unlinks, a new
    turn reuses the same content-addressed bytes, commits its admission and ends its lease. The
    cleanup must not unlink bytes the new admission now owns."""
    from gateway.session_ingress_media import release_admission_media, restore_attachments
    from gateway.platforms.base import cache_image_from_bytes
    authority = await _authority(tmp_path, monkeypatch)
    staged = cache_image_from_bytes(PNG, '.png')
    old = await _submit(authority, 'old', staged)
    authority.persistence.admissions.cancel(epoch=authority.epoch, admission_id=old.admission_id)
    store = authority_persistence(authority)
    real = store.admissions.unheld_media
    loop = asyncio.get_running_loop()
    admitted = []

    def snapshot_then_new_turn(*args, **kwargs):
        found = real(*args, **kwargs)
        if not admitted:
            admitted.append(asyncio.run_coroutine_threadsafe(_submit(authority, 'new', staged), loop).result(10))
        return found
    monkeypatch.setattr(store.admissions, 'unheld_media', snapshot_then_new_turn)
    await asyncio.to_thread(release_admission_media, store, old.admission_id)
    row = authority.persistence.admissions.get(admission_id=admitted[0].admission_id)
    assert admitted[0].status == 'queued'
    restore_attachments(row.payload)  # raises storage_unavailable when the bytes were unlinked


@pytest.mark.asyncio
async def test_refused_native_admission_leaves_no_fresh_capture(tmp_path, monkeypatch):
    """admit_native captured the message's media, then the write refused it (admission_conflict
    on a changed redelivery): the fresh capture no admission owns must not stay retained."""
    from gateway.session_ingress_media import _media_root, capture_native_media
    from persistence.core.errors import AuthorityRefusal
    authority = await _authority(tmp_path, monkeypatch)
    from gateway.platforms.base import cache_image_from_bytes

    async def prepare(runner, event):
        return {'text': 'x', 'native_text_v1': {'source': {}, 'event': {'message_id': 'm'},
                'media': capture_native_media([cache_image_from_bytes(PNG + b'fresh', '.png')])}}
    monkeypatch.setattr('gateway.session_envelope.prepare_native', prepare)
    monkeypatch.setattr('gateway.session_envelope.restore_native_source',
                        lambda payload: authority.sessions['s'].source)

    def refused(*args, **kwargs):
        raise AuthorityRefusal('admission_conflict')
    monkeypatch.setattr(authority_persistence(authority).admissions, 'admit', refused)
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        await authority.admit_native(object())
    assert not [p for p in _media_root().rglob('*') if p.is_file()], 'a refused capture stayed retained'
    assert Path(_media_root()).is_dir()
