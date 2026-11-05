"""Public ``prompt.submit`` attachments: scoped to the profile staging dir, committed as bytes."""
from types import SimpleNamespace

import os
import pytest

from persistence.core.errors import AuthorityRefusal
from tests.fixtures.storage_seed import seed_session

_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)


async def _authority(tmp_path, monkeypatch, answer):
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(_session_db=store.persistence, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    seed_session(authority.persistence, 's', 'telegram')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    return authority


@pytest.mark.asyncio
async def test_submit_rejects_attachment_outside_profile_staging_dir(tmp_path, monkeypatch):
    from gateway.session_contract import SessionActor, SessionRef, Submission

    seen = []
    async def answer(event):
        seen.append(event)
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    outside = tmp_path / 'elsewhere.png'
    outside.write_bytes(_ONE_PX_PNG)
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        await authority.submit(actor, Submission('r1', SessionRef('p', 's'),
            {'text': 'look', 'attachments': [{'path': str(outside), 'mime': 'image/png'}]}, 'queue'))
    assert authority.persistence.admissions.list(session_id='s', pending_only=False) == []
    assert seen == []


@pytest.mark.asyncio
async def test_staged_attachment_reaches_message_event_media(tmp_path, monkeypatch):
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_contract import SessionActor, SessionRef, Submission

    seen = []
    async def answer(event):
        seen.append(event)
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    staged = cache_image_from_bytes(_ONE_PX_PNG, '.png')
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    receipt = await authority.submit(actor, Submission('r2', SessionRef('p', 's'),
        {'text': 'look', 'attachments': [{'path': staged, 'mime': 'image/png'}]}, 'queue'))
    await authority.sessions['s'].task
    assert receipt.status == 'queued'
    (event,) = seen
    assert event.text == 'look'
    assert event.media_types == ['image/png']
    (path,) = event.media_urls
    assert path != staged, 'execution must read committed bytes, not the mutable staging file'
    assert 'native-inputs' in path
    # Retained bytes are exact-retry evidence by digest only; settlement releases them.
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='off-loop media collection')
    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_lost_ack_retry_reconciles_a_queued_image_after_staging_is_gone(tmp_path, monkeypatch):
    """Lost-ACK recovery of a still-queued image admission must not depend on the client's
    disposable staging file: the committed bytes and digest are the evidence. A changed payload
    under the same request id still conflicts, and a fresh request id still needs real staging."""
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_authority import SessionAuthority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    staged = cache_image_from_bytes(_ONE_PX_PNG, '.png')
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')
    def submit(request_id, text='look', path=staged):
        return authority.submit(actor, Submission(request_id, SessionRef('p', 's'),
            {'text': text, 'attachments': [{'path': path, 'mime': 'image/png'}]}, 'queue'))
    first = await submit('r-lost')
    assert (await submit('r-lost')).admission_id == first.admission_id
    os.unlink(staged)
    assert (await submit('r-lost')).admission_id == first.admission_id
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        await submit('r-lost', text='changed')
    with pytest.raises(AuthorityRefusal, match='invalid_params'):
        await submit('r-fresh')
    assert [row.request_id for row in authority.persistence.admissions.list(session_id='s')] == ['r-lost']


@pytest.mark.asyncio
async def test_retry_with_the_same_image_restaged_under_a_new_name_replays(tmp_path, monkeypatch):
    """Retry identity is the committed bytes, not the client's disposable staging filename: the
    same request re-staged under a new name replays the admission instead of conflicting, and
    different bytes under the same request id still conflict."""
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_authority import SessionAuthority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    def submit(path):
        return authority.submit(actor, Submission('r-same', SessionRef('p', 's'),
            {'text': 'look', 'attachments': [{'path': path, 'mime': 'image/png'}]}, 'queue'))
    first = await submit(cache_image_from_bytes(_ONE_PX_PNG, '.png'))
    restaged = cache_image_from_bytes(_ONE_PX_PNG, '.png')
    assert (await submit(restaged)).admission_id == first.admission_id
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        await submit(cache_image_from_bytes(_ONE_PX_PNG + b'\0', '.png'))
    assert len(authority.persistence.admissions.list(session_id='s')) == 1


@pytest.mark.asyncio
async def test_discarding_a_lost_image_turn_releases_its_committed_bytes(tmp_path, monkeypatch):
    """A turn lost across an owner restart and Discarded is terminal: its committed image bytes
    are released like a settled turn's, never left on disk for the life of the profile."""
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_contract import SessionActor, SessionRef, Submission
    from gateway.session_authority import SessionAuthority

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    from gateway import session_local_recovery
    monkeypatch.setattr(session_local_recovery, 'transcript_target', lambda authority, ref: ref.session_id)
    actor = SessionActor('human', 'p', frozenset({'session:submit', 'session:control'}), 't')
    receipt = await authority.submit(actor, Submission('img', SessionRef('p', 's'),
        {'text': 'look', 'attachments': [{'path': cache_image_from_bytes(_ONE_PX_PNG, '.png'),
                                          'mime': 'image/png'}]}, 'queue'))
    store = authority.persistence
    committed = store.admissions.get(admission_id=receipt.admission_id).payload[
        'attachments_v1']['media'][0]['path']
    row = store.admissions.claim(epoch=authority.epoch, session_id='s')
    authority.epoch = store.runtime.begin_epoch(instance_id='restarted')
    store.admissions.recover(epoch=authority.epoch)
    await authority.resolve_unknown(actor, SessionRef('p', 's'), receipt.admission_id, row.generation)
    assert store.admissions.get(admission_id=receipt.admission_id).status == 'terminal'
    from gateway.session_runtime_workers import mutation_tasks
    from tests.waiting import async_wait_until
    await async_wait_until(lambda: not mutation_tasks(authority), timeout=5, what='off-loop media collection')
    assert not os.path.exists(committed)


@pytest.mark.asyncio
async def test_refused_image_capture_is_released_without_masking_the_refusal(tmp_path, monkeypatch):
    """A local submit captures image bytes before admission can refuse them. The refused capture is
    released holder-aware (the accepted admission's bytes stay), and a collection failure never
    replaces the refusal the client must see."""
    from gateway import session_ingress_media
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef, Submission

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    actor = SessionActor('human', 'p', frozenset({'session:submit'}), 't')

    def submit(data):
        return authority.submit(actor, Submission('r-conflict', SessionRef('p', 's'), {'text': 'look', 'attachments': [
            {'path': cache_image_from_bytes(data, '.png'), 'mime': 'image/png'}]}, 'queue'))
    await submit(_ONE_PX_PNG)
    retained = lambda: sorted(p.parent.name for p in session_ingress_media._media_root().glob('*/*'))
    accepted = retained()
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        await submit(_ONE_PX_PNG + b'B')
    assert retained() == accepted, 'the refused capture must be released, the accepted one kept'

    release = session_ingress_media.release_unheld_media
    def collection_fails(*args, **kwargs):
        raise OSError('collection write failed')
    monkeypatch.setattr(session_ingress_media, 'release_unheld_media', collection_fails)
    with pytest.raises(AuthorityRefusal, match='admission_conflict'):
        await submit(_ONE_PX_PNG + b'C')
    assert len(authority.persistence.admissions.list(session_id='s')) == 1
    # The failed collection stays a durable obligation the next retirement sweep honours.
    monkeypatch.setattr(session_ingress_media, 'release_unheld_media', release)
    from session_retention import collect_retired_media
    collect_retired_media(authority.persistence, authority.persistence.scope.profile_dir)
    assert retained() == accepted
