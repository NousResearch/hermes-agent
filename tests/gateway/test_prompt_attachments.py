"""Public ``prompt.submit`` attachments: scoped to the profile staging dir, committed as bytes."""
from types import SimpleNamespace

import os
import pytest

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
    runner = SimpleNamespace(_session_db=store._db, session_store=store, _draining=False,
                             _handle_message=answer, _adapter_for_source=lambda source: None)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    store._db.create_session('s', source='telegram')
    authority.sessions['s'] = LiveSession(SessionSource(platform=Platform.TELEGRAM, chat_id='c'), 's')
    return authority


@pytest.mark.asyncio
async def test_submit_rejects_attachment_outside_profile_staging_dir(tmp_path, monkeypatch):
    from gateway.session_contract import Principal, SessionRef, Submission
    from hermes_state_runtime import RuntimeStoreError, list_session_admissions

    seen = []
    async def answer(event):
        seen.append(event)
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    outside = tmp_path / 'elsewhere.png'
    outside.write_bytes(_ONE_PX_PNG)
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
    with pytest.raises(RuntimeStoreError, match='invalid_params'):
        await authority.submit(actor, Submission('r1', SessionRef('p', 's'),
            {'text': 'look', 'attachments': [{'path': str(outside), 'mime': 'image/png'}]}, 'queue'))
    assert list_session_admissions(authority.db, session_id='s', pending_only=False) == []
    assert seen == []


@pytest.mark.asyncio
async def test_staged_attachment_reaches_message_event_media(tmp_path, monkeypatch):
    from gateway.platforms.base import cache_image_from_bytes
    from gateway.session_contract import Principal, SessionRef, Submission

    seen = []
    async def answer(event):
        seen.append(event)
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    staged = cache_image_from_bytes(_ONE_PX_PNG, '.png')
    actor = Principal('human', 'p', frozenset({'session:submit'}), 't')
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
    assert not os.path.exists(path)
