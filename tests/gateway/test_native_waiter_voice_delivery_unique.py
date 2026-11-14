"""Voice delivery for a settled native waiter and an uncertain settlement."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway import session_ingress, session_results, session_settlement_recovery as recovery
from hermes_state_runtime import get_session_admission
from tests.gateway.test_session_authority_cancel_settlement import _authority, _submit, ACTOR, REF


@pytest.mark.asyncio
@pytest.mark.parametrize('outcome', ['completed', 'failed'])
async def test_native_waiter_only_speaks_committed_success(tmp_path, monkeypatch, outcome):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    authority.runner._should_send_voice_reply = lambda *args, **kwargs: True
    authority.runner._send_voice_reply = AsyncMock()
    answer = 'the answer'
    execution = {'final_response': answer, 'failed': outcome == 'failed'}
    event = SimpleNamespace(_deferred_voice_response=answer, _deferred_voice_execution=execution,
                            _deferred_voice_messages=[])

    async def execute(owner, ref, row):
        aid = row['admission_id']
        owner.pending_results[aid] = {'result': execution, 'usage': {}}
        session_ingress.waiter_replies(owner)[aid] = (object(), event, 'original-route', answer, None)
        return answer

    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(recovery, '_release_media', lambda *args: None)
    with db:
        receipt = await _submit(authority, 'voice')
        authority.native_waiters.add(receipt.admission_id)
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        await authority._drain(REF)
        assert get_session_admission(db, admission_id=receipt.admission_id)['outcome'] == outcome
        assert await waiter == answer
        if outcome == 'completed':
            authority.runner._send_voice_reply.assert_awaited_once_with(event, answer)
        else:
            authority.runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
async def test_unknown_discard_never_speaks_unproven_retained_candidate(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    monkeypatch.setattr(recovery, '_SETTLE_RETRY_DELAYS_S', ())
    authority.runner._should_send_voice_reply = lambda *args, **kwargs: True
    authority.runner._send_voice_reply = AsyncMock()
    answer = 'answer'
    execution = {'final_response': answer}
    event = SimpleNamespace(_deferred_voice_response=answer, _deferred_voice_execution=execution,
                            _deferred_voice_messages=[])
    async def execute(owner, ref, row):
        aid = row['admission_id']
        owner.pending_results[aid] = {'result': execution, 'usage': {}}
        session_ingress.waiter_replies(owner)[aid] = (object(), event, 'original-route', answer, None)
        return answer
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(session_results, 'finish_result', lambda *a, **k: (_ for _ in ()).throw(OSError('offline')))
    monkeypatch.setattr(recovery, '_release_media', lambda *args: None)
    with db:
        receipt = await _submit(authority, 'voice')
        await authority._drain(REF)
        row = get_session_admission(db, admission_id=receipt.admission_id)
        assert row['status'] == 'unknown'
        # A different captured execution with equal text must not authorize old audio.
        authority.pending_results[receipt.admission_id] = {'result': {'final_response': answer}, 'usage': {}}
        monkeypatch.setattr(session_ingress, 'deliver_response', AsyncMock(return_value=True))
        await authority.resolve_unknown(ACTOR, REF, receipt.admission_id, row['generation'])
        authority.runner._send_voice_reply.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["media", "local_path"])
async def test_native_waiter_completion_extracts_before_speech(tmp_path, monkeypatch, kind):
    import json
    from pathlib import Path
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource
    from tests.gateway.test_auto_voice_reply_format import _make_runner
    from tests.gateway.test_base_auto_tts_profile_scope import _Adapter

    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)
    runner = _make_runner()
    adapter = _Adapter()
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._voice_mode['telegram:123'] = 'voice_only'
    runner._session_key_for_source = lambda source: 'session'
    runner._deliver_voice_reply = AsyncMock()
    authority.runner = runner
    adapter.filter_media_delivery_paths = lambda files, session_key='': files
    adapter.filter_local_delivery_paths = lambda files, session_key='': files
    file = tmp_path / 'private.png'
    file.write_bytes(b'image')
    answer = f'Finished. {"MEDIA:" if kind == "media" else ""}{file}'
    execution = {'final_response': answer}
    event = MessageEvent(text='completion', internal=True, metadata={'presentation_origin': 'voice'},
                         source=SessionSource(platform=Platform.TELEGRAM, chat_id='123'))
    event._deferred_voice_response = answer
    event._deferred_voice_execution = execution
    event._deferred_voice_messages = []
    spoken = []
    def tts(*, text, output_path):
        spoken.append(text)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b'audio')
        return json.dumps({'success': True, 'file_path': output_path})
    monkeypatch.setattr('tools.tts_tool.text_to_speech_tool', tts)
    async def execute(owner, ref, row):
        aid = row['admission_id']
        owner.pending_results[aid] = {'result': execution, 'usage': {}}
        session_ingress.waiter_replies(owner)[aid] = (adapter, event, 'session', answer, None)
        return answer
    monkeypatch.setattr('gateway.session_finite.execute_finite_admission', execute)
    monkeypatch.setattr(recovery, '_release_media', lambda *args: None)
    with db:
        receipt = await _submit(authority, 'voice')
        authority.native_waiters.add(receipt.admission_id)
        waiter = authority.waiters.setdefault(receipt.admission_id, asyncio.get_running_loop().create_future())
        await authority._drain(REF)
        assert await waiter == answer
        assert spoken == ([] if kind == 'media' else ['Finished.'])
