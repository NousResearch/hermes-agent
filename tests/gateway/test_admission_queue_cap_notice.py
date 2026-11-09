"""A messaging follow-up past the FIFO cap gets a plain queue_full notice, not a turn error (N62)."""
import pytest

from gateway.platforms.event import MessageEvent
from gateway.session_contract import Submission
from gateway.session_ingress import admit_message
from hermes_state_runtime import ADMISSION_QUEUE_MAX, list_session_admissions
from tests.gateway.test_session_authority_cancel_settlement import ACTOR, REF, _authority, _submit


@pytest.mark.asyncio
async def test_messaging_overflow_is_answered_with_a_queue_full_notice(tmp_path, monkeypatch):
    db, authority = _authority(tmp_path, monkeypatch)
    monkeypatch.setattr(authority, '_schedule', lambda ref: None)  # the running turn never drains

    async def admit_native(event):
        return await authority.submit(ACTOR, Submission(event.text, REF, {'text': event.text}, 'queue'))
    monkeypatch.setattr(authority, 'admit_native', admit_native)
    with db:
        for n in range(ADMISSION_QUEUE_MAX):
            await _submit(authority, f'r{n}')
        reply = await admit_message(authority, MessageEvent(text='overflow', source=authority.sessions['s'].source))
        assert reply is not None and 'Too many messages' in reply, reply
        assert len(list_session_admissions(db, session_id='s')) == ADMISSION_QUEUE_MAX
