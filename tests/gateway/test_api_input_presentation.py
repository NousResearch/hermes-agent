"""An admitted API input is a user turn, not a gateway-produced notice (#156802)."""
from types import SimpleNamespace

import pytest

from gateway.platforms.event import MessageEvent
from gateway.response_filters import (
    INTERNAL_NOTIFICATION_DISPLAY_KIND,
    display_kind_for_event,
    silence_allowed,
)
from gateway.session_api_turn import admit_api_turn
from gateway.session_finite import execute_finite_admission
from gateway.session_results import execution_result
from tests.gateway.session_claims import claim


@pytest.mark.asyncio
async def test_admitted_api_input_keeps_trust_without_becoming_machinery(api, owner):
    _, ref, row = admit_api_turn(
        api, session_id='api-input', request_id='run_input',
        user_message='Is the build done?', conversation_history=[])
    seen = []

    async def handle(event):
        from gateway.response_filters import persisted_user_metadata_for_event

        kind = display_kind_for_event(event)
        seen.append((kind, event.internal, event.allow_gateway_control,
                     persisted_user_metadata_for_event(event)))
        assert not silence_allowed(kind, reply_expected=True)
        # A self-injected follow-up on the same API route is still machinery.
        notice = MessageEvent(text='job finished', source=event.source, internal=True,
                              message_id='notice', metadata={})
        assert display_kind_for_event(notice) == INTERNAL_NOTIFICATION_DISPLAY_KIND
        execution_result.get()['result'] = {'final_response': 'ok'}
        return 'ok'

    owner.runner._handle_message = handle
    assert await execute_finite_admission(owner, ref, claim(owner, ref, row)) == 'ok'
    assert seen == [(None, True, False, {'submission_id': 'run_input'})]


def test_unadmitted_metadata_does_not_turn_a_notice_into_user_input():
    event = SimpleNamespace(internal=True, message_id='unadmitted', metadata={
        'submission_id': 'run_input', 'api_user_input': True})
    assert display_kind_for_event(event) == INTERNAL_NOTIFICATION_DISPLAY_KIND
    assert silence_allowed(display_kind_for_event(event), reply_expected=True)
