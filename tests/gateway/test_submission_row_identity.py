"""The user row a canonical admission persists names the client's ``submission_id``.

A Desktop window that closes after the owner admitted a send but before the ack came back reads the
stored transcript on reopen to tell a send that landed from one that never did (its composer outbox).
"""
from types import SimpleNamespace

import pytest

from tests.gateway.test_surface_admission import _authority


@pytest.mark.asyncio
async def test_the_admitted_turns_user_row_names_its_submission_id(tmp_path, monkeypatch):
    from gateway.run_turn_persistence import GatewayTurnPersistenceMixin
    from gateway.session_controls import AuthorityConnection

    def stored_metadata(event):
        prepared = SimpleNamespace(persist_user_message=None, message_text=event.text, persist_user_timestamp=None,
                                   persist_user_display_kind=None, persistence_owner=None)
        return GatewayTurnPersistenceMixin._hmwa_user_transcript_entry(event, prepared, 0.0).get('display_metadata', {})

    seen = []
    async def answer(event):
        # A follow-up event on the same task (a /queue chain) is another input: it never inherits the id.
        stray = SimpleNamespace(message_id='other', internal=False, metadata={}, text='stray')
        seen.append((event.text, stored_metadata(event), stored_metadata(stray)))
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        for rid, extra in (('send-1', {}), ('send-2', {'title_preview': 'Notes'})):
            reply = await connection.dispatch({'id': 2, 'method': 'prompt.submit', 'params': {
                'session_id': 's', 'submission_id': rid, 'text': rid, **extra}})
            assert reply.get('result', {}).get('status') == 'queued', reply
            await authority.sessions['s'].task
    finally:
        await connection.close()

    assert seen == [
        ('send-1', {'submission_id': 'send-1'}, {}),
        ('send-2', {'title_preview': 'Notes', 'submission_id': 'send-2'}, {}),
    ]


@pytest.mark.asyncio
async def test_a_retry_of_an_admitted_id_is_an_info_line_never_an_error(tmp_path, monkeypatch, caplog):
    """The client settles this refusal as the dedupe it is: errors.log must not report it as a failure."""
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        await connection.dispatch({'id': 1, 'method': 'session.resume', 'params': {'session_id': 's'}})
        for rid, text in ((2, 'look at this'), (3, 'look at this, edited')):
            with caplog.at_level('DEBUG', logger='gateway.session_controls'):
                reply = await connection.dispatch({'id': rid, 'method': 'prompt.submit', 'params': {
                    'session_id': 's', 'submission_id': 'send-1', 'text': text}})
            await authority.sessions['s'].task
    finally:
        await connection.close()

    assert reply['error']['data']['field'] == 'request_id', reply
    refusals = [r for r in caplog.records if 'refused for profile' in r.getMessage()]
    assert [r.levelname for r in refusals] == ['INFO'], [(r.levelname, r.getMessage()) for r in refusals]


@pytest.mark.asyncio
async def test_a_malformed_create_request_id_stays_a_warning(tmp_path, monkeypatch, caplog):
    """Only the admitted-send retry is lowered: session.create's ``invalid_params field=request_id`` still reaches errors.log."""
    from gateway.session_controls import AuthorityConnection

    async def answer(event):
        return 'ok'
    authority = await _authority(tmp_path, monkeypatch, answer)
    connection = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        with caplog.at_level('DEBUG', logger='gateway.session_controls'):
            reply = await connection.dispatch({'id': 1, 'method': 'session.create', 'params': {'request_id': 'x' * 300}})
    finally:
        await connection.close()

    assert reply['error']['data']['field'] == 'request_id', reply
    refusals = [r for r in caplog.records if 'refused for profile' in r.getMessage()]
    assert [r.levelname for r in refusals] == ['WARNING'], [(r.levelname, r.getMessage()) for r in refusals]
