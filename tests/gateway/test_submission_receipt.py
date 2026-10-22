"""A lost admission reply is recovered by durable identity, without resubmitting."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.session_authority import LiveSession, SessionAuthority
from gateway.session_controls import AuthorityConnection
from hermes_state import SessionDB
from hermes_state_runtime import admit_session_input, begin_runtime_epoch, cancel_session_input
from tests.gateway.test_canonical_wire_contract import owner as _wire_owner


@pytest.fixture
def owner(tmp_path):
    yield from _wire_owner.__wrapped__(tmp_path)


@pytest.mark.asyncio
async def test_submission_receipt_survives_settlement_and_new_authority_epoch(owner, tmp_path):
    connection, db = owner
    authority = connection.authority
    row = admit_session_input(db, epoch=authority.epoch, principal_id=connection.actor.subject,
                              session_id='s', request_id='lost-reply', payload={'text': 'once'})
    cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
    db.close()
    db = SessionDB(db_path=tmp_path / 'state.db')
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id='owned',
        instance_id='replacement', db=db, epoch=begin_runtime_epoch(db, instance_id='replacement'))
    authority.sessions['s'] = LiveSession(None, 'route')  # no event replay survives
    fresh = AuthorityConnection(authority, object(), {'user_id': 'owner'})
    try:
        before = db._conn.total_changes
        reply = await fresh.dispatch({'id': 1, 'method': 'prompt.receipt', 'params': {
            'session_id': 's', 'submission_id': 'lost-reply', 'include_result': True}})
        assert reply.get('result', {}).get('admission_id') == row['admission_id'], reply
        assert reply['result']['status'] == 'terminal'
        assert reply['result']['outcome'] == 'cancelled'
        assert db._conn.total_changes == before

        db.create_session('sibling', source='test')
        authority.sessions['sibling'] = LiveSession(None, 'other')
        for actor, session, reason in (
            (fresh.actor, 'sibling', 'not_found'),
            (replace(fresh.actor, subject='foreign'), 's', 'not_found'),
            (replace(fresh.actor, profile_id='foreign'), 's', 'profile_mismatch'),
        ):
            fresh.actor = actor
            refused = await fresh.dispatch({'id': 2, 'method': 'prompt.receipt', 'params': {
                'session_id': session, 'submission_id': 'lost-reply'}})
            assert refused['error']['data']['reason'] == reason, refused
    finally:
        await fresh.close()
        await connection.close()
        db.close()


@pytest.mark.asyncio
async def test_receipt_requires_exactly_one_nonempty_selector(owner):
    connection, _ = owner
    try:
        for selectors in ({}, {'admission_id': 'a', 'submission_id': 's'},
                          {'submission_id': ''}, {'admission_id': ''}, {'submission_id': 3}):
            reply = await connection.dispatch({'id': 1, 'method': 'prompt.receipt',
                                                'params': {'session_id': 's', **selectors}})
            assert reply['error']['data']['reason'] == 'invalid_params', reply
    finally:
        await connection.close()


@pytest.mark.asyncio
async def test_history_pages_durable_inputs_without_hidden_text_or_sibling_rows(owner):
    connection, db = owner
    authority = connection.authority
    rows = []
    for n, payload in enumerate(({'text': 'visible'}, {'text': 'secret', 'display_v1': {'kind': 'hidden'}},
                                 {'text': 'later'})):
        row = admit_session_input(db, epoch=authority.epoch, principal_id=connection.actor.subject,
                                  session_id='s', request_id=str(n), payload=payload)
        cancel_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'])
        rows.append(row)
    db.create_session('sibling', source='test')
    admit_session_input(db, epoch=authority.epoch, principal_id=connection.actor.subject,
                        session_id='sibling', request_id='0', payload={'text': 'sibling private input'})
    try:
        reply = await connection.dispatch({'id': 1, 'method': 'prompt.history', 'params': {
            'session_id': 's', 'after_sequence': 0, 'limit': 2}})
        page = reply.get('result')
        assert page is not None, reply
        assert [entry['receipt']['admission_id'] for entry in page['admissions']] == [r['admission_id'] for r in rows[:2]]
        assert page['admissions'][0]['input'] == {'text': 'visible'}
        assert page['admissions'][1]['input'] is None
        assert 'secret' not in str(page) and page['has_more']
        next_page = await connection.dispatch({'id': 2, 'method': 'prompt.history', 'params': {
            'session_id': 's', 'after_sequence': page['last_sequence'], 'limit': 2}})
        assert next_page['result']['admissions'][0]['receipt']['admission_id'] == rows[2]['admission_id']
        assert not next_page['result']['has_more']
        for invalid in ({'limit': 0}, {'limit': 65}, {'after_sequence': -1}, {'limit': True}):
            refused = await connection.dispatch({'id': 3, 'method': 'prompt.history', 'params': {
                'session_id': 's', 'after_sequence': 0, **invalid}})
            assert refused['error']['data']['reason'] == 'invalid_params'
        connection.actor = replace(connection.actor, profile_id='foreign')
        refused = await connection.dispatch({'id': 4, 'method': 'prompt.history', 'params': {
            'session_id': 's', 'after_sequence': 0}})
        assert refused['error']['data']['reason'] == 'profile_mismatch'
    finally:
        await connection.close()
