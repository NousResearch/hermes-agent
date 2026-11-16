"""RT-32 (#140156): Desktop's Bot Chat lookup adopts a legacy local row stamped with the caller's own subject.

Older builds stamped ``user_id`` on local rows. The legacy migration refused every stamp except main's Desktop
Nous-account label, so ``session.list {profile, title: "Bot Chat"}`` answered ``not_found`` for a row the
caller already owns by ``authorize_history``'s rule (``user_id == actor.subject``). A stamp naming another
principal stays refused. Driven through a multiplexed runner with a named primary and a host-scoped
connection, the way Desktop opens a Bot Chat.
"""
from pathlib import Path

import pytest

from tests.gateway.test_session_authorities_multiplex import _reserve_homes, _runner


class _Transport:
    def write(self, frame):
        return True

    def close(self):
        pass


_OWN = 'uid:desk'
# A stamp naming any other principal: another local user, a dashboard login as v0.22 stamped it, an
# authenticated gateway subject, and a Nous label on a non-Desktop row.
_FOREIGN = ('uid:4242', 'basic:alice', 'auth:v1:["dashboard","","alice"]', 'nous:acct-1')


async def _bot_chat_lookup(tmp_path, monkeypatch, source, user_id):
    from gateway.run_runtime import initialize_gateway_runtime
    from gateway.runtime_ownership import process_ownership
    from gateway.session_authorities import authority_for_profile_id, owner_scope
    from gateway.session_controls import AuthorityConnection
    root, homes = _reserve_homes(tmp_path, monkeypatch)
    process_ownership.reserve([home for _, home in homes])
    connections, runner, replies = [], None, {}
    try:
        runner = _runner(root, homes)
        await initialize_gateway_runtime(runner)
        store = runner.session_ticket_store
        ids = {name: str(home.resolve()) for name, home in homes}
        ticket = store.mint(profile_id=ids['alpha'], subject=_OWN, purpose='interactive', scope='host')
        grant = store.redeem(ticket, profile_id=None, purpose='interactive')
        authority = authority_for_profile_id(runner, grant['profile_id'])
        identity = {'user_id': grant['subject'], 'provider': 'local', 'profile_id': grant['profile_id'],
                    'instance_id': grant['instance_id'], 'capabilities': grant['capabilities'],
                    'native_bootstrap': True, 'profile_scope': grant.get('scope', 'profile')}
        with owner_scope(authority):
            host = AuthorityConnection(authority, _Transport(), identity, operator=True)
        connections.append(host)
        for prof in ('default', 'beta'):
            db = authority_for_profile_id(runner, ids[prof]).db
            db.create_session(prof + '-bot-chat', source=source, model='old', cwd=str(root), user_id=user_id)
            db.set_session_title(prof + '-bot-chat', 'Bot Chat')
            db.set_session_hidden(prof + '-bot-chat', True)
        for prof in ('default', 'beta'):
            with owner_scope(host.authority):
                replies[prof] = await host.dispatch({'id': 1, 'method': 'session.list', 'params': {
                    'profile': prof, 'title': 'Bot Chat', 'include_hidden': True, 'limit': 200}})
    finally:
        for connection in connections:
            with owner_scope(connection.authority):
                await connection.close()
        for authority in list(getattr(runner, 'session_authorities', []) or []):
            authority.db.close()
        for _, home in homes:
            process_ownership.release(Path(home))
    return replies


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['cli', 'desktop'])
async def test_bot_chat_stamped_with_the_callers_own_subject_opens(tmp_path, monkeypatch, source):
    replies = await _bot_chat_lookup(tmp_path, monkeypatch, source, _OWN)
    for prof, reply in replies.items():
        assert 'error' not in reply, (prof, reply)
        assert [s.get('id') or s.get('session_id') for s in reply['result']['sessions']] == [prof + '-bot-chat'], (
            prof, reply)


@pytest.mark.asyncio
@pytest.mark.parametrize('user_id', _FOREIGN)
async def test_bot_chat_stamped_with_another_principal_stays_refused(tmp_path, monkeypatch, user_id):
    replies = await _bot_chat_lookup(tmp_path, monkeypatch, 'cli', user_id)
    for prof, reply in replies.items():
        assert reply.get('error', {}).get('message') == 'not_found', (prof, reply)


def test_legacy_validation_names_the_subject_it_adopts_for(tmp_path):
    from hermes_state import SessionDB
    from hermes_state_local_migration import validate_legacy_row
    from hermes_state_runtime import RuntimeStoreError
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        db.create_session('own', source='cli', user_id=_OWN)
        validate_legacy_row(db, db.get_session('own'), subject=_OWN)
        with pytest.raises(RuntimeStoreError):  # no subject named: a stamped row is someone's
            validate_legacy_row(db, db.get_session('own'))
        with pytest.raises(RuntimeStoreError):
            validate_legacy_row(db, db.get_session('own'), subject='uid:4242')
        # Composes with the login-stamp rule: a Desktop row a served login stamped adopts for that login
        # only; the native peer's own-subject rule does not reach it.
        db.create_session('login', source='desktop', user_id='basic:alice')
        validate_legacy_row(db, db.get_session('login'), subject='auth:v1:["basic","","alice"]')
        for other in (_OWN, 'auth:v1:["basic","","bob"]', 'auth:v1:["basic","iss","alice"]'):
            with pytest.raises(RuntimeStoreError):
                validate_legacy_row(db, db.get_session('login'), subject=other)
    finally:
        db.close()
