"""A chat controls Group Chats only with an owner grant and an allowlisted sender."""
import json
import time
from types import SimpleNamespace

import pytest

from gateway import group_chat_access as access
from gateway.config import Platform
from gateway.group_chat_identity import is_private_source
from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.group_chat_fixtures import OWNER, Bot, authority_for, message, runner_for


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with SessionDB(tmp_path / 'state.db') as db:
        authority = authority_for(tmp_path, db)
        bot = Bot()
        runner = runner_for(authority, bot)
        yield SimpleNamespace(authority=authority, bot=bot, runner=runner, verb=access.control_verb(runner))


def connect(setup, event, subject=OWNER):
    chat, adapter = access.resolve_chat(setup.runner, event)
    code, _ = access.request_code(setup.runner, setup.authority, chat, event.source, adapter)
    return chat, setup.verb({'action': 'allow', 'code': access.display_code(code)}, subject)


def test_only_allowlisted_people_resolve_and_the_scope_picks_the_list(setup):
    private, _ = access.resolve_chat(setup.runner, message('/group'))
    assert (private.kind, private.user_id, private.chat_label) == ('private', 'alice', 'chat-1')
    shared, _ = access.resolve_chat(setup.runner, message('/group', user='bob', chat_type='group', chat_name='Team'))
    assert (shared.kind, shared.chat_label, shared.admins) == ('shared', 'Team', 'group_allow_admin_from')
    # A Slack DM can hold several people: shared audience, but still the DM admin list.
    slack, _ = access.resolve_chat(setup.runner, message('/group', platform=Platform.SLACK))
    assert (slack.kind, slack.admins) == ('shared', 'allow_admin_from')
    # Bob is a group admin, not a DM admin; Carol is neither; an empty list allows nobody.
    for event in (message('/group', user='bob'), message('/group', user='carol', chat_type='group')):
        with pytest.raises(access.GroupChatDenied, match='admin'):
            access.resolve_chat(setup.runner, event)
    setup.bot.config.extra = {}
    with pytest.raises(access.GroupChatDenied, match='allow_admin_from'):
        access.resolve_chat(setup.runner, message('/group'))
    setup.runner._transport_owner = lambda source: None
    with pytest.raises(access.GroupChatDenied, match='available'):
        access.resolve_chat(setup.runner, message('/group'))


@pytest.mark.parametrize('change', [
    {'is_bot': True}, {'message_is_edit': True}, {'delivered_via_upstream_relay': True},
    {'profile_route_rejected': True}, {'user': 'unknown'}, {'chat_type': 'channel', 'user': '-100'},
])
def test_machines_edits_and_unclassified_relays_are_not_people(setup, change):
    with pytest.raises(access.GroupChatDenied, match='person'):
        access.resolve_chat(setup.runner, message('/group', **change))


def test_private_needs_a_provable_one_to_one_chat():
    # Slack calls multi-person DMs "dm": that chat is shared, while a Telegram DM is private.
    assert is_private_source(message('/x').source)
    assert not is_private_source(message('/x', platform=Platform.SLACK).source)
    assert is_private_source(message('/x', platform=Platform.SLACK, is_one_to_one=True).source)
    assert not is_private_source(message('/x', chat_type='group').source)
    assert not is_private_source(message('/x', delivered_via_upstream_relay=True).source)


def test_grant_key_names_the_person_only_in_private_chats(setup):
    alice, _ = access.resolve_chat(setup.runner, message('/group', chat='team', chat_type='group'))
    bob, _ = access.resolve_chat(setup.runner, message('/group', user='bob', chat='team', chat_type='group'))
    assert alice.key == bob.key
    thread, _ = access.resolve_chat(setup.runner, message('/group', chat='team', chat_type='group', thread='t1'))
    assert thread.key != alice.key
    dm, _ = access.resolve_chat(setup.runner, message('/group', chat='team'))
    assert dm.key != alice.key


def test_codes_are_reused_per_chat_bounded_and_expire(setup, monkeypatch):
    event = message('/group')
    chat, adapter = access.resolve_chat(setup.runner, event)
    first, ttl = access.request_code(setup.runner, setup.authority, chat, event.source, adapter)
    assert ttl == access.CODE_TTL_SECONDS and len(first) == 8
    assert access.request_code(setup.runner, setup.authority, chat, event.source, adapter)[0] == first
    later = time.monotonic() + access.CODE_TTL_SECONDS + 1
    monkeypatch.setattr(access.time, 'monotonic', lambda: later)
    assert access.request_code(setup.runner, setup.authority, chat, event.source, adapter)[0] != first
    monkeypatch.setattr(access, 'MAX_PENDING_CODES', 1)
    other, _ = access.resolve_chat(setup.runner, message('/group', chat='other'))
    with pytest.raises(access.GroupChatDenied, match='waiting'):
        access.request_code(setup.runner, setup.authority, other, event.source, adapter)


def test_allow_binds_the_cli_account_once_and_announces_to_the_chat(setup, monkeypatch):
    event = message('/group', chat='team', chat_type='group', chat_name='Team', thread='topic')
    chat, adapter = access.resolve_chat(setup.runner, event)
    code, _ = access.request_code(setup.runner, setup.authority, chat, event.source, adapter)
    scheduled = []
    monkeypatch.setattr('asyncio.run_coroutine_threadsafe',
                        lambda coroutine, loop: scheduled.append((coroutine.cr_frame.f_locals, loop))
                        or coroutine.close())
    verb = access.control_verb(setup.runner, 'loop')
    described = verb({'action': 'describe', 'code': code.lower()}, OWNER)
    assert (described['kind'], described['chat'], described['user_id']) == ('shared', 'Team', 'alice')
    assert described['admins'] == 'group_allow_admin_from'
    allowed = verb({'action': 'allow', 'code': access.display_code(code)}, OWNER)
    assert allowed['grant'] == chat.key[:8]
    (sent, loop), = scheduled
    assert loop == 'loop' and sent['chat_id'] == 'team' and sent['metadata'] == {'thread_id': 'topic'}
    assert 'Everyone here can read' in sent['content']
    grant = access.current_grant(setup.authority, chat)
    assert (grant['owner'], grant['thread_id'], grant['kind']) == (OWNER, 'topic', 'shared')
    assert verb({'action': 'allow', 'code': code}, OWNER) == {'error': 'unknown_code'}


def test_another_account_cannot_take_over_and_sees_none_of_the_grants(setup):
    chat, allowed = connect(setup, message('/group'))
    assert 'grant' in allowed
    _, taken = connect(setup, message('/group'), subject='uid:999')
    assert taken == {'error': 'permission_denied'}
    assert setup.verb({'action': 'list'}, 'uid:999') == {'chats': []}
    assert setup.verb({'action': 'revoke', 'grant': chat.key[:8]}, 'uid:999') == {'error': 'unknown_grant'}
    listed = setup.verb({'action': 'list'}, OWNER)['chats']
    assert [(row['grant'], row['kind'], row['user']) for row in listed] == [(chat.key[:8], 'private', 'Alice')]
    assert access.current_grant(setup.authority, chat)['owner'] == OWNER


def test_revoke_by_unique_prefix_removes_only_that_grant(setup):
    chat, _ = connect(setup, message('/group'))
    other, _ = connect(setup, message('/group', chat='other'))
    assert setup.verb({'action': 'revoke', 'grant': 'zz'}, OWNER) == {'error': 'unknown_grant'}
    twin = {**access.current_grant(setup.authority, chat), 'grant_id': chat.key[:8] + 'f' * 56}
    setup.authority.db._execute_write(lambda conn: access._save(conn, twin))
    assert setup.verb({'action': 'revoke', 'grant': chat.key[:8]}, OWNER) == {'error': 'ambiguous_grant'}
    revoked = setup.verb({'action': 'revoke', 'grant': chat.key.upper()}, OWNER)
    assert revoked['revoked'] == chat.key[:8]
    assert access.current_grant(setup.authority, chat) is None
    assert access.current_grant(setup.authority, other) is not None


def test_room_numbers_are_stable_and_never_reused(setup):
    chat, _ = connect(setup, message('/group'))
    grant = access.current_grant(setup.authority, chat)
    grant = access.assign_refs(setup.authority, grant, ['a', 'b'])
    assert grant['refs'] == {'a': 1, 'b': 2}
    grant = access.assign_refs(setup.authority, grant, ['b', 'c'])
    assert grant['refs'] == {'b': 2, 'c': 3}
    assert access.room_for_ref(grant, 1) is None and access.room_for_ref(grant, 3) == 'c'
    setup.verb({'action': 'revoke', 'grant': chat.key[:8]}, OWNER)
    with pytest.raises(RuntimeStoreError, match='permission_denied'):
        access.assign_refs(setup.authority, grant, ['d'])


def test_a_damaged_or_misfiled_grant_never_authorizes(setup):
    chat, _ = connect(setup, message('/group'))
    key = access.GRANT_PREFIX + chat.key
    with setup.authority.db._read_ctx() as conn:
        good = json.loads(conn.execute('SELECT value FROM state_meta WHERE key=?', (key,)).fetchone()[0])
    for damaged in ('not json', json.dumps({**good, 'kind': 'everyone'}),
                    json.dumps({**good, 'extra': 1}), json.dumps({**good, 'grant_id': 'f' * 64}),
                    json.dumps({**good, 'refs': {'a': 5}, 'next_ref': 2}), json.dumps({**good, 'version': True})):
        setup.authority.db._execute_write(
            lambda conn: conn.execute('UPDATE state_meta SET value=? WHERE key=?', (damaged, key)))
        assert access.current_grant(setup.authority, chat) is None
        assert setup.verb({'action': 'list'}, OWNER) == {'chats': []}


def test_unknown_actions_and_unauthenticated_calls_are_refused(setup):
    assert setup.verb({'action': 'grant-everything'}, OWNER) == {'error': 'invalid_request'}
    assert setup.verb({'action': 'list'}, '') == {'error': 'invalid_request'}
    assert setup.verb({'action': 'describe', 'code': 'not-a-code'}, OWNER) == {'error': 'unknown_code'}


def test_author_names_the_person_and_platform():
    chat = access.Chat('k', 'default', 'telegram', 'c', None, None, '42', 'private', 'c', 'Alice', 'allow_admin_from')
    assert chat.author() == {'kind': 'user', 'id': 'telegram:42', 'display_name': 'Alice via Telegram'}
    odd = access.Chat('k', 'default', 'matrix', 'c', None, None, '@alice:example.org', 'shared', 'c', 'Alice',
                      'allow_admin_from')
    assert odd.author()['id'].startswith('matrix:sha256-')
