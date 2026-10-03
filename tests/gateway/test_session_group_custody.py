"""Custody on the canonical surface: custody-only invitations, consent, owner controls and reading a copy."""
import pytest

from gateway import hosted_room_custody as custody
from gateway import hosted_room_peer as peer
from gateway import hosted_room_replicas as replicas
from gateway import hosted_rooms as rooms
from gateway.session_controls import AuthorityConnection
from types import SimpleNamespace

from tests.gateway.fixtures.passive_copy import HOME, MEMBERS, append, catalog, member
from tests.gateway.test_session_group_replication import IDENTITY, call, gateway, permissions  # noqa: F401
from tui_gateway.contracts import groups_bot_relay as contract

CUSTODY = dict(room_id='room', home_install_id=HOME, authority_gateway_id=HOME, authority_epoch=1)


def _copy_here(gateway, tmp_path, room_id='room'):
    """A copy of another gateway's room in this gateway's store, as its home's pages left it."""
    source = tmp_path / f'{room_id}-home.db'
    rooms.create_room(source, room_id=room_id, name='Workshop', members=MEMBERS, authority_gateway_id=HOME)
    append(source, 'hello', 'Café')
    page = rooms.read_events(source, room_id=room_id)
    replicas.ingest_page(gateway.authority.db.db_path, room_id=room_id, room_name='Workshop', members=MEMBERS,
                         page=page)
    return page


@pytest.mark.asyncio
async def test_custody_methods_are_canonical_and_contracted(gateway):
    methods = (await call(gateway.owner, 'groups.capabilities'))['methods']
    assert {'groups.custody.status', 'groups.custody.designate', 'groups.custody.add', 'groups.custody.remove',
            'groups.custody.allow'} <= set(methods)


@pytest.mark.asyncio
async def test_a_custody_only_invitation_mints_a_copy_only_grant(gateway):
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    minted = await call(gateway.owner, 'groups.peer.invite', **CUSTODY, custody_only=True, successor=True)
    claims = peer.decode_room_grant(peer.gateway_room_grant_secret(), minted['grant'], permission='status')
    assert (claims['member_id'], claims['permissions']) == (custody.CUSTODY_MEMBER_ID, ['replicate', 'status', 'successor'])
    plain = await call(gateway.owner, 'groups.peer.invite', **CUSTODY, custody_only=True)
    assert permissions(plain['grant']) == ['replicate', 'status']
    for flags in ({'member_id': 'reviewer'}, {'work_records': True}, {'replication': False}, {'passive_only': False}):
        assert await call(gateway.owner, 'groups.peer.invite', **CUSTODY, custody_only=True,
                          **flags) == 'invalid_params', flags
    assert await call(gateway.owner, 'groups.peer.invite', **CUSTODY, custody_only=False) == 'invalid_params'
    assert custody.local_consent(gateway.authority.db.db_path, 'room') is False  # the latest invitation's choice


@pytest.mark.asyncio
async def test_the_operator_allows_continuation_here_and_sees_it_unconfirmed_until_the_host_reports(gateway):
    gateway.adapter.gateway_runner = SimpleNamespace(session_authority=gateway.authority)
    await call(gateway.owner, 'groups.peer.invite', **IDENTITY, successor=True)
    assert custody.local_consent(gateway.authority.db.db_path, 'room') is True
    allowed = await call(gateway.owner, 'groups.custody.allow', room_id='room', successor=False)
    assert allowed == {'room_id': 'room', 'install_id': rooms.local_authority_gateway_id(), 'allowed': False,
                       'confirmed': False}
    contract.GroupsCustodyAllowResult.model_validate(allowed)
    assert custody.local_consent(gateway.authority.db.db_path, 'room') is False
    member_only = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    assert await call(member_only, 'groups.custody.allow', room_id='room', successor=True) == 'permission_denied'
    assert await call(gateway.owner, 'groups.custody.allow', room_id='room', successor='yes') == 'invalid_params'


@pytest.mark.asyncio
async def test_a_copy_reads_like_a_room_for_its_operator_and_recorded_owner_only(gateway, tmp_path):
    page = _copy_here(gateway, tmp_path)
    listed = (await call(gateway.owner, 'groups.list'))['rooms']
    assert [(room['room_id'], room['copy'], room['revision'], room['latest_seq']) for room in listed] == [
        ('room', True, 0, page['latest_seq'])]
    contract.GroupsListResult.model_validate({'rooms': listed, 'next_offset': None})
    state = await call(gateway.owner, 'groups.state', room_id='room')
    assert (state['room']['copy'], state['driver_status']) == (True, None)
    assert state['room']['custody'] == {'configuration_seq': 0, 'at_risk_after_seq': 0, 'custodians': []}
    contract.GroupsStateResult.model_validate(state)
    log = await call(gateway.owner, 'groups.log', room_id='room')
    assert [event['payload']['text'] for event in log['events']] == ['Café']
    assert log['authority'] == page['authority']
    status = await call(gateway.owner, 'groups.custody.status', room_id='room')
    assert status['role'] == 'custodian'
    contract.GroupsCustodyStatusResult.model_validate(status)
    # Anyone else sees nothing of it, unless recorded as the room's owner on this installation.
    stranger = AuthorityConnection(gateway.authority, object(), {'user_id': 'stranger'})
    assert (await call(stranger, 'groups.list'))['rooms'] == []
    assert await call(stranger, 'groups.state', room_id='room') == 'permission_denied'
    assert await call(stranger, 'groups.log', room_id='room') == 'permission_denied'
    gateway.authority.db._execute_write(lambda conn: conn.execute(
        'INSERT INTO state_meta(key, value) VALUES (?, ?)', ('gateway.hosted.owner.v1:room', stranger.actor.subject)))
    assert [room['room_id'] for room in (await call(stranger, 'groups.list'))['rooms']] == ['room']
    assert (await call(stranger, 'groups.log', room_id='room'))['events']
    # A copy is never written to here.
    assert await call(gateway.owner, 'groups.rename', room_id='room', event_id='rename', name='Mine') != {}
    assert replicas.replica_state(gateway.authority.db.db_path, room_id='room')['name'] == 'Workshop'


@pytest.mark.asyncio
async def test_a_shared_messaging_chat_never_changes_custody(gateway):
    """A messaging chat other people read carries the owner's subject, yet it never acts for the owner."""
    from dataclasses import replace
    await call(gateway.owner, 'groups.create', room_id='hosted', name='Hosted', members=[
        MEMBERS[0], member('reviewer', target='install:participant')])
    shared = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    shared.actor = replace(gateway.owner.actor, transport_id='messaging:shared:grant-1')
    for method, params in (('groups.custody.designate', {'install_id': 'install:nobody', 'successor': True}),
                           ('groups.custody.add', {'target_url': 'https://backup.example', 'grant': 'g',
                                                   'catalog': catalog('install:backup').as_mapping()}),
                           ('groups.custody.remove', {'install_id': 'install:nobody'})):
        assert await call(shared, method, room_id='hosted', **params) == 'permission_denied', method
    assert (await call(shared, 'groups.custody.status', room_id='hosted'))['role'] == 'authority'
    private = AuthorityConnection(gateway.authority, object(), {'user_id': 'owner'})
    private.actor = replace(gateway.owner.actor, transport_id='messaging:private:grant-2')
    assert await call(private, 'groups.custody.designate', room_id='hosted', install_id='install:nobody',
                      successor=True) == 'room_custody_invalid'  # the owner's private chat reaches the check


@pytest.mark.asyncio
async def test_owner_controls_refuse_what_custody_cannot_be(gateway):
    room = (await call(gateway.owner, 'groups.create', room_id='hosted', name='Hosted', members=[
        MEMBERS[0], member('reviewer', target='install:participant')]))['room']
    assert room['authority_gateway_id'] == rooms.local_authority_gateway_id()
    assert await call(gateway.owner, 'groups.custody.designate', room_id='hosted', install_id='install:nobody',
                      successor=True) == 'room_custody_invalid'
    status = await call(gateway.owner, 'groups.custody.status', room_id='hosted')
    assert (status['role'], status['custodians'], status['at_risk_after_seq']) == ('authority', [], 0)
    contract.GroupsCustodyStatusResult.model_validate(status)
    # A member installation is a custodian through its Bots already, never also custodian-only.
    grant = peer.issue_room_grant(
        b'p' * 32, grant_id='g', room_id='hosted', home_install_id=room['authority_gateway_id'],
        authority_gateway_id=room['authority_gateway_id'], authority_epoch=1, member_id=custody.CUSTODY_MEMBER_ID,
        target_install_id='install:participant', target_profile='default', permissions=('replicate', 'status'),
        ttl_seconds=600)
    assert await call(gateway.owner, 'groups.custody.add', room_id='hosted', target_url='https://backup.example',
                      catalog=catalog('install:participant').as_mapping(), grant=grant) == 'peer_target_mismatch'
    assert await call(gateway.owner, 'groups.custody.add', room_id='hosted', target_url='https://backup.example',
                      catalog=catalog('install:backup').as_mapping(), grant=grant.replace('.', '!')) == 'invalid_params'
    assert await call(gateway.owner, 'groups.custody.remove', room_id='hosted',
                      install_id='install:nobody') == 'room_custody_invalid'
