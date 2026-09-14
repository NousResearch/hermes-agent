"""Canonical Files registration reaches the real store without a legacy server."""
import base64
from dataclasses import replace
from types import SimpleNamespace

import pytest

from gateway.session_group_controls import dispatch_group_control
from hermes_state_runtime import RuntimeStoreError
from tests.gateway.test_session_group_files import files, share  # noqa: F401


@pytest.mark.asyncio
async def test_registered_files_remain_readable_without_running_coordinator(files):
    service, actor, gateway, db = files
    service.authority.hosted_room_service = service
    saved = share(service, actor, 1)
    connection = SimpleNamespace(authority=service.authority, actor=actor)
    assert not service.runtime.status()['running']
    capabilities = await dispatch_group_control(connection, 'groups.capabilities', {})
    assert not capabilities['driver']
    assert 'groups.attachment.list' in capabilities['methods']
    page = await dispatch_group_control(connection, 'groups.attachment.list', {'room_id': 'room'})
    item, = page['items']
    assert item['attachment_id'] == saved['attachment_id']
    data = await dispatch_group_control(connection, 'groups.attachment.download', {
        'room_id': 'room', 'event_id': item['event_id'], 'attachment_id': item['attachment_id'],
        'authority_gateway_id': gateway, 'authority_epoch': 1})
    assert base64.b64decode(data['data_base64']) == b'version 1'
    assert not service.runtime.status()['running']


@pytest.mark.asyncio
async def test_registration_retains_schema_and_principal_boundaries(files):
    service, actor, gateway, db = files
    service.authority.hosted_room_service = service
    share(service, actor, 1)
    connection = SimpleNamespace(authority=service.authority, actor=actor)
    for params in ({'room_id': 'room', 'path': '/not-a-capability'},
                   {'room_id': 'room', 'authority_epoch': 2, 'authority_gateway_id': gateway},
                   {'room_id': 'room', 'profile': 'foreign'}):
        with pytest.raises(RuntimeStoreError):
            await dispatch_group_control(connection, 'groups.attachment.list', params)
    for denied in (replace(actor, subject='bob'), replace(actor, capabilities=frozenset()),
                   replace(actor, profile_id='foreign')):
        with pytest.raises(RuntimeStoreError):
            await dispatch_group_control(SimpleNamespace(authority=service.authority, actor=denied),
                'groups.attachment.list', {'room_id': 'room'})
