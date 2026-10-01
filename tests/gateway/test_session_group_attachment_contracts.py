"""Declared canonical byte RPCs accept their actual handler inputs and outputs."""
import base64
import json

import pytest

from gateway.session_hosted_attachments import append_user_event
from tests.gateway.test_session_group_peers import call, gateway  # noqa: F401
from tui_gateway.contracts.groups_bot_relay import (
    GroupsAttachmentUploadParams, GroupsAttachmentResult,
    GroupsAttachmentDownloadParams, GroupsAttachmentDownloadResult)


@pytest.mark.asyncio
async def test_canonical_attachment_contracts_match_upload_and_authorized_download(gateway):
    helper = gateway.home / 'profiles' / 'helper'
    helper.mkdir(parents=True)
    (gateway.home / 'config.yaml').write_text('hosted_rooms:\n  profiles:\n    helper: ' + json.dumps(str(helper)) + '\n')
    created = await call(gateway.owner, 'groups.create', room_id='room', name='Room', members=[
        {'member_id': 'writer', 'profile': 'default', 'handle': 'writer'},
        {'member_id': 'helper', 'profile': 'helper', 'handle': 'helper'}])
    assert isinstance(created, dict), created
    room = created['room']
    params = dict(room_id='room', upload_id='file', kind='file', name='note.txt', mime='text/plain',
                  data_base64=base64.b64encode(b'private bytes').decode())
    GroupsAttachmentUploadParams.model_validate(params)
    uploaded = await call(gateway.owner, 'groups.attachment.upload', **params)
    GroupsAttachmentResult.model_validate(uploaded)
    manifest = [{key: uploaded[key] for key in ('attachment_id', 'kind', 'name', 'size', 'mime')}]
    append_user_event(gateway.service, room_id='room', event_id='message', payload={'text': 'Read', 'attachments': manifest},
                      gateway_id=room['authority_gateway_id'], epoch=room['authority_epoch'])
    params = dict(room_id='room', event_id='message', attachment_id=uploaded['attachment_id'])
    GroupsAttachmentDownloadParams.model_validate(params)
    downloaded = await call(gateway.owner, 'groups.attachment.download', **params)
    GroupsAttachmentDownloadResult.model_validate(downloaded)
    assert base64.b64decode(downloaded['data_base64']) == b'private bytes'
