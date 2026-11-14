"""An email that is fetched again after a gateway restart (it is still unread) is the same admission.

The email adapter has no provider time on its event, so each fetch stamps local arrival time, and
that ``timestamp`` must not make an exact redelivery look like a changed replay."""
import asyncio
import os
from unittest.mock import patch

import pytest

ENV = {
    'EMAIL_ADDRESS': 'hermes@test.com', 'EMAIL_PASSWORD': 'secret', 'EMAIL_IMAP_HOST': 'imap.test.com',
    'EMAIL_IMAP_PORT': '993', 'EMAIL_SMTP_HOST': 'smtp.test.com', 'EMAIL_SMTP_PORT': '587',
    'EMAIL_ALLOWED_USERS': 'owner@example.test',
}


def _mail(body='please check the report'):
    return {'uid': b'1', 'sender_addr': 'owner@example.test', 'sender_name': 'Owner', 'subject': 'Report',
            'body': body, 'message_id': '<fixture-1@example.test>', 'in_reply_to': '', 'attachments': [],
            'headers': {}, 'sender_authenticated': True, 'auth_configuration_missing': False,
            'auth_reason': 'test'}


async def _fetch(adapter, mail):
    events = []

    async def handle(event):
        events.append(event)
    adapter.handle_message = handle
    await adapter._dispatch_message(dict(mail))
    return events[0]


@pytest.mark.asyncio
async def test_refetched_email_after_restart_is_its_committed_admission(tmp_path, monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from gateway.session_ingress_context import native_callback
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import (RuntimeStoreError, claim_session_input, list_session_admissions,
                                      settle_session_input)
    from plugins.platforms.email.adapter import EmailAdapter
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    with patch.dict(os.environ, ENV, clear=False):
        runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
        adapter = EmailAdapter(PlatformConfig(enabled=True))
        adapter.gateway_runner, runner.adapters = runner, {Platform.EMAIL: adapter}
        authority = await initialize_session_authority(runner, profile_id='default', instance_id='email')
        runner._wire_adapter_handlers(adapter)

        async def admit(event):
            with native_callback(runner, event, get_hermes_home()):
                return await authority.admit_native(event)
        try:
            first = await admit(await _fetch(adapter, _mail()))
            sid = first.ref.session_id
            claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=sid)
            settle_session_input(authority.db, epoch=authority.epoch, admission_id=first.admission_id,
                                 generation=claim['generation'], outcome='interrupted')
            again = await admit(await _fetch(adapter, _mail()))
            assert again.admission_id == first.admission_id and again.status == 'terminal'
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                await admit(await _fetch(adapter, _mail(body='wire the money instead')))
            assert len(list_session_admissions(authority.db, session_id=sid, pending_only=False)) == 1
        finally:
            await asyncio.to_thread(authority.db.close)
