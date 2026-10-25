"""A native admission refusal names the failing clause in the error and the log, and in the chat
reply only for content clauses; trust and provenance clauses stay generic there."""
import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.run import GatewayRunner
from gateway.session_authority import initialize_session_authority
from gateway.session_envelope import prepare_native
from gateway.session_ingress import admit_message
from hermes_state_runtime import RuntimeStoreError, list_session_admissions

CHAT = '15551234567@s.whatsapp.net'


async def _authority(tmp_path, monkeypatch):
    from plugins.platforms.whatsapp.adapter import WhatsAppAdapter
    monkeypatch.setenv('WHATSAPP_ALLOWED_USERS', '15551234567')
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = WhatsAppAdapter(PlatformConfig(enabled=True, extra={'session_path': str(tmp_path / 'wa')}))
    adapter.gateway_runner, runner.adapters = runner, {Platform.WHATSAPP: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='named')
    runner._wire_adapter_handlers(adapter)
    return authority, adapter.build_source(chat_id=CHAT, user_id=CHAT, chat_type='dm')


@pytest.mark.asyncio
@pytest.mark.parametrize('fields, clause', [
    ({'media_urls': 'https://example.com/a.png'}, 'media_urls'),
    ({'reply_expected': 'yes'}, 'reply_expected'),
    ({'auto_skill': 7}, 'auto_skill'),
    # Overlapping failures: the first clause in the original check's order is the one named.
    ({'reply_expected': 'yes', 'auto_skill': 7}, 'reply_expected'),
    ({'media_urls': 'https://example.com/a.png', 'media_types': 'image/png'}, 'media_urls'),
])
async def test_invalid_params_names_the_failing_clause(tmp_path, monkeypatch, caplog, fields, clause):
    authority, source = await _authority(tmp_path, monkeypatch)
    try:
        with caplog.at_level(logging.WARNING, logger='gateway.session_envelope'):
            with pytest.raises(RuntimeStoreError) as raised:
                await prepare_native(authority.runner, MessageEvent(text='hi', source=source, **fields))
        # A string still refuses after MSG-2's None -> [] normalisation. The wire reason is unchanged; the clause rides beside it (was a bare invalid_params).
        assert raised.value.reason == 'invalid_params' and raised.value.field == clause
        assert any(clause in record.getMessage() and 'invalid_params' in record.getMessage()
                   for record in caplog.records), [r.getMessage() for r in caplog.records]
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_messaging_user_sees_named_refusal_not_generic_error(tmp_path, monkeypatch):
    authority, source = await _authority(tmp_path, monkeypatch)
    try:
        reply = await admit_message(authority, MessageEvent(text='hi', source=source, message_id='m1',
                                                            media_urls='https://example.com/a.png'))
        # Was: RuntimeStoreError escaped -> "Sorry, I encountered an error (RuntimeStoreError)".
        assert 'invalid_params' in reply and 'media_urls' in reply, reply
        # Refused before commit: nothing is queued for a retry or a drain to run.
        assert not any(list_session_admissions(authority.db, session_id=live, pending_only=False)
                       for live in authority.sessions)
    finally:
        authority.db.close()


def _forge_multiplex(authority, source):
    authority.runner.config.multiplex_profiles = True


def _forge_authorization_home(authority, source):
    source._authorization_profile_home = '/forged/home'


@pytest.mark.asyncio
@pytest.mark.parametrize('forge, clause', [
    (_forge_multiplex, 'multiplex_without_provenance'),
    (_forge_authorization_home, 'authorization_home_without_provenance'),
])
async def test_trust_clause_refusal_is_not_named_to_the_chat(tmp_path, monkeypatch, caplog, forge, clause):
    authority, source = await _authority(tmp_path, monkeypatch)
    try:
        forge(authority, source)
        with caplog.at_level(logging.WARNING, logger='gateway.session_envelope'):
            with pytest.raises(RuntimeStoreError) as raised:
                await prepare_native(authority.runner, MessageEvent(text='hi', source=source))
            reply = await admit_message(authority, MessageEvent(text='hi', source=source, message_id='m1'))
        assert raised.value.reason == 'invalid_params' and raised.value.field == clause
        # The reply reaches everyone in a group chat: a trust/provenance check stays generic there,
        # so a sender probing forged trust fields learns nothing; the operator log keeps the clause.
        assert 'invalid_params' in reply and clause not in reply and 'provenance' not in reply, reply
        assert any(clause in record.getMessage() and 'invalid_params' in record.getMessage()
                   for record in caplog.records), [r.getMessage() for r in caplog.records]
        assert not any(list_session_admissions(authority.db, session_id=live, pending_only=False)
                       for live in authority.sessions)
    finally:
        authority.db.close()
