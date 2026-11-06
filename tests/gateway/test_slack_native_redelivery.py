"""Slack redelivery keeps one durable admission despite a new local arrival clock."""
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session_authority import SessionAuthority, initialize_session_authority
from gateway.session_ingress_context import native_callback
from hermes_constants import get_hermes_home
from hermes_state_runtime import (RuntimeStoreError, claim_session_input, list_session_admissions,
                                  settle_session_input)
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.mark.asyncio
@pytest.mark.parametrize('legacy_arrival', [False, True], ids=['provider-time', 'legacy-arrival-time'])
@pytest.mark.parametrize('legacy_schema', [False, True], ids=['current-schema', 'legacy-schema'])
async def test_slack_redelivery_reuses_admission_and_rejects_changed_content(
        tmp_path, monkeypatch, legacy_arrival, legacy_schema):
    monkeypatch.setenv('SLACK_ALLOWED_USERS', 'U_TEST')
    monkeypatch.setattr(SessionAuthority, '_schedule', lambda self, ref: None)
    runner = GatewayRunner(GatewayConfig(sessions_dir=tmp_path / 'sessions'))
    adapter = SlackAdapter(PlatformConfig(enabled=True, token='fixture-token'))
    runner.adapters = {Platform.SLACK: adapter}
    adapter._resolve_user_name = AsyncMock(return_value='Fixture User')
    adapter._resolve_channel_name = AsyncMock(return_value='fixture')
    adapter._humanize_user_mentions = AsyncMock(side_effect=lambda text, **_: text)
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='slack-retry')
    runner._wire_adapter_handlers(adapter)
    raw = {'type': 'message', 'user': 'U_TEST', 'channel': 'C_TEST',
           'team': 'T_TEST', 'text': 'synthetic message', 'ts': '1792485702.680459'}

    async def event():
        return await adapter._build_message_event(
            dict(raw), text=raw['text'], original_text=raw['text'], command_probe_text=raw['text'],
            is_command_text=False, channel_id='C_TEST', team_id='T_TEST', ts=raw['ts'],
            user_id='U_TEST', thread_ts=raw['ts'], is_dm=False, media_urls=[], media_types=[],
            media_text_inlined=[], channel_context=None)

    async def admit(value):
        with native_callback(runner, value, get_hermes_home()):
            return await authority.admit_native(value)

    try:
        first_event = await event()
        provider_time = first_event.timestamp
        if legacy_arrival:
            first_event = replace(first_event, timestamp=datetime(2026, 10, 20, 17, 41, 57, 652398))
        if legacy_schema:
            # Model the supported older persisted shape, not weaker current authorization.
            import gateway.session_envelope as envelopes
            actual_prepare = envelopes.prepare_native

            async def legacy_prepare(runner, event):
                payload = await actual_prepare(runner, event)
                payload['native_text_v1'].pop('provenance', None)
                return payload

            with monkeypatch.context() as older:
                older.setattr(envelopes, 'prepare_native', legacy_prepare)
                first = await admit(first_event)
        else:
            first = await admit(first_event)
        second_event = await event()
        second = await admit(second_event)
        assert second.admission_id == first.admission_id
        assert provider_time == datetime.fromtimestamp(1792485702.680459, UTC)
        assert second_event.timestamp == provider_time
        rows = list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False)
        assert len(rows) == 1
        original = rows[0]['payload']
        for changed in [replace(second_event, text='different synthetic content'),
                        replace(second_event, reply_expected=False),
                        replace(second_event, channel_context='new synthetic context')]:
            with pytest.raises(RuntimeStoreError, match='admission_conflict'):
                await admit(changed)
        with pytest.raises(RuntimeStoreError, match='not_found'):
            await admit(replace(second_event, source=replace(second_event.source, scope_id='T_OTHER')))
        claim = claim_session_input(authority.db, epoch=authority.epoch, session_id=first.ref.session_id)
        settle_session_input(authority.db, epoch=authority.epoch, admission_id=first.admission_id,
                             generation=claim['generation'], outcome='completed')
        late = await admit(await event())
        assert late.admission_id == first.admission_id and late.status == 'terminal'
        rows = list_session_admissions(authority.db, session_id=first.ref.session_id, pending_only=False)
        assert len(rows) == 1 and rows[0]['payload'] == original
    finally:
        authority.db.close()
