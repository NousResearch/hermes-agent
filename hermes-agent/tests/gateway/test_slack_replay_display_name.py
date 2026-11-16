"""A Slack display-name refresh is not a new message after restart (#149150).

Composed Slack ingress, GatewayRunner, and SQLite across two owner epochs. The
restart/message_changed scenario is adapted from DygaJ's fix/slack-admitted-replay
harness; unlike blanket conflict suppression, content and sender changes must
still fail admission. Only the Slack API and inference are replaced.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_native_redelivery_after_restart import _gateway

TEAM = 'T_REPLAY'
BOT = 'U_BOT'
USER = 'U_ORIGINAL'
OTHER = 'U_OTHER'
DM = 'D_REPLAY'
TS = '1793819016.301339'
TEXT = 'Show the comparison.'


async def _slack_gateway(home, monkeypatch, display_name, instance):
    from gateway.config import Platform, PlatformConfig
    from gateway.session_authority import initialize_session_authority
    from plugins.platforms.slack import adapter as slack

    runner, _, sent, calls = await _gateway(home, monkeypatch)
    monkeypatch.setenv('SLACK_ALLOWED_USERS', f'{USER},{OTHER}')
    monkeypatch.setattr(slack, '_AGENT_SESSIONS_SUPPORTED', False)
    adapter = slack.SlackAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False))
    adapter._app = MagicMock()
    adapter._app.client = AsyncMock()
    adapter._app.client.users_info.return_value = {
        'user': {'is_bot': False, 'profile': {'display_name': display_name}, 'real_name': display_name}}
    adapter._app.client.conversations_info.return_value = {'channel': {'name': 'direct'}}
    adapter._app.client.conversations_replies.return_value = {'messages': []}
    adapter._bot_user_id = BOT
    adapter._running = True
    monkeypatch.setattr(adapter, 'send', runner.adapters[Platform.TELEGRAM].send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.SLACK: adapter}
    authority = await initialize_session_authority(runner, profile_id=str(home), instance_id=instance)
    runner._wire_adapter_handlers(adapter)
    return runner, adapter, authority, sent, calls


async def _deliver(adapter, event):
    await adapter._handle_slack_message(event, {'team_id': TEAM})
    if adapter._background_tasks:
        await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 10)
    drains = [live.task for live in adapter.gateway_runner.session_authority.sessions.values() if live.task]
    if drains:
        await asyncio.wait_for(asyncio.gather(*drains), 10)


def _rows(authority):
    from hermes_state_runtime import list_session_admissions
    return [row for sid in authority.sessions
            for row in list_session_admissions(authority.db, session_id=sid, pending_only=False)]


@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['identical', 'display-name', 'content'])
async def test_slack_answered_root_restart_replay_preserves_identity(tmp_path, monkeypatch, change):
    from gateway.session_ingress import ALREADY_ADMITTED_REPLY
    from hermes_state_runtime import ADMITTED_REQUEST_CONFLICT, RuntimeStoreError

    original = {'type': 'message', 'channel': DM, 'channel_type': 'im', 'team': TEAM,
                'user': USER, 'text': TEXT, 'ts': TS, 'event_ts': TS, 'client_msg_id': 'fixture'}
    runner, adapter, authority, sent, calls = await _slack_gateway(tmp_path, monkeypatch, 'Before', 'first')
    try:
        await _deliver(adapter, original)
        assert calls == [TEXT]
        assert [text for text in sent if 'home channel' not in text] == ['the answer']
        before = _rows(authority)
        assert len(before) == 1 and before[0]['status'] == 'terminal'
    finally:
        runner.session_store.close_all_db_handles()
        authority.db.close()

    # Fresh runner, adapter guards, name cache and owner epoch; the same durable home.
    runner, adapter, authority, sent, calls = await _slack_gateway(
        tmp_path, monkeypatch, 'After' if change != 'identical' else 'Before', 'second')
    outcomes = []
    admit = authority.admit_native

    async def record_admission(event):
        try:
            receipt = await admit(event)
        except RuntimeStoreError as exc:
            outcomes.append((exc.reason, exc.field))
            raise
        outcomes.append(receipt.admission_id)
        return receipt

    monkeypatch.setattr(authority, 'admit_native', record_admission)
    message = {**original, 'thread_ts': TS, 'reply_count': 3, 'latest_reply': '1793821177.880909'}
    if change == 'content':
        message.update(text='A different request.', edited={'user': USER, 'ts': '1793821400.000000'})
    replay = {'type': 'message', 'subtype': 'message_changed', 'hidden': True,
              'channel': DM, 'channel_type': 'im', 'team': TEAM,
              'ts': '1793821400.000200', 'event_ts': '1793821400.000200', 'message': message}
    try:
        await _deliver(adapter, replay)
        after = _rows(authority)
        assert calls == [], 'neither the answered request nor a changed copy may execute again'
        assert after == before, 'replay must not alter or replace the committed admission'
        if change in ('identical', 'display-name'):
            assert sent == [], 'metadata-only replay must not emit an admission-conflict reply'
            assert outcomes == [before[0]['admission_id']], 'must reconcile, not merely suppress the error'
        else:
            assert outcomes == [('admission_conflict', ADMITTED_REQUEST_CONFLICT['field'])]
            assert sent == [ALREADY_ADMITTED_REPLY], 'meaningful conflicts remain visible'
    finally:
        runner.session_store.close_all_db_handles()
        authority.db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('sender_location', ['source', 'event'])
async def test_slack_name_reconcile_does_not_erase_sender_ids(tmp_path, monkeypatch, sender_location):
    """Within an admitted identity, neither copy of the sender id is display metadata.

    A different source user normally derives a different principal (and is not a
    retry). Pin the existing principal here to exercise the payload digest guard.
    """
    from copy import deepcopy
    from hermes_state_runtime import ADMITTED_REQUEST_CONFLICT, RuntimeStoreError

    runner, adapter, authority, _sent, calls = await _slack_gateway(tmp_path, monkeypatch, 'Before', 'first')
    try:
        await _deliver(adapter, {'type': 'message', 'channel': DM, 'channel_type': 'im', 'team': TEAM,
                                 'user': USER, 'text': TEXT, 'ts': TS, 'event_ts': TS})
        before = _rows(authority)
        assert len(before) == 1 and before[0]['status'] == 'terminal'
        row = before[0]
        payload = deepcopy(row['payload'])
        payload['native_text_v1']['source']['user_name'] = 'After'
        payload['native_text_v1'][sender_location]['user_id'] = OTHER
        with pytest.raises(RuntimeStoreError) as error:
            await asyncio.to_thread(authority._admit_native_write,
                                    principal_id=row['principal_id'], session_id=row['target_session_id'],
                                    request_id=TS, payload=payload, authorize=None)
        assert (error.value.reason, error.value.field) == ('admission_conflict', ADMITTED_REQUEST_CONFLICT['field'])
        assert _rows(authority) == before
        assert calls == [TEXT]
    finally:
        runner.session_store.close_all_db_handles()
        authority.db.close()
