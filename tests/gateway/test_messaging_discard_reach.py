"""Messaging ``/discard`` reaches every lost turn it promises to (review #141357, GW-10).

Slack has no native ``/discard`` (its slash cap), so the pause notice must name ``/hermes discard``;
a conversation brought back by ``/resume`` after a restart is not in the owner's live map, and
``/discard`` must still find its lost turn. Real runner dispatch and SQLite; transport stubbed."""
import asyncio

import pytest


def test_slack_pause_notice_names_the_registered_umbrella_command():
    from types import SimpleNamespace
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_contract import SessionRef
    from gateway.session_ingress import pause_notice
    from hermes_cli.commands_platforms import slack_native_slashes, slack_subcommand_map
    assert 'discard' not in [row[0].lstrip('/') for row in slack_native_slashes()]
    assert slack_subcommand_map().get('discard')
    for platform, expected in ((Platform.SLACK, '/hermes discard'), (Platform.TELEGRAM, 'Send /discard')):
        live = SimpleNamespace(source=SessionSource(platform=platform, chat_id='D42'), pause_notified=False)
        notice = pause_notice(SimpleNamespace(sessions={'s': live}), SessionRef('p', 's'), 'unknown_execution')
        assert expected in notice, notice


@pytest.mark.asyncio
@pytest.mark.parametrize('suspended', [False, True], ids=['cold', 'cold-and-stuck-loop-suspended'])
async def test_discard_after_resume_into_a_cold_conversation_resolves_its_lost_turn(tmp_path, monkeypatch, suspended):
    """``suspended``: startup's stuck-loop guard suspends a route active across repeated restarts, which
    the get-or-create lookup rotates to a NEW session; /discard must act on the peeked one instead."""
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.run_runtime import recover_profile_native_sessions
    from gateway.session_authority import initialize_session_authority
    from gateway.session_ingress_context import native_callback, register_transport_home
    from hermes_constants import get_hermes_home
    from hermes_state_runtime import claim_session_input, get_session_admission
    from tests.gateway.test_command_bypass_active_session import _StubAdapter

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('TELEGRAM_ALLOWED_USERS', '42')
    monkeypatch.setattr('gateway.run._hermes_home', tmp_path)
    monkeypatch.setattr('hermes_state.DEFAULT_DB_PATH', tmp_path / 'state.db')  # conftest repoints it per test
    runner = GatewayRunner(GatewayConfig(multiplex_profiles=False, sessions_dir=tmp_path / 'sessions'))
    adapter = _StubAdapter(PlatformConfig(enabled=True, token='fixture', typing_indicator=False), Platform.TELEGRAM)
    sent = []

    async def send(chat_id, content, **kwargs):
        sent.append(content)
        return SendResult(success=True, message_id=str(len(sent)))
    monkeypatch.setattr(adapter, 'send', send)
    adapter.gateway_runner = runner
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._wire_adapter_handlers(adapter)
    register_transport_home(runner, None, tmp_path)
    source = adapter.build_source(chat_id='42', chat_type='dm', user_id='42')
    first = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='old')
    first._schedule = lambda ref: None
    previous = first.register(source).session_id
    first.db.append_message(previous, 'user', 'older conversation')
    runner.session_store.get_or_create_session(source, force_new=True)
    head = MessageEvent(text='lost human input', source=source, message_id='head')
    with native_callback(runner, head, get_hermes_home()):
        receipt = await first.admit_native(head)
    sid = receipt.ref.session_id
    claimed = claim_session_input(first.db, epoch=first.epoch, session_id=sid)
    key = runner._session_key_for_source(source)
    try:
        # A restart fences the turn unknown; /resume moves away, a second restart, /resume back:
        # the historical conversation is the chat's route again but not in the owner's live map.
        recovered = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='mid')
        recovered._schedule = lambda ref: None
        await recover_profile_native_sessions(runner, recovered)
        await runner._handle_message(MessageEvent(text='/resume ' + previous, source=source, message_id='away'))
        second = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id='third')
        second._schedule = lambda ref: None
        await recover_profile_native_sessions(runner, second)
        await runner._handle_message(MessageEvent(text='/resume ' + sid, source=source, message_id='back'))
        assert runner.session_store.peek_session_id(key) == sid and sid not in second.sessions
        assert get_session_admission(second.db, admission_id=claimed['admission_id'])['status'] == 'unknown'
        if suspended:
            runner.session_store._entries[key].suspended = True  # as _suspend_stuck_loop_sessions does
        await adapter.handle_message(MessageEvent(text='/discard', source=source, message_id='discard'))
        if adapter._background_tasks:
            await asyncio.wait_for(asyncio.gather(*list(adapter._background_tasks)), 10)
        row = get_session_admission(second.db, admission_id=claimed['admission_id'])
        assert (row['status'], row['outcome']) == ('terminal', 'interrupted'), (row, sent)
        assert runner.session_store.peek_session_id(key) == sid  # /discard never rotates the chat
    finally:
        runner.session_store.close_all_db_handles()
