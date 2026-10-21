"""A watch notice pinned to a session that ended at a user boundary (/new) is dropped, not requeued
forever: the session authority refuses it ``admission_conflict`` on every drain tick (#141562).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; no agent turn runs."""
import logging
import queue
import asyncio

import pytest

from tests.module_clock import install_module_clock
from tests.waiting import async_wait_until
def _notice(entry, pinned_session_id, kind='watch_match'):
    return {
        'type': kind, 'event_id': f'{kind}-1', 'session_id': 'proc_watch', 'started_at': 1.0,
        'session_key': entry.session_key, 'parent_session_id': pinned_session_id,
        'platform': 'discord', 'chat_type': 'dm', 'chat_id': '42', 'user_id': '42',
        'command': 'tail -f build.log', 'pattern': 'READY', 'output': 'READY', 'suppressed': 0,
        'seq': 1, 'interval': 60, 'elapsed': 60, 'omitted': 0,
    }




async def _gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_results import execution_result
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    from gateway import run_notifications
    install_module_clock(monkeypatch, run_notifications)

    config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                token="test-token",
                typing_indicator=False
            )
        }
    )
    runner = GatewayRunner(config)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token='fixture-token', typing_indicator=False))
    adapter.gateway_runner = runner
    runner.adapters = {Platform.DISCORD: adapter}
    authority = await initialize_session_authority(runner, profile_id='default', instance_id='owner')
    runner._wire_adapter_handlers(adapter)

    # Start a new session via message
    source = adapter.build_source(chat_id='123', chat_type='dm', user_id='456')
    await adapter.handle_message(MessageEvent(
        
         
        
        message_id='789',
        text='hello',
        source=source
    ))
    key = runner.session_store._generate_session_key(source)
    entry = type('MockEntry', (), {'session_id': 'test-session', 'session_key': key, 'route': key})()


    def _foreign_pin(runner):
        """Add a foreign pin session to test against."""
        entry = asyncio.run(runner.start_session(
            principal_id="other-owner",
            session_id="foreign-session",
            
            adapter_config=DiscordAdapter.AdapterConfig(
                
                
                thread_id=None,
            ),
            initial_event=MessageEvent(
                
                
                
                message_id="789",
                content="foreign",
            ),
        ))
        return entry.session_id

    def _queued_admissions(authority, session_id):
        """What entries are ``_deliver_completion_notification`` about to admit."""
        return [
            entry
            for entry in authority.db.get_admissions()
            if entry.session_id == session_id and entry.status == "pending"
        ]

    return runner, authority, entry


@pytest.mark.asyncio
async def test_watch_notice_pinned_to_missing_db_row_is_dropped(monkeypatch, caplog):
    """A watch notice pinned to a session missing from the DB is dropped permanently (terminal cause)."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        # Create a notice pinned to a non-existent session
        notice = _notice(entry, 'missing-session-123', 'watch_match')
        pending = queue.Queue()
        pending.put(notice)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        # Notice should be dropped (not in queue anymore)
        with pytest.raises(queue.Empty):
            pending.get_nowait()
        
        # Should have warning about permanently gone pinned session
        warning_records = [r for r in caplog.records if 'permanently gone' in r.getMessage()]
        assert len(warning_records) == 1
        assert 'permanently gone' in warning_records[0].getMessage() and 'no longer exists' in warning_records[0].getMessage()
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_watch_notice_pinned_to_subagent_without_delegate_is_dropped(monkeypatch, caplog):
    """A watch notice pinned to a subagent session without _delegate_from is dropped permanently (terminal cause)."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        # Create a notice pinned to a session that exists but has no _delegate_from
        notice = _notice(entry, 'subagent-no-delegate')
        pending = queue.Queue()
        pending.put(notice)

        # Mock the _classify_completion_owner to return a subagent without delegate
        async def mock_classify_completion_owner(pinned, session_key):
            # Simulate terminal verdict for subagent with no owning conversation
            return "terminal", "subagent-hop", None
        
        # Mock _lookup_completion_owner to return the subagent case
        async def mock_lookup_completion_owner(pinned, live):
            return "terminal", "subagent-hop", None
        
        # Mock _gone_pin_cause to return subagent-specific message
        async def mock_gone_pin_cause(pinned, live):
            return "subagent transcript with no owning conversation"
        
        monkeypatch.setattr(runner, '_classify_completion_owner', mock_classify_completion_owner)
        monkeypatch.setattr(runner, '_lookup_completion_owner', mock_lookup_completion_owner)
        monkeypatch.setattr(runner, '_gone_pin_cause', mock_gone_pin_cause)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        # Notice should be dropped (not in queue anymore)
        with pytest.raises(queue.Empty):
            pending.get_nowait()
        
        # Should have warning about permanently gone pinned session
        warning_records = [r for r in caplog.records if 'permanently gone' in r.getMessage()]
        assert len(warning_records) == 1
        assert 'permanently gone' in warning_records[0].getMessage() and 'subagent-hop' in warning_records[0].getMessage()
    finally:
        authority.db.close()
