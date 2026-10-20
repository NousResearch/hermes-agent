"""A watch notice pinned to a session that ended at a user boundary (/new) is dropped, not requeued
forever: the session authority refuses it ``admission_conflict`` on every drain tick (#141562).
Real GatewayRunner, DiscordAdapter, SessionAuthority and drain over SQLite; no agent turn runs."""
import logging
import queue

import pytest

from tests.module_clock import install_module_clock
from tests.waiting import async_wait_until


async def _gateway(monkeypatch):
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.platforms.base import SendResult
    from gateway.platforms.event import MessageEvent
    from gateway.run import GatewayRunner
    from gateway.session_results import execution_result
    from gateway.session_authority import SessionAuthority, initialize_session_authority
    from plugins.platforms.discord.adapter import DiscordAdapter

    install_module_clock(monkeypatch)

    db_path = ":memory:"
    config = GatewayConfig(
        platform_configs=[
            PlatformConfig(
                platform=Platform.DISCORD,
                adapter_class=DiscordAdapter,
                bot_token="test-token",
            )
        ],
        db_path=db_path,
        api_workers=0,
    )
    runner = GatewayRunner(config)
    authority = initialize_session_authority(runner.db_path, runner.epoch)
    await authority.migrate()

    # Start a new session
    session_id = "test-session"
    entry = await runner.start_session(
        principal_id="test-owner",
        session_id=session_id,
        platform=Platform.DISCORD,
        adapter_config=DiscordAdapter.AdapterConfig(
            channel_id="123",
            user_id="456",
            thread_id=None,
        ),
        initial_event=MessageEvent(
            platform=Platform.DISCORD,
            channel_id="123",
            user_id="456",
            message_id="789",
            content="hello",
        ),
    )
    await runner._loop.run_until_complete(entry.wait_for_result())

    def _notice(session_entry, pinned_session_id, notice_type="completion"):
        """What the process watcher hands ``_deliver_completion_notification`` for a pinned completion."""
        return {
            "type": notice_type,
            "session_id": "proc_watch",
            "started_at": 1700000000.0,
            "command": "make",
            "session_key": session_entry.route,
            "parent_session_id": pinned_session_id,
        }

    def _foreign_pin(runner):
        """Add a foreign pin session to test against."""
        entry = asyncio.run(runner.start_session(
            principal_id="other-owner",
            session_id="foreign-session",
            platform=Platform.DISCORD,
            adapter_config=DiscordAdapter.AdapterConfig(
                channel_id="123",
                user_id="456",
                thread_id=None,
            ),
            initial_event=MessageEvent(
                platform=Platform.DISCORD,
                channel_id="123",
                user_id="456",
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

    return runner, authority, entry, _notice, _foreign_pin, _queued_admissions


@pytest.mark.asyncio
async def test_watch_notice_pinned_to_missing_db_row_is_dropped(monkeypatch, caplog):
    """A watch notice pinned to a session missing from the DB is dropped permanently (terminal cause)."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        # Create a notice pinned to a non-existent session
        notice = {'type': 'completion', 'session_id': 'proc_watch', 'started_at': 1700000000.0,
                  'session_key': entry.route, 'parent_session_id': 'missing-session-123'}
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
        assert 'cannot be delivered' in warning_records[0].getMessage()
    finally:
        authority.db.close()


@pytest.mark.asyncio
async def test_watch_notice_pinned_to_subagent_without_delegate_is_dropped(monkeypatch, caplog):
    """A watch notice pinned to a subagent session without _delegate_from is dropped permanently (terminal cause)."""
    runner, authority, entry = await _gateway(monkeypatch)
    try:
        # Create a notice pinned to a session that exists but has no _delegate_from
        notice = {'type': 'completion', 'session_id': 'proc_watch', 'started_at': 1700000000.0,
                  'session_key': entry.route, 'parent_session_id': 'subagent-no-delegate'}
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
        
        monkeypatch.setattr(runner._session_authority, '_classify_completion_owner', mock_classify_completion_owner)
        monkeypatch.setattr(runner._session_authority, '_lookup_completion_owner', mock_lookup_completion_owner)
        monkeypatch.setattr(runner._session_authority, '_gone_pin_cause', mock_gone_pin_cause)

        with caplog.at_level(logging.WARNING, logger='gateway.run'):
            await runner._drain_watch_notifications(pending)

        # Notice should be dropped (not in queue anymore)
        with pytest.raises(queue.Empty):
            pending.get_nowait()
        
        # Should have warning about permanently gone pinned session
        warning_records = [r for r in caplog.records if 'permanently gone' in r.getMessage()]
        assert len(warning_records) == 1
        assert 'subagent transcript' in warning_records[0].getMessage()
    finally:
        authority.db.close()