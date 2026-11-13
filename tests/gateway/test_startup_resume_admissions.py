"""Settled admissions must not strand shutdown-interrupted sessions at boot."""
import asyncio
from contextlib import closing
from datetime import datetime
import sqlite3
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import Platform
from gateway.session import SessionEntry
from gateway.session_authorities import holds_admission
from tests.gateway.restart_test_helpers import make_restart_runner, make_restart_source


@pytest.mark.asyncio
@pytest.mark.parametrize('status', ['terminal', 'queued', 'unknown', 'started'])
async def test_startup_resume_defers_only_to_outstanding_admissions(status):
    runner, adapter = make_restart_runner()
    entry = SessionEntry(
        session_key='agent:main:telegram:dm:c', session_id='physical',
        created_at=datetime.now(), updated_at=datetime.now(),
        origin=make_restart_source(chat_id='c'), platform=Platform.TELEGRAM,
        chat_type='dm', resume_pending=True, resume_reason='shutdown_timeout',
        last_resume_marked_at=datetime.now(),
    )
    runner.session_store._entries = {entry.session_key: entry}
    adapter.handle_message = AsyncMock()
    # Execute the ownership predicate's SQL; mock only the surrounding authority.
    with closing(sqlite3.connect(':memory:')) as conn:
        conn.execute('CREATE TABLE session_admissions (target_session_id, status)')
        conn.execute('INSERT INTO session_admissions VALUES (?, ?)', (entry.session_id, status))
        runner.session_authorities = [SimpleNamespace(
            logical_owner=lambda sid: None,
            db=SimpleNamespace(_read_one=lambda sql, params: conn.execute(sql, params).fetchone()),
        )]
        assert runner._schedule_resume_pending_sessions() == (1 if status == 'terminal' else 0)
        await asyncio.sleep(0)
        if status == 'terminal':
            adapter.handle_message.assert_awaited_once()
            event = adapter.handle_message.call_args.args[0]
            assert event.internal is True
            assert event.text == ''
        else:
            adapter.handle_message.assert_not_called()


@pytest.mark.parametrize('retiring', [True, False])
def test_unreadable_admission_store_fails_closed_only_when_retiring(retiring):
    authority = SimpleNamespace(
        retiring=retiring, logical_owner=lambda sid: None,
        db=SimpleNamespace(_read_one=Mock(side_effect=sqlite3.OperationalError('disk I/O error'))),
    )
    if retiring:
        assert holds_admission(authority, 'physical') is True
    else:
        with pytest.raises(sqlite3.OperationalError, match='disk I/O error'):
            holds_admission(authority, 'physical')
