"""Typed watcher terminal origin with an actual registry process, not adapter.send provenance."""
import json

import pytest

from tests.gateway.test_webhook_finalization_recovery import settled_before_finalizer
from tests.gateway.test_webhook_prepared_origins import records
from tests.waiting import async_wait_until


@pytest.mark.asyncio
@pytest.mark.platforms('posix')
@pytest.mark.parametrize('mode,exit_code,muted,expected', [
    ('result', 0, False, 'pending'), ('result', 4, False, 'pending'),
    ('result', 4, True, 'suppressed'), ('error', 0, False, 'suppressed'), ('off', 0, False, 'suppressed')])
async def test_typed_process_terminal_owns_origin_and_notification_policy(monkeypatch, mode, exit_code, muted, expected):
    from tools.process_registry import process_registry
    from gateway import warning_notifications
    async with settled_before_finalizer(deliver='bot-chat') as (_, _, _, runner, authority, original):
        native = json.loads(original['payload_json'])['native_text_v1']
        session = process_registry.spawn_local(f'printf RAW_PROCESS_FINAL; exit {exit_code}',
            cwd=str(authority.db.db_path.parent), session_key=native['route'])
        try:
            await async_wait_until(lambda: session.exited, timeout=10, what='actual process exited')
            monkeypatch.setattr(runner, '_load_background_notifications_mode', lambda: mode)
            monkeypatch.setattr(warning_notifications, 'warning_notifications_enabled', lambda *a, **kw: not muted)
            watcher = dict(session_id=session.id, check_interval=0.01, platform='webhook',
                session_key=native['route'], chat_id=native['source']['chat_id'],
                parent_session_id=original['target_session_id'], notify_on_complete=False)
            await runner._watch_process(watcher)
            saved = [r for r in records(authority) if r['origin_kind'] == 'process_final']
            assert len(saved) == 1, 'typed process final must have a real durable origin'
            value = saved[0]
            assert value['process_id'] == session.id and value['process_started_at'] == session.started_at
            assert value['state'] == expected
            assert value['category'] == ('diagnostic' if exit_code else 'result')
            assert value['source_outcome'] == ('failed' if exit_code else 'completed')
            await runner._watch_process(watcher)
            assert len([r for r in records(authority) if r['origin_kind'] == 'process_final']) == 1
            assert len(authority.db._read_all('SELECT * FROM session_admissions')) == 1
        finally:
            if session.process is not None:
                session.process.wait(timeout=10)
