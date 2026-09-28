import pytest

from agent.outbound_context import pending, acknowledge
from gateway.mirror import mirror_to_session
from hermes_state import SessionDB
from tools.session_search_tool import _session_link, session_search


def test_delivery_waits_for_next_turn_without_changing_transcript(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME',str(tmp_path))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM','telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID','123')
    # Delivery can precede the destination's first conversation.
    assert mirror_to_session('telegram','123','The report is ready', source_label='cron')
    ids, note = pending('new-session')
    assert 'The report is ready' in note
    assert 'Context only' in note
    acknowledge(ids, 'new-session')
    assert pending('new-session') == ([], '')
    monkeypatch.setenv('HERMES_HOME',str(tmp_path/'other'))
    assert pending('new-session') == ([], '')


def test_compact_reference_resolves_native_session_and_rejects_collision(tmp_path):
    db = SessionDB(db_path=tmp_path/'state.db')
    try:
        sid = '2026-09-27-abcd-123456789abc'
        db.create_session(session_id=sid, source='cli', model='test')
        db.append_message(sid, role='user',content='The release plan')
        reference = _session_link(sid)
        assert sid not in reference
        result = session_search(session_id=reference,db=db)
        assert 'The release plan' in result
        db.create_session(session_id='other-'+sid[-12:], source='cli', model='test')
        assert 'ambiguous' in session_search(session_id=reference,db=db)
    finally:
        db.close()


def test_child_and_review_cannot_consume_parent_delivery(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from tools.skill_provenance import set_current_write_origin, reset_current_write_origin
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', '123')
    assert mirror_to_session('telegram', '123', 'parent only', source_label='cron')
    assert pending('child', agent=SimpleNamespace(_delegate_depth=1)) == ([], '')
    token = set_current_write_origin('background_review')
    try:
        assert pending('parent') == ([], '')
    finally:
        reset_current_write_origin(token)
    ids, note = pending('parent', agent=SimpleNamespace(_delegate_depth=0))
    assert 'parent only' in note
    acknowledge(ids, 'parent')
    assert pending('parent') == ([], '')


@pytest.mark.parametrize('moa_active', [False, True])
def test_failed_transcript_flush_keeps_delivery_until_committed(tmp_path, monkeypatch, moa_active):
    from unittest.mock import patch
    from tests.agent.test_api_content_row_addressed_backfill import _RealPersistenceAgent
    from tests.agent.test_api_content_sidecar import _build
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', '123')
    db = SessionDB(db_path=tmp_path / 'state.db')
    db.create_session('parent', source='cli')
    try:
        agent = _RealPersistenceAgent(db, 'parent')
        assert mirror_to_session('telegram', '123', 'keep until durable', source_label='cron')
        with patch('agent.session_persistence._db_flush_write', side_effect=OSError('disk full')):
            _build(agent, moa_active=moa_active)
        assert pending('parent')[0]
        ctx = _build(agent, moa_active=moa_active)
        assert 'keep until durable' in ctx.messages[ctx.current_turn_user_idx]['api_content']
        assert pending('parent') == ([], '')
        assert 'keep until durable' in db.get_messages('parent')[-1]['api_content']
    finally:
        db.close()


def test_isolated_destination_waits_for_its_participant(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('group_sessions_per_user: true\nthread_sessions_per_user: true\n')
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', '-123')
    monkeypatch.setenv('HERMES_SESSION_THREAD_ID', '42')
    assert mirror_to_session('telegram', '-123', 'For Alex', thread_id='42', user_id='alex')
    monkeypatch.setenv('HERMES_SESSION_USER_ID', 'sam')
    assert pending('sam-session') == ([], '')
    monkeypatch.setenv('HERMES_SESSION_USER_ID', 'alex')
    ids, note = pending('alex-session')
    assert 'For Alex' in note
    acknowledge(ids, 'alex-session')
    assert pending('alex-session') == ([], '')


def test_dm_delivery_does_not_inherit_origin_participant(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('group_sessions_per_user: true\nthread_sessions_per_user: true\n')
    assert mirror_to_session('telegram', 'bob-dm', 'From Alice', user_id='alice')
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', 'bob-dm')
    monkeypatch.setenv('HERMES_SESSION_CHAT_TYPE', 'dm')
    monkeypatch.setenv('HERMES_SESSION_USER_ID', 'bob')
    ids, note = pending('bob-session')
    assert 'From Alice' in note
    acknowledge(ids, 'bob-session')
    assert pending('bob-session') == ([], '')


def test_shared_group_report_reaches_each_isolated_conversation_once(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text('group_sessions_per_user: true\n')
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', '-123')
    monkeypatch.setenv('HERMES_SESSION_CHAT_TYPE', 'group')
    assert mirror_to_session('telegram', '-123', 'Shared report', source_label='webhook')
    for user in ('alex', 'sam'):
        monkeypatch.setenv('HERMES_SESSION_USER_ID', user)
        ids, note = pending(user + '-session')
        assert 'Shared report' in note
        acknowledge(ids, user + '-session')
        assert pending(user + '-session') == ([], '')
        assert pending(user + '-new-session') == ([], '')
    monkeypatch.setenv('HERMES_SESSION_USER_ID', 'alex')
    assert pending('alex-session') == ([], '')


def test_delivery_backlog_has_bounded_retention_and_incremental_consumption(tmp_path, monkeypatch):
    import json
    from agent.outbound_context import enqueue, connection, MAX_CONTEXT_CHARS, MAX_QUEUED_PER_DESTINATION
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for index in range(MAX_QUEUED_PER_DESTINATION + 5):
        enqueue('session', f'{index}:' + '<' * 10000, 'cron')
    with connection() as db:
        assert db.execute('SELECT count(*) FROM deliveries').fetchone()[0] == MAX_QUEUED_PER_DESTINATION
        oldest, content = db.execute('SELECT id,content FROM deliveries ORDER BY rowid LIMIT 1').fetchone()
        assert json.loads(content)['message'].startswith('5:')
        db.execute("UPDATE deliveries SET created=datetime('now','-8 days') WHERE id=?", (oldest,))
        db.execute('INSERT INTO consumed VALUES (?,?)', (oldest, 'other'))
    ids, note = pending('session')
    assert len(note) <= MAX_CONTEXT_CHARS and 'truncated' in note
    assert len(ids) < MAX_QUEUED_PER_DESTINATION and oldest not in ids
    assert 'More queued' in note
    acknowledge(ids, 'session')
    next_ids, next_note = pending('session')
    assert next_ids and not set(ids).intersection(next_ids)
    assert len(next_note) <= MAX_CONTEXT_CHARS
    with connection() as db:
        assert not db.execute('SELECT 1 FROM consumed WHERE delivery=?', (oldest,)).fetchone()


def test_telegram_general_delivery_matches_native_inbound_topic(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from plugins.platforms.telegram.adapter import TelegramAdapter
    from agent.outbound_context import destination_key
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_SESSION_PLATFORM', 'telegram')
    monkeypatch.setenv('HERMES_SESSION_CHAT_ID', '-100123')
    message = SimpleNamespace(chat=SimpleNamespace(type='supergroup', is_forum=True),
                              message_thread_id=None, is_topic_message=False)
    topic = TelegramAdapter._effective_message_thread_id(message)
    monkeypatch.setenv('HERMES_SESSION_THREAD_ID', topic)
    assert mirror_to_session('telegram', '-100123', 'General report')
    ids, note = pending('forum-session')
    assert ids and 'General report' in note
    monkeypatch.setenv('HERMES_SESSION_THREAD_ID', '42')
    assert pending('other-topic') == ([], '')
    assert destination_key('telegram', '123', '1') != destination_key('telegram', '123')
