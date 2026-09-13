"""An explicit forget removes stored payloads without replacing the transcript."""
import json

import pytest

from hermes_state import SessionDB, SessionCompressionInProgressError
from hermes_state_errors import SessionTurnLeaseLostError


def _rows(db):
    return {row['id']: dict(row) for row in db._conn.execute('SELECT * FROM messages ORDER BY id')}


def test_selected_payloads_and_fts_are_erased_atomically_without_losing_other_turns(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('reader', source='cli', session_key='reader-route')
        untouched = [db.append_message('reader', 'user', 'Keep the meeting at nine'),
                     db.append_message('reader', 'assistant', 'The meeting remains at nine')]
        selected = [db.append_message('reader', 'user', [
            {'type': 'text', 'text': 'Describe forgettoken'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,forgettoken'}}],
            api_content='original question plus forgettoken recall',
            display_metadata={'attachment_caption': 'forgettoken', 'gateway_input_owner': 'accepted-input-1'})]
        selected.append(db.append_message('reader', 'assistant', None, tool_calls=[{
            'id': 'read-1', 'type': 'function',
            'function': {'name': 'read_source', 'arguments': '{"source":"forgettoken"}'}}]))
        selected.append(db.append_message('reader', 'tool', 'forgettoken original result', tool_call_id='read-1'))
        selected.append(db.append_message('reader', 'assistant', 'forgettoken is green',
            reasoning='forgettoken reasoning', reasoning_content='forgettoken chain',
            reasoning_details=[{'text': 'forgettoken detail'}],
            codex_message_items=[{'text': 'forgettoken item'}],
            codex_reasoning_items=[{'text': 'forgettoken summary'}]))
        untouched.extend([db.append_message('reader', 'user', 'Keep the second task open'),
                          db.append_message('reader', 'assistant', 'The second task stays open')])
        # A compaction archive is an owned stored copy even when hidden from ordinary replay.
        archived = db.append_message('reader', 'assistant', 'forgettoken archived answer')
        db._execute_write(lambda conn: conn.execute('UPDATE messages SET active=0,compacted=1 WHERE id=?', (archived,)))
        selected.append(archived)
        before = _rows(db)
        session_before = db.get_session('reader')
        assert db.search_messages('forgettoken', include_inactive=True)
        expected = db.get_message_redaction_snapshot('reader', selected)

        stale = [dict(row) for row in expected]
        stale[-1]['sha256'] = '0' * 64
        with pytest.raises(ValueError, match='preimage changed'):
            db.redact_message_payloads('reader', stale)
        assert _rows(db) == before  # earlier rows cannot be partially erased on a later conflict

        assert db.try_acquire_session_turn_lease('reader', 'other-turn')
        with pytest.raises(SessionTurnLeaseLostError):
            db.redact_message_payloads('reader', expected)
        db.release_session_turn_lease('reader', 'other-turn')
        assert db.try_acquire_compression_lock('reader', 'compressor')
        with pytest.raises(SessionCompressionInProgressError):
            db.redact_message_payloads('reader', expected)
        db.release_compression_lock('reader', 'compressor')

        result = db.redact_message_payloads('reader', expected)
        after = _rows(db)
        assert result['status'] == 'redacted' and result['redacted_ids'] == selected
        assert db.get_transcript_redaction_revision('reader') == 1
        assert [after[row] for row in untouched] == [before[row] for row in untouched]
        assert db.get_session('reader') == session_before
        assert set(after) == set(before)
        for row_id in selected:
            assert 'forgettoken' not in json.dumps(after[row_id])
            for field in ('id', 'session_id', 'role', 'timestamp', 'tool_call_id',
                          'active', 'compacted', 'platform_message_id'):
                assert after[row_id][field] == before[row_id][field]
            # Native display identities hash payloads and are invalidated by the existing
            # content-update trigger. Stable row/routing IDs, not that old hash, are retained.
            assert after[row_id]['display_identity'] is None
        calls = json.loads(after[selected[1]]['tool_calls'])
        assert calls == [{'id': 'read-1', 'type': 'function',
                          'function': {'name': 'read_source', 'arguments': '{}'}}]
        assert db.has_gateway_input_owner('reader', 'accepted-input-1')
        assert not db.search_messages('forgettoken', include_inactive=True)
        # The receipt makes replay idempotent without storing the erased payload.
        assert db.redact_message_payloads('reader', expected)['redacted_ids'] == []
        assert db.get_transcript_redaction_revision('reader') == 1
    finally:
        db.close()


def test_compressed_parent_waits_for_child_writer_and_changes_descendant_revision(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('parent', source='cli')
        target = db.append_message('parent', 'user', 'forgettoken parent')
        db.end_session('parent', 'compression')
        db.create_session('child', source='cli', parent_session_id='parent')
        kept = db.append_message('child', 'user', 'Unrelated continued task')
        db.create_session('fork', source='cli', parent_session_id='parent',
                          model_config={'_branched_from': 'parent'})
        fork_row = db.append_message('fork', 'user', 'Independent fork task')
        assert db.try_acquire_session_turn_lease('fork', 'long-running-unrelated-fork')
        expected = db.get_message_redaction_snapshot('parent', [target])
        assert db.get_transcript_dependents('parent') == ['parent', 'child']
        assert db.try_acquire_session_turn_lease('child', 'active-child')
        with pytest.raises(SessionTurnLeaseLostError):
            db.redact_message_payloads('parent', expected)
        db.release_session_turn_lease('child', 'active-child')
        before = _rows(db)[kept]
        fork_before = _rows(db)[fork_row]
        db.redact_message_payloads('parent', expected)
        assert _rows(db)[kept] == before
        assert _rows(db)[fork_row] == fork_before
        assert db.get_transcript_redaction_revision('fork') == 0
        assert db.get_transcript_redaction_revision('child') == 1
        assert db.get_session('child')['parent_session_id'] == 'parent'
        assert db.get_transcript_dependents('child') == ['parent', 'child']
        child_target = db.append_message('child', 'assistant', 'forgettoken continued answer')
        db.redact_message_payloads('child', db.get_message_redaction_snapshot('child', [child_target]))
        assert db.get_transcript_redaction_revision('parent') == 2
        db.release_session_turn_lease('fork', 'long-running-unrelated-fork')
    finally:
        db.close()


def test_recalled_api_payload_is_erased_without_removing_the_human_question(tmp_path):
    db = SessionDB(tmp_path / 'state.db')
    try:
        db.create_session('reader', source='api_server')
        target = db.append_message('reader', 'user', 'What should we work on next?',
            api_content='What should we work on next? injected forgettoken reference',
            display_metadata={'gateway_input_owner': 'accepted-2', 'ordinary': 'keep this context'})
        before = _rows(db)[target]
        expected = db.get_message_redaction_snapshot('reader', [target], mode='api_content')
        db.redact_message_payloads('reader', expected)
        after = _rows(db)[target]
        assert after['api_content'] is None
        for field in before.keys() - {'api_content', 'display_metadata', 'display_identity', 'display_order'}:
            assert after[field] == before[field]
        assert json.loads(after['display_metadata'])['ordinary'] == 'keep this context'
        assert db.has_gateway_input_owner('reader', 'accepted-2')
        assert 'forgettoken' not in str(db.get_messages_as_conversation('reader'))
        assert db.redact_message_payloads('reader', expected)['redacted_ids'] == []
    finally:
        db.close()
