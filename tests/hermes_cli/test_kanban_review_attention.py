"""A producer badge is handoff-scoped display, never a lifecycle transition."""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli.kanban_db_attention import current_attention, set_review_attention
from hermes_cli.kanban_db_connect import connect_closing


@pytest.fixture
def board(monkeypatch):
    monkeypatch.setenv('HERMES_PROFILE', 'forge')
    kb.init_db()
    with connect_closing() as conn:
        yield conn


def handoff(conn, reason='awaiting-test'):
    tid = kb.create_task(conn, title='Review me', assignee='forge', priority=7)
    run = kb.claim_task(conn, tid)
    assert kb.request_review(conn, tid, expected_run_id=run.current_run_id,
                             metadata={'wait_reason': reason}, summary='Preserved approval')
    event = next(e for e in reversed(kb.list_events(conn, tid)) if e.kind == 'review_requested')
    return tid, event.id


def projection(conn, tid):
    return current_attention(conn.execute('SELECT * FROM tasks WHERE id=?', (tid,)).fetchone(),
        conn.execute('SELECT * FROM task_events WHERE task_id=? ORDER BY id', (tid,)).fetchall(),
        conn.execute('SELECT * FROM task_runs WHERE task_id=? ORDER BY id', (tid,)).fetchall())


def test_set_repeat_clear_preserve_task_and_review_provenance(board):
    tid, eid = handoff(board)
    before = kb.get_task(board, tid), kb.list_runs(board, tid)
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge={'label': 'Acceptance ready', 'tone': 'success'})
    assert projection(board, tid)['badge'] == {'label': 'Acceptance ready', 'tone': 'success'}
    assert projection(board, tid)['handoff_event_id'] == eid
    count = len(kb.list_events(board, tid))
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge={'label': 'Acceptance ready', 'tone': 'success'})
    assert len(kb.list_events(board, tid)) == count
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge=None)
    assert 'badge' not in projection(board, tid)
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge=None)
    assert len(kb.list_events(board, tid)) == count + 1
    assert (kb.get_task(board, tid), kb.list_runs(board, tid)) == before


@pytest.mark.parametrize('case', ['answer', 'stale', 'claim', 'done', 'archive', 'wrong_run'])
def test_setter_rejects_noncurrent_testing_handoff(board, case):
    tid, eid = handoff(board, 'awaiting-answer' if case == 'answer' else 'awaiting-test')
    if case in {'claim', 'done', 'archive'}:
        assert kb.reopen_review_task(board, tid)
        if case == 'claim':
            kb.claim_task(board, tid)
        elif case == 'done':
            kb.complete_task(board, tid, result='Done')
        else:
            kb.archive_task(board, tid)
    elif case == 'wrong_run':
        # Corrupted historical event identity cannot authorize an annotation.
        board.execute('UPDATE task_events SET run_id=NULL WHERE id=?', (eid,))
        board.commit()
    before = kb.get_task(board, tid), kb.list_runs(board, tid), kb.list_events(board, tid)
    assert not set_review_attention(board, tid, expected_handoff_event_id=eid + (case == 'stale'), badge={'label': 'Acceptance ready', 'tone': 'success'})
    assert (kb.get_task(board, tid), kb.list_runs(board, tid), kb.list_events(board, tid)) == before


def test_same_second_reopen_and_new_review_cannot_reuse_marker(board, monkeypatch):
    monkeypatch.setattr(kb.time, 'time', lambda: 1000)
    tid, eid = handoff(board)
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge={'label': 'Acceptance ready', 'tone': 'success'})
    assert kb.reopen_review_task(board, tid)
    assert projection(board, tid) is None
    run = kb.claim_task(board, tid)
    assert kb.request_review(board, tid, expected_run_id=run.current_run_id, metadata={'wait_reason': 'awaiting-test'})
    assert 'badge' not in projection(board, tid)
    assert not set_review_attention(board, tid, expected_handoff_event_id=eid, badge={'label': 'Acceptance ready', 'tone': 'success'})


@pytest.mark.parametrize('payload', ['{broken', '[]', '{"badge":"future"}', '{"badge":{},"handoff_event_id":false}'])
def test_bad_latest_marker_fails_closed_without_reviving_old_set(board, payload):
    tid, eid = handoff(board)
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge={'label': 'Acceptance ready', 'tone': 'success'})
    with kb.write_txn(board):
        kb._append_event(board, tid, 'review_attention', {}, run_id=kb.latest_run(board, tid).id)
        board.execute('UPDATE task_events SET payload=? WHERE id=(SELECT MAX(id) FROM task_events)', (payload,))
    attention = projection(board, tid)
    assert attention['kind'] == 'awaiting-test' and 'badge' not in attention


@pytest.mark.parametrize('badge', [False, 'label', {}, {'label': ' ', 'tone': 'info'},
    {'label': 'x' * 81, 'tone': 'info'}, {'label': 'Ready', 'tone': 'arbitrary-css'},
    {'label': 'Ready', 'tone': []}])
def test_invalid_badge_does_not_mutate_current_handoff(board, badge):
    tid, eid = handoff(board)
    before = kb.list_events(board, tid)
    assert not set_review_attention(board, tid, expected_handoff_event_id=eid, badge=badge)
    assert kb.list_events(board, tid) == before


@pytest.mark.parametrize('tone', ['success', 'info', 'warning', 'danger'])
def test_producer_label_and_tone_survive_projection(board, tone):
    tid, eid = handoff(board)
    badge = {'label': '<b>Release acceptance</b>', 'tone': tone}
    assert set_review_attention(board, tid, expected_handoff_event_id=eid, badge=badge)
    assert projection(board, tid)['badge'] == badge


def test_legacy_project_marker_is_not_reinterpreted(board):
    tid, eid = handoff(board)
    with kb.write_txn(board):
        kb._append_event(board, tid, 'review_attention', {'handoff_event_id': eid, 'final_pr': True},
                         run_id=kb.latest_run(board, tid).id)
    assert 'badge' not in projection(board, tid)
