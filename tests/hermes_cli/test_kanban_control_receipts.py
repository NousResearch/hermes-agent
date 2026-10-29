"""Native event provenance for response-loss reconciliation, not another authority store."""
import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path / "kanban"))
    from hermes_cli import kanban_db_connect as kc
    kc.init_db()
    with kc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Guarded request", initial_status="blocked")
        yield conn, task_id


def test_guarded_resume_retains_request_and_exact_predecessor(board):
    conn, task_id = board
    token = kb.list_events(conn, task_id)[-1].id
    assert kb.unblock_task(conn, task_id, expected_event_id=token, request_id="integration-request")
    receipt = kb.list_events(conn, task_id)[-1]
    assert receipt.kind == "unblocked"
    assert receipt.payload == {"request_id": "integration-request", "expected_event_id": token}
    assert not kb.unblock_task(conn, task_id, expected_event_id=token, request_id="integration-request")
    assert kb.list_events(conn, task_id)[-1].id == receipt.id


def test_other_operator_resume_is_not_an_integration_receipt(board):
    conn, task_id = board
    token = kb.list_events(conn, task_id)[-1].id
    assert kb.unblock_task(conn, task_id)
    assert not kb.unblock_task(conn, task_id, expected_event_id=token, request_id="integration-request")
    assert kb.list_events(conn, task_id)[-1].payload is None


def test_later_stop_remains_newer_than_original_request_receipt(board):
    conn, task_id = board
    token = kb.list_events(conn, task_id)[-1].id
    assert kb.unblock_task(conn, task_id, expected_event_id=token, request_id="integration-request")
    assert kb.block_task(conn, task_id, reason="operator Stop", kind="needs_input")
    stop = kb.list_events(conn, task_id)[-1]
    assert not kb.unblock_task(conn, task_id, expected_event_id=token, request_id="integration-request")
    assert kb.get_task(conn, task_id).status == "blocked"
    assert kb.list_events(conn, task_id)[-1].id == stop.id


@pytest.mark.parametrize("token,label", [(None, "request"), (2, ""), (2, " "), (2, 12), (2, "x" * 201)])
def test_invalid_request_labels_never_mutate_native_state(board, token, label):
    conn, task_id = board
    before = list(conn.iterdump())
    with pytest.raises(ValueError):
        kb.unblock_task(conn, task_id, expected_event_id=token, request_id=label)
    assert list(conn.iterdump()) == before
