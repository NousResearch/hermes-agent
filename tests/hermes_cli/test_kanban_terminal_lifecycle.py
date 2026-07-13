from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _delivery_gates():
    return {
        "delivery": {
            "review": {"verdict": "PASS", "head": "final-head"},
            "merge": {"commit": "merge-commit"},
            "production": {"deployed": True, "evidence": "deployment-url"},
            "migration": {"required": False},
            "e2e": {"passed": True, "evidence": "production-smoke"},
        }
    }


def test_cancelled_task_requires_explicit_reopen_before_new_run(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="cancel me", assignee="developer")
        first = kb.claim_task(conn, task_id, claimer="first")
        assert first is not None
        assert kb.cancel_task(conn, task_id, reason="user stopped", expected_run_id=first.current_run_id)

        assert kb.get_task(conn, task_id).status == "cancelled"
        assert kb.claim_task(conn, task_id, claimer="forbidden") is None

        assert kb.reopen_task(conn, task_id, actor="mika", reason="resume explicitly")
        second = kb.claim_task(conn, task_id, claimer="second")

        assert second is not None
        assert second.current_run_id != first.current_run_id
        assert [run.outcome for run in kb.list_runs(conn, task_id)] == ["cancelled", None]


def test_done_task_cannot_run_again_without_explicit_reopen(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="done", assignee="developer")
        assert kb.complete_task(conn, task_id, summary="finished")

        assert kb.claim_task(conn, task_id, claimer="forbidden") is None
        assert kb.reopen_task(conn, task_id, actor="mika", reason="new acceptance")
        assert kb.claim_task(conn, task_id, claimer="allowed") is not None


def test_child_reads_only_current_handoff_after_reopen(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="developer")
        child = kb.create_task(conn, title="child", assignee="developer", parents=[parent])
        assert kb.complete_task(conn, parent, summary="old handoff")
        assert kb.reopen_task(conn, parent, actor="mika", reason="correct old result")
        assert kb.complete_task(conn, parent, summary="current handoff")

        context = kb.build_worker_context(conn, child)

    assert "current handoff" in context
    assert "old handoff" not in context
    assert "handoff v2" in context


def test_coding_done_waits_for_delivery_gates_and_notification_ack(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="ship code",
            assignee="developer",
            workspace_kind="worktree",
        )
        claimed = kb.claim_task(conn, task_id, claimer="coder")
        assert claimed is not None

        with pytest.raises(kb.CompletionGateError, match="production, e2e"):
            kb.complete_task(
                conn,
                task_id,
                summary="not shipped",
                metadata={
                    "delivery": {
                        "review": {"verdict": "PASS", "head": "final-head"},
                        "merge": {"commit": "merge-commit"},
                        "migration": {"required": False},
                    }
                },
                expected_run_id=claimed.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"

        assert kb.complete_task(
            conn,
            task_id,
            summary="shipped",
            metadata=_delivery_gates(),
            expected_run_id=claimed.current_run_id,
        )
        pending = kb.get_task(conn, task_id)
        assert pending.status == "review"
        assert pending.pending_completion_event_id is not None

        assert not kb.ack_completion_notification(
            conn,
            task_id,
            event_id=pending.pending_completion_event_id + 1,
            platform="slack",
            chat_id="C0BGA1TAYRY",
            message_id="wrong",
        )
        assert kb.get_task(conn, task_id).status == "review"

        assert kb.ack_completion_notification(
            conn,
            task_id,
            event_id=pending.pending_completion_event_id,
            platform="slack",
            chat_id="C0BGA1TAYRY",
            message_id="123.456",
        )
        done = kb.get_task(conn, task_id)
        assert done.status == "done"
        assert done.pending_completion_event_id is None
        assert not kb.ack_completion_notification(
            conn,
            task_id,
            event_id=pending.pending_completion_event_id,
            platform="slack",
            chat_id="C0BGA1TAYRY",
            message_id="duplicate",
        )
        assert [event.kind for event in kb.list_events(conn, task_id)].count("completion_acknowledged") == 1
