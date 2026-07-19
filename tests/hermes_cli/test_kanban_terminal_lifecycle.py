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
    head = "a" * 40
    merge = "b" * 40
    return {
        "delivery": {
            "final_head": head,
            "review": {"verdict": "PASS", "head": head, "reviewer": "reviewer", "evidence": "https://example.test/review"},
            "merge": {"head": head, "commit": merge, "evidence": "https://example.test/merge"},
            "production": {"deployed": True, "commit": merge, "evidence": "https://example.test/deploy"},
            "migration": {"required": False},
            "e2e": {"passed": True, "commit": merge, "evidence": "https://example.test/e2e"},
        }
    }


def _record_delivery_gates(conn, task_id, *, repo="example/repo"):
    head = "a" * 40
    merge = "b" * 40
    kb.record_delivery_attestation(conn, task_id, repo=repo, gate="review", head=head, subject=head, actor="reviewer", evidence="review-1")
    kb.record_delivery_attestation(conn, task_id, repo=repo, gate="merge", head=head, subject=merge, actor="github-app", evidence="merge-1")
    for gate in ("production", "e2e"):
        kb.record_delivery_attestation(conn, task_id, repo=repo, gate=gate, head=merge, subject=f"{gate}-1", actor="delivery-runner", evidence=f"{gate}-1")
    kb.record_delivery_attestation(conn, task_id, repo=repo, gate="migration", head=merge, subject="not-required", actor="delivery-runner", evidence="migration-inventory-1")


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


def test_cancel_with_stale_run_does_not_signal_current_worker(kanban_home, monkeypatch):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="stale cancel", assignee="developer")
        claimed = kb.claim_task(conn, task_id, claimer=kb._claimer_id())
        assert claimed is not None
        assert claimed.current_run_id is not None
        calls = []
        monkeypatch.setattr(
            kb,
            "_terminate_reclaimed_worker",
            lambda *args, **kwargs: calls.append(args) or {"terminated": True},
        )

        assert not kb.cancel_task(
            conn,
            task_id,
            reason="stale request",
            expected_run_id=claimed.current_run_id + 1,
        )
        assert calls == []
        task = kb.get_task(conn, task_id)
        assert task is not None and task.status == "running"


def test_cancel_refuses_to_release_remote_worker_claim(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="remote worker", assignee="developer")
        claimed = kb.claim_task(conn, task_id, claimer="remote-host:123")
        assert claimed is not None
        conn.execute("UPDATE tasks SET worker_pid = 999999 WHERE id = ?", (task_id,))

        assert not kb.cancel_task(
            conn,
            task_id,
            reason="cannot stop remotely",
            expected_run_id=claimed.current_run_id,
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        assert task.status == "running"
        assert task.claim_lock == "remote-host:123"


def test_done_task_cannot_run_again_without_explicit_reopen(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="done", assignee="developer")
        assert kb.complete_task(conn, task_id, summary="finished")

        assert kb.claim_task(conn, task_id, claimer="forbidden") is None
        assert kb.reopen_task(conn, task_id, actor="mika", reason="new acceptance")
        assert kb.claim_task(conn, task_id, claimer="allowed") is not None


def test_reopen_preserves_handoff_and_audits_child_demotion(kanban_home):
    with kb.connect() as conn:
        parent = kb.create_task(conn, title="parent", assignee="developer")
        child = kb.create_task(conn, title="child", assignee="developer", parents=[parent])
        assert kb.complete_task(conn, parent, result="historical", summary="historical")
        assert kb.reopen_task(conn, parent, actor="mika", reason="new work")

        assert kb.get_task(conn, parent).result == "historical"
        assert kb.get_task(conn, child).status == "todo"
        assert any(
            event.kind == "status" and event.payload.get("status") == "todo"
            for event in kb.list_events(conn, child)
        )


def test_reopen_snapshots_legacy_result_before_next_completion(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="legacy", assignee="developer")
        assert kb.complete_task(conn, task_id, result="historical")
        conn.execute("DELETE FROM task_runs WHERE task_id = ?", (task_id,))

        assert kb.reopen_task(conn, task_id, actor="mika", reason="new generation")
        assert kb.complete_task(conn, task_id, result="current")

        assert [run.summary for run in kb.list_runs(conn, task_id)] == [
            "historical",
            "current",
        ]


def test_done_handoff_cannot_be_edited_in_place(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="done", assignee="developer")
        assert kb.complete_task(conn, task_id, result="original", summary="original")
        assert not kb.edit_completed_task_result(conn, task_id, result="rewritten")
        assert kb.get_task(conn, task_id).result == "original"
        assert kb.latest_summary(conn, task_id) == "original"


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
            delivery_required=True,
        )
        claimed = kb.claim_task(conn, task_id, claimer="coder")
        assert claimed is not None

        with pytest.raises(kb.CompletionGateError, match="review, merge, production, migration, e2e"):
            kb.complete_task(
                conn,
                task_id,
                summary="not shipped",
                metadata={"delivery": {"migration": {"required": False}}},
                expected_run_id=claimed.current_run_id,
            )
        assert kb.get_task(conn, task_id).status == "running"

        _record_delivery_gates(conn, task_id)
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
        kb.add_notify_sub(
            conn, task_id=task_id, platform="slack", chat_id="C0BGA1TAYRY",
            notifier_profile="developer",
        )

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
            notifier_profile="developer",
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
            notifier_profile="developer",
        )
        assert [event.kind for event in kb.list_events(conn, task_id)].count("completion_acknowledged") == 1


@pytest.mark.parametrize("workspace_kind", ["scratch", "dir", "worktree"])
def test_developer_tasks_cannot_bypass_delivery_gates_by_workspace(kanban_home, workspace_kind):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="coding", assignee="developer", workspace_kind=workspace_kind, task_kind="coding")
        with pytest.raises(kb.CompletionGateError):
            kb.complete_task(conn, task_id, summary="forged", metadata=_delivery_gates())
        assert kb.get_task(conn, task_id).status == "ready"


def test_general_worktree_is_not_coding_gated(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="research", assignee="researcher", workspace_kind="worktree", task_kind="general")
        assert kb.complete_task(conn, task_id, summary="done")
        assert kb.get_task(conn, task_id).status == "done"


def test_pending_completion_route_is_bound_once(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(conn, title="route", assignee="developer", task_kind="coding")
        _record_delivery_gates(conn, task_id)
        assert kb.complete_task(conn, task_id, summary="ready")
        kb.add_notify_sub(conn, task_id=task_id, platform="slack", chat_id="CORCH", notifier_profile="developer")
        kb.add_notify_sub(conn, task_id=task_id, platform="slack", chat_id="CORCH", notifier_profile="developer")
        with pytest.raises(ValueError, match="already bound"):
            kb.add_notify_sub(conn, task_id=task_id, platform="slack", chat_id="CWRONG", notifier_profile="developer")
        _, _, events = kb.claim_unseen_events_for_sub(conn, task_id=task_id, platform="slack", chat_id="CORCH", kinds=["completed"])
        assert len(events) == 1


def test_cancelled_pending_completion_cannot_be_claimed_for_delivery(kanban_home):
    with kb.connect() as conn:
        task_id = kb.create_task(
            conn,
            title="withdraw completion",
            assignee="developer",
            delivery_required=True,
        )
        _record_delivery_gates(conn, task_id)
        assert kb.complete_task(
            conn,
            task_id,
            summary="must not send",
            metadata=_delivery_gates(),
        )
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="C0BGA1TAYRY",
            notifier_profile="developer",
        )
        assert kb.cancel_task(conn, task_id, reason="withdrawn")

        _, _, events = kb.claim_unseen_events_for_sub(
            conn,
            task_id=task_id,
            platform="slack",
            chat_id="C0BGA1TAYRY",
            kinds=["completed"],
        )

        assert events == []
