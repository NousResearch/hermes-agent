from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_discord_projection as kdp


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def test_issue_without_parent_thread_is_not_dispatchable(kanban_home):
    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="needs parent thread")
        kdp.mark_projection_required(conn, issue_id)

        assert kdp.is_dispatchable(conn, issue_id) is False
        assert kb.claim_task(conn, issue_id, claimer="worker") is None
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (issue_id,))
        conn.commit()
        assert kb.claim_review_task(conn, issue_id, claimer="reviewer") is None


def test_unmarked_ready_issue_without_parent_thread_cannot_be_claimed(kanban_home):
    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="unmarked ready needs parent thread")

        assert kdp.is_dispatchable(conn, issue_id) is False
        claimed = kb.claim_task(conn, issue_id, claimer="worker")
        task = kb.get_task(conn, issue_id)
        assert claimed is None
        assert task is not None
        assert task.status == "ready"


def test_unmarked_review_issue_without_parent_thread_cannot_be_claimed(kanban_home):
    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="unmarked review needs parent thread")
        conn.execute("UPDATE tasks SET status = 'review' WHERE id = ?", (issue_id,))
        conn.commit()

        assert kdp.is_dispatchable(conn, issue_id) is False
        claimed = kb.claim_review_task(conn, issue_id, claimer="reviewer")
        task = kb.get_task(conn, issue_id)
        assert claimed is None
        assert task is not None
        assert task.status == "review"


def test_link_parent_thread_is_idempotent_and_rejects_duplicates(kanban_home):
    with kb.connect() as conn:
        first = kb.create_task(conn, title="first")
        second = kb.create_task(conn, title="second")

        linked = kdp.link_parent_thread(
            conn,
            issue_id=first,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key="link:first",
        )
        again = kdp.link_parent_thread(
            conn,
            issue_id=first,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key="link:first",
        )

        assert linked.created is True
        assert again.created is False
        assert kdp.get_parent_thread(conn, first).thread_id == "thread-1"

        with pytest.raises(kdp.ParentThreadConflict):
            kdp.link_parent_thread(
                conn,
                issue_id=first,
                guild_id="guild-1",
                channel_id="channel-1",
                thread_id="thread-2",
                idempotency_key="link:first:other",
            )

        with pytest.raises(kdp.ParentThreadConflict):
            kdp.link_parent_thread(
                conn,
                issue_id=second,
                guild_id="guild-1",
                channel_id="channel-1",
                thread_id="thread-1",
                idempotency_key="link:second",
            )


@pytest.mark.parametrize("terminal_status", ["closed", "archived"])
def test_parent_thread_reuse_after_terminal_status_is_rejected(kanban_home, terminal_status):
    with kb.connect() as conn:
        first = kb.create_task(conn, title=f"first {terminal_status}")
        second = kb.create_task(conn, title=f"second {terminal_status}")
        kdp.link_parent_thread(
            conn,
            issue_id=first,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key=f"link:first:{terminal_status}",
        )
        conn.execute("UPDATE tasks SET status = ? WHERE id = ?", (terminal_status, first))
        conn.commit()

        with pytest.raises(kdp.ParentThreadConflict):
            kdp.link_parent_thread(
                conn,
                issue_id=second,
                guild_id="guild-1",
                channel_id="channel-1",
                thread_id="thread-1",
                idempotency_key=f"link:second:{terminal_status}",
            )


def test_status_change_outbox_projector_is_idempotent(kanban_home):
    sent: list[tuple[str, str, str]] = []

    def send(thread_id: str, content: str, idempotency_key: str) -> str:
        sent.append((thread_id, content, idempotency_key))
        return f"msg-{len(sent)}"

    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="status projection")
        kdp.link_parent_thread(
            conn,
            issue_id=issue_id,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key="link:status",
        )

        event_id = kdp.record_status_change(
            conn,
            issue_id=issue_id,
            from_status="ready",
            to_status="running",
            actor="developer",
            reason="claimed for implementation",
            next_step="post progress checkpoint",
            sequence=7,
        )
        assert event_id == kdp.record_status_change(
            conn,
            issue_id=issue_id,
            from_status="ready",
            to_status="running",
            actor="developer",
            reason="claimed for implementation",
            next_step="post progress checkpoint",
            sequence=7,
        )

        result = kdp.project_next_outbox_event(conn, send)
        assert result.delivered is True
        assert sent == [
            (
                "thread-1",
                "Kanban status changed: ready -> running\nActor: developer\nReason: claimed for implementation\nNext: post progress checkpoint",
                f"kanban-status:{issue_id}:7:ready->running",
            )
        ]
        assert kdp.project_next_outbox_event(conn, send).delivered is False


def test_failed_projection_does_not_mutate_lifecycle_and_retry_dedupes(kanban_home):
    attempts = 0

    def flaky_send(thread_id: str, content: str, idempotency_key: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("discord unavailable")
        return "discord-msg-1"

    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="retry projection")
        kdp.link_parent_thread(
            conn,
            issue_id=issue_id,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key="link:retry",
        )
        kdp.record_status_change(
            conn,
            issue_id=issue_id,
            from_status="ready",
            to_status="done",
            actor="developer",
            reason="finished",
            sequence=1,
        )

        failed = kdp.project_next_outbox_event(conn, flaky_send)
        assert failed.delivered is False
        assert kb.get_task(conn, issue_id).status == "ready"

        delivered = kdp.project_next_outbox_event(conn, flaky_send)
        assert delivered.delivered is True
        assert attempts == 2
        assert kdp.project_next_outbox_event(conn, flaky_send).delivered is False


def test_agent_checkpoint_targets_parent_thread_and_dedupes(kanban_home):
    with kb.connect() as conn:
        issue_id = kb.create_task(conn, title="checkpoint")
        kdp.link_parent_thread(
            conn,
            issue_id=issue_id,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-1",
            idempotency_key="link:checkpoint",
        )

        first = kdp.emit_agent_checkpoint(
            conn,
            issue_id=issue_id,
            checkpoint_type="progress",
            agent_profile="developer",
            session_id="session-1",
            run_id="run-1",
            summary="implemented helper",
            next_step="run tests",
            evidence="unit test added",
            now=1000,
        )
        duplicate = kdp.emit_agent_checkpoint(
            conn,
            issue_id=issue_id,
            checkpoint_type="progress",
            agent_profile="developer",
            session_id="session-1",
            run_id="run-1",
            summary="implemented helper",
            next_step="run tests",
            evidence="unit test added",
            now=1010,
        )

        assert first.created is True
        assert duplicate.created is False
        pending = kdp.list_outbox(conn)
        assert len(pending) == 1
        assert pending[0].parent_thread_id == "thread-1"
        assert "Agent checkpoint: progress" in pending[0].payload["content"]


def test_typing_indicator_is_best_effort_and_rate_limited():
    calls: list[str] = []

    def typing(thread_id: str) -> None:
        calls.append(thread_id)
        raise RuntimeError("rate limited")

    state = kdp.TypingIndicatorState()

    assert kdp.send_typing_best_effort("thread-1", typing, state=state, now=1000) is False
    assert kdp.send_typing_best_effort("thread-1", typing, state=state, now=1001) is False
    assert calls == ["thread-1"]


def test_archive_dry_run_gates_status_projection_and_active_work(kanban_home):
    with kb.connect() as conn:
        done_issue = kb.create_task(conn, title="done must not auto archive")
        archive_issue = kb.create_task(conn, title="archive candidate")
        blocked_issue = kb.create_task(conn, title="active blocker")

        for issue_id in (done_issue, archive_issue, blocked_issue):
            kdp.link_parent_thread(
                conn,
                issue_id=issue_id,
                guild_id="guild-1",
                channel_id="channel-1",
                thread_id=f"thread-{issue_id}",
                idempotency_key=f"link:{issue_id}",
            )

        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (done_issue,))
        conn.execute("UPDATE tasks SET status = 'archive_candidate' WHERE id = ?", (archive_issue,))
        conn.execute("UPDATE tasks SET status = 'closed', current_run_id = 123 WHERE id = ?", (blocked_issue,))
        conn.commit()
        kdp.record_status_change(
            conn,
            issue_id=archive_issue,
            from_status="done",
            to_status="archive_candidate",
            actor="pm",
            reason="review passed",
            sequence=1,
        )
        kdp.mark_outbox_delivered(conn, f"kanban-status:{archive_issue}:1:done->archive_candidate", "msg-1")

        done_plan = kdp.plan_archive_dry_run(conn, done_issue)
        archive_plan = kdp.plan_archive_dry_run(conn, archive_issue)
        blocked_plan = kdp.plan_archive_dry_run(conn, blocked_issue)

        assert done_plan.allowed is False
        assert "status done is not archive-eligible in MVP" in done_plan.reasons
        assert archive_plan.allowed is True
        assert archive_plan.live_mode is False
        assert blocked_plan.allowed is False
        assert "active worker/session present" in blocked_plan.reasons
        assert kb.get_task(conn, archive_issue).status == "archive_candidate"
