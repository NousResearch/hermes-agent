import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


class RecordingAdapter:
    def __init__(self):
        self.sent = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})


class FailsOnSecondSendAdapter(RecordingAdapter):
    async def send(self, chat_id, text, metadata=None):
        if len(self.sent) == 1:
            raise RuntimeError("send failed on second event")
        await super().send(chat_id, text, metadata=metadata)


class ReturnsFailureAdapter(RecordingAdapter):
    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})
        return SimpleNamespace(success=False, error="adapter reported failure")


class DisconnectedAdapters(dict):
    """Expose a platform during collection, then simulate disconnect on get()."""

    def get(self, key, default=None):
        return None


async def _run_one_notifier_tick(monkeypatch, runner):
    real_sleep = asyncio.sleep

    async def fake_sleep(delay):
        if delay == 5:
            return None
        runner._running = False
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    await runner._kanban_notifier_watcher(interval=1)


def _make_runner(adapter):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    return runner


def _create_completed_subscription(summary="done once"):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="notify once", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.complete_task(conn, tid, summary=summary)
        return tid
    finally:
        conn.close()


def _unseen_terminal_events(tid):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_dedupes_board_slugs_pointing_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-kanban.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")

    tid = _create_completed_subscription()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert "*작업 완료*" in adapter.sent[0]["text"]
    assert tid in adapter.sent[0]["text"]


def test_kanban_notifier_claim_prevents_second_watcher_send(tmp_path, monkeypatch):
    db_path = tmp_path / "single-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    tid = _create_completed_subscription()

    adapter1 = RecordingAdapter()
    adapter2 = RecordingAdapter()

    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter1)))
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter2)))

    assert len(adapter1.sent) == 1
    assert adapter2.sent == []


def test_kanban_notifier_delivers_comment_events(tmp_path, monkeypatch):
    db_path = tmp_path / "comment-events.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="agent handoff", assignee="news-scout")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.add_comment(
            conn,
            tid,
            author="news-scout",
            body="Status:\nScout complete\nEvidence:\nCanonical URLs verified",
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    text = adapter.sent[0]["text"]
    assert text.startswith("*새 코멘트*")
    assert tid in text
    assert "• 작성: `news-scout`" in text
    assert "Scout complete" in text
    assert "Status:" not in text
    assert "Evidence: Canonical URLs verified" in text


def test_kanban_notifier_rewinds_claim_if_adapter_disconnects(tmp_path, monkeypatch):
    db_path = tmp_path / "adapter-disconnect.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = DisconnectedAdapters({Platform.TELEGRAM: RecordingAdapter()})
    runner._kanban_sub_fail_counts = {}

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_kanban_notifier_retries_from_first_failed_event(tmp_path, monkeypatch):
    db_path = tmp_path / "partial-send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="partial delivery", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.add_comment(conn, tid, author="worker", body="first event delivered")
        kb.add_comment(conn, tid, author="worker", body="second event should retry")
    finally:
        conn.close()

    first_adapter = FailsOnSecondSendAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(first_adapter)))

    assert len(first_adapter.sent) == 1
    assert "first event delivered" in first_adapter.sent[0]["text"]

    second_adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(second_adapter)))

    assert len(second_adapter.sent) == 1
    assert "first event delivered" not in second_adapter.sent[0]["text"]
    assert "second event should retry" in second_adapter.sent[0]["text"]


def test_kanban_notifier_rewinds_when_adapter_returns_failure(tmp_path, monkeypatch):
    db_path = tmp_path / "send-result-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="return failure", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb.add_comment(conn, tid, author="worker", body="retry me")
    finally:
        conn.close()

    failing_adapter = ReturnsFailureAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(failing_adapter)))

    assert len(failing_adapter.sent) == 1

    retry_adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(retry_adapter)))

    assert len(retry_adapter.sent) == 1
    assert "retry me" in retry_adapter.sent[0]["text"]


def test_kanban_db_path_is_test_isolated_from_real_home():
    hermes_home = Path(kb.kanban_home())
    production_db = Path.home() / ".hermes" / "kanban.db"
    assert kb.kanban_db_path().resolve() != production_db.resolve()


def test_kanban_create_task_idempotency_key_returns_existing_task(tmp_path, monkeypatch):
    db_path = tmp_path / "idempotency.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        first = kb.create_task(conn, title="first", assignee="worker", idempotency_key="slack:T:C:171.1")
        second = kb.create_task(conn, title="second", assignee="worker", idempotency_key="slack:T:C:171.1")
        rows = conn.execute(
            "SELECT id, title FROM tasks WHERE idempotency_key = ?",
            ("slack:T:C:171.1",),
        ).fetchall()
    finally:
        conn.close()

    assert second == first
    assert [(row["id"], row["title"]) for row in rows] == [(first, "first")]


def test_kanban_handoff_comment_creates_next_agent_task(tmp_path, monkeypatch):
    db_path = tmp_path / "handoff-comment.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        source_id = kb.create_task(
            conn,
            title="frontend intake",
            assignee="frontend-engineer",
        )
        kb.add_notify_sub(
            conn,
            task_id=source_id,
            platform="slack",
            chat_id="C_FRONTEND",
            thread_id="171.1",
            user_id="U_USER",
            notifier_profile="orchestrator",
        )
        kb.add_comment(
            conn,
            source_id,
            author="frontend-engineer",
            body=(
                "@accessibility-reviewer 다음 액션: 버튼 focus state와 live region을 검토해 주세요.\n"
                "근거: keyboard user flow가 추가되었습니다.\n"
                "완료 기준: 접근성 이슈와 수정안을 comment로 남깁니다."
            ),
        )
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    created = runner._kanban_process_handoff_events_once(kb)

    assert len(created) == 1
    child_id = created[0]["task_id"]
    conn = kb.connect()
    try:
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.assignee == "accessibility-reviewer"
        assert child.status == "ready"
        assert "버튼 focus state" in (child.body or "")
        subs = kb.list_notify_subs(conn, child_id)
        assert len(subs) == 1
        assert subs[0]["platform"] == "slack"
        assert subs[0]["chat_id"] == "C_FRONTEND"
        assert subs[0]["thread_id"] == "171.1"
    finally:
        conn.close()


def test_kanban_handoff_body_keeps_thread_collaboration_contract():
    runner = GatewayRunner.__new__(GatewayRunner)

    body = runner._kanban_handoff_body(
        board="ai-frontend",
        source_task_id="t_source",
        source_title="frontend intake",
        author="frontend-engineer",
        target_agent="accessibility-reviewer",
        comment_id=7,
        action=(
            "버튼 focus state를 검토해 주세요.\n"
            "근거: keyboard user flow가 추가되었습니다.\n"
            "완료 기준: 수정안을 comment로 남깁니다."
        ),
    )

    assert "Thread summary:" in body
    assert "Open questions:" in body
    assert "Reaction rules:" in body
    assert "accessibility-reviewer" in body
    assert "보고서 덤프" in body


def test_kanban_handoff_comment_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "handoff-idempotent.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        source_id = kb.create_task(conn, title="security intake", assignee="threat-modeler")
        kb.add_comment(
            conn,
            source_id,
            author="threat-modeler",
            body="@security-reviewer 다음 액션: threat model 결과를 검토해 주세요.",
        )
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    first = runner._kanban_process_handoff_events_once(kb)
    second = runner._kanban_process_handoff_events_once(kb)

    assert len(first) == 1
    assert second == []


def test_kanban_owner_closeout_created_after_handoff_child_completion(tmp_path, monkeypatch):
    db_path = tmp_path / "owner-closeout.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("ai-frontend", name="AI Frontend")

    conn = kb.connect(board="ai-frontend")
    try:
        source_id = kb.create_task(conn, title="frontend intake", assignee="frontend-engineer")
        child_id = kb.create_task(
            conn,
            title="[handoff] accessibility-reviewer: 접근성 검토",
            body=(
                "자동 handoff task입니다.\n\n"
                "- board: ai-frontend\n"
                f"- source_task: {source_id} - frontend intake\n"
                "- handoff_author: frontend-engineer\n"
                "- next_agent: accessibility-reviewer\n\n"
                "다음 액션:\n접근성 검토"
            ),
            assignee="accessibility-reviewer",
            created_by="handoff:frontend-engineer",
            idempotency_key="kanban-handoff:ai-frontend:10:accessibility-reviewer",
        )
        kb.add_notify_sub(
            conn,
            task_id=source_id,
            platform="slack",
            chat_id="C_FRONTEND",
            thread_id="171.2",
            user_id="U_USER",
            notifier_profile="orchestrator",
        )
        kb.complete_task(
            conn,
            child_id,
            summary="접근성 검토 완료. focus state와 live region은 적절합니다.",
        )
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    created = runner._kanban_process_owner_closeout_events_once(kb)

    assert len(created) == 1
    closeout_id = created[0]["task_id"]
    conn = kb.connect(board="ai-frontend")
    try:
        closeout = kb.get_task(conn, closeout_id)
        assert closeout is not None
        assert closeout.assignee == "frontend"
        assert closeout.status == "ready"
        assert "접근성 검토 완료" in (closeout.body or "")
        assert child_id in (closeout.body or "")
        subs = kb.list_notify_subs(conn, closeout_id)
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "C_FRONTEND"
        assert subs[0]["thread_id"] == "171.2"
    finally:
        conn.close()


def test_kanban_owner_closeout_is_idempotent(tmp_path, monkeypatch):
    db_path = tmp_path / "owner-closeout-idempotent.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("ai-security", name="AI Security")

    conn = kb.connect(board="ai-security")
    try:
        child_id = kb.create_task(
            conn,
            title="[handoff] security-reviewer: 검토",
            body=(
                "자동 handoff task입니다.\n\n"
                "- board: ai-security\n"
                "- source_task: t_source - security intake\n"
                "- handoff_author: threat-modeler\n"
                "- next_agent: security-reviewer\n\n"
                "다음 액션:\n검토"
            ),
            assignee="security-reviewer",
            created_by="handoff:threat-modeler",
        )
        kb.complete_task(conn, child_id, summary="보안 검토 완료")
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    first = runner._kanban_process_owner_closeout_events_once(kb)
    second = runner._kanban_process_owner_closeout_events_once(kb)

    assert len(first) == 1
    assert second == []


def test_kanban_memory_evolution_created_after_owner_closeout(tmp_path, monkeypatch):
    db_path = tmp_path / "memory-evolution.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("ai-policy", name="AI Policy")

    conn = kb.connect(board="ai-policy")
    try:
        source_id = kb.create_task(conn, title="정책 변경 요청", assignee="policy-guardian")
        closeout_id = kb.create_task(
            conn,
            title="[closeout] policy: 정책 운영 기준 정리",
            body=(
                "자동 owner closeout task입니다.\n\n"
                "- board: ai-policy\n"
                "- owner: policy\n"
                "- specialist_task: t_specialist - 정책 검토\n"
                "- specialist_agent: policy-guardian\n"
                f"- source_task: {source_id} - 정책 변경 요청\n\n"
                "specialist 결과:\n"
                "새 Slack channel policy 변경은 owner closeout에서 최종 승인 후 공지해야 합니다.\n\n"
                "다음 액션:\n최종 정리"
            ),
            assignee="policy",
            created_by="owner-closeout:policy-guardian",
            idempotency_key="kanban-owner-closeout:ai-policy:10:t_specialist",
        )
        kb.add_notify_sub(
            conn,
            task_id=source_id,
            platform="slack",
            chat_id="C_POLICY",
            thread_id="172.1",
            user_id="U_USER",
            notifier_profile="orchestrator",
        )
        kb.complete_task(
            conn,
            closeout_id,
            summary="정책 결정: Slack channel policy 변경은 owner closeout 승인 후 공지한다.",
        )
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    created = runner._kanban_process_memory_evolution_events_once(kb)

    assert len(created) == 1
    memory_id = created[0]["task_id"]
    conn = kb.connect(board="ai-policy")
    try:
        memory_task = kb.get_task(conn, memory_id)
        assert memory_task is not None
        assert memory_task.assignee == "memory-curator"
        assert memory_task.status == "ready"
        assert "정책 결정" in (memory_task.body or "")
        assert closeout_id in (memory_task.body or "")
        subs = kb.list_notify_subs(conn, memory_id)
        assert len(subs) == 1
        assert subs[0]["chat_id"] == "C_POLICY"
        assert subs[0]["thread_id"] == "172.1"
    finally:
        conn.close()


def test_kanban_memory_evolution_skips_smoke_closeouts(tmp_path, monkeypatch):
    db_path = tmp_path / "memory-evolution-smoke.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("ai-frontend", name="AI Frontend")

    conn = kb.connect(board="ai-frontend")
    try:
        closeout_id = kb.create_task(
            conn,
            title="[closeout] frontend: owner closeout smoke",
            body=(
                "자동 owner closeout task입니다.\n\n"
                "- board: ai-frontend\n"
                "- owner: frontend\n"
                "- specialist_task: t_smoke - smoke\n"
                "- specialist_agent: accessibility-reviewer\n"
                "- source_task: t_source - smoke\n\n"
                "specialist 결과:\nowner closeout smoke 검증 완료"
            ),
            assignee="frontend",
            created_by="owner-closeout:accessibility-reviewer",
        )
        kb.complete_task(conn, closeout_id, summary="owner closeout smoke 완료")
    finally:
        conn.close()

    runner = GatewayRunner.__new__(GatewayRunner)
    assert runner._kanban_process_memory_evolution_events_once(kb) == []


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


def test_kanban_notifier_rewinds_claim_on_send_exception(tmp_path, monkeypatch):
    """A raising adapter rewinds the claim so the next tick can retry.

    This is the second rewind path (distinct from the adapter-disconnect path
    in test_kanban_notifier_rewinds_claim_if_adapter_disconnects). Here the
    adapter is connected and the send call actually fires; the claim must
    still rewind so the event isn't lost when send() raises mid-tick.
    """
    db_path = tmp_path / "send-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    tid = _create_completed_subscription()

    adapter = FailingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Send was attempted (so we exercised the failure path, not just the
    # disconnect path) and the claim was rewound — the unseen-events query
    # still returns the event for retry on the next tick.
    assert adapter.attempts >= 1, "send should have been attempted at least once"
    assert [ev.kind for ev in _unseen_terminal_events(tid)] == ["completed"]


def test_notifier_redelivers_same_kind_on_dispatch_cycle(tmp_path, monkeypatch):
    """A retry cycle (crashed → reclaimed → crashed) notifies the user twice.

    Before #21398 the notifier auto-unsubscribed on any terminal event kind
    (gave_up / crashed / timed_out), so the second crash in a respawn cycle
    silently dropped — the subscription was already gone. This test pins the
    new contract: subscription survives non-final terminal events; the
    cursor handles dedup.

    Two crashes ten seconds apart on the same task — both should land on
    the adapter.
    """
    db_path = tmp_path / "redeliver-cycle.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="cycle test", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        # First crash — fired by the dispatcher when the worker PID dies.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # First crash delivered.
    assert len(adapter.sent) == 1
    assert "*작업 오류*" in adapter.sent[0]["text"]

    # Subscription survives — the cursor advanced past event #1, but the
    # row is still there.
    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, (
            "Subscription must survive a crashed event so a respawn-cycle "
            "second crash also notifies the user (issue #21398)."
        )

        # Second crash — same task, same dispatcher (or a respawn). Append
        # another event to simulate the dispatcher firing crashed a second
        # time during retry.
        kb._append_event(conn, tid, kind="crashed")
    finally:
        conn.close()

    # New tick: the second event has a fresh id past the cursor advance,
    # so it gets claimed and delivered.
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 2, (
        f"Second crashed event should also notify; got {len(adapter.sent)} "
        f"deliveries (texts: {[d['text'] for d in adapter.sent]})"
    )
    assert "*작업 오류*" in adapter.sent[1]["text"]
