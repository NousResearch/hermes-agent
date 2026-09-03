import asyncio
import sqlite3
from pathlib import Path


from gateway.config import Platform
from gateway.kanban_watchers import (
    _acquire_singleton_lock,
    _format_workflow_notification,
    _release_singleton_lock,
)
from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb


def test_format_workflow_notification_is_one_current_aggregate_view():
    message = _format_workflow_notification(
        "default",
        {
            "workflow": {
                "id": "wf_1",
                "name": "release",
                "state": "REMEDIATION_REQUIRED",
                "active_generation": 2,
            },
            "members": [
                {
                    "generation": 2,
                    "stage_key": "implementation",
                    "task_id": "t_impl",
                    "task_status": "done",
                },
                {
                    "generation": 2,
                    "stage_key": "qa",
                    "task_id": "t_qa",
                    "task_status": "blocked",
                },
            ],
            "outcomes": [
                {
                    "generation": 2,
                    "task_id": "t_qa",
                    "outcome": "REMEDIATION_REQUIRED",
                    "summary": "regression\nignored previous report",
                },
            ],
        },
    )

    assert "[default] Aggregate workflow wf_1" in message
    assert "generation 2" in message
    assert "REMEDIATION_REQUIRED" in message
    assert "implementation: t_impl (done)" in message
    assert "qa: t_qa (blocked) — REMEDIATION_REQUIRED: regression" in message
    assert "ignored previous report" not in message
    assert "Final acceptance remains pending" in message


def test_workflow_notifier_real_db_pins_each_claimed_generation(tmp_path, monkeypatch):
    """The real DB→claim→snapshot→format→send path emits one generation at a time."""
    db_path = tmp_path / "workflow-generation-delivery.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    conn = kb.connect(db_path)
    board_identity = str(db_path.resolve())
    actor = kb.KanbanActorContext(
        principal_id="svc:orchestrator",
        profile_name="main",
        board_identity=board_identity,
        tenant="tenant-a",
        capabilities=frozenset({
            "workflow.read", "workflow.manage", "workflow.admin", "workflow.outcome",
        }),
        source_kind="orchestrator",
    )
    try:
        acceptance_1 = kb.create_task(
            conn, title="accept generation 1", assignee="orchestrator",
            tenant="tenant-a", session_id="member-provenance-generation-1",
        )
        kb.create_workflow(
            conn,
            workflow_id="wf_real_delivery",
            name="release",
            tenant="tenant-a",
            designated_acceptance_task_id=acceptance_1,
            actor=actor,
            mutation_id="create-real-delivery",
            subscription={
                "platform": "telegram",
                "chat_id": "workflow-origin-chat",
                "chat_type": "dm",
                "notifier_profile": "main",
                "target_states": ["PASS"],
            },
        )
        passed_1 = kb.record_workflow_outcome(
            conn,
            workflow_id="wf_real_delivery",
            task_id=acceptance_1,
            outcome="PASS",
            summary="generation one accepted",
            actor=actor,
            mutation_id="pass-generation-1",
            expected_version=1,
        )
        acceptance_2 = kb.create_task(
            conn, title="accept generation 2", assignee="orchestrator",
            tenant="tenant-a", session_id="member-provenance-generation-2",
        )
        remediation_2 = kb.create_task(
            conn, title="remediate generation 2", assignee="builder",
            tenant="tenant-a",
        )
        reverification_2 = kb.create_task(
            conn, title="reverify generation 2", assignee="x_qa",
            tenant="tenant-a",
        )
        reopened = kb.reopen_workflow(
            conn,
            workflow_id="wf_real_delivery",
            designated_acceptance_task_id=acceptance_2,
            members=[
                {
                    "task_id": acceptance_2,
                    "stage_key": "acceptance-2",
                    "stage_role": "acceptance",
                    "required": True,
                },
                {
                    "task_id": remediation_2,
                    "stage_key": "remediation-2",
                    "stage_role": "remediation",
                    "required": True,
                },
                {
                    "task_id": reverification_2,
                    "stage_key": "reverification-2",
                    "stage_role": "reverification",
                    "required": True,
                },
            ],
            actor=actor,
            mutation_id="reopen-generation-2",
            expected_version=passed_1["workflow"]["version"],
            reason="second release generation",
        )
        remediated = kb.record_workflow_outcome(
            conn,
            workflow_id="wf_real_delivery",
            task_id=remediation_2,
            outcome="PASS",
            summary="generation two remediation complete",
            actor=actor,
            mutation_id="remediate-generation-2",
            expected_version=reopened["workflow"]["version"],
        )
        reverified = kb.record_workflow_outcome(
            conn,
            workflow_id="wf_real_delivery",
            task_id=reverification_2,
            outcome="PASS",
            summary="generation two independently verified",
            actor=actor,
            mutation_id="reverify-generation-2",
            expected_version=remediated["workflow"]["version"],
        )
        kb.record_workflow_outcome(
            conn,
            workflow_id="wf_real_delivery",
            task_id=acceptance_2,
            outcome="PASS",
            summary="generation two accepted",
            actor=actor,
            mutation_id="pass-generation-2",
            expected_version=reverified["workflow"]["version"],
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"
    runner._authorization_adapter = lambda platform, profile=None: adapter

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    runner._running = True
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [item["chat_id"] for item in adapter.sent] == [
        "workflow-origin-chat", "workflow-origin-chat",
    ]
    first, second = [item["text"] for item in adapter.sent]
    assert "generation 1" in first
    assert f"acceptance: {acceptance_1}" in first
    assert "PASS: generation one accepted" in first
    assert acceptance_2 not in first
    assert "generation two accepted" not in first
    assert "member-provenance-generation-1" not in repr(adapter.sent)

    assert "generation 2" in second
    assert f"acceptance-2: {acceptance_2}" in second
    assert "PASS: generation two accepted" in second
    assert acceptance_1 not in second
    assert "generation one accepted" not in second
    assert "member-provenance-generation-2" not in repr(adapter.sent)


def _workflow_delivery(sub, *, snapshot=None):
    return {
        "workflow_delivery": True,
        "sub": sub,
        "old_cursor": 3,
        "cursor": 7,
        "events": [{"id": 7, "kind": "aggregate_changed"}],
        "snapshot": snapshot or {
            "workflow": {
                "id": sub["workflow_id"], "name": "release",
                "state": "PASS", "active_generation": 1,
            },
            "members": [],
            "outcomes": [],
        },
        "board": "default",
    }


class _FakeWorkflowKB:
    def __init__(self):
        self.completed = []
        self.failed = []

    class _Conn:
        def close(self):
            return None

    def connect(self, board=None):
        assert board == "default"
        return self._Conn()

    def complete_workflow_delivery(self, conn, **kwargs):
        self.completed.append(kwargs)
        return {"retry_count": 0}

    def fail_workflow_delivery(self, conn, **kwargs):
        self.failed.append(kwargs)
        return {"retry_count": 1, "dead_lettered_at": None}


def test_workflow_push_delivery_uses_only_subscription_route(monkeypatch):
    sub = {
        "workflow_id": "wf_1", "role": "origin", "platform": "telegram",
        "chat_id": "workflow-chat", "chat_type": "dm", "thread_id": "topic-7",
        "user_id": "user-1", "notifier_profile": "origin-profile",
        "delivery_metadata": {"chat_type": "dm", "thread_id": "topic-7"},
    }
    snapshot = {
        "workflow": {
            "id": "wf_1", "name": "release", "state": "PASS",
            "active_generation": 1,
        },
        "members": [{
            "generation": 1, "task_id": "t_member", "stage_key": "qa",
            "task_status": "done", "session_id": "attacker-controlled-session",
        }],
        "outcomes": [],
    }
    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._authorization_adapter = lambda platform, profile=None: adapter
    fake_kb = _FakeWorkflowKB()

    asyncio.run(runner._deliver_workflow_notification(
        _workflow_delivery(sub, snapshot=snapshot), Platform, fake_kb,
    ))

    assert adapter.sent[0]["chat_id"] == "workflow-chat"
    assert adapter.sent[0]["metadata"]["thread_id"] == "topic-7"
    assert adapter.sent[0]["metadata"]["idempotency_key"] == "workflow:wf_1:event:7"
    assert adapter.handled[0].source.chat_id == "workflow-chat"
    assert adapter.handled[0].source.thread_id == "topic-7"
    assert "attacker-controlled-session" not in repr(adapter.handled[0])
    assert len(fake_kb.completed) == 1
    assert fake_kb.failed == []


def test_workflow_api_server_wake_uses_subscription_chat_id(monkeypatch):
    class ApiAdapter(RecordingAdapter):
        supports_async_delivery = False

    adapter = ApiAdapter()
    runner = _make_runner(adapter)
    runner._authorization_adapter = lambda platform, profile=None: adapter
    fake_kb = _FakeWorkflowKB()
    wakes = []

    async def fake_deliver_wake(adapter_arg, **kwargs):
        wakes.append(kwargs)

    monkeypatch.setattr("gateway.wake.deliver_wake", fake_deliver_wake)
    sub = {
        "workflow_id": "wf_api", "role": "origin", "platform": "api_server",
        "chat_id": "workflow-origin-session", "chat_type": None, "thread_id": "",
        "user_id": None, "notifier_profile": "origin-profile",
        "delivery_metadata": {},
    }

    asyncio.run(runner._deliver_workflow_notification(
        _workflow_delivery(sub), Platform, fake_kb,
    ))

    assert adapter.sent == []
    assert wakes == [{
        "text": _format_workflow_notification("default", _workflow_delivery(sub)["snapshot"]),
        "session_id": "workflow-origin-session",
    }]
    assert len(fake_kb.completed) == 1
    assert fake_kb.failed == []


def test_workflow_unavailable_profile_fails_closed_and_rewinds():
    sub = {
        "workflow_id": "wf_1", "role": "origin", "platform": "telegram",
        "chat_id": "workflow-chat", "chat_type": "dm", "thread_id": "",
        "user_id": None, "notifier_profile": "missing-profile",
        "delivery_metadata": {},
    }
    runner = _make_runner(RecordingAdapter())
    runner._authorization_adapter = lambda platform, profile=None: None
    fake_kb = _FakeWorkflowKB()

    asyncio.run(runner._deliver_workflow_notification(
        _workflow_delivery(sub), Platform, fake_kb,
    ))

    assert fake_kb.completed == []
    assert len(fake_kb.failed) == 1
    assert fake_kb.failed[0]["claimed_cursor"] == 7
    assert fake_kb.failed[0]["old_cursor"] == 3
    assert "Unavailable" in fake_kb.failed[0]["error_class"]


def test_notifier_polls_workflow_subscription_without_task_subscriptions(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "workflow-only.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    sub = {
        "workflow_id": "wf_only", "role": "origin", "platform": "telegram",
        "chat_id": "workflow-chat", "chat_type": "dm", "thread_id": "",
        "user_id": None, "notifier_profile": "main", "tenant": "tenant-a",
        "delivery_metadata": "{}", "disabled_at": None, "dead_lettered_at": None,
    }

    class FakeCursor:
        def fetchall(self):
            return [sub]

    class FakeConn:
        def execute(self, sql, params=None):
            assert "kanban_workflow_subscriptions" in sql
            return FakeCursor()

        def close(self):
            return None

    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{
        "slug": "default", "db_path": str(db_path),
    }])
    monkeypatch.setattr(kb, "count_notify_subs", lambda **kwargs: 0)
    monkeypatch.setattr(
        kb, "count_workflow_subscriptions_readonly", lambda board=None: 1,
        raising=False,
    )
    monkeypatch.setattr(kb, "connect", lambda board=None: FakeConn())
    monkeypatch.setattr(kb, "list_notify_subs", lambda conn, **kwargs: [])
    monkeypatch.setattr(
        kb, "claim_workflow_events_for_subscription",
        lambda conn, workflow_id, role="origin": (
            0, 7, [{
                "id": 7,
                "kind": "aggregate_changed",
                "generation": 1,
                "payload": {
                    "generation": 1,
                    "resulting_version": 2,
                    "resulting_state": "PASS",
                },
            }],
        ),
        raising=False,
    )
    monkeypatch.setattr(
        kb, "KanbanActorContext", lambda **kwargs: kwargs, raising=False,
    )
    monkeypatch.setattr(
        kb, "get_workflow", lambda conn, workflow_id, **kwargs: {
            "workflow": {
                "id": workflow_id, "name": "release", "state": "ACTIVE",
                "active_generation": 2, "version": 3,
            },
            "members": [], "outcomes": [],
        },
        raising=False,
    )

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"
    delivered = []

    async def fake_deliver(delivery, platform_enum, kb_module):
        delivered.append(delivery)
        return True

    runner._deliver_workflow_notification = fake_deliver
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(delivered) == 1
    assert delivered[0]["sub"]["workflow_id"] == "wf_only"
    assert delivered[0]["cursor"] == 7
    assert delivered[0]["snapshot"]["workflow"]["state"] == "PASS"
    assert delivered[0]["snapshot"]["workflow"]["active_generation"] == 1
    assert delivered[0]["snapshot"]["workflow"]["version"] == 2


def test_workflow_collection_failure_after_claim_rewinds_for_durable_retry(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "workflow-collection-failure.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    sub = {
        "workflow_id": "wf_failed_collect", "role": "origin", "platform": "telegram",
        "chat_id": "workflow-chat", "chat_type": "dm", "thread_id": "",
        "user_id": None, "notifier_profile": "main", "tenant": "tenant-a",
        "delivery_metadata": "{}", "disabled_at": None, "dead_lettered_at": None,
    }
    failed = []

    class FakeCursor:
        def fetchall(self):
            return [sub]

    class FakeConn:
        def execute(self, sql, params=None):
            assert "kanban_workflow_subscriptions" in sql
            return FakeCursor()

        def close(self):
            return None

    monkeypatch.setattr(kb, "list_boards", lambda include_archived=False: [{
        "slug": "default", "db_path": str(db_path),
    }])
    monkeypatch.setattr(kb, "count_notify_subs", lambda **kwargs: 0)
    monkeypatch.setattr(
        kb, "count_workflow_subscriptions_readonly", lambda board=None: 1,
        raising=False,
    )
    monkeypatch.setattr(kb, "connect", lambda board=None: FakeConn())
    monkeypatch.setattr(kb, "list_notify_subs", lambda conn, **kwargs: [])
    monkeypatch.setattr(
        kb, "claim_workflow_events_for_subscription",
        lambda conn, workflow_id, role="origin": (
            3, 7, [{
                "id": 7, "kind": "aggregate_changed", "generation": 1,
                "payload": {"generation": 1, "resulting_version": 2,
                            "resulting_state": "PASS"},
            }],
        ),
    )
    monkeypatch.setattr(kb, "KanbanActorContext", lambda **kwargs: kwargs)
    monkeypatch.setattr(
        kb, "get_workflow",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("snapshot corrupt")),
    )

    def record_failure(conn, **kwargs):
        failed.append(kwargs)
        return {"retry_count": 1, "dead_lettered_at": None}

    monkeypatch.setattr(kb, "fail_workflow_delivery", record_failure)
    runner = _make_runner(RecordingAdapter())
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert failed == [{
        "workflow_id": "wf_failed_collect",
        "role": "origin",
        "claimed_cursor": 7,
        "old_cursor": 3,
        "error_class": "RuntimeError",
    }]


class RecordingAdapter:
    def __init__(self):
        self.sent = []
        self.handled = []

    async def send(self, chat_id, text, metadata=None):
        self.sent.append({"chat_id": chat_id, "text": text, "metadata": metadata or {}})

    async def handle_message(self, event):
        self.handled.append(event)


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
    # Most tests model the default gateway after its dispatcher acquired the
    # singleton lock. Tests for startup or non-owner gateways clear this.
    runner._kanban_dispatcher_lock_handle = object()
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


def test_kanban_notifier_replays_telegram_dm_topic_delivery_metadata(tmp_path, monkeypatch):
    db_path = tmp_path / "dm-topic-metadata.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm topic task",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            thread_id="20197",
            delivery_mode="notify+wake",
            delivery_metadata={
                "chat_type": "dm",
                "direct_messages_topic_id": "20197",
                "telegram_dm_topic_reply_fallback": True,
                "telegram_reply_to_message_id": "462",
                "thread_id": "20197",
            },
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.sent[0]["metadata"] == {
        "chat_type": "dm",
        "direct_messages_topic_id": "20197",
        "telegram_dm_topic_reply_fallback": True,
        "telegram_reply_to_message_id": "462",
        "thread_id": "20197",
    }
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"
    assert adapter.handled[0].source.thread_id == "20197"


def test_active_named_profile_subscription_is_delivered(tmp_path, monkeypatch):
    """A sub stamped with the gateway's own named profile uses self.adapters.

    Regression for #71340: on a standalone (non-multiplex) gateway running a
    named profile, _authorization_adapter() used to treat the active name as a
    multiplex secondary, find no _profile_adapters entry, fail closed, and
    rewind the claim forever — silent zero-delivery.
    """
    db_path = tmp_path / "actionable-block.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    reason = "AGE-39 — https://linear.example/AGE-39 — publishing verified."
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="approval", assignee="publisher")
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            notifier_profile="main",
        )
        kb.block_task(conn, tid, reason=reason, kind="needs_input")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "main"

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    message = adapter.sent[0]["text"]
    assert tid in message
    assert "blocked" in message


def test_non_dispatch_gateway_claims_only_its_profile_subscriptions(
    tmp_path, monkeypatch,
):
    """A profile gateway delivers its events while another gateway dispatches."""
    db_path = tmp_path / "cross-profile-notifier.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        foreign_tid = kb.create_task(
            conn, title="default-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=foreign_tid,
            platform="telegram",
            chat_id="default-chat",
            notifier_profile="default",
        )
        kb.complete_task(conn, foreign_tid, summary="default done")

        owned_tid = kb.create_task(
            conn, title="writer-owned", assignee="worker",
        )
        kb.add_notify_sub(
            conn,
            task_id=owned_tid,
            platform="telegram",
            chat_id="writer-chat",
            notifier_profile="writer",
        )
        kb.complete_task(conn, owned_tid, summary="writer done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "writer"
    runner._kanban_dispatcher_lock_handle = None

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert [delivery["chat_id"] for delivery in adapter.sent] == ["writer-chat"]
    assert owned_tid in adapter.sent[0]["text"]
    assert len(_unseen_terminal_events_for(foreign_tid, "default-chat")) == 1


def test_legacy_subscription_requires_confirmed_dispatcher_lock_owner(
    tmp_path, monkeypatch,
):
    """Startup and lock-losing gateways cannot claim legacy notifications."""
    db_path = tmp_path / "legacy-lock-owner.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(conn, title="legacy", assignee="worker")
        kb.add_notify_sub(
            conn,
            task_id=task_id,
            platform="telegram",
            chat_id="legacy-chat",
        )
        kb.complete_task(conn, task_id, summary="legacy done")
    finally:
        conn.close()

    startup_adapter = RecordingAdapter()
    startup_runner = _make_runner(startup_adapter)
    startup_runner._kanban_dispatcher_lock_handle = None
    asyncio.run(_run_one_notifier_tick(monkeypatch, startup_runner))
    assert startup_adapter.sent == []
    assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

    lock_path = tmp_path / ".dispatcher.lock"
    winner_handle, winner_state = _acquire_singleton_lock(lock_path)
    loser_handle, loser_state = _acquire_singleton_lock(lock_path)
    try:
        assert winner_state == "held"
        assert loser_state == "contended"

        loser_adapter = RecordingAdapter()
        loser_runner = _make_runner(loser_adapter)
        loser_runner._kanban_dispatcher_lock_handle = loser_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, loser_runner))
        assert loser_adapter.sent == []
        assert len(_unseen_terminal_events_for(task_id, "legacy-chat")) == 1

        winner_adapter = RecordingAdapter()
        winner_runner = _make_runner(winner_adapter)
        winner_runner._kanban_dispatcher_lock_handle = winner_handle
        asyncio.run(_run_one_notifier_tick(monkeypatch, winner_runner))
        assert [item["chat_id"] for item in winner_adapter.sent] == ["legacy-chat"]
        assert task_id in winner_adapter.sent[0]["text"]
    finally:
        _release_singleton_lock(loser_handle)
        _release_singleton_lock(winner_handle)


class FailingAdapter:
    """Adapter whose send() always raises, simulating a transient send error."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        raise RuntimeError("simulated send failure")


class ReportedFailureAdapter:
    """Adapter that REPORTS failure via SendResult(success=False) instead of
    raising — the exact contract the Telegram adapter uses for 'Not connected'
    and degraded-send paths."""

    def __init__(self):
        self.attempts = 0

    async def send(self, chat_id, text, metadata=None):
        self.attempts += 1
        from gateway.platforms.base import SendResult
        return SendResult(success=False, error="Not connected")


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
    assert "crashed" in adapter.sent[0]["text"].lower()

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
    assert "crashed" in adapter.sent[1]["text"].lower()


def test_notifier_subscription_survives_done_reopen_until_archive(
    tmp_path, monkeypatch,
):
    """Done is reversible; archive alone ends notification ownership."""
    db_path = tmp_path / "done-reopen-archive.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="review continuation",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="origin-chat",
            thread_id="origin-thread",
            user_id="origin-user",
            chat_type="group",
            notifier_profile="reviewer",
            delivery_mode="notify+wake",
        )
        assert kb.complete_task(conn, tid, summary="first completion")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.sent[0]["chat_id"] == "origin-chat"
    assert adapter.sent[0]["metadata"]["thread_id"] == "origin-thread"
    assert adapter.handled[0].source.thread_id == "origin-thread"
    assert adapter.handled[0].source.profile == "reviewer"

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1, "completion must retain the origin subscription"
        first_cursor = subs[0]["last_event_id"]
    finally:
        conn.close()

    # A quiet tick proves the completed event cannot replay after its cursor
    # was advanced, even though the subscription now remains present.
    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))
    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "status", {"status": "ready"})
        assert kb.complete_task(conn, tid, summary="corrected completion")
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The reopen status and second completion each deliver once, while only
    # completion wakes the exact original session/thread.
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    assert all(item["chat_id"] == "origin-chat" for item in adapter.sent)
    assert adapter.handled[-1].source.thread_id == "origin-thread"
    assert adapter.handled[-1].source.profile == "reviewer"

    conn = kb.connect()
    try:
        subs = kb.list_notify_subs(conn, tid)
        assert len(subs) == 1
        assert subs[0]["last_event_id"] > first_cursor
        assert kb.archive_task(conn, tid)
    finally:
        conn.close()

    runner = _make_runner(adapter)
    runner._active_profile_name = lambda: "reviewer"
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # Archive itself is intentionally silent, but consumes its event and
    # removes the subscription so no later historical event can replay.
    assert len(adapter.sent) == 3
    assert len(adapter.handled) == 2
    conn = kb.connect()
    try:
        assert kb.list_notify_subs(conn, tid) == []
    finally:
        conn.close()


def test_notifier_wakeup_uses_subscription_chat_type(tmp_path, monkeypatch):
    db_path = tmp_path / "chat-type-wakeup.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="dm requester",
            assignee="worker",
            session_id="origin-session",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-dm",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb.complete_task(conn, tid, summary="done")
    finally:
        conn.close()

    adapter = RecordingAdapter()
    asyncio.run(_run_one_notifier_tick(monkeypatch, _make_runner(adapter)))

    assert len(adapter.sent) == 1
    assert len(adapter.handled) == 1
    assert adapter.handled[0].source.chat_type == "dm"

    # The wake must resume the creator's real DM session key — the whole bug
    # was that a hardcoded chat_type="group" made build_session_key() produce
    # a group-scoped key (a NEW session) instead of the ":dm:<chat_id>" shape
    # the original conversation runs under (#56580 / #68874).
    from gateway.session import build_session_key

    wake_key = build_session_key(adapter.handled[0].source)
    assert wake_key == "agent:main:telegram:dm:chat-dm"
    assert ":group:" not in wake_key


def _unseen_terminal_events_for(tid, chat_id):
    conn = kb.connect()
    try:
        _, events = kb.unseen_events_for_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id=chat_id,
            kinds=["completed", "blocked", "gave_up", "crashed", "timed_out"],
        )
        return events
    finally:
        conn.close()


def test_kanban_notifier_isolates_per_subscription_failure(tmp_path, monkeypatch):
    """One bad subscription must not block delivery for all others.

    Regression for #59269: when claim_unseen_events_for_sub raises for one
    subscription, the entire notifier tick used to abort — silently blocking
    delivery for every other subscription.
    """
    db_path = tmp_path / "isolation.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    # Create two tasks with subscriptions and complete both. The BAD task is
    # created first: list_notify_subs() has no ORDER BY, so SQLite's natural
    # scan returns insertion order — the failing subscription must be
    # processed BEFORE the good one or this test passes even without the
    # per-subscription isolation (the good delivery happens before the tick
    # aborts). A deterministic-order shim below removes the reliance on the
    # scan order entirely.
    conn = kb.connect()
    try:
        tid_bad = kb.create_task(conn, title="bad task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_bad, platform="telegram", chat_id="chat-bad")
        kb.complete_task(conn, tid_bad, summary="done")

        tid_good = kb.create_task(conn, title="good task", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid_good, platform="telegram", chat_id="chat-good")
        kb.complete_task(conn, tid_good, summary="done")
    finally:
        conn.close()

    original_claim = kb.claim_unseen_events_for_sub

    def selective_claim(conn, task_id, **kwargs):
        if task_id == tid_bad:
            raise RuntimeError("simulated DB corruption for bad task")
        return original_claim(conn, task_id=task_id, **kwargs)

    monkeypatch.setattr(kb, "claim_unseen_events_for_sub", selective_claim)

    # Force the failing subscription to be iterated FIRST regardless of the
    # unordered SELECT's scan order.
    original_list = kb.list_notify_subs

    def bad_first(conn, task_id=None, **kwargs):
        subs = original_list(conn, task_id, **kwargs)
        return sorted(subs, key=lambda s: 0 if s["task_id"] == tid_bad else 1)

    monkeypatch.setattr(kb, "list_notify_subs", bad_first)

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    # The good task must still be delivered despite the bad task failing.
    assert len(adapter.sent) == 1
    assert tid_good in adapter.sent[0]["text"]


def test_notifier_delivers_block_loop_detected_triage_ping(tmp_path, monkeypatch):
    """A `block_loop_detected` event must reach the subscriber as a triage ping.

    Regression for the silent-triage gap (PR #62712): kanban_db routes a task
    to `triage` after BLOCK_RECURRENCE_LIMIT re-blocks for the same cause and
    emits ONLY a `block_loop_detected` event — no `blocked`/`status` event.
    Before `block_loop_detected` joined TERMINAL_KINDS with its own message
    branch, that one transition (the whole point of which is to force human
    attention) produced zero notification and the task stalled in triage
    silently.
    """
    db_path = tmp_path / "block-loop.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="loops forever", assignee="worker")
        kb.add_notify_sub(conn, task_id=tid, platform="telegram", chat_id="chat-1")
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)

    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "block_loop_detected must produce a notification"
    text = adapter.sent[0]["text"]
    assert "TRIAGE" in text
    assert tid in text
    assert "needs credentials" in text
    # Cursor advanced: the event is claimed and not re-delivered.
    conn = kb.connect()
    try:
        _, remaining = kb.unseen_events_for_sub(
            conn, task_id=tid, platform="telegram", chat_id="chat-1",
            kinds=["block_loop_detected"],
        )
    finally:
        conn.close()
    assert remaining == []


# ---------------------------------------------------------------------------
# Handoffs that hand a decision back to the origin must wake it, not only ping
# it: `review_requested` (implementation done, waiting for a reviewer) and
# `block_loop_detected` (routed to triage) are terminal kinds just like
# `blocked`.
# ---------------------------------------------------------------------------


def _wake_text(adapter):
    """Text of the single synthetic wake turn injected into the adapter."""
    assert len(adapter.handled) == 1, (
        f"expected exactly one wake turn, got {len(adapter.handled)}"
    )
    return getattr(adapter.handled[0], "text", "") or ""


def _review_handoff_task(
    *,
    delivery_mode="notify+wake",
    summary="PR ready: https://example.invalid/pr/7\nfull details below",
):
    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="implement the thing",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode=delivery_mode,
        )
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(
            conn, tid, summary=summary, expected_run_id=run_id,
        ) is True
        return tid
    finally:
        conn.close()


def test_review_requested_wakes_the_origin_session(tmp_path, monkeypatch):
    """A review handoff wakes the origin and carries the worker's summary."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-wake.db"))
    kb.init_db()
    tid = _review_handoff_task()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1, "the passive review ping is unchanged"
    assert "ready for review" in adapter.sent[0]["text"]

    wake = _wake_text(adapter)
    assert tid in wake
    assert "PR ready: https://example.invalid/pr/7" in wake, (
        "the worker's handoff must ride the wake turn like it does for "
        "`completed`, otherwise the woken reviewer has to re-read the board"
    )


def test_block_loop_detected_wakes_the_origin_session(tmp_path, monkeypatch):
    """A triage escalation wakes the origin so a decision gets made."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "triage-wake.db"))
    kb.init_db()

    conn = kb.connect()
    try:
        tid = kb.create_task(
            conn,
            title="loops forever",
            assignee="worker",
            session_id="agent:main:telegram:dm:chat-1",
        )
        kb.add_notify_sub(
            conn,
            task_id=tid,
            platform="telegram",
            chat_id="chat-1",
            chat_type="dm",
            delivery_mode="notify+wake",
        )
        kb._append_event(
            conn, tid, "block_loop_detected",
            {"reason": "needs credentials", "kind": "needs_input",
             "recurrences": 2, "limit": kb.BLOCK_RECURRENCE_LIMIT},
        )
    finally:
        conn.close()

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert tid in _wake_text(adapter)


def test_review_requested_does_not_wake_a_notify_only_subscription(
    tmp_path, monkeypatch,
):
    """delivery_mode still decides whether a wake-worthy kind wakes at all."""
    monkeypatch.setenv("HERMES_KANBAN_DB", str(tmp_path / "review-notify.db"))
    kb.init_db()
    _review_handoff_task(delivery_mode="notify")

    adapter = RecordingAdapter()
    runner = _make_runner(adapter)
    asyncio.run(_run_one_notifier_tick(monkeypatch, runner))

    assert len(adapter.sent) == 1
    assert adapter.handled == [], (
        "notify-only subscriptions must not be woken by a review handoff"
    )
