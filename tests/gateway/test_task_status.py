import asyncio

import pytest
from unittest.mock import patch

from gateway.platforms.base import SendResult
from gateway.task_status import (
    DeliveryRoute,
    PublicationKind,
    TaskStatusButton,
    TaskStatusDestination,
    TaskStatusPublication,
    TaskStatusPublisher,
    make_task_status_callback_data,
    resolve_exact_task_status_destination,
    resolve_task_status_callback,
)
from hermes_cli import kanban_db as kb
from hermes_cli.config import DEFAULT_CONFIG


class FakeTelegramAdapter:
    def __init__(self):
        self.sends = []
        self.edits = []
        self.pushes = []
        self.wakeups = []
        self.next_message_id = 100

    async def handle_message(self, event):
        self.wakeups.append(event)

    async def send_task_status(self, chat_id, content, *, buttons, metadata):
        self.sends.append(
            {
                "chat_id": chat_id,
                "content": content,
                "buttons": buttons,
                "metadata": metadata,
            }
        )
        message_id = str(self.next_message_id)
        self.next_message_id += 1
        return SendResult(success=True, message_id=message_id)

    async def edit_task_status(
        self, chat_id, message_id, content, *, buttons, metadata
    ):
        self.edits.append(
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "content": content,
                "buttons": buttons,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id=message_id)

    async def send_task_status_push(self, chat_id, content, *, buttons, metadata):
        self.pushes.append(
            {
                "chat_id": chat_id,
                "content": content,
                "buttons": buttons,
                "metadata": metadata,
            }
        )
        return SendResult(success=True, message_id="push-1")


def test_task_status_defaults_are_opt_in_and_suppress_worker_noise():
    config = DEFAULT_CONFIG["kanban"]["task_status"]

    assert config["enabled"] is False
    assert config["publisher_profile"] == "default"
    assert config["dedup_seconds"] == 180
    assert config["worker_lifecycle_pushes"] is False
    assert config["raw_event_pushes"] is False


def test_task_status_schema_migration_is_idempotent_and_profile_scoped(
    tmp_path, monkeypatch
):
    home = tmp_path / "profile-home"
    home.mkdir()
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))

    first = kb.init_db()
    second = kb.init_db()

    assert first == second
    assert first.parent == home
    conn = kb.connect()
    try:
        tables = {
            row["name"]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(task_status_bindings)")
        }
        callback_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(task_status_callback_messages)")
        }
    finally:
        conn.close()
    assert "task_status_bindings" in tables
    assert {
        "chat_id",
        "message_thread_id",
        "message_id",
        "session_id",
        "kanban_task_id",
        "linear_issue_key",
        "lifecycle_state",
        "state_version",
    } <= columns
    assert {
        "kanban_task_id",
        "state_version",
        "chat_id",
        "message_thread_id",
        "message_id",
    } <= callback_columns


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    kb.init_db()
    conn = kb.connect()
    try:
        task_id = kb.create_task(
            conn,
            title="Ship task status",
            assignee="coder",
            session_id="session-1",
        )
        yield conn, task_id
    finally:
        conn.close()


def test_board_fixture_ignores_inherited_live_board_pins(tmp_path, monkeypatch, request):
    live_db = tmp_path.parent / "registered-live-board" / "kanban.db"
    live_db.parent.mkdir()
    monkeypatch.setenv("HERMES_KANBAN_DB", str(live_db))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "registered-live-board")

    conn, _ = request.getfixturevalue("board")

    assert conn is not None
    resolved = kb.kanban_db_path()
    assert resolved == tmp_path / ".hermes" / "kanban.db"
    assert resolved.is_relative_to(tmp_path)


def _config():
    return {
        "enabled": True,
        "publisher_profile": "default",
        "dedup_seconds": 180,
        "routes": {
            "control": {"chat_id": "control-chat", "thread_id": "900"},
            "briefings": {"chat_id": "brief-chat", "thread_id": "901"},
            "system": {"chat_id": "system-chat", "thread_id": ""},
        },
    }


def test_composite_dedup_window_accepts_again_after_180_seconds(board):
    conn, task_id = board
    key = "same-composite-publication"

    first = kb.claim_task_status_publication(
        conn,
        dedup_key=key,
        kanban_task_id=task_id,
        window_seconds=180,
        now=1_000,
    )
    within_window = kb.claim_task_status_publication(
        conn,
        dedup_key=key,
        kanban_task_id=task_id,
        window_seconds=180,
        now=1_179,
    )
    after_window = kb.claim_task_status_publication(
        conn,
        dedup_key=key,
        kanban_task_id=task_id,
        window_seconds=180,
        now=1_181,
    )

    assert first is True
    assert within_window is False
    assert after_window is True


def test_destination_routes_are_exact_and_never_inferred():
    config = _config()
    origin = TaskStatusDestination("project-chat", "77")

    assert resolve_exact_task_status_destination(
        DeliveryRoute.CONTROL, origin=origin, config=config
    ) == TaskStatusDestination("control-chat", "900")
    assert resolve_exact_task_status_destination(
        DeliveryRoute.BRIEFINGS, origin=origin, config=config
    ) == TaskStatusDestination("brief-chat", "901")
    assert resolve_exact_task_status_destination(
        DeliveryRoute.PROJECT, origin=origin, config=config
    ) == origin
    assert resolve_exact_task_status_destination(
        DeliveryRoute.SYSTEM, origin=origin, config=config
    ) == TaskStatusDestination("system-chat", "")

    del config["routes"]["briefings"]
    with pytest.raises(ValueError, match="briefings destination"):
        resolve_exact_task_status_destination(
            DeliveryRoute.BRIEFINGS, origin=origin, config=config
        )


def test_structured_create_event_requires_exact_destination(board):
    conn, task_id = board

    with pytest.raises(ValueError, match="exact destination"):
        kb.append_task_status_publication(
            conn,
            task_id,
            {
                "operation": "create",
                "kanban_task_id": task_id,
                "session_id": "session-1",
                "lifecycle_state": "in_progress",
                "state_version": 1,
                "content": "Starting.",
            },
        )


@pytest.mark.asyncio
async def test_create_persists_exact_binding_and_routine_update_edits_same_message(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publisher = TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=_config(),
        profile_name="default",
    )

    created = await publisher.publish(
        TaskStatusPublication(
            operation="create",
            kanban_task_id=task_id,
            session_id="session-1",
            linear_issue_key="OPS-42",
            lifecycle_state="in_progress",
            state_version=1,
            content="Implementation is in progress.",
            kind=PublicationKind.ROUTINE,
            destination=TaskStatusDestination(
                chat_id="project-chat", message_thread_id="77"
            ),
        )
    )

    assert created.ok is True
    assert created.message_id == "100"
    assert len(adapter.sends) == 1
    assert adapter.sends[0]["metadata"] == {"thread_id": "77", "notify": False}
    binding = kb.get_task_status_binding(conn, task_id)
    assert binding is not None
    assert binding.chat_id == "project-chat"
    assert binding.message_thread_id == "77"
    assert binding.message_id == "100"
    assert binding.session_id == "session-1"
    assert binding.kanban_task_id == task_id
    assert binding.linear_issue_key == "OPS-42"
    assert binding.lifecycle_state == "in_progress"
    assert binding.state_version == 1

    updated = await publisher.publish(
        TaskStatusPublication(
            operation="update",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="qa_review",
            state_version=2,
            content="Independent QA is reviewing the implementation.",
            kind=PublicationKind.ROUTINE,
        )
    )

    assert updated.ok is True
    assert len(adapter.sends) == 1
    assert adapter.pushes == []
    assert adapter.edits == [
        {
            "chat_id": "project-chat",
            "message_id": "100",
            "content": "Independent QA is reviewing the implementation.",
            "buttons": [],
            "metadata": {"thread_id": "77", "notify": False},
        }
    ]
    binding = kb.get_task_status_binding(conn, task_id)
    assert binding.lifecycle_state == "qa_review"
    assert binding.state_version == 2


@pytest.mark.asyncio
async def test_duplicate_create_is_rejected_without_orphaning_another_message(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publisher = TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=_config(),
        profile_name="default",
    )
    publication = TaskStatusPublication(
        operation="create",
        kanban_task_id=task_id,
        session_id="session-1",
        lifecycle_state="in_progress",
        state_version=1,
        content="Starting.",
        kind=PublicationKind.ROUTINE,
        destination=TaskStatusDestination("project-chat", "77"),
    )

    first = await publisher.publish(publication)
    second = await publisher.publish(publication)

    assert first.ok is True
    assert second.ok is False
    assert "already bound" in second.error.lower()
    assert len(adapter.sends) == 1
    assert kb.get_task_status_binding(conn, task_id).message_id == "100"


async def _create_binding(conn, task_id, adapter):
    publisher = TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=_config(),
        profile_name="default",
    )
    result = await publisher.publish(
        TaskStatusPublication(
            operation="create",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="in_progress",
            state_version=1,
            content="Starting.",
            kind=PublicationKind.ROUTINE,
            destination=TaskStatusDestination("project-chat", "77"),
        )
    )
    assert result.ok is True
    return publisher


def _seed_task_status_correlation_rows(conn, task_id, adapter):
    asyncio.run(_create_binding(conn, task_id, adapter))
    assert kb.claim_task_status_publication(
        conn,
        dedup_key=f"delete-regression:{task_id}",
        kanban_task_id=task_id,
    )
    kb.register_task_status_callback_message(
        conn,
        kanban_task_id=task_id,
        state_version=1,
        platform="telegram",
        chat_id="project-chat",
        message_thread_id="77",
        message_id="callback-1",
    )
    assert kb.claim_task_status_callback(
        conn,
        kanban_task_id=task_id,
        state_version=1,
        action="approve",
    )


def _assert_no_task_status_correlation_rows(conn, task_id):
    task_columns = {
        "task_status_bindings": "kanban_task_id",
        "task_status_publication_dedup": "kanban_task_id",
        "task_status_callback_messages": "kanban_task_id",
        "task_status_callback_claims": "kanban_task_id",
    }
    for table, column in task_columns.items():
        count = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (task_id,)
        ).fetchone()[0]
        assert count == 0, f"{table} retained orphan rows for {task_id}"


def test_delete_task_removes_all_task_status_correlation_rows(board):
    conn, task_id = board
    _seed_task_status_correlation_rows(conn, task_id, FakeTelegramAdapter())

    assert kb.delete_task(conn, task_id) is True

    _assert_no_task_status_correlation_rows(conn, task_id)


def test_delete_archived_task_removes_all_task_status_correlation_rows(board):
    conn, task_id = board
    _seed_task_status_correlation_rows(conn, task_id, FakeTelegramAdapter())
    assert kb.archive_task(conn, task_id) is True

    assert kb.delete_archived_task(conn, task_id) is True

    _assert_no_task_status_correlation_rows(conn, task_id)


@pytest.mark.asyncio
async def test_structured_update_event_rejects_destination_mismatch(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    await _create_binding(conn, task_id, adapter)
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="project-chat",
        thread_id="77",
        notifier_profile="default",
    )

    with pytest.raises(ValueError, match="does not match"):
        kb.append_task_status_publication(
            conn,
            task_id,
            {
                "operation": "update",
                "kanban_task_id": task_id,
                "session_id": "session-1",
                "lifecycle_state": "qa_review",
                "state_version": 2,
                "content": "QA review.",
                "destination": {
                    "chat_id": "wrong-chat",
                    "message_thread_id": "99",
                },
            },
        )


@pytest.mark.asyncio
async def test_composite_dedup_suppresses_equivalent_update_and_allows_change(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publisher = await _create_binding(conn, task_id, adapter)

    qa = TaskStatusPublication(
        operation="update",
        kanban_task_id=task_id,
        session_id="session-1",
        lifecycle_state="qa_review",
        state_version=2,
        content="QA review.",
        kind=PublicationKind.ROUTINE,
        recommended_action="Wait for QA.",
    )
    first = await publisher.publish(qa)
    duplicate = await publisher.publish(qa)
    changed = await publisher.publish(
        TaskStatusPublication(
            operation="update",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="donna_review",
            state_version=3,
            content="Donna review.",
            kind=PublicationKind.ROUTINE,
            recommended_action="Wait for Donna.",
        )
    )
    changed_action = await publisher.publish(
        TaskStatusPublication(
            operation="update",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="donna_review",
            state_version=3,
            content="Donna review; recommendation changed.",
            kind=PublicationKind.ROUTINE,
            recommended_action="Approve the pilot.",
        )
    )

    assert first.ok is True
    assert duplicate.ok is True and duplicate.deduplicated is True
    assert changed.ok is True
    assert changed_action.ok is True
    assert len(adapter.edits) == 3


@pytest.mark.asyncio
async def test_decision_pushes_once_to_control_while_routine_never_pushes(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publisher = await _create_binding(conn, task_id, adapter)

    decision = TaskStatusPublication(
        operation="update",
        kanban_task_id=task_id,
        session_id="session-1",
        lifecycle_state="waiting_for_approval",
        state_version=2,
        content="A decision is required.",
        push_content="Approve or deny the protected pilot.",
        kind=PublicationKind.DECISION,
        recommended_action="Approve the protected pilot.",
        buttons=(
            TaskStatusButton("Approve", "approve"),
            TaskStatusButton("Deny", "deny"),
        ),
    )

    first = await publisher.publish(decision)
    duplicate = await publisher.publish(decision)

    assert first.ok is True and first.pushed is True
    assert duplicate.ok is True and duplicate.deduplicated is True
    assert len(adapter.pushes) == 1
    assert adapter.pushes[0]["chat_id"] == "control-chat"
    assert adapter.pushes[0]["metadata"] == {"thread_id": "900", "notify": True}
    callback_data = adapter.pushes[0]["buttons"][0]["callback_data"]
    assert callback_data == f"ts:approve:{task_id}:2"
    assert len(callback_data.encode("utf-8")) <= 64
    resolution = resolve_task_status_callback(
        conn,
        callback_data=callback_data,
        chat_id="control-chat",
        message_thread_id="900",
        message_id="push-1",
        allowed_actions=("approve", "deny"),
    )
    assert resolution.ok is True
    assert resolution.kanban_task_id == task_id
    assert resolution.state_version == 2


@pytest.mark.asyncio
async def test_accepted_completion_finalizes_card_and_pushes_to_origin_only(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publisher = await _create_binding(conn, task_id, adapter)

    result = await publisher.publish(
        TaskStatusPublication(
            operation="update",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="complete",
            state_version=2,
            content="The accepted outcome is complete.",
            push_content="Completed: the protected outcome was accepted.",
            kind=PublicationKind.ACCEPTED_COMPLETION,
        )
    )

    assert result.ok is True and result.pushed is True
    assert len(adapter.edits) == 1
    assert adapter.pushes == [
        {
            "chat_id": "project-chat",
            "content": "Completed: the protected outcome was accepted.",
            "buttons": [],
            "metadata": {"thread_id": "77", "notify": True},
        }
    ]


@pytest.mark.asyncio
async def test_missing_control_destination_fails_closed_before_transport(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    await _create_binding(conn, task_id, adapter)
    bad_config = _config()
    del bad_config["routes"]["control"]
    publisher = TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=bad_config,
        profile_name="default",
    )

    result = await publisher.publish(
        TaskStatusPublication(
            operation="update",
            kanban_task_id=task_id,
            session_id="session-1",
            lifecycle_state="blocked",
            state_version=2,
            content="Blocked.",
            push_content="A blocker needs attention.",
            kind=PublicationKind.BLOCKER,
        )
    )

    assert result.ok is False
    assert "control destination" in result.error
    assert adapter.edits == []
    assert adapter.pushes == []


@pytest.mark.asyncio
async def test_wrong_publisher_profile_and_missing_binding_fail_closed(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    publication = TaskStatusPublication(
        operation="update",
        kanban_task_id=task_id,
        session_id="session-1",
        lifecycle_state="qa_review",
        state_version=2,
        content="QA review.",
    )

    wrong_profile = await TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=_config(),
        profile_name="coder",
    ).publish(publication)
    missing_binding = await TaskStatusPublisher(
        conn=conn,
        adapter=adapter,
        config=_config(),
        profile_name="default",
    ).publish(publication)

    assert wrong_profile.ok is False
    assert "profile mismatch" in wrong_profile.error
    assert missing_binding.ok is False
    assert "no complete task-status binding" in missing_binding.error
    assert adapter.sends == [] and adapter.edits == [] and adapter.pushes == []


@pytest.mark.asyncio
async def test_callbacks_require_exact_task_message_topic_and_current_version(board):
    conn, task_id = board
    adapter = FakeTelegramAdapter()
    await _create_binding(conn, task_id, adapter)
    valid = make_task_status_callback_data(task_id, 1, "approve")

    stale = resolve_task_status_callback(
        conn,
        callback_data=make_task_status_callback_data(task_id, 2, "approve"),
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    wrong_message = resolve_task_status_callback(
        conn,
        callback_data=valid,
        chat_id="project-chat",
        message_thread_id="77",
        message_id="999",
        allowed_actions=("approve", "deny"),
    )
    wrong_topic = resolve_task_status_callback(
        conn,
        callback_data=valid,
        chat_id="project-chat",
        message_thread_id="other",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    wrong_task = resolve_task_status_callback(
        conn,
        callback_data=make_task_status_callback_data(
            "t_wrongtask", 1, "approve"
        ),
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    unknown_action = resolve_task_status_callback(
        conn,
        callback_data=make_task_status_callback_data(task_id, 1, "continue"),
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    bare = resolve_task_status_callback(
        conn,
        callback_data="approve",
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    bare_number = resolve_task_status_callback(
        conn,
        callback_data="1",
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    accepted = resolve_task_status_callback(
        conn,
        callback_data=valid,
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    replay = resolve_task_status_callback(
        conn,
        callback_data=valid,
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )
    conflicting_replay = resolve_task_status_callback(
        conn,
        callback_data=make_task_status_callback_data(task_id, 1, "deny"),
        chat_id="project-chat",
        message_thread_id="77",
        message_id="100",
        allowed_actions=("approve", "deny"),
    )

    assert stale.ok is False and "stale" in stale.error
    assert wrong_message.ok is False and "message" in wrong_message.error
    assert wrong_topic.ok is False and "topic" in wrong_topic.error
    assert wrong_task.ok is False and "no task-status binding" in wrong_task.error
    assert unknown_action.ok is False and "unknown" in unknown_action.error
    assert bare.ok is False and bare_number.ok is False
    assert accepted.ok is True
    assert accepted.kanban_task_id == task_id
    assert accepted.state_version == 1
    assert accepted.action == "approve"
    assert replay.ok is False and "already resolved" in replay.error
    assert conflicting_replay.ok is False and "already resolved" in conflicting_replay.error


@pytest.mark.asyncio
async def test_gateway_notifier_integration_uses_structured_events_and_silences_raw_events(
    board, monkeypatch
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    conn, task_id = board
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="project-chat",
        thread_id="77",
        notifier_profile="default",
    )
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="other-chat",
        thread_id="88",
        notifier_profile="default",
    )
    kb.append_task_status_publication(
        conn,
        task_id,
        {
            "operation": "create",
            "kanban_task_id": task_id,
            "session_id": "session-1",
            "lifecycle_state": "in_progress",
            "state_version": 1,
            "content": "Implementation is in progress.",
            "kind": "routine",
            "destination": {
                "chat_id": "project-chat",
                "message_thread_id": "77",
            },
        },
    )

    adapter = FakeTelegramAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "default"
    real_sleep = asyncio.sleep

    async def one_tick(delay):
        if delay == 5:
            return
        runner._running = False
        await real_sleep(0)

    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "task_status": _config(),
        }
    }
    with patch("hermes_cli.config.load_config", return_value=config), patch(
        "gateway.kanban_watchers.asyncio.sleep", side_effect=one_tick
    ):
        await runner._kanban_notifier_watcher(interval=1)

    assert len(adapter.sends) == 1
    assert kb.get_task_status_binding(conn, task_id).message_id == "100"

    # A raw blocked event and a structured routine update arrive together.
    # Living-card mode must suppress the raw terminal push and edit only.
    kb.block_task(conn, task_id, reason="wait", kind="needs_input")
    kb.append_task_status_publication(
        conn,
        task_id,
        {
            "operation": "update",
            "kanban_task_id": task_id,
            "session_id": "session-1",
            "lifecycle_state": "qa_review",
            "state_version": 2,
            "content": "QA review is in progress.",
            "kind": "routine",
        },
    )
    runner._running = True
    with patch("hermes_cli.config.load_config", return_value=config), patch(
        "gateway.kanban_watchers.asyncio.sleep", side_effect=one_tick
    ):
        await runner._kanban_notifier_watcher(interval=1)

    assert len(adapter.edits) == 1
    assert adapter.sends[0]["chat_id"] == "project-chat"
    assert adapter.pushes == []
    assert adapter.wakeups == []


@pytest.mark.asyncio
async def test_worker_completion_keeps_route_until_explicit_accepted_completion(
    board, monkeypatch
):
    from gateway.config import Platform
    from gateway.run import GatewayRunner

    conn, task_id = board
    kb.add_notify_sub(
        conn,
        task_id=task_id,
        platform="telegram",
        chat_id="project-chat",
        thread_id="77",
        notifier_profile="default",
    )
    kb.append_task_status_publication(
        conn,
        task_id,
        {
            "operation": "create",
            "kanban_task_id": task_id,
            "session_id": "session-1",
            "lifecycle_state": "in_progress",
            "state_version": 1,
            "content": "Implementation is in progress.",
            "kind": "routine",
            "destination": {
                "chat_id": "project-chat",
                "message_thread_id": "77",
            },
        },
    )

    adapter = FakeTelegramAdapter()
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._kanban_sub_fail_counts = {}
    runner._kanban_notifier_profile = "default"
    config = {
        "kanban": {
            "dispatch_in_gateway": True,
            "task_status": _config(),
        }
    }
    real_sleep = asyncio.sleep

    async def run_one_tick():
        async def one_tick(delay):
            if delay == 5:
                return
            runner._running = False
            await real_sleep(0)

        runner._running = True
        with patch("hermes_cli.config.load_config", return_value=config), patch(
            "gateway.kanban_watchers.asyncio.sleep", side_effect=one_tick
        ):
            await runner._kanban_notifier_watcher(interval=1)

    await run_one_tick()
    assert kb.get_task_status_binding(conn, task_id).message_id == "100"

    assert kb.complete_task(conn, task_id, summary="Worker artifact is ready") is True
    await run_one_tick()

    assert len(kb.list_notify_subs(conn, task_id=task_id)) == 1
    assert adapter.edits == []
    assert adapter.pushes == []

    kb.append_task_status_publication(
        conn,
        task_id,
        {
            "operation": "update",
            "kanban_task_id": task_id,
            "session_id": "session-1",
            "lifecycle_state": "complete",
            "state_version": 2,
            "content": "The accepted outcome is complete.",
            "push_content": "Completed: the protected outcome was accepted.",
            "kind": "accepted_completion",
        },
    )
    await run_one_tick()

    assert len(adapter.edits) == 1
    assert len(adapter.pushes) == 1
    assert kb.list_notify_subs(conn, task_id=task_id) == []
