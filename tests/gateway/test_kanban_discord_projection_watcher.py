from __future__ import annotations

from unittest.mock import patch

from gateway.run import GatewayRunner
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_discord_projection as kdp


def _runner(send=None):
    runner = GatewayRunner.__new__(GatewayRunner)
    runner._running = True
    if send is not None:
        runner._kanban_discord_projection_send = send
    return runner


def _cfg(enabled=True, per_tick=1):
    return {
        "kanban": {
            "discord_projection_outbox_drain_mock": enabled,
            "discord_projection_outbox_drain_per_tick": per_tick,
        }
    }


def _linked_status_event():
    conn = kb.connect()
    try:
        issue_id = kb.create_task(conn, title="gateway projection", assignee="worker")
        kdp.link_parent_thread(
            conn,
            issue_id=issue_id,
            guild_id="guild-1",
            channel_id="channel-1",
            thread_id="thread-parent-1",
            idempotency_key="link:gateway-projection",
        )
        kdp.record_status_change(
            conn,
            issue_id=issue_id,
            from_status="ready",
            to_status="running",
            actor="developer",
            reason="mock drain",
            sequence=42,
        )
        return issue_id
    finally:
        conn.close()


def _outbox_rows():
    conn = kb.connect()
    try:
        return kdp.list_outbox(conn)
    finally:
        conn.close()


def test_discord_projection_drain_is_disabled_by_default():
    runner = _runner(send=lambda *_: "msg-never")
    with patch("hermes_cli.config.load_config", return_value={"kanban": {}}):
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            assert runner._kanban_discord_projection_drain_tick() == []
    mock_connect.assert_not_called()


def test_discord_projection_drain_requires_injected_mock_sender():
    runner = _runner()
    with patch("hermes_cli.config.load_config", return_value=_cfg(True)):
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            assert runner._kanban_discord_projection_drain_tick() == []
    mock_connect.assert_not_called()


def test_mocked_discord_projection_drain_delivers_one_outbox_event_with_parent_thread_metadata():
    issue_id = _linked_status_event()
    sent = []

    def fake_send(thread_id: str, content: str, idempotency_key: str) -> str:
        sent.append((thread_id, content, idempotency_key))
        return "discord-message-1"

    runner = _runner(send=fake_send)
    with patch("hermes_cli.config.load_config", return_value=_cfg(True, per_tick=1)):
        results = runner._kanban_discord_projection_drain_tick()

    assert len(results) == 1
    assert results[0].delivered is True
    assert sent == [
        (
            "thread-parent-1",
            "Kanban status changed: ready -> running\nActor: developer\nReason: mock drain",
            f"kanban-status:{issue_id}:42:ready->running",
        )
    ]
    rows = _outbox_rows()
    assert len(rows) == 1
    assert rows[0].delivery_status == "delivered"
    assert rows[0].parent_thread_id == "thread-parent-1"

    # Idempotency: a second tick sees no pending event and performs no send.
    with patch("hermes_cli.config.load_config", return_value=_cfg(True, per_tick=1)):
        again = runner._kanban_discord_projection_drain_tick()
    assert again == []
    assert len(sent) == 1


def test_mocked_discord_projection_failure_is_retryable_and_does_not_mutate_task_or_archive():
    issue_id = _linked_status_event()
    attempts = []

    def failing_send(thread_id: str, content: str, idempotency_key: str) -> str:
        attempts.append((thread_id, idempotency_key))
        raise RuntimeError("mock discord down")

    runner = _runner(send=failing_send)
    with patch("hermes_cli.config.load_config", return_value=_cfg(True, per_tick=1)):
        results = runner._kanban_discord_projection_drain_tick()

    assert len(results) == 1
    assert results[0].delivered is False
    assert results[0].error == "mock discord down"
    assert attempts == [("thread-parent-1", f"kanban-status:{issue_id}:42:ready->running")]

    conn = kb.connect()
    try:
        task = kb.get_task(conn, issue_id)
        assert task is not None
        assert task.status == "ready"
        row = conn.execute(
            "SELECT delivery_status, attempts, last_error FROM discord_projection_outbox WHERE issue_id = ?",
            (issue_id,),
        ).fetchone()
        assert row["delivery_status"] == "failed"
        assert row["attempts"] == 1
        assert row["last_error"] == "mock discord down"
        audits = conn.execute("SELECT COUNT(*) AS n FROM discord_close_audit").fetchone()["n"]
        assert audits == 0
    finally:
        conn.close()


def test_mocked_discord_projection_drain_dedupes_board_aliases_to_same_db(tmp_path, monkeypatch):
    db_path = tmp_path / "shared-projection.db"
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))
    kb.init_db()
    kb.write_board_metadata("alias-a", name="Alias A")
    kb.write_board_metadata("alias-b", name="Alias B")
    _linked_status_event()

    sent = []

    def fake_send(thread_id: str, content: str, idempotency_key: str) -> str:
        sent.append(idempotency_key)
        return "discord-message-1"

    runner = _runner(send=fake_send)
    with patch("hermes_cli.config.load_config", return_value=_cfg(True, per_tick=3)):
        results = runner._kanban_discord_projection_drain_tick()

    assert len(results) == 1
    assert len(sent) == 1

def test_discord_projection_drain_string_false_is_disabled():
    runner = _runner(send=lambda *_: "msg-never")
    with patch(
        "hermes_cli.config.load_config",
        return_value=_cfg("false", per_tick=1),
    ):
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            assert runner._kanban_discord_projection_drain_tick() == []
    mock_connect.assert_not_called()


def test_gateway_startup_wires_discord_projection_watcher_into_lifecycle():
    source = GatewayRunner.start.__code__.co_names
    assert "_kanban_discord_projection_watcher" in source

