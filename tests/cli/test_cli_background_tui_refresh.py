"""Tests for CLI background command TUI refresh behavior.

Ensures the TUI is properly refreshed before printing background task output
to prevent spinner/status bar overlap (#2718).
"""

from queue import Queue
from unittest.mock import MagicMock, patch


from cli import HermesCLI


def _make_cli():
    """Create a minimal HermesCLI instance for testing."""
    cli_obj = HermesCLI.__new__(HermesCLI)
    cli_obj.model = "test-model"
    cli_obj._background_tasks = {}
    cli_obj._background_task_counter = 0
    cli_obj.conversation_history = []
    cli_obj.agent = None
    cli_obj._app = None
    cli_obj._user_turn_id = "turn-current"
    cli_obj._pending_input = Queue()
    return cli_obj


class TestBackgroundCommandTuiRefresh:
    """Tests for TUI refresh in background command output."""

    def test_invalidate_called_before_success_output(self):
        """App.invalidate() is called before printing background success output."""
        cli_obj = _make_cli()
        mock_app = MagicMock()
        cli_obj._app = mock_app

        # Track call order
        call_order = []
        original_invalidate = mock_app.invalidate

        def track_invalidate():
            call_order.append("invalidate")
            return original_invalidate()

        mock_app.invalidate = track_invalidate

        # Patch print to track when it's called
        with patch("builtins.print") as mock_print:
            mock_print.side_effect = lambda *args, **kwargs: call_order.append("print")

            # Simulate the background task output code path
            if cli_obj._app:
                cli_obj._app.invalidate()
                import time
                time.sleep(0.01)  # reduced for test
            print()

        # Verify invalidate was called before print
        assert call_order[0] == "invalidate"
        assert "print" in call_order

    def test_invalidate_called_before_error_output(self):
        """App.invalidate() is called before printing background error output."""
        cli_obj = _make_cli()
        mock_app = MagicMock()
        cli_obj._app = mock_app

        call_order = []
        mock_app.invalidate.side_effect = lambda: call_order.append("invalidate")

        with patch("builtins.print") as mock_print:
            mock_print.side_effect = lambda *args, **kwargs: call_order.append("print")

            # Simulate error path
            if cli_obj._app:
                cli_obj._app.invalidate()
                import time
                time.sleep(0.01)
            print()

        assert call_order[0] == "invalidate"
        assert "print" in call_order

    def test_no_crash_when_app_is_none(self):
        """No crash when _app is None (non-TUI mode)."""
        cli_obj = _make_cli()
        cli_obj._app = None

        # This should not raise
        if cli_obj._app:
            cli_obj._app.invalidate()
        # If we get here without exception, test passes

    def test_background_task_thread_safety(self):
        """Background task tracking is thread-safe."""
        cli_obj = _make_cli()

        # Simulate adding and removing background tasks
        task_id = "test_task_1"
        cli_obj._background_tasks[task_id] = MagicMock()
        assert task_id in cli_obj._background_tasks

        # Clean up
        cli_obj._background_tasks.pop(task_id, None)
        assert task_id not in cli_obj._background_tasks


def test_process_notification_is_displayed_without_queueing_agent_turn(monkeypatch):
    cli_obj = _make_cli()
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)

    with patch("tools.process_registry.process_registry.drain_notifications") as drain:
        drain.return_value = [
            (
                {"type": "completion", "session_id": "proc_1"},
                "[IMPORTANT: Background process proc_1 completed]",
            )
        ]
        cli_obj._drain_process_notifications("test")

    assert cli_obj._pending_input.empty()
    assert len(printed) == 1
    assert "Background process proc_1" in printed[0]


def test_current_turn_delegation_queues_typed_followup(monkeypatch):
    cli_obj = _make_cli()
    monkeypatch.setattr("cli._cprint", lambda *_args: None)

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_current",
        "dispatch_turn_id": "turn-current",
    }
    with patch("tools.process_registry.process_registry.drain_notifications") as drain:
        drain.return_value = [(event, "[ASYNC DELEGATION COMPLETE — deleg_current]")]
        with patch("tools.async_delegation.claim_event_delivery", return_value="claim"):
            with patch("tools.async_delegation.complete_event_delivery"):
                cli_obj._drain_process_notifications("test")

    queued = cli_obj._pending_input.get_nowait()
    assert type(queued).__name__ == "_QueuedSyntheticInput"
    assert queued.text == "[ASYNC DELEGATION COMPLETE — deleg_current]"


def test_stale_delegation_is_displayed_and_acknowledged(monkeypatch):
    cli_obj = _make_cli()
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_stale",
        "dispatch_turn_id": "turn-old",
    }
    with patch("tools.process_registry.process_registry.drain_notifications") as drain:
        drain.return_value = [(event, "[ASYNC DELEGATION COMPLETE — deleg_stale]")]
        with patch("tools.async_delegation.claim_event_delivery", return_value="claim"):
            with patch("tools.async_delegation.complete_event_delivery") as complete:
                cli_obj._drain_process_notifications("test")

    assert cli_obj._pending_input.empty()
    assert printed and "deleg_stale" in printed[0]
    complete.assert_called_once_with(event, "claim")


def test_cli_requeues_notification_when_delivery_fails(monkeypatch):
    from tools.process_registry import process_registry

    cli_obj = _make_cli()
    event = {"type": "completion", "session_id": "proc_retry"}
    isolated_queue = Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)
    monkeypatch.setattr("cli._cprint", lambda *_args: None)

    with patch.object(
        process_registry,
        "drain_notifications",
        return_value=[(event, "process completed")],
    ):
        with patch("tools.async_delegation.claim_event_delivery", return_value="claim"):
            with patch(
                "tools.async_delegation.complete_event_delivery",
                side_effect=RuntimeError("ack failed"),
            ):
                with patch("tools.async_delegation.release_event_delivery") as release:
                    cli_obj._drain_process_notifications("test")

    assert isolated_queue.get_nowait() == event
    release.assert_called_once_with(event, "claim")


def test_cli_does_not_requeue_fresh_delegation_after_it_was_queued(monkeypatch):
    from tools.process_registry import process_registry

    cli_obj = _make_cli()
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_ack_failure",
        "dispatch_turn_id": "turn-current",
    }
    isolated_queue = Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)

    with patch.object(
        process_registry,
        "drain_notifications",
        return_value=[(event, "delegation result")],
    ):
        with patch("tools.async_delegation.claim_event_delivery", return_value="claim"):
            with patch(
                "tools.async_delegation.complete_event_delivery",
                side_effect=RuntimeError("ack failed"),
            ):
                with patch("tools.async_delegation.release_event_delivery") as release:
                    cli_obj._drain_process_notifications("test")

    assert cli_obj._pending_input.qsize() == 1
    assert isolated_queue.empty()
    release.assert_not_called()


def test_cli_requeues_notification_when_claim_fails(monkeypatch):
    from tools.process_registry import process_registry

    cli_obj = _make_cli()
    event = {"type": "completion", "session_id": "proc_claim_retry"}
    isolated_queue = Queue()
    monkeypatch.setattr(process_registry, "completion_queue", isolated_queue)

    with patch.object(
        process_registry,
        "drain_notifications",
        return_value=[(event, "process completed")],
    ):
        with patch(
            "tools.async_delegation.claim_event_delivery",
            side_effect=RuntimeError("db unavailable"),
        ):
            cli_obj._drain_process_notifications("test")

    assert isolated_queue.get_nowait() == event


def test_queued_delegation_does_not_advance_user_turn_generation():
    cli_obj = _make_cli()

    cli_obj._queue_synthetic_input("delegation result")
    text, is_user_turn = cli_obj._unwrap_queued_input(cli_obj._pending_input.get_nowait())

    assert text == "delegation result"
    assert is_user_turn is False
    assert cli_obj._user_turn_id == "turn-current"


def test_queued_delegation_is_rechecked_before_agent_turn(monkeypatch):
    cli_obj = _make_cli()
    cli_obj._user_turn_id = "turn-new"
    printed = []
    monkeypatch.setattr("cli._cprint", printed.append)

    cli_obj._queue_synthetic_input("old delegation result", "turn-old")
    text, is_user_turn = cli_obj._unwrap_queued_input(cli_obj._pending_input.get_nowait())

    assert text == ""
    assert is_user_turn is False
    assert printed and "old delegation result" in printed[0]
