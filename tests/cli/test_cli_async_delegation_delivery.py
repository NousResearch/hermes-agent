"""Regression coverage for CLI async-delegation completion ownership."""

import queue
from types import SimpleNamespace

from cli import HermesCLI
from hermes_cli.cli_process_notifications import _ProcessNotificationBatch


def test_cli_completion_drain_uses_visible_session_identity(monkeypatch):
    """A CLI window must not claim another window's restored completion."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_visible",
        "session_key": "visible-session",
    }
    calls = []

    class FakeRegistry:
        def drain_notifications(self, *, session_key="", owns_event=None):
            calls.append((session_key, owns_event(event)))
            return [(event, "completion payload")]

    claimed = []
    completed = []

    monkeypatch.setattr(
        "tools.process_registry.process_registry",
        FakeRegistry(),
    )
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda evt, consumer: claimed.append((evt, consumer)) or "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )

    cli._drain_process_notifications("cli-idle")

    assert calls == [("visible-session", True)]
    assert cli._pending_input.get_nowait() == "completion payload"
    assert claimed == [(event, "cli-idle")]
    assert completed == [(event, "claim-token")]


def test_cli_completion_drain_batches_owned_backlog_into_one_turn(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    events = [
        {
            "type": "completion",
            "session_id": f"proc_{index}",
            "session_key": "visible-session",
        }
        for index in range(12)
    ]

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, f"completion-{index}") for index, event in enumerate(events)]

        @staticmethod
        def is_completion_consumed(_session_id):
            return False

    registry = FakeRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda _event, _consumer: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda _event, _claim: None)

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 1
    batch = cli._pending_input.get_nowait()
    assert isinstance(batch, _ProcessNotificationBatch)
    rendered = batch.render(registry)
    assert rendered is not None
    assert "12 background notifications" in rendered
    assert rendered.index("completion-0") < rendered.index("completion-11")

    turns = []
    cli._pending_resume_sessions = []
    cli._typed_voice_stop = lambda _text: False
    cli.handle_bang_shell = lambda _text: False
    cli._print_user_message_preview = lambda _text: None
    cli._turn_summary_begin = lambda: None
    cli._app = SimpleNamespace(invalidate=lambda: None)
    cli.chat = lambda text, **_kwargs: turns.append(text)
    cli._tui_after_turn = lambda: None
    cli._tui_process_one_input(batch)

    assert turns == [rendered]


def test_cli_completion_batch_keeps_async_delegation_individual(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    delegation = {
        "type": "async_delegation",
        "delegation_id": "deleg_individual",
        "session_key": "visible-session",
    }
    completions = [
        {"type": "completion", "session_id": f"proc_{index}", "session_key": "visible-session"}
        for index in range(2)
    ]

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [
                (delegation, "delegation payload"),
                (completions[0], "completion-0"),
                (completions[1], "completion-1"),
            ]

        @staticmethod
        def is_completion_consumed(_session_id):
            return False

    registry = FakeRegistry()
    monkeypatch.setattr("tools.process_registry.process_registry", registry)
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda _event, _consumer: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda _event, _claim: None)

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.qsize() == 2
    assert cli._pending_input.get_nowait() == "delegation payload"
    batch = cli._pending_input.get_nowait()
    assert isinstance(batch, _ProcessNotificationBatch)
    rendered = batch.render(registry)
    assert rendered is not None
    assert "completion-0" in rendered and "completion-1" in rendered


def test_cli_completion_batch_rechecks_consumed_events_before_turn():
    from tools.process_registry import ProcessRegistry

    stale = {"type": "completion", "session_id": "proc_stale"}
    live = {"type": "completion", "session_id": "proc_live"}
    registry = ProcessRegistry()
    registry._completion_consumed.add("proc_stale")
    batch = _ProcessNotificationBatch(((stale, "stale completion"), (live, "live completion")))

    try:
        assert batch.render(registry) == "live completion"
        registry._completion_consumed.add("proc_live")
        assert batch.render(registry) is None
    finally:
        registry._completion_consumed.difference_update({"proc_stale", "proc_live"})


def test_cli_completion_ownership_rejects_foreign_session():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None

    assert not cli._owns_process_notification(
        {"type": "async_delegation", "session_key": "foreign-session"}
    )


def test_cli_completion_ownership_accepts_compression_lineage():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            assert session_id == "pre-compression-session"
            return "visible-session"

    cli._session_db = FakeSessionDB()

    assert cli._owns_process_notification(
        {
            "type": "async_delegation",
            "session_key": "pre-compression-session",
        }
    )
