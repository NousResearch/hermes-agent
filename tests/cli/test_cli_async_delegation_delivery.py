"""Regression coverage for CLI async-delegation completion ownership."""

import queue
import threading

import cli as cli_mod
from cli import HermesCLI
from tools import async_delegation as ad
from tools.process_registry import process_registry


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


def test_oneshot_drain_waits_for_owned_delegation_and_runs_completion(monkeypatch):
    """A one-shot parent must consume the required child result before exit."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "parent-session"
    cli._pending_input = queue.Queue()

    live_checks = iter([True, True, False, False])
    selector_calls = []
    drain_calls = []
    responses = []

    def has_live_for_session(**selectors):
        selector_calls.append(selectors)
        return next(live_checks)

    def drain(consumer):
        drain_calls.append(consumer)
        if len(drain_calls) == 2:
            cli._pending_input.put("delegation completion")

    monkeypatch.setattr(
        "tools.async_delegation.has_live_for_session",
        has_live_for_session,
    )
    monkeypatch.setattr(cli, "_drain_process_notifications", drain)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda _seconds: None)

    count = cli_mod._drain_oneshot_async_delegations(
        cli,
        run_turn=lambda message: responses.append(message),
    )

    assert count == 1
    assert responses == ["delegation completion"]
    assert drain_calls == ["cli-one-shot", "cli-one-shot", "cli-one-shot"]
    assert selector_calls
    assert all(
        call == {"parent_session_id": "parent-session"}
        for call in selector_calls
    )


def test_oneshot_drain_is_noop_without_owned_delegations(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "parent-session"
    cli._pending_input = queue.Queue()
    turns = []

    monkeypatch.setattr(
        "tools.async_delegation.has_live_for_session",
        lambda **_selectors: False,
    )
    monkeypatch.setattr(cli, "_drain_process_notifications", lambda _consumer: None)

    assert cli_mod._drain_oneshot_async_delegations(
        cli,
        run_turn=turns.append,
    ) == 0
    assert turns == []


def test_oneshot_drain_real_dispatch_delivers_before_return():
    """Exercise dispatch -> completion queue -> CLI synthetic follow-up end to end."""
    ad._reset_for_tests()
    while not process_registry.completion_queue.empty():
        process_registry.completion_queue.get_nowait()

    gate = threading.Event()
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "parent-session"
    cli._session_db = None
    cli._pending_input = queue.Queue()
    turns = []

    def runner():
        gate.wait(timeout=5)
        return {
            "status": "completed",
            "summary": "review verdict",
            "api_calls": 1,
            "duration_seconds": 0.01,
            "model": "test-model",
        }

    dispatched = ad.dispatch_async_delegation(
        goal="review candidate",
        context=None,
        toolsets=None,
        role="leaf",
        model="test-model",
        session_key="parent-session",
        parent_session_id="parent-session",
        runner=runner,
        max_async_children=1,
    )
    timer = threading.Timer(0.05, gate.set)
    assert dispatched["status"] == "dispatched"
    timer.start()
    try:
        assert cli_mod._drain_oneshot_async_delegations(
            cli,
            run_turn=turns.append,
        ) == 1
        assert len(turns) == 1
        assert "ASYNC DELEGATION" in turns[0]
        assert "review verdict" in turns[0]
        assert ad.active_count() == 0
    finally:
        gate.set()
        timer.cancel()
        ad._reset_for_tests()
        while not process_registry.completion_queue.empty():
            process_registry.completion_queue.get_nowait()


def test_oneshot_drain_keeps_pre_rotation_parent_identity(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "post-rotation-session"
    cli._pending_input = queue.Queue()
    drain_calls = []
    selector_calls = []
    responses = []

    def has_live_for_session(**selectors):
        selector_calls.append(selectors)
        return (
            selectors.get("parent_session_id") == "pre-rotation-session"
            and len(drain_calls) < 2
        )

    def drain(_consumer):
        drain_calls.append(True)
        if len(drain_calls) == 2:
            cli._pending_input.put("pre-rotation completion")

    monkeypatch.setattr(
        "tools.async_delegation.has_live_for_session",
        has_live_for_session,
    )
    monkeypatch.setattr(cli, "_drain_process_notifications", drain)
    monkeypatch.setattr(cli_mod.time, "sleep", lambda _seconds: None)

    processed = cli_mod._drain_oneshot_async_delegations(
        cli,
        run_turn=responses.append,
        owned_session_ids={"pre-rotation-session"},
    )

    assert processed == 1
    assert responses == ["pre-rotation completion"]
    assert {"parent_session_id": "pre-rotation-session"} in selector_calls
