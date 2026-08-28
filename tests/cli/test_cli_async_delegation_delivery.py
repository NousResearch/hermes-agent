"""Regression coverage for CLI async-delegation completion ownership."""

import queue

from cli import (
    HermesCLI,
    _InternalContinuation,
    _retry_failed_closeout_continuation,
)


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


def test_cli_closeout_drain_retains_exact_typed_identity(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli._internal_continuations = queue.Queue()
    event = {
        "type": "async_delegation_work_closeout",
        "session_key": "visible-session",
        "origin_work_id": "work-a",
        "work_generation": 7,
        "delivery_id": "delivery-a",
        "claim_id": "claim-a",
    }

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "aggregate text")]

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_a: "")

    cli._drain_process_notifications("cli-idle")

    item = cli._internal_continuations.get_nowait()
    assert item == (
        "aggregate text", "work-a", 7, "delivery-a", "claim-a"
    )
    assert cli._pending_input.empty()


def test_failed_cli_closeout_handoff_releases_and_requeues(monkeypatch):
    calls = []
    replacement_queue = queue.Queue()

    monkeypatch.setattr(
        "tools.async_delegation.release_enqueued_work_group_event",
        lambda event: calls.append(("release", event.copy())) or True,
    )
    monkeypatch.setattr(
        "tools.async_delegation.recover_and_enqueue_work_groups",
        lambda **kwargs: calls.append(("recover", kwargs)) or [{"replacement": True}],
    )
    monkeypatch.setattr(
        "tools.process_registry.process_registry.completion_queue",
        replacement_queue,
    )

    item = _InternalContinuation("closeout", "work", 3, "delivery", "claim")
    assert _retry_failed_closeout_continuation(item)
    release_calls = [call for call in calls if call[0] == "release"]
    assert release_calls == [(
        "release",
        {
            "type": "async_delegation_work_closeout",
            "origin_work_id": "work",
            "work_generation": 3,
            "delivery_id": "delivery",
            "claim_id": "claim",
        },
    )]
    retry_calls = [
        call for call in calls
        if call[0] == "recover" and call[1].get("consumer") == "cli-closeout-retry"
    ]
    assert len(retry_calls) == 1
    assert retry_calls[0][1]["target_queue"] is replacement_queue
