"""Regression coverage for CLI async-delegation completion ownership."""

import queue

import pytest

from cli import HermesCLI


@pytest.mark.parametrize("event_type", ["async_delegation", "completion"])
def test_cli_completion_drain_uses_visible_session_identity(monkeypatch, event_type):
    """A CLI window must not claim another window's restored completion."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()

    event = {
        "type": event_type,
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
    queued = cli._pending_input.get_nowait()
    assert queued == "completion payload"
    expected_kind = (
        "async_delegation_complete" if event_type == "async_delegation" else None
    )
    assert getattr(queued, "display_kind", None) == expected_kind
    if expected_kind:
        assert (
            getattr(queued, "display_metadata")["delegation_id"]
            == event["delegation_id"]
        )
    assert claimed == [(event, "cli-idle")]
    assert completed == [(event, "claim-token")]


def test_cli_completion_ownership_rejects_foreign_session():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = None

    assert not cli._owns_process_notification({
        "type": "async_delegation",
        "session_key": "foreign-session",
    })


def test_cli_completion_ownership_accepts_compression_lineage():
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"

    class FakeSessionDB:
        def resolve_resume_session_id(self, session_id):
            assert session_id == "pre-compression-session"
            return "visible-session"

    cli._session_db = FakeSessionDB()

    assert cli._owns_process_notification({
        "type": "async_delegation",
        "session_key": "pre-compression-session",
    })
