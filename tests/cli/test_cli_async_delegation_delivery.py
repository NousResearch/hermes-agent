"""Regression coverage for CLI async-delegation completion ownership."""

import queue

from cli import HermesCLI


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


def test_cli_completion_is_durable_display_only_not_a_recursive_turn(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli.conversation_history = [{"role": "user", "content": "earlier turn"}]
    persisted = []
    cli.__dict__["_session_db"] = type("DB", (), {
        "append_message": lambda _self, *args, **kwargs: persisted.append((args, kwargs)),
    })()
    event = {"type": "async_delegation", "delegation_id": "deleg", "session_key": "visible-session"}

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: "claim")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *_args: None)

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.empty()
    assert persisted == [(
        ("visible-session", "user"),
        {"content": "completion payload", "display_kind": "async_delegation_complete"},
    )]
    # The immediate next authorized turn reads this live list, not SQLite.
    assert cli.conversation_history[-1] == {
        "role": "user",
        "content": "completion payload",
        "display_kind": "async_delegation_complete",
        "_db_persisted": True,
    }


def test_cli_duplicate_completion_is_persisted_once(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    persisted = []
    cli._session_db = type("DB", (), {
        "append_message": lambda _self, *args, **kwargs: persisted.append((args, kwargs)),
    })()
    event = {"type": "async_delegation", "delegation_id": "deleg", "session_key": "visible-session"}

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload"), (event, "completion payload")]

    claims = iter(("claim", None))
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: next(claims))
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *_args: None)

    cli._drain_process_notifications("cli-idle")

    assert len(persisted) == 1


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
