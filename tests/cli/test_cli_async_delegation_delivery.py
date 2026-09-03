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
    monkeypatch.setattr("cli._cli_visible_print", lambda _text: None)

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.empty()
    assert persisted == [(
        ("visible-session", "user"),
        {"content": "completion payload", "display_kind": "async_delegation_complete",
         "display_metadata": {"delegation_id": "deleg"}},
    )]
    # The immediate next authorized turn reads this live list, not SQLite.
    assert cli.conversation_history[-1] == {
        "role": "user",
        "content": "completion payload",
        "display_kind": "async_delegation_complete",
        "display_metadata": {"delegation_id": "deleg"},
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
    assert not hasattr(cli, "conversation_history")
    assert cli._pending_input.empty()


def test_cli_stale_replay_after_crash_does_not_duplicate_durable_or_live_history(
    monkeypatch, tmp_path,
):
    """A crash after durable append but before ack replays the same identity once."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("visible-session", source="cli")
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-crash-window",
        "session_key": "visible-session",
    }
    # Simulate the first process committing the row and dying before its ack.
    db.append_async_delegation_completion(
        "visible-session", "completion payload", {"delegation_id": "deleg-crash-window"},
    )

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = db
    cli._pending_input = queue.Queue()
    cli.conversation_history = db.get_messages_as_conversation("visible-session")
    delivered = []
    visible = []

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    # This token represents the stale-claim reclaim after restart.
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: "stale-claim")
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda *_args: delivered.append(_args),
    )
    monkeypatch.setattr("cli._cli_visible_print", visible.append)

    cli._drain_process_notifications("cli-idle")

    rows = db.get_messages("visible-session")
    assert len(rows) == 1
    assert sum(
        message.get("display_metadata", {}).get("delegation_id") == "deleg-crash-window"
        for message in cli.conversation_history
    ) == 1
    assert cli._pending_input.empty()
    # The durable row predates this replay, so it was already accepted for
    # display. A reclaimed stale claim still gets acknowledged, but does not
    # print the completion again.
    assert visible == []
    assert delivered == [(event, "stale-claim")]


def test_cli_ack_failure_releases_claim_then_replay_acks_without_reprinting(monkeypatch, tmp_path):
    """Ack failures retry immediately without duplicating durable/live/visible delivery."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("visible-session", source="cli")
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-ack-retry",
        "session_key": "visible-session",
    }
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._session_db = db
    cli._pending_input = queue.Queue()
    cli.conversation_history = []
    visible = []
    released = []
    acknowledged = []
    claims = iter(("first-claim", "retry-claim"))

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    def complete(_event, claim):
        if claim == "first-claim":
            raise OSError("ack unavailable")
        acknowledged.append((_event, claim))

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: next(claims))
    monkeypatch.setattr("tools.async_delegation.release_event_delivery", lambda *args: released.append(args))
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", complete)
    monkeypatch.setattr("cli._cli_visible_print", visible.append)

    cli._drain_process_notifications("cli-idle")
    cli._drain_process_notifications("cli-idle")

    assert len(db.get_messages("visible-session")) == 1
    assert sum(
        message.get("display_metadata", {}).get("delegation_id") == "deleg-ack-retry"
        for message in cli.conversation_history
    ) == 1
    assert visible == ["completion payload"]
    assert released == [(event, "first-claim")]
    assert acknowledged == [(event, "retry-claim")]


def test_cli_releases_claim_when_durable_delivery_fails(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    cli.conversation_history = []
    event = {"type": "async_delegation", "delegation_id": "deleg", "session_key": "visible-session"}
    released = []

    class FakeRegistry:
        def drain_notifications(self, **_kwargs):
            return [(event, "completion payload")]

    class FailingDb:
        def append_message(self, *_args, **_kwargs):
            raise AssertionError("legacy append must not run")

        def append_async_delegation_completion(self, *_args):
            raise OSError("database busy")

    cli.__dict__["_session_db"] = FailingDb()
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_args: "claim")
    monkeypatch.setattr("tools.async_delegation.release_event_delivery", lambda *args: released.append(args))
    acknowledged = []
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda *args: acknowledged.append(args),
    )

    cli._drain_process_notifications("cli-idle")

    assert released == [(event, "claim")]
    assert acknowledged == []
    assert cli.conversation_history == []
    assert cli._pending_input.empty()


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
