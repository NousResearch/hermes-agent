"""Regression coverage for CLI async-delegation completion ownership."""

import queue
from unittest.mock import MagicMock

import pytest
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


def test_cli_goal_owned_intermediate_completion_is_acked_without_parent_turn(
    monkeypatch,
):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    event = {
        "type": "async_delegation",
        "delegation_id": "deleg_joining",
        "session_key": "visible-session",
        "goal_id": "goal-id",
        "requires_goal_join": True,
    }

    class FakeRegistry:
        def drain_notifications(self, *, session_key="", owns_event=None):
            assert session_key == "visible-session"
            assert owns_event is not None
            assert owns_event(event)
            return [(event, "legacy completion payload")]

    completed = []
    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr(
        "tools.async_delegation.claim_event_delivery",
        lambda _evt, _consumer: "claim-token",
    )
    monkeypatch.setattr(
        "tools.async_delegation.complete_event_delivery",
        lambda evt, token: completed.append((evt, token)),
    )
    monkeypatch.setattr(
        "hermes_cli.goals.prepare_goal_delegation_delivery",
        lambda *_args, **_kwargs: {
            "classification": "current",
            "prompt": None,
            "status_message": "waiting for one more",
        },
    )

    cli._drain_process_notifications("cli-idle")

    assert cli._pending_input.empty()
    assert completed == [(event, "claim-token")]


def test_cli_goal_owned_claim_is_enqueued_as_internal_envelope(monkeypatch):
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = queue.Queue()
    event = {"type": "async_delegation", "delegation_id": "deleg_join"}

    class FakeRegistry:
        @staticmethod
        def drain_notifications(**_kwargs):
            return [(event, "legacy completion")]

    monkeypatch.setattr("tools.process_registry.process_registry", FakeRegistry())
    monkeypatch.setattr("tools.async_delegation.claim_event_delivery", lambda *_a, **_k: "transport")
    monkeypatch.setattr("tools.async_delegation.complete_event_delivery", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "hermes_cli.goals.prepare_goal_delegation_delivery",
        lambda *_a, **_k: {
            "classification": "current",
            "prompt": "reconcile this batch",
            "status_message": "",
            "reconciliation_claim": "recon-1",
            "reconciliation_attempt": 2,
            "goal_id": "goal-1",
            "goal_session_id": "visible-session",
        },
    )

    cli._drain_process_notifications(consumer="cli-test")

    envelope = cli._pending_input.get_nowait()
    assert envelope["text"] == "reconcile this batch"
    assert envelope["goal_reconciliation"] == {
        "claim_id": "recon-1",
        "goal_id": "goal-1",
        "session_id": "visible-session",
        "attempt": 2,
    }


def test_cli_enqueue_failure_releases_transport_and_reconciliation(monkeypatch):
    from hermes_cli import goals
    from tools import async_delegation as ad

    event = {
        "type": "async_delegation",
        "delegation_id": "deleg-owned-fail",
        "session_key": "visible-session",
        "goal_id": "goal-id",
        "requires_goal_join": True,
        "status": "completed",
        "task": {"goal": "work"},
    }

    class _Registry:
        completion_queue = queue.Queue()

        @staticmethod
        def drain_notifications(**_kwargs):
            return [(event, "raw completion")]

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "visible-session"
    cli._pending_input = MagicMock()
    cli._pending_input.put.side_effect = RuntimeError("queue closed")
    monkeypatch.setattr("tools.process_registry.process_registry", _Registry())
    monkeypatch.setattr(ad, "claim_event_delivery", lambda *_a, **_k: "transport-claim")
    release_transport = MagicMock()
    monkeypatch.setattr(ad, "release_event_delivery", release_transport)
    monkeypatch.setattr(
        goals,
        "prepare_goal_delegation_delivery",
        lambda *_a, **_k: {
            "classification": "current",
            "prompt": "joined results",
            "reconciliation_claim": "recon-1",
            "goal_id": "goal-id",
            "goal_session_id": "visible-session",
            "reconciliation_attempt": 1,
        },
    )
    release_reconciliation = MagicMock(return_value=True)
    monkeypatch.setattr(
        goals, "release_goal_reconciliation_turn", release_reconciliation
    )

    with pytest.raises(RuntimeError, match="queue closed"):
        cli._drain_process_notifications("cli-idle")

    release_transport.assert_called_once_with(event, "transport-claim")
    release_reconciliation.assert_called_once_with(
        "recon-1",
        goal_id="goal-id",
        session_id="visible-session",
        attempt=1,
        turn_started=False,
        requeue=False,
    )
    assert _Registry.completion_queue.get_nowait() == event


def test_cli_goal_resume_enqueues_the_next_turn(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_cli.goals import GoalManager

    manager = GoalManager("resume-session")
    manager.set("finish the release")
    manager.pause("test")

    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "resume-session"
    cli._pending_input = queue.Queue()

    assert cli._handle_goal_command("/goal resume") is None
    prompt = cli._pending_input.get_nowait()
    assert "Continue working toward this goal" in prompt
    assert "finish the release" in prompt


def test_cli_goal_resume_releases_claim_when_enqueue_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import hermes_cli.goals as goals

    manager = goals.GoalManager("resume-session")
    manager.set("reconcile before continuing")
    manager.pause("test")
    monkeypatch.setattr(
        goals,
        "prepare_goal_resume",
        lambda *_a, **_k: {
            "prompt": "internal reconciliation",
            "reconciliation_claim": "recon-resume",
            "reconciliation_attempt": 2,
            "goal_id": "goal-id",
            "goal_session_id": "resume-session",
            "delegation_ids": ["deleg-a"],
        },
    )
    release = MagicMock(return_value={"released": True})
    monkeypatch.setattr(goals, "release_goal_reconciliation_turn", release)
    broken_queue = MagicMock()
    broken_queue.put.side_effect = RuntimeError("queue closed")
    cli = HermesCLI.__new__(HermesCLI)
    cli.session_id = "resume-session"
    cli._pending_input = broken_queue

    assert cli._handle_goal_command("/goal resume") is None

    release.assert_called_once_with(
        "recon-resume",
        session_id="resume-session",
        goal_id="goal-id",
        attempt=2,
        turn_started=False,
    )
