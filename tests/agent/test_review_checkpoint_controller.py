"""Session-state contract for automatic review checkpoints."""

from __future__ import annotations

from dataclasses import replace
import threading

import pytest

from agent.review_checkpoints import (
    ReviewCheckpointConfig,
    ReviewCheckpointController,
)
from agent.review_runner import ReviewRequest, ReviewResult


def _request(**overrides):
    request = ReviewRequest(
        checkpoint_id="turn-1:plan:0",
        session_id="session-1",
        phase="plan",
        objective="Make the requested change",
        constraints=("Keep the core cache-safe",),
        candidate={"summary": "Edit one file after focused tests"},
        provider="openai-codex",
        model="gpt-review",
        main_provider="openai-codex",
        main_model="gpt-economy",
    )
    return replace(request, **overrides)


def _result(verdict="PASS", **overrides):
    result = ReviewResult(
        checkpoint_id="turn-1:plan:0",
        status="completed",
        verdict=verdict,
        summary=f"Reviewer returned {verdict}.",
        actual_route={
            "profile": "default",
            "provider": "openai-codex",
            "model": "gpt-review",
            "credential_kind": "subscription_oauth",
        },
    )
    return replace(result, **overrides)


def _runner(result):
    def run(request):
        run.calls.append(request)
        return replace(result, checkpoint_id=request.checkpoint_id)

    run.calls = []
    return run


def test_disabled_controller_has_zero_review_calls_and_no_behavioral_effect():
    run = _runner(_result())
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=False),
        run_review=run,
    )

    decision = controller.evaluate(_request())

    assert decision.action == "continue"
    assert decision.result is None
    assert run.calls == []


@pytest.mark.parametrize(
    ("verdict", "expected_action"),
    [
        ("PASS", "continue"),
        ("REVISE", "revise"),
        ("ASK_USER", "ask_user"),
        ("BLOCK", "block"),
    ],
)
def test_completed_verdicts_map_to_explicit_actions(verdict, expected_action):
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=_runner(_result(verdict)),
    )

    decision = controller.evaluate(_request())

    assert decision.action == expected_action
    assert decision.result.verdict == verdict


def test_revision_count_is_bounded_and_escalates_to_user():
    run = _runner(_result("REVISE"))
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True, max_revisions=2),
        run_review=run,
    )

    first = controller.evaluate(_request(attempt=0))
    second = controller.evaluate(_request(checkpoint_id="turn-1:plan:1", attempt=1))
    exhausted = controller.evaluate(_request(checkpoint_id="turn-1:plan:2", attempt=2))

    assert first.action == "revise"
    assert second.action == "revise"
    assert exhausted.action == "ask_user"
    assert exhausted.reason == "revision_limit_reached"
    assert len(run.calls) == 3


def test_unavailable_is_visible_and_fail_open_by_default():
    events = []
    run = _runner(_result(
        verdict=None,
        status="unavailable",
        unavailable_reason="route_unavailable",
    ))
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=run,
        emit=events.append,
    )

    decision = controller.evaluate(_request())

    assert decision.action == "continue"
    assert decision.reason == "route_unavailable"
    assert events[-1]["status"] == "unavailable"
    assert events[-1]["failure_policy"] == "continue"


def test_unavailable_can_be_configured_to_block():
    run = _runner(_result(
        verdict=None,
        status="timed_out",
        unavailable_reason="timeout",
    ))
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True, failure_policy="block"),
        run_review=run,
    )

    decision = controller.evaluate(_request())

    assert decision.action == "block"
    assert decision.reason == "timeout"


def test_replayed_checkpoint_id_returns_cached_decision_without_second_call():
    run = _runner(_result("PASS"))
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=run,
    )

    first = controller.evaluate(_request())
    replay = controller.evaluate(_request())

    assert replay is first
    assert len(run.calls) == 1


def test_cancelled_inflight_checkpoint_discards_late_result():
    entered = threading.Event()
    release = threading.Event()
    decisions = []

    def slow_run(request):
        entered.set()
        assert release.wait(timeout=5)
        return replace(_result("PASS"), checkpoint_id=request.checkpoint_id)

    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=slow_run,
    )
    worker = threading.Thread(
        target=lambda: decisions.append(controller.evaluate(_request())),
        daemon=True,
    )
    worker.start()
    assert entered.wait(timeout=5)

    controller.cancel("turn-1:plan:0")
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert decisions[0].action == "cancelled"
    assert decisions[0].result.status == "cancelled"
    assert controller.evaluate(_request()) is decisions[0]


def test_concurrent_duplicate_waits_for_owner_and_bills_once():
    entered = threading.Event()
    release = threading.Event()
    calls = []
    decisions = []

    def slow_run(request):
        calls.append(request)
        entered.set()
        assert release.wait(timeout=5)
        return replace(_result("PASS"), checkpoint_id=request.checkpoint_id)

    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=slow_run,
    )
    workers = [
        threading.Thread(
            target=lambda: decisions.append(controller.evaluate(_request())),
            daemon=True,
        )
        for _ in range(2)
    ]
    workers[0].start()
    assert entered.wait(timeout=5)
    workers[1].start()
    release.set()
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert len(calls) == 1
    assert len(decisions) == 2
    assert decisions[0] is decisions[1]


def test_session_mismatch_is_rejected_before_runner_call():
    run = _runner(_result())
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=run,
    )

    decision = controller.evaluate(_request(session_id="session-2"))

    assert decision.action == "block"
    assert decision.reason == "session_mismatch"
    assert run.calls == []


def test_events_expose_sanitized_state_not_request_or_credential_objects():
    events = []
    controller = ReviewCheckpointController(
        session_id="session-1",
        config=ReviewCheckpointConfig(enabled=True),
        run_review=_runner(_result("PASS")),
        emit=events.append,
    )

    controller.evaluate(_request(candidate={"api_key": "must-not-leak"}))

    serialized = str(events)
    assert "must-not-leak" not in serialized
    assert "credential_handle" not in serialized
    assert events[-1]["actual_route"]["credential_kind"] == "subscription_oauth"


@pytest.mark.parametrize(
    "config",
    [
        ReviewCheckpointConfig(enabled=True, max_revisions=-1),
        ReviewCheckpointConfig(enabled=True, failure_policy="retry"),
    ],
)
def test_invalid_controller_config_fails_fast(config):
    with pytest.raises(ValueError):
        ReviewCheckpointController(
            session_id="session-1",
            config=config,
            run_review=_runner(_result()),
        )
