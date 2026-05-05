"""Tests for anticipation policy decisions."""

from datetime import datetime, timedelta, timezone
from math import inf, nan

import pytest

from agent.anticipation import AnticipationPermission, AnticipationRuntimeConfig
from agent.anticipation_policy import (
    AnticipationCandidate,
    AnticipationDecisionHistory,
    decide_anticipation_action,
)

NOW = datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc)


def runtime_config(**overrides):
    values = {
        "enabled": True,
        "loop_enabled": True,
        "loop_permission": AnticipationPermission.SUGGEST,
        "min_confidence": 0.7,
        "quiet_hours_enabled": False,
        "quiet_hours_start": "22:00",
        "quiet_hours_end": "08:00",
        "max_per_day": 3,
        "min_minutes_between": 120,
    }
    values.update(overrides)
    return AnticipationRuntimeConfig(**values)


def candidate(**overrides):
    values = {
        "loop_id": "stale_task_resurfacer",
        "title": "Stale thread worth picking back up",
        "body": "We paused at turning anticipation into a concrete Hermes plan.",
        "confidence": 0.8,
        "proposed_permission": AnticipationPermission.SUGGEST,
        "dedupe_key": "session:123",
        "created_at": NOW,
    }
    values.update(overrides)
    return AnticipationCandidate(**values)


def test_decision_suggests_when_candidate_passes_all_gates():
    decision = decide_anticipation_action(candidate(), runtime_config(), AnticipationDecisionHistory(), now=NOW)

    assert decision.action == "suggest"
    assert decision.reason == "passed"


def test_decision_skips_when_anticipation_disabled():
    decision = decide_anticipation_action(
        candidate(),
        runtime_config(enabled=False),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "anticipation_disabled"


def test_decision_skips_when_loop_disabled():
    decision = decide_anticipation_action(
        candidate(),
        runtime_config(loop_enabled=False),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "loop_disabled"


def test_decision_skips_when_confidence_below_threshold():
    decision = decide_anticipation_action(
        candidate(confidence=0.69),
        runtime_config(min_confidence=0.7),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "below_confidence_threshold"


@pytest.mark.parametrize("confidence", [nan, inf, -inf])
def test_decision_skips_when_candidate_confidence_is_not_finite(confidence):
    decision = decide_anticipation_action(
        candidate(confidence=confidence),
        runtime_config(min_confidence=0.7),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "below_confidence_threshold"


def test_decision_skips_empty_candidate_body():
    decision = decide_anticipation_action(
        candidate(body="   "),
        runtime_config(),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "empty_candidate_body"


def test_decision_skips_duplicate_inside_quieting_window():
    history = AnticipationDecisionHistory(
        recent_dedupe_keys={"session:123": NOW - timedelta(minutes=10)},
    )

    decision = decide_anticipation_action(candidate(), runtime_config(), history, now=NOW)

    assert decision.action == "skip"
    assert decision.reason == "duplicate_dedupe_key"


def test_decision_skips_when_notification_budget_exhausted():
    history = AnticipationDecisionHistory(notifications_today=3)

    decision = decide_anticipation_action(candidate(), runtime_config(max_per_day=3), history, now=NOW)

    assert decision.action == "skip"
    assert decision.reason == "notification_budget_exhausted"


def test_decision_uses_budget_reason_when_last_notification_was_too_recent():
    history = AnticipationDecisionHistory(last_notification_at=NOW - timedelta(minutes=10))

    decision = decide_anticipation_action(candidate(), runtime_config(), history, now=NOW)

    assert decision.action == "skip"
    assert decision.reason == "notification_budget_exhausted"


def test_decision_skips_inside_quiet_hours():
    quiet_now = datetime(2026, 5, 5, 23, 0, tzinfo=timezone.utc)

    decision = decide_anticipation_action(
        candidate(created_at=quiet_now),
        runtime_config(quiet_hours_enabled=True, quiet_hours_start="22:00", quiet_hours_end="08:00"),
        AnticipationDecisionHistory(),
        now=quiet_now,
    )

    assert decision.action == "skip"
    assert decision.reason == "inside_quiet_hours"


def test_decision_allows_silent_log_inside_quiet_hours_and_exhausted_budget():
    quiet_now = datetime(2026, 5, 5, 23, 0, tzinfo=timezone.utc)

    decision = decide_anticipation_action(
        candidate(
            created_at=quiet_now,
            proposed_permission=AnticipationPermission.SILENT_LOG,
        ),
        runtime_config(
            loop_permission=AnticipationPermission.SILENT_LOG,
            quiet_hours_enabled=True,
            quiet_hours_start="22:00",
            quiet_hours_end="08:00",
            max_per_day=0,
        ),
        AnticipationDecisionHistory(notifications_today=99),
        now=quiet_now,
    )

    assert decision.action == "silent_log"
    assert decision.reason == "passed"


def test_decision_skips_when_candidate_permission_exceeds_loop_ceiling():
    decision = decide_anticipation_action(
        candidate(proposed_permission=AnticipationPermission.ASK_TO_EXECUTE),
        runtime_config(loop_permission=AnticipationPermission.SUGGEST),
        AnticipationDecisionHistory(),
        now=NOW,
    )

    assert decision.action == "skip"
    assert decision.reason == "permission_exceeds_ceiling"
