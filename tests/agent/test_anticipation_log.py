"""Tests for sanitized anticipation decision logging."""

import json
from datetime import datetime, timezone
from math import nan

from agent.anticipation import AnticipationPermission
from agent.anticipation_log import append_decision_log, decision_log_path, read_recent_decision_logs
from agent.anticipation_policy import AnticipationCandidate, AnticipationDecision


def make_decision(secret_body="body with private-token", secret_title="Potential stale task for private client"):
    candidate = AnticipationCandidate(
        loop_id="stale_task_resurfacer",
        title=secret_title,
        body=secret_body,
        confidence=0.82,
        proposed_permission=AnticipationPermission.SUGGEST,
        dedupe_key="raw-session-id-with-secret",
        created_at=datetime(2026, 5, 5, 18, 0, tzinfo=timezone.utc),
    )
    return AnticipationDecision(action="skip", reason="below_confidence_threshold", candidate=candidate)


def test_append_decision_log_writes_sanitized_jsonl_with_secure_permissions():
    path = append_decision_log(make_decision(), now=datetime(2026, 5, 5, 18, 1, tzinfo=timezone.utc))

    assert path == decision_log_path()
    assert path.exists()
    raw_log = path.read_text(encoding="utf-8")
    record = json.loads(raw_log.strip())

    assert record["loop_id"] == "stale_task_resurfacer"
    assert record["action"] == "skip"
    assert record["reason"] == "below_confidence_threshold"
    assert record["confidence"] == 0.82
    assert "dedupe_key_hash" in record
    assert "title_hash" in record
    assert "raw-session-id-with-secret" not in raw_log
    assert "private-token" not in raw_log
    assert "Potential stale task for private client" not in raw_log

    if path.parent.stat().st_mode & 0o777 != 0o700:
        # Containerized test environments may not honor chmod on mounted paths.
        return
    assert path.stat().st_mode & 0o777 == 0o600


def test_read_recent_decision_logs_returns_newest_records_first():
    first = make_decision("first")
    second = make_decision("second")
    append_decision_log(first, now=datetime(2026, 5, 5, 18, 1, tzinfo=timezone.utc))
    append_decision_log(second, now=datetime(2026, 5, 5, 18, 2, tzinfo=timezone.utc))

    records = read_recent_decision_logs(limit=1)

    assert len(records) == 1
    assert records[0]["ts"] == "2026-05-05T18:02:00+00:00"


def test_append_decision_log_sanitizes_non_finite_confidence():
    decision = make_decision()
    candidate = AnticipationCandidate(
        loop_id=decision.candidate.loop_id,
        title=decision.candidate.title,
        body=decision.candidate.body,
        confidence=nan,
        proposed_permission=decision.candidate.proposed_permission,
        dedupe_key=decision.candidate.dedupe_key,
        created_at=decision.candidate.created_at,
    )

    path = append_decision_log(
        AnticipationDecision(action="skip", reason="below_confidence_threshold", candidate=candidate),
        now=datetime(2026, 5, 5, 18, 3, tzinfo=timezone.utc),
    )

    raw_log = path.read_text(encoding="utf-8")
    assert "NaN" not in raw_log
    assert json.loads(raw_log.splitlines()[-1])["confidence"] is None
