from gateway.orchestrator.lanes import LaneResult, LaneStatus
from gateway.orchestrator.synthesis import synthesize


def lane(lane_id, agent, status, output=None, error=None):
    return LaneResult(lane_id=lane_id, agent=agent, status=status, output=output, error=error, duration_s=0.1, exit_code=0 if status is LaneStatus.SUCCEEDED else 1, log_path=None)


def test_synthesize_chooses_first_success_and_reports_other_statuses():
    result = synthesize([
        lane("codex-1", "codex", LaneStatus.SKIPPED, error="agent degraded"),
        lane("ccd-1", "ccd", LaneStatus.SUCCEEDED, output="good plan"),
        lane("ccg-1", "ccg", LaneStatus.FAILED, error="failed"),
    ])

    assert result.chosen_lane_id == "ccd-1"
    assert result.confidence == "medium"
    assert "codex-1" in result.summary
    assert "ccg-1" in result.summary
    assert [digest.status for digest in result.lanes] == ["skipped", "succeeded", "failed"]


def test_synthesize_reports_low_confidence_when_no_lane_succeeds():
    result = synthesize([
        lane("codex-1", "codex", LaneStatus.TIMED_OUT, error="timeout"),
        lane("ccm-1", "ccm", LaneStatus.FAILED, error="bad"),
    ])

    assert result.chosen_lane_id is None
    assert result.confidence == "low"
    assert "성공 lane 없음" in result.summary


def test_synthesize_redacts_lane_digests_and_detects_conflicts():
    result = synthesize([
        lane("ccd-1", "ccd", LaneStatus.SUCCEEDED, output="answer A sk-123...cdef"),
        lane("ccg-1", "ccg", LaneStatus.SUCCEEDED, output="answer B"),
    ])

    digests = "\n".join(digest.digest for digest in result.lanes)
    assert "sk-" not in digests
    assert result.conflicts
    assert result.confidence == "medium"
