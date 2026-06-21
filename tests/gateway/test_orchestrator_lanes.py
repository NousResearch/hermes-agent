from gateway.orchestrator.lanes import LaneRequest, LaneResult, LaneStatus


def test_lane_request_defaults_are_safe_for_dry_run():
    req = LaneRequest(lane_id="l1", agent="ccd", prompt="review this")

    assert req.effort is None
    assert req.timeout_s == 60.0
    assert req.metadata == {}


def test_lane_result_factories_and_terminal_statuses():
    req = LaneRequest(lane_id="l1", agent="codex", prompt="review")

    failed = LaneResult.failed(req, "boom", duration_s=1.2, exit_code=2)
    timed_out = LaneResult.timed_out(req, duration_s=60.0)
    running = LaneResult(lane_id="l1", agent="codex", status=LaneStatus.RUNNING, output=None, error=None, duration_s=0, exit_code=None, log_path=None)

    assert failed.status is LaneStatus.FAILED
    assert failed.error == "boom"
    assert timed_out.status is LaneStatus.TIMED_OUT
    assert failed.is_terminal() is True
    assert timed_out.is_terminal() is True
    assert running.is_terminal() is False


def test_lane_result_to_dict_serializes_enum_values():
    result = LaneResult(lane_id="l1", agent="ccd", status=LaneStatus.SUCCEEDED, output="ok", error=None, duration_s=0.1, exit_code=0, log_path="/tmp/l1.log")

    assert result.to_dict()["status"] == "succeeded"
