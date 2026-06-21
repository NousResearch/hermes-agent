import json
import threading
import time

from gateway.orchestrator.executors import FakeLaneExecutor
from gateway.orchestrator.lanes import LaneRequest, LaneResult, LaneStatus
from gateway.orchestrator.runner import run_lanes


def test_run_lanes_collects_results_and_writes_redacted_artifacts(tmp_path):
    reqs = [
        LaneRequest(lane_id="ccd-1", agent="ccd", prompt="p1"),
        LaneRequest(lane_id="codex-1", agent="codex", prompt="p2"),
    ]
    executor = FakeLaneExecutor({
        "ccd-1": LaneResult(lane_id="ccd-1", agent="ccd", status=LaneStatus.SUCCEEDED, output="ok sk-123...cdef", error=None, duration_s=0.1, exit_code=0, log_path=None),
        "codex-1": LaneStatus.TIMED_OUT,
    })

    results = run_lanes(reqs, executor=executor, max_workers=2, log_dir=tmp_path)
    by_id = {result.lane_id: result for result in results}

    assert by_id["ccd-1"].status is LaneStatus.SUCCEEDED
    assert by_id["codex-1"].status is LaneStatus.TIMED_OUT
    assert "sk-" not in (tmp_path / "ccd-1.log").read_text(encoding="utf-8")
    assert (tmp_path / "ccd-1.result.md").exists()
    status_payload = json.loads((tmp_path / "ccd-1.status.json").read_text(encoding="utf-8"))
    assert status_payload["status"] == "succeeded"
    assert by_id["ccd-1"].artifacts["log"].endswith("ccd-1.log")


def test_run_lanes_skips_degraded_agents_without_calling_executor(tmp_path):
    reqs = [
        LaneRequest(lane_id="codex-1", agent="codex", prompt="p"),
        LaneRequest(lane_id="ccd-1", agent="ccd", prompt="p"),
    ]
    executor = FakeLaneExecutor({"ccd-1": LaneStatus.SUCCEEDED})

    results = run_lanes(reqs, executor=executor, degraded_agents={"codex"}, log_dir=tmp_path)
    by_id = {result.lane_id: result for result in results}

    assert by_id["codex-1"].status is LaneStatus.SKIPPED
    assert by_id["codex-1"].error == "agent degraded"
    assert [req.lane_id for req in executor.seen] == ["ccd-1"]


def test_run_lanes_converts_executor_exceptions_to_failed_results(tmp_path):
    req = LaneRequest(lane_id="ccg-1", agent="ccg", prompt="p")
    executor = FakeLaneExecutor({"ccg-1": RuntimeError("tool exploded")})

    result = run_lanes([req], executor=executor, log_dir=tmp_path)[0]

    assert result.status is LaneStatus.FAILED
    assert "tool exploded" in result.error


def test_run_lanes_respects_max_workers_limit(tmp_path):
    class CountingExecutor:
        def __init__(self):
            self.active = 0
            self.max_active = 0
            self.lock = threading.Lock()

        def execute(self, req):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.05)
            with self.lock:
                self.active -= 1
            return LaneResult(lane_id=req.lane_id, agent=req.agent, status=LaneStatus.SUCCEEDED, output=req.prompt, error=None, duration_s=0.05, exit_code=0, log_path=None)

    executor = CountingExecutor()
    reqs = [LaneRequest(lane_id=f"ccd-{i}", agent="ccd", prompt=str(i)) for i in range(5)]

    results = run_lanes(reqs, executor=executor, max_workers=2, log_dir=tmp_path)

    assert all(result.status is LaneStatus.SUCCEEDED for result in results)
    assert 1 < executor.max_active <= 2
