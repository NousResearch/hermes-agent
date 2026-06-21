from gateway.orchestrator.command import CommandResult, FakeCommandRunner
from gateway.orchestrator.doctor import run_doctor
from gateway.orchestrator.executors import FakeLaneExecutor
from gateway.orchestrator.lanes import LaneRequest, LaneStatus
from gateway.orchestrator.runner import run_lanes
from gateway.orchestrator.synthesis import synthesize


def test_doctor_degraded_agent_can_feed_parallel_runner_skip_policy(tmp_path):
    runner = FakeCommandRunner({
        ("codex", "--version"): CommandResult(0, "codex-cli 0.141.0\n", ""),
        ("bash", "-ic", "type -t ccd"): CommandResult(0, "function\n", ""),
        ("bash", "-ic", "type -t ccg"): CommandResult(0, "function\n", ""),
        ("bash", "-ic", "type -t ccm"): CommandResult(0, "function\n", ""),
        ("bwrap", "--unshare-user", "--uid", "0", "--gid", "0", "true"): CommandResult(1, "", "Operation not permitted"),
        ("unshare", "-Ur", "true"): CommandResult(0, "", ""),
        ("unshare", "-n", "true"): CommandResult(0, "", ""),
    })
    doctor = run_doctor(which_fn=lambda name: f"/bin/{name}" if name in {"codex", "bwrap", "unshare"} else None, runner=runner)
    degraded = {agent.name for agent in doctor.agents if agent.status == "degraded"}

    results = run_lanes(
        [
            LaneRequest(lane_id="codex-1", agent="codex", prompt="review"),
            LaneRequest(lane_id="ccd-1", agent="ccd", prompt="review"),
        ],
        executor=FakeLaneExecutor({"ccd-1": LaneStatus.SUCCEEDED}),
        degraded_agents=degraded,
        log_dir=tmp_path,
    )
    synthesis = synthesize(results)

    assert degraded == {"codex"}
    assert {result.lane_id: result.status for result in results}["codex-1"] is LaneStatus.SKIPPED
    assert synthesis.chosen_lane_id == "ccd-1"
