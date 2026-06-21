from pathlib import Path

from gateway.orchestrator.command import CommandResult
from gateway.orchestrator.executors import CodexExternalIsolationExecutor, FakeLaneExecutor, RealLaneExecutor
from gateway.orchestrator.lanes import LaneRequest, LaneResult, LaneStatus


def test_fake_lane_executor_returns_programmed_results_and_records_prompts():
    req = LaneRequest(lane_id="ccd-1", agent="ccd", prompt="plan", effort="max")
    expected = LaneResult(lane_id="ccd-1", agent="ccd", status=LaneStatus.SUCCEEDED, output="good", error=None, duration_s=0.2, exit_code=0, log_path=None)
    executor = FakeLaneExecutor({"ccd-1": expected})

    result = executor.execute(req)

    assert result is expected
    assert executor.seen == [req]


def test_fake_lane_executor_can_program_status_or_exception():
    failed_req = LaneRequest(lane_id="f", agent="codex", prompt="x")
    boom_req = LaneRequest(lane_id="boom", agent="ccg", prompt="x")
    executor = FakeLaneExecutor({"f": LaneStatus.FAILED, "boom": RuntimeError("broken")})

    failed = executor.execute(failed_req)
    assert failed.status is LaneStatus.FAILED
    assert failed.agent == "codex"

    try:
        executor.execute(boom_req)
    except RuntimeError as exc:
        assert str(exc) == "broken"
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")


def test_real_lane_executor_is_phase_three_placeholder():
    req = LaneRequest(lane_id="real", agent="ccd", prompt="do it")

    try:
        RealLaneExecutor().execute(req)
    except NotImplementedError as exc:
        assert "Phase 3" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected NotImplementedError")


def test_codex_external_isolation_executor_invokes_codex_in_lane_workspace(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "README.md").write_text("hello\n", encoding="utf-8")
    seen: dict[str, object] = {}

    class Runner:
        def run(self, argv, timeout, *, cwd=None, input_text=None, env=None):
            seen["argv"] = list(argv)
            seen["timeout"] = timeout
            seen["cwd"] = cwd
            seen["input_text"] = input_text
            seen["env"] = dict(env or {})
            output_path = argv[argv.index("-o") + 1]
            assert "do the task" not in argv
            assert input_text == "do the task"
            Path(output_path).write_text("codex result\n", encoding="utf-8")
            return CommandResult(0, "stdout noise", "")

    req = LaneRequest(lane_id="codex-review", agent="codex", prompt="do the task", effort="xhigh", timeout_s=17)
    executor = CodexExternalIsolationExecutor(
        codex_path="/opt/bin/codex",
        source_dir=source,
        artifact_root=tmp_path / "artifacts",
        runner=Runner(),
    )

    result = executor.execute(req)

    lane_root = tmp_path / "artifacts" / "codex-review"
    workdir = lane_root / "workdir"
    codex_home = lane_root / "codex-home"
    prompt_file = lane_root / "prompt.md"
    output_file = lane_root / "codex-output.md"

    assert result.status is LaneStatus.SUCCEEDED
    assert result.output == "codex result"
    assert result.exit_code == 0
    assert result.artifacts["workdir"] == str(workdir)
    assert result.artifacts["codex_home"] == str(codex_home)
    assert result.artifacts["prompt"] == str(prompt_file)
    assert result.artifacts["output"] == str(output_file)
    assert result.artifacts["invocation"] == str(lane_root / "invocation.json")
    assert result.artifacts["log"] == str(lane_root / "codex.log")
    assert (workdir / "README.md").read_text(encoding="utf-8") == "hello\n"
    assert prompt_file.read_text(encoding="utf-8") == "do the task"
    assert oct(codex_home.stat().st_mode & 0o777) == "0o700"
    assert seen["cwd"] == str(workdir)
    assert seen["timeout"] == 17
    assert seen["env"]["CODEX_HOME"] == str(codex_home)
    assert seen["argv"] == [
        "/opt/bin/codex",
        "--ask-for-approval",
        "never",
        "--sandbox",
        "danger-full-access",
        "--cd",
        str(workdir),
        "--config",
        "model_reasoning_effort=xhigh",
        "exec",
        "--skip-git-repo-check",
        "-o",
        str(output_file),
        "-",
    ]


def test_codex_external_isolation_executor_redacts_failed_output(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    class Runner:
        def run(self, argv, timeout, *, cwd=None, input_text=None, env=None):
            output_path = argv[argv.index("-o") + 1]
            Path(output_path).write_text("raw sk-123...cdef output\n", encoding="utf-8")
            return CommandResult(2, "stdout ***", "token sk-123...cdef failed")

    req = LaneRequest(lane_id="codex-review", agent="codex", prompt="p", timeout_s=5)
    executor = CodexExternalIsolationExecutor(
        codex_path="codex",
        source_dir=source,
        artifact_root=tmp_path / "artifacts",
        runner=Runner(),
    )

    result = executor.execute(req)

    assert result.status is LaneStatus.FAILED
    assert result.exit_code == 2
    assert "sk-" not in result.error
    assert "[REDACTED]" in result.error
    assert "***" not in Path(result.artifacts["log"]).read_text(encoding="utf-8")
    assert "sk-" not in Path(result.artifacts["log"]).read_text(encoding="utf-8")
    assert "sk-" not in Path(result.artifacts["output"]).read_text(encoding="utf-8")


def test_codex_external_isolation_executor_reports_timeout(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    class Runner:
        def run(self, argv, timeout, *, cwd=None, input_text=None, env=None):
            return CommandResult(-1, "", "timed out", timed_out=True)

    req = LaneRequest(lane_id="codex-review", agent="codex", prompt="p", timeout_s=5)
    executor = CodexExternalIsolationExecutor(
        codex_path="codex",
        source_dir=source,
        artifact_root=tmp_path / "artifacts",
        runner=Runner(),
    )

    result = executor.execute(req)

    assert result.status is LaneStatus.TIMED_OUT
    assert result.error == "timed out after 5s"
