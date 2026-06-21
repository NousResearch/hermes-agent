from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str, *, platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=platform,
            chat_id="channel-1",
            chat_type="channel",
            user_id="user-1",
        ),
    )


def test_parallel_review_command_is_registered_as_safe_gateway_command():
    from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, is_gateway_known_command, resolve_command

    cmd = resolve_command("parallel_review")

    assert cmd is not None
    assert cmd.name == "parallel_review"
    assert cmd.category == "Info"
    assert cmd.args_hint == "dryrun [--save] <request>"
    assert cmd.gateway_only is True
    assert cmd.gateway_platforms == ("discord",)
    assert is_gateway_known_command("parallel_review") is True
    assert "parallel_review" in ACTIVE_SESSION_BYPASS_COMMANDS


@pytest.mark.asyncio
async def test_parallel_review_handler_rejects_non_discord_platform():
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_parallel_review_command(
        _event('/parallel_review dryrun "검토"', platform=Platform.TELEGRAM)
    )

    assert "Discord" in result
    assert "dry-run" not in result


@pytest.mark.asyncio
async def test_parallel_review_dryrun_formats_plan_without_model_calls(monkeypatch):
    import gateway.run as gateway_run
    from gateway.orchestrator.doctor import AgentReport, DoctorReport, ExternalIsolationHealth, SandboxHealth

    report = DoctorReport(
        agents=[
            AgentReport("ccd", "shell_function", "available", notes=["available via bash -ic"]),
            AgentReport(
                "codex",
                "binary",
                "degraded",
                version="codex-cli 0.141.0",
                sandbox=SandboxHealth("degraded", "namespace smoke", "Operation not permitted"),
                external_isolation=ExternalIsolationHealth("available", "danger-full-access", "external worktree required"),
                execution_mode="external-isolated",
                notes=["sandbox degraded", "external isolation available"],
            ),
            AgentReport("ccg", "shell_function", "available", notes=["available via bash -ic"]),
            AgentReport("ccm", "shell_function", "available", notes=["output suppressed for sensitive wrapper"]),
        ]
    )
    monkeypatch.setattr("gateway.orchestrator.doctor.run_doctor", lambda: report)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_parallel_review_command(_event('/parallel_review dryrun "이 설계 검토해줘"'))

    assert "Parallel review dry-run" in result
    assert "이 설계 검토해줘" in result
    assert "ccd-review" in result and "would run" in result
    assert "codex-review" in result and "would run" in result and "external isolation" in result
    assert "codex-review" not in result.split("ccg-review", 1)[0] or "skipped" not in result.split("codex-review", 1)[1].split("ccg-review", 1)[0]
    assert "ccg-review" in result and "would run" in result
    assert "ccm-review" in result and "sensitive output suppressed" in result
    assert "No model calls were made" in result
    assert "No files were changed" in result
    assert "claude" not in result
    assert "emd" not in result


@pytest.mark.asyncio
async def test_parallel_review_dryrun_save_writes_redacted_artifacts(monkeypatch, tmp_path):
    import json

    import gateway.run as gateway_run
    from gateway.orchestrator.doctor import AgentReport, DoctorReport, ExternalIsolationHealth, SandboxHealth

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    report = DoctorReport(
        agents=[
            AgentReport("ccd", "shell_function", "available", notes=["available via bash -ic"]),
            AgentReport(
                "codex",
                "binary",
                "degraded",
                version="codex-cli 0.141.0",
                sandbox=SandboxHealth("degraded", "namespace smoke", "Operation not permitted"),
                external_isolation=ExternalIsolationHealth("available", "danger-full-access", "external worktree required"),
                execution_mode="external-isolated",
                notes=["sandbox degraded", "external isolation available"],
            ),
            AgentReport("ccm", "shell_function", "available", notes=["output suppressed for sensitive wrapper"]),
        ]
    )
    monkeypatch.setattr("gateway.orchestrator.doctor.run_doctor", lambda: report)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_parallel_review_command(
        _event('/parallel_review dryrun --save "검토 sk-1234567abc"')
    )

    runs_root = tmp_path / "hermes-home" / "orchestrator" / "runs"
    run_dirs = [path for path in runs_root.iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    assert "Artifacts saved:" in result
    assert str(run_dir) in result
    assert "No model calls were made" in result
    assert "No tmux sessions were created" in result
    assert "sk-1234567abc" not in result

    request_payload = json.loads((run_dir / "request.json").read_text(encoding="utf-8"))
    doctor_payload = json.loads((run_dir / "doctor.json").read_text(encoding="utf-8"))
    planned_payload = json.loads((run_dir / "planned_lanes.json").read_text(encoding="utf-8"))
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")

    assert request_payload["command"] == "parallel_review"
    assert request_payload["mode"] == "dryrun"
    assert request_payload["request"] == "검토 [REDACTED]"
    assert request_payload["source"]["platform"] == "discord"
    assert request_payload["model_calls"] == 0
    assert request_payload["tmux_sessions_created"] == 0
    assert doctor_payload["agents"][0]["name"] == "ccd"
    assert doctor_payload["agents"][1]["name"] == "codex"
    assert doctor_payload["agents"][1]["execution_mode"] == "external-isolated"
    assert doctor_payload["agents"][1]["external_isolation"]["status"] == "available"
    assert [lane["lane_id"] for lane in planned_payload["lanes"]] == ["ccd-review", "codex-review", "ccm-review"]
    codex_lane = next(lane for lane in planned_payload["lanes"] if lane["agent"] == "codex")
    assert codex_lane["metadata"]["execution_mode"] == "external-isolated"
    assert planned_payload["external_isolation_agents"] == ["codex"]
    results_by_lane = {lane["lane_id"]: lane for lane in planned_payload["results"]}
    assert results_by_lane["ccd-review"]["status"] == "succeeded"
    assert results_by_lane["codex-review"]["status"] == "succeeded"
    assert (run_dir / "lanes" / "ccd-review.status.json").exists()
    assert (run_dir / "lanes" / "codex-review.status.json").exists()
    assert "Parallel review dry-run" in summary
    assert "sk-1234567abc" not in summary


@pytest.mark.asyncio
async def test_parallel_review_rejects_non_dryrun_until_real_mode_exists():
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_parallel_review_command(_event('/parallel_review run "실제 호출"'))

    assert "dryrun" in result
    assert "실제 agent/model 호출은 아직 비활성화" in result


@pytest.mark.asyncio
async def test_parallel_review_dryrun_requires_request_text():
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_parallel_review_command(_event('/parallel_review dryrun'))

    assert "사용법" in result
    assert "/parallel_review dryrun" in result
