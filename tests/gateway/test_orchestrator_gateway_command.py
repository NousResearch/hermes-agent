from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="channel-1",
            chat_type="channel",
            user_id="user-1",
        ),
    )


def test_agent_doctor_command_is_registered_as_gateway_info_command():
    from hermes_cli.commands import ACTIVE_SESSION_BYPASS_COMMANDS, is_gateway_known_command, resolve_command

    cmd = resolve_command("agent_doctor")

    assert cmd is not None
    assert cmd.name == "agent_doctor"
    assert cmd.category == "Info"
    assert cmd.args_hint == "[json]"
    assert cmd.gateway_only is True
    assert is_gateway_known_command("agent_doctor") is True
    assert "agent_doctor" in ACTIVE_SESSION_BYPASS_COMMANDS


@pytest.mark.asyncio
async def test_agent_doctor_handler_formats_read_only_summary_for_operational_agents(monkeypatch):
    import gateway.run as gateway_run
    from gateway.orchestrator.doctor import AgentReport, DoctorReport, ExternalIsolationHealth, SandboxHealth

    report = DoctorReport(
        agents=[
            AgentReport("ccd", "shell_function", "available", notes=["available via bash -ic"]),
            AgentReport(
                "codex",
                "binary",
                "degraded",
                path="/bin/codex",
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

    result = await runner._handle_agent_doctor_command(_event("/agent_doctor"))

    assert "Agent doctor" in result
    assert "read-only" in result
    assert "ccd" in result and "available" in result
    assert "codex" in result and "degraded" in result
    assert "execution=external-isolated" in result
    assert "external_isolation=available" in result
    assert "ccg" in result and "available" in result
    assert "ccm" in result and "available" in result
    assert "Operation not permitted" in result
    assert "claude" not in result
    assert "emd" not in result


@pytest.mark.asyncio
async def test_agent_doctor_handler_can_return_json(monkeypatch):
    import json
    import gateway.run as gateway_run
    from gateway.orchestrator.doctor import AgentReport, DoctorReport

    report = DoctorReport(agents=[AgentReport("ccd", "shell_function", "available")])
    monkeypatch.setattr("gateway.orchestrator.doctor.run_doctor", lambda: report)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace()

    result = await runner._handle_agent_doctor_command(_event("/agent_doctor json"))

    assert result.startswith("```json")
    payload = result.removeprefix("```json\n").removesuffix("\n```")
    data = json.loads(payload)
    assert data["agents"][0]["name"] == "ccd"
    assert data["agents"][0]["status"] == "available"
