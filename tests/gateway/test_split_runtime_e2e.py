"""Production-path integration tests for API-server split runtime."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import ClientSession

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


def _write_config(text: str) -> None:
    hermes_home = Path(os.environ["HERMES_HOME"])
    hermes_home.mkdir(parents=True, exist_ok=True)
    (hermes_home / "config.yaml").write_text(text, encoding="utf-8")


def _point_gateway_at_test_home(monkeypatch) -> None:
    import gateway.run

    monkeypatch.setattr(gateway.run, "_hermes_home", Path(os.environ["HERMES_HOME"]))


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": 0,
                "key": "split-e2e-server-key",
            },
        )
    )


def _chat_response(*, content=None, tool_calls=None, finish_reason="stop"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls or [],
                    reasoning=None,
                    reasoning_content=None,
                    refusal=None,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        model="test-model",
    )


def _server_port(adapter: APIServerAdapter) -> int:
    sockets = adapter._site._server.sockets
    return int(sockets[0].getsockname()[1])


@pytest.mark.asyncio
async def test_configured_production_routes_drive_real_agent_split_runtime(monkeypatch):
    _write_config(
        """
model:
  provider: openrouter
  default: test-model
platform_toolsets:
  api_server:
    - file
gateway:
  api_server:
    split_runtime:
      enabled: true
      routed_toolsets:
        - file
      request_timeout_seconds: 10
""".strip()
        + "\n"
    )
    _point_gateway_at_test_home(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")

    server_workspace = Path(os.environ["HERMES_HOME"]) / "server-workspace" / "src"
    server_workspace.mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_CWD", str(server_workspace.parent))
    (server_workspace / "AGENTS.md").write_text(
        "SERVER_ONLY_SUBDIRECTORY_INSTRUCTIONS",
        encoding="utf-8",
    )

    tool_call = SimpleNamespace(
        id="call_e2e_read",
        type="function",
        function=SimpleNamespace(
            name="read_file",
            arguments=json.dumps({"path": str(server_workspace / "x.py")}),
        ),
        extra_content=None,
        model_extra={},
    )
    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = [
        _chat_response(content=None, tool_calls=[tool_call], finish_reason="tool_calls"),
        _chat_response(content="model completed after local read"),
    ]

    adapter = _adapter()
    try:
        with patch("run_agent.OpenAI", return_value=fake_client):
            assert await adapter.connect() is True
            assert adapter._split_runtime_enabled is True
            assert adapter._split_runtime_is_effective() is True
            base_url = f"http://127.0.0.1:{_server_port(adapter)}"
            headers = {"Authorization": "Bearer split-e2e-server-key"}

            async with ClientSession(headers=headers) as client:
                start = await client.post(f"{base_url}/v1/runs", json={"input": "read it"})
                assert start.status == 202
                run_id = (await start.json())["run_id"]

                events = await client.get(
                    f"{base_url}/v1/runs/{run_id}/events?tool_executor=1",
                    headers={"X-Hermes-Tool-Executor-Token": "e2e-executor"},
                )
                assert events.status == 200

                request_event = None
                for _ in range(20):
                    raw = await events.content.readline()
                    line = raw.decode().strip()
                    if line.startswith("data: "):
                        event = json.loads(line[6:])
                        if event.get("event") == "tool.request":
                            request_event = event
                            break
                assert request_event is not None
                assert request_event["tool_name"] == "read_file"

                result = await client.post(
                    f"{base_url}/v1/runs/{run_id}/tool_result",
                    headers={"X-Hermes-Tool-Executor-Token": "e2e-executor"},
                    json={
                        "request_id": request_event["request_id"],
                        "result": "LOCAL E2E CONTENT" + ("x" * 500_000),
                    },
                )
                assert result.status == 200

                stream_body = await events.text()
                assert "run.completed" in stream_body
                assert "model completed after local read" in stream_body

            assert fake_client.chat.completions.create.call_count == 2
            second_messages = fake_client.chat.completions.create.call_args_list[1].kwargs["messages"]
            tool_message = next(
                message for message in second_messages if message.get("role") == "tool"
            )
            assert "LOCAL E2E CONTENT" in tool_message["content"]
            assert "Full output could not be saved to sandbox" in tool_message["content"]
            assert "Full output saved to:" not in tool_message["content"]
            assert "SERVER_ONLY_SUBDIRECTORY_INSTRUCTIONS" not in tool_message["content"]
    finally:
        await adapter.disconnect()


def test_split_runtime_is_default_off_from_real_config(monkeypatch):
    _write_config(
        """
model:
  provider: openrouter
  default: test-model
platform_toolsets:
  api_server:
    - file
""".strip()
        + "\n"
    )

    _point_gateway_at_test_home(monkeypatch)
    adapter = _adapter()

    assert adapter._split_runtime_enabled is False
    assert adapter._split_runtime_is_effective() is False


def test_enabled_split_runtime_is_ineffective_without_file_toolset(monkeypatch):
    _write_config(
        """
model:
  provider: openrouter
  default: test-model
platform_toolsets:
  api_server:
    - clarify
gateway:
  api_server:
    split_runtime:
      enabled: true
      routed_toolsets:
        - file
""".strip()
        + "\n"
    )

    _point_gateway_at_test_home(monkeypatch)
    adapter = _adapter()

    assert adapter._split_runtime_enabled is True
    assert adapter._split_runtime_is_effective() is False
