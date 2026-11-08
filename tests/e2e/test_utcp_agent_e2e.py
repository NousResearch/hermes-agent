"""End-to-end: a UTCP tool invoked through the normal Hermes agent flow.

Runs the *real* CLI (``hermes chat -q``) in a subprocess, with:

  * a local UTCP provider (``tests/tools/utcp_mock_provider.py``) as the tool
    source — the agent discovers it at startup, exactly like a user's provider;
  * a local OpenAI-compatible stub (``tests/tools/openai_stub_server.py``) as
    the model, so the turn is deterministic and needs no credentials or network;
  * an isolated ``HERMES_HOME`` so the user's config is never touched.

What it proves: the model's tool call reaches the UTCP provider over HTTP, the
provider's structured result reaches the model, and the final assistant message
carries it. That is the full path — config → discovery → registry → agent tool
snapshot → dispatch → UTCP transport → response — not a direct SDK call.

Marked ``integration`` (excluded by the default pytest addopts) because it
spawns a subprocess and takes ~10s::

    pytest tests/e2e/test_utcp_agent_e2e.py -m integration
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from tests.tools.openai_stub_server import OpenAIStubServer
from tests.tools.utcp_mock_provider import MockUtcpProvider, utcp_tool as mock_tool

pytestmark = [pytest.mark.integration]

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_home(tmp_path: Path, *, manual_url: str, base_url: str) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    config = {
        "model": {
            "default": "stub-model",
            "provider": "custom",
            "base_url": base_url,
        },
        "toolsets": ["hermes-cli"],
        # Keep the catalog eager so the stub model can call the tool directly
        # instead of going through the tool_search bridge.
        "tools": {"tool_search": {"enabled": "off"}},
        "utcp": {
            "enabled": True,
            "providers": [
                {
                    "name": "demo",
                    "call_template_type": "http",
                    "url": manual_url,
                    "timeout": 30,
                }
            ],
        },
        "agent": {"max_turns": 4},
    }
    (home / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (home / ".env").write_text("OPENAI_API_KEY=stub-key\n", encoding="utf-8")
    return home


def _run_cli(home: Path, prompt: str, *, timeout: int = 240) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(
        {
            "HERMES_HOME": str(home),
            "PYTHONPATH": str(REPO_ROOT),
            "HERMES_NO_UPDATE_CHECK": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from hermes_cli.main import main; sys.exit(main())",
            "chat",
            "-q",
            prompt,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_agent_calls_utcp_tool_end_to_end(tmp_path):
    provider = MockUtcpProvider(
        [
            mock_tool(
                "echo",
                "/echo",
                description="Echo a message back through the UTCP mock provider.",
                inputs={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string", "description": "Message to echo"}
                    },
                    "required": ["message"],
                },
            )
        ]
    ).start()
    stub = OpenAIStubServer(
        tool_name="utcp__demo__echo",
        tool_arguments={"message": "utcp-e2e-ok"},
    ).start()
    try:
        home = _write_home(tmp_path, manual_url=provider.manual_url, base_url=stub.base_url)
        proc = _run_cli(
            home,
            "Call utcp__demo__echo with message='utcp-e2e-ok', then report its output.",
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, f"hermes exited {proc.returncode}:\n{output[-4000:]}"

        # 1. The provider's manual was fetched at startup.
        assert [r for r in provider.requests if r.path == "/.well-known/utcp"], (
            f"UTCP discovery never ran:\n{output[-4000:]}"
        )
        # 2. The model's tool call reached the provider with its arguments.
        echo_calls = [r for r in provider.requests if r.path == "/echo"]
        assert echo_calls, f"UTCP tool was never executed:\n{output[-4000:]}"
        assert echo_calls[0].query.get("message") == ["utcp-e2e-ok"]
        # 3. The tool result came back into the agent's final answer.
        assert "TOOL_RESULT" in output
        assert "utcp-e2e-ok" in output
        # 4. Two model round trips: tool call, then final answer.
        assert len(stub.requests) >= 2
    finally:
        stub.stop()
        provider.stop()


def test_agent_reports_utcp_tool_in_its_inventory(tmp_path):
    """The tool is visible to the model, not just registered internally."""
    provider = MockUtcpProvider([mock_tool("ping", "/echo")]).start()
    stub = OpenAIStubServer(tool_name="utcp__demo__ping", tool_arguments={}).start()
    try:
        home = _write_home(tmp_path, manual_url=provider.manual_url, base_url=stub.base_url)
        proc = _run_cli(home, "Call utcp__demo__ping with no arguments.")
        output = (proc.stdout or "") + (proc.stderr or "")
        assert proc.returncode == 0, output[-4000:]
        tools_sent = {
            tool["function"]["name"]
            for request in stub.requests
            for tool in (request.get("tools") or [])
        }
        assert "utcp__demo__ping" in tools_sent, (
            "the UTCP tool was not in the model's tool list; "
            f"got {sorted(tools_sent)[:20]}"
        )
        assert [r for r in provider.requests if r.path == "/echo"], output[-4000:]
    finally:
        stub.stop()
        provider.stop()
