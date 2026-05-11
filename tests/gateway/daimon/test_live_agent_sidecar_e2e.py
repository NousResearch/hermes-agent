from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from run_agent import AIAgent


REPO_ROOT = Path(__file__).resolve().parents[3]
COMPOSE_FILE = REPO_ROOT / "docker" / "daimon-sandbox" / "docker-compose.yml"


def _tool_call(name: str, arguments: dict, call_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


def _assistant_message(*tool_calls: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(tool_calls=list(tool_calls))


def _make_agent() -> AIAgent:
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Execute shell commands",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    with (
        patch("run_agent.get_tool_definitions", return_value=tool_defs),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key",
            base_url="https://example.invalid/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            enabled_toolsets=["terminal"],
        )


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("DAIMON_LIVE_AGENT_E2E") != "1" or not os.getenv("GH_TOKEN_PATH"),
    reason="set DAIMON_LIVE_AGENT_E2E=1 and GH_TOKEN_PATH to run live Daimon sidecar tests",
)
def test_live_agent_terminal_paths_use_sidecar_without_token_extraction():
    original_terminal_env = {
        key: os.environ.get(key)
        for key in (
            "TERMINAL_ENV",
            "TERMINAL_CWD",
            "TERMINAL_DOCKER_IMAGE",
            "TERMINAL_DOCKER_EXEC_USER",
            "TERMINAL_DOCKER_NETWORK",
            "TERMINAL_DOCKER_VOLUMES",
        )
    }
    env = dict(os.environ)
    try:
        subprocess.run(
            ["docker", "compose", "-f", str(COMPOSE_FILE), "up", "-d", "--build"],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            timeout=300,
        )

        os.environ["TERMINAL_ENV"] = "docker"
        os.environ["TERMINAL_CWD"] = "/workspaces"
        os.environ["TERMINAL_DOCKER_IMAGE"] = os.getenv(
            "TERMINAL_DOCKER_IMAGE",
            "daimon-sandbox-daimon-sandbox:latest",
        )
        os.environ["TERMINAL_DOCKER_EXEC_USER"] = "1000:1000"
        os.environ["TERMINAL_DOCKER_NETWORK"] = "daimon-sandbox_daimon-net"
        os.environ["TERMINAL_DOCKER_VOLUMES"] = "[]"

        agent = _make_agent()
        messages: list[dict] = []

        allowed = _tool_call(
            "terminal",
            {"command": "gh issue list --search sidecar --limit 1"},
            "allowed",
        )
        auth_status = _tool_call(
            "terminal",
            {"command": "gh auth status"},
            "auth-status",
        )
        denied = _tool_call(
            "terminal",
            {"command": "gh auth token"},
            "denied",
        )
        probe = _tool_call(
            "terminal",
            {
                "command": (
                    "python - <<'PY'\n"
                    "import json, socket\n"
                    "s=socket.create_connection(('daimon-github-broker', 7842), timeout=5)\n"
                    "s.sendall(json.dumps({'argv':['auth','token']}).encode())\n"
                    "s.shutdown(socket.SHUT_WR)\n"
                    "print(s.recv(65536).decode())\n"
                    "PY"
                )
            },
            "probe",
        )
        sandbox_checks = _tool_call(
            "terminal",
            {
                "command": (
                    "sh -lc \"test ! -e /run/secrets/gh_token && "
                    "test ! -S /run/git-credentials.sock && "
                    "out=$(printf 'protocol=https\\nhost=github.com\\n\\n' | git credential fill 2>/dev/null || true); "
                    "! printf '%s' \"$out\" | grep -q '^password=' && printf no-secrets\""
                )
            },
            "sandbox-checks",
        )

        agent._execute_tool_calls_sequential(_assistant_message(allowed, auth_status, denied), messages, "daimon-live-seq")
        agent._execute_tool_calls_concurrent(_assistant_message(probe, sandbox_checks), messages, "daimon-live-conc")

        by_id = {message["tool_call_id"]: message["content"] for message in messages}
        assert "Error executing tool" not in by_id["allowed"]
        assert "Error: GitHub broker" not in by_id["allowed"]
        assert "config.yml" not in by_id["allowed"]
        assert "permission" not in by_id["allowed"].lower()
        assert "Authenticated via Daimon GitHub broker" in by_id["auth-status"]

        combined = "\n".join(message["content"] for message in messages)
        assert "Denied" in combined
        assert "auth" in combined
        assert "no-secrets" in combined
        assert "password=" not in combined
        assert "github_pat_" not in combined
        assert os.getenv("GH_TOKEN_PATH", "") not in combined
    finally:
        for key, value in original_terminal_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
