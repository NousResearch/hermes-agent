"""Session sidecar behavior for external cmux prompt enhancers."""

import json
import os
from pathlib import Path
from unittest.mock import patch

from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _make_agent(*, quiet_mode: bool = True, platform: str = "cli") -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        return AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=quiet_mode,
            platform=platform,
            skip_context_files=True,
            skip_memory=True,
        )


def test_cmux_surface_sidecar_is_opt_in_for_foreground_cli(monkeypatch):
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "workspace-abc")
    monkeypatch.setenv("CMUX_SURFACE_ID", "surface-123")
    monkeypatch.setenv("CMUX_SOCKET_PATH", "/tmp/cmux-test.sock")

    agent = _make_agent(quiet_mode=True, platform="cli")
    agent._publish_cmux_surface_sidecar = True
    agent._write_current_session_sidecar()

    hermes_home = Path(os.environ["HERMES_HOME"])
    sidecar = hermes_home / "current-sessions" / "by-cmux-surface" / "surface-123.json"
    assert sidecar.exists()

    payload = json.loads(sidecar.read_text())
    assert payload["session_id"] == agent.session_id
    assert payload["pid"] == os.getpid()
    assert payload["session_json"] == str(hermes_home / "sessions" / f"session_{agent.session_id}.json")
    assert payload["cmux"] == {
        "workspace_id": "workspace-abc",
        "surface_id": "surface-123",
        "socket_path": "/tmp/cmux-test.sock",
        "tty": "",
    }


def test_cmux_surface_sidecar_not_written_until_foreground_cli_opts_in(monkeypatch):
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "workspace-abc")
    monkeypatch.setenv("CMUX_SURFACE_ID", "surface-background")

    _make_agent(quiet_mode=True, platform="cli")

    hermes_home = Path(os.environ["HERMES_HOME"])
    sidecar = hermes_home / "current-sessions" / "by-cmux-surface" / "surface-background.json"
    assert not sidecar.exists()


def test_cmux_surface_sidecar_key_cannot_escape_directory(monkeypatch):
    monkeypatch.setenv("CMUX_WORKSPACE_ID", "workspace-abc")
    monkeypatch.setenv("CMUX_SURFACE_ID", "../surface-bad")

    agent = _make_agent(quiet_mode=True, platform="cli")
    agent._publish_cmux_surface_sidecar = True
    agent._write_current_session_sidecar()

    hermes_home = Path(os.environ["HERMES_HOME"])
    escaped = hermes_home / "current-sessions" / "surface-bad.json"
    sidecar = hermes_home / "current-sessions" / "by-cmux-surface" / ".._surface-bad.json"
    assert not escaped.exists()
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["session_id"] == agent.session_id
