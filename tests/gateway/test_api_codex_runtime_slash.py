"""API-server /codex-runtime slash dispatch."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def api_server():
    from gateway.platforms.api_server import APIServerAdapter

    server = object.__new__(APIServerAdapter)
    server._active_run_agents = {}
    server._run_statuses = {}
    server._background_tasks = set()
    server._persist_subagent_event = MagicMock(side_effect=lambda event, session_id=None: event)
    return server


def test_codex_runtime_query_returns_current_state(api_server):
    with patch("hermes_cli.codex_runtime_switch.get_current_runtime", return_value="auto"):
        with patch("hermes_cli.codex_runtime_switch.check_codex_binary_ok", return_value=(True, "0.130.0")):
            with patch("hermes_cli.config.load_config", return_value={"model": {}}):
                result = api_server._dispatch_slash_command("/codex-runtime", "session-1")

    assert result["type"] == "text"
    assert "openai_runtime: auto" in result["content"]


def test_codex_runtime_enable_persists(api_server):
    cfg = {"model": {"openai_runtime": "auto"}}

    with patch("hermes_cli.config.load_config", return_value=cfg):
        with patch("hermes_cli.config.save_config") as save_config:
            with patch("hermes_cli.codex_runtime_switch.check_codex_binary_ok", return_value=(True, "0.130.0")):
                with patch("hermes_cli.codex_runtime_plugin_migration.migrate"):
                    result = api_server._dispatch_slash_command(
                        "/codex-runtime on",
                        "session-1",
                    )

    assert result["type"] == "text"
    assert cfg["model"]["openai_runtime"] == "codex_app_server"
    save_config.assert_called_once()


def test_codex_runtime_blocks_while_agent_running(api_server):
    api_server._run_statuses["run-1"] = {
        "session_id": "session-1",
        "status": "running",
    }

    result = api_server._dispatch_slash_command("/codex-runtime off", "session-1")

    assert result["type"] == "error"
    assert "Agent is running" in result["message"]
