"""Tests for the bundled wave-role-board plugin."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


def _load_plugin():
    repo_root = Path(__file__).resolve().parents[2]
    plugin_dir = repo_root / "plugins" / "wave-role-board"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.wave_role_board",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.wave_role_board"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.wave_role_board"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("WAVE_HUB_HOME", raising=False)
    return hermes_home


def test_wave_progress_writes_message_and_agent_status(isolated_home):
    plugin = _load_plugin()

    result = json.loads(plugin.wave_progress_handler({
        "role": "Coda",
        "message": "implementation started",
        "kind": "progress",
        "status": "running",
        "task": "build",
        "source": "test",
    }))

    assert result["success"] is True
    hub = isolated_home / "wave-hub"
    messages = (hub / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(messages) == 1
    rec = json.loads(messages[0])
    assert rec["role"] == "Coda"
    assert rec["text"] == "implementation started"
    assert rec["status"] == "running"
    assert rec["task"] == "build"

    agents = json.loads((hub / "agents.json").read_text(encoding="utf-8"))
    assert agents["Coda"]["status"] == "running"
    assert agents["Coda"]["task"] == "build"
    assert agents["Coda"]["message"] == "implementation started"


def test_board_status_handles_missing_viewers_and_restore_script(isolated_home):
    plugin = _load_plugin()
    plugin.append_progress("Nova", "status check", status="done", task="status")

    status = plugin.board_status()

    assert status["hub"] == str(isolated_home / "wave-hub")
    assert status["all_roles_running"] is False
    assert status["restore_available"] is False
    assert status["agents"]["Nova"]["status"] == "done"

    restore = plugin.restore_board()
    assert restore["restored"] is False
    assert "missing" in restore["reason"]


def test_tool_summarizer_maps_codex_claude_and_kanban():
    plugin = _load_plugin()

    assert plugin._summarize_tool("terminal", {"command": "codex exec task"})[0] == "Coda"
    assert plugin._summarize_tool("terminal", {"command": "claude review"})[0] == "Clara"
    assert plugin._summarize_tool("terminal", {"command": "hermes kanban list"})[0] == "Nova"
    assert plugin._summarize_tool("search_files", {"pattern": "x"})[0] == "Mira"
