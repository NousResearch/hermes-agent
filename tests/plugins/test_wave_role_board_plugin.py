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


def test_mode_router_classifies_and_sets_context(isolated_home):
    plugin = _load_plugin()

    council = plugin.classify_request("이건 council mode로 판단해줘")
    assert council["mode"] == "council"
    assert council["scope"] == "global"
    assert council["recommended_action"] == "gather_coda_clara_mira_nova_opinions_then_synthesize"

    scratch = plugin.classify_request("새 프로젝트 아이디어 잡아보자")
    assert scratch["mode"] == "scratch"
    assert scratch["scope"] == "scratch"

    project = plugin.classify_request("이 repo에서 테스트 고쳐줘")
    assert project["mode"] == "project"
    assert project["needs_project"] is True

    result = json.loads(plugin.wave_set_mode_handler({"mode": "scratch"}))
    assert result["success"] is True
    assert result["data"]["mode"] == "scratch"
    assert result["data"]["scope"] == "scratch"


def test_route_request_emits_chat_notes(isolated_home):
    plugin = _load_plugin()

    result = json.loads(plugin.wave_route_request_handler({
        "text": "방식 추천해줘",
        "emit_notes": True,
    }))

    assert result["success"] is True
    assert result["data"]["route"]["mode"] == "chat"
    messages = (isolated_home / "wave-hub" / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    roles = {json.loads(line)["role"] for line in messages}
    assert roles == {"Coda", "Clara", "Mira", "Nova"}


def test_council_note_handler_writes_opinions(isolated_home):
    plugin = _load_plugin()

    result = json.loads(plugin.wave_council_note_handler({
        "topic": "decision",
        "coda": "build it",
        "clara": "test it",
        "mira": "explain it",
        "nova": "operate it",
    }))

    assert result["success"] is True
    assert set(result["data"]["emitted_roles"]) == {"Coda", "Clara", "Mira", "Nova"}
    messages = [json.loads(line) for line in (isolated_home / "wave-hub" / "messages.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(msg["mode"] == "council" for msg in messages)
    assert {msg["status"] for msg in messages} == {"opinion"}


def test_pre_llm_call_injects_mode_context_and_notes(isolated_home):
    plugin = _load_plugin()

    ret = plugin._pre_llm_call(user_message="이거 어떻게 생각해?")

    assert ret and "Wave role-board routing" in ret["context"]
    assert "mode: chat" in ret["context"]
    messages = (isolated_home / "wave-hub" / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(messages) == 4
