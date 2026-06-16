"""Tests for agent_context.py — AgentContext dataclass (CICS TCB model)."""

import json
import tempfile
from pathlib import Path

import pytest

from src.orchestration.agent_context import AgentContext


# ── helpers ──────────────────────────────────────────────────────────

@pytest.fixture
def tmp_agents_dir(tmp_path, monkeypatch):
    """Redirect ~/.hermes/ to a temp directory."""
    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    monkeypatch.setattr(
        "src.orchestration.agent_context.get_hermes_home",
        lambda: fake_home,
    )
    return fake_home / "agents"


# ── tests ────────────────────────────────────────────────────────────


class TestAgentContextFromDisk:

    def test_new_agent_creates_directory_with_defaults(self, tmp_agents_dir):
        """from_disk() with no prior files returns an empty context."""
        ctx = AgentContext.from_disk("test-agent")
        assert ctx.id == "test-agent"
        assert ctx.workspace == tmp_agents_dir / "test-agent"
        assert ctx.workspace.is_dir()
        assert ctx.memory == {}
        assert ctx.skills == []
        assert ctx.state == {}

    def test_loads_memory_from_md(self, tmp_agents_dir):
        """MEMORY.md is parsed into the memory dict."""
        ws = tmp_agents_dir / "mem-agent"
        ws.mkdir(parents=True)
        (ws / "MEMORY.md").write_text("lang: python\neditor: vscode\n")
        ctx = AgentContext.from_disk("mem-agent")
        assert ctx.memory == {"lang": "python", "editor": "vscode"}

    def test_loads_state_from_json(self, tmp_agents_dir):
        """state.json is deserialised on load."""
        ws = tmp_agents_dir / "state-agent"
        ws.mkdir(parents=True)
        (ws / "state.json").write_text(
            json.dumps({"last_turn": 42, "mode": "thinking"})
        )
        ctx = AgentContext.from_disk("state-agent")
        assert ctx.state == {"last_turn": 42, "mode": "thinking"}


class TestAgentContextPersistence:

    def test_save_state_writes_json(self, tmp_agents_dir):
        """save_state() persists state to state.json."""
        ctx = AgentContext(id="persist", workspace=tmp_agents_dir / "persist")
        ctx.workspace.mkdir(parents=True)
        ctx.state = {"key": "value", "nested": [1, 2]}
        ctx.save_state()
        loaded = json.loads((ctx.workspace / "state.json").read_text())
        assert loaded == {"key": "value", "nested": [1, 2]}

    def test_save_memory_writes_md(self, tmp_agents_dir):
        """save_memory() persists memory to MEMORY.md."""
        ctx = AgentContext(id="persist", workspace=tmp_agents_dir / "persist")
        ctx.workspace.mkdir(parents=True)
        ctx.memory = {"x": "1", "y": "2"}
        ctx.save_memory()
        content = (ctx.workspace / "MEMORY.md").read_text()
        assert "x: 1" in content
        assert "y: 2" in content
