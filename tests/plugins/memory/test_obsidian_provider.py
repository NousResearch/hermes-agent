"""Tests for the Obsidian memory provider plugin."""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.memory.obsidian import ObsidianMemoryProvider


@pytest.fixture()
def vault(tmp_path: Path, monkeypatch):
    vault = tmp_path / "vault"
    (vault / "Agent-Shared").mkdir(parents=True)
    (vault / "Agent-Hermes" / "daily").mkdir(parents=True)
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    monkeypatch.setattr(
        "plugins.memory.obsidian._load_config",
        lambda: {
            "memory": {
                "obsidian_checkpoint_tool_calls": 4,
                "obsidian_read_char_limit": 2500,
            }
        },
    )
    return vault


class TestObsidianProvider:
    def test_is_available_when_vault_exists(self, vault):
        provider = ObsidianMemoryProvider()
        assert provider.is_available() is True

    def test_initialize_creates_expected_layout(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")

        assert (vault / "Agent-Shared" / "user-profile.md").exists()
        assert (vault / "Agent-Shared" / "project-state.md").exists()
        assert (vault / "Agent-Shared" / "decisions-log.md").exists()
        assert (vault / "Agent-Hermes" / "working-context.md").exists()
        assert (vault / "Agent-Hermes" / "mistakes.md").exists()
        assert (vault / "Agent-Hermes" / "daily").exists()

    def test_system_prompt_block_mentions_required_notes(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")

        block = provider.system_prompt_block()
        assert "Obsidian vault memory layer" in block
        assert "Agent-Shared/user-profile.md" in block
        assert "Agent-Hermes/working-context.md" in block
        assert "Never write inside Agent-Aria/" in block

    def test_prefetch_reads_shared_and_private_notes(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")
        (vault / "Agent-Shared" / "user-profile.md").write_text("# Shared user profile\n\n- Danny prefers brevity\n")
        (vault / "Agent-Shared" / "project-state.md").write_text("# Shared project state\n\n- Hermes work matters\n")
        (vault / "Agent-Hermes" / "working-context.md").write_text("# Hermes working context\n\n- active task\n")
        today = next((vault / "Agent-Hermes" / "daily").glob("*.md"))
        today.write_text("# 2026-04-12\n\n- breadcrumb\n")

        context = provider.prefetch("what were we working on?", session_id="sess-1")
        assert "# Obsidian vault context" in context
        assert "Danny prefers brevity" in context
        assert "Hermes work matters" in context
        assert "active task" in context
        assert "breadcrumb" in context

    def test_on_turn_start_logs_session_start_once(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")

        provider.on_turn_start(1, "complete this task")
        provider.on_turn_start(2, "follow up")

        daily = next((vault / "Agent-Hermes" / "daily").glob("*.md")).read_text()
        working = (vault / "Agent-Hermes" / "working-context.md").read_text()
        assert daily.count("## Hermes session start") == 1
        assert "first task: complete this task" in daily
        assert "Initial request: complete this task" in working

    def test_tool_call_checkpoint_writes_every_configured_interval(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")
        provider.on_turn_start(1, "ship it")

        for i in range(3):
            provider.on_tool_call_complete("read_file", {"path": f"f{i}.md"}, "ok")

        daily_path = next((vault / "Agent-Hermes" / "daily").glob("*.md"))
        assert "## Hermes checkpoint" not in daily_path.read_text()

        provider.on_tool_call_complete("read_file", {"path": "f4.md"}, "ok")
        daily = daily_path.read_text()
        working = (vault / "Agent-Hermes" / "working-context.md").read_text()
        assert "## Hermes checkpoint" in daily
        assert "tools total: 4" in daily
        assert "read_file(path) -> ok" in daily
        assert "Turn 1, tool 4: read_file(path) -> ok" in working

    def test_on_pre_compress_marks_prefetch_for_next_turn(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")

        provider.prefetch("first")
        assert provider.prefetch("boring unrelated query") == ""

        provider.on_pre_compress([
            {"role": "user", "content": "compress me"},
            {"role": "assistant", "content": "done"},
        ])
        refreshed = provider.prefetch("boring unrelated query")
        assert "# Obsidian vault context" in refreshed

    def test_on_session_end_writes_summary(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")
        provider.on_turn_start(1, "wrap up")

        provider.on_session_end([
            {"role": "user", "content": "wrap up"},
            {"role": "assistant", "content": "done"},
        ])

        daily = next((vault / "Agent-Hermes" / "daily").glob("*.md")).read_text()
        working = (vault / "Agent-Hermes" / "working-context.md").read_text()
        assert "## Hermes session end" in daily
        assert "turns observed: 1" in daily
        assert "tool calls observed: 0" in daily
        assert "summary: user: wrap up | assistant: done" in daily
        assert "Session sess-1 ended." in working

    def test_on_memory_write_routes_user_entries_to_shared_profile(self, vault):
        provider = ObsidianMemoryProvider()
        provider.initialize("sess-1", platform="telegram")

        provider.on_memory_write("add", "user", "Danny likes concise updates")
        provider.on_memory_write("add", "memory", "Environment note")

        profile = (vault / "Agent-Shared" / "user-profile.md").read_text()
        working = (vault / "Agent-Hermes" / "working-context.md").read_text()
        assert "Danny likes concise updates" in profile
        assert "Environment note" in working
