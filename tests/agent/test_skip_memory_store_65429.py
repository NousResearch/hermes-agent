"""Regression test for issue #65429.

An agent built with ``skip_memory=True`` AND ``enabled_toolsets=["memory"]``
used to get a memory tool wired to ``store=None`` (the built-in
``MemoryStore`` was skipped along with the external provider), so every
memory call failed silently and the main auto-capture path was dead.

The fix creates the built-in store whenever memory is enabled in config OR the
memory toolset is explicitly enabled, while the external-provider block stays
gated on ``skip_memory``.
"""

import pytest

from run_agent import AIAgent


class _FakeOpenAI:
    def __init__(self, **kw):
        self.api_key = kw.get("api_key", "test")
        self.base_url = kw.get("base_url", "http://test")

    def close(self):
        pass


def _make_agent(
    monkeypatch, enabled_toolsets=None, disabled_toolsets=None, skip_memory=True
):
    monkeypatch.setattr("model_tools.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("model_tools.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", _FakeOpenAI)
    return AIAgent(
        api_key="test-key",
        base_url="http://test",
        provider="openrouter",
        api_mode="chat_completions",
        max_iterations=1,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=skip_memory,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
    )


def test_skip_memory_with_memory_toolset_creates_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)
    assert agent._memory_store is not None, (
        "memory toolset enabled despite skip_memory=True must still build "
        "the built-in store (#65429)"
    )






def test_skip_memory_memory_tool_handler_works_and_provider_skipped(
    monkeypatch, tmp_path
):
    """End-to-end behavioral check for #65429.

    The memory tool handler must actually WORK (not return the
    "Memory is not available" store=None error) on a skip_memory=True agent
    with the memory toolset enabled, while the external memory provider
    sync/prefetch stays skipped (no MemoryManager is created).
    """
    import json

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)

    # Provider sync/prefetch must remain skipped: skip_memory still gates the
    # external memory provider block.
    assert agent._memory_manager is None, (
        "skip_memory=True must still skip the external memory provider"
    )

    # Dispatch through the same entry point the tool executor uses
    # (agent/tool_executor.py wires store=agent._memory_store).
    from tools.memory_tool import memory_tool

    raw = memory_tool(
        action="add",
        target="memory",
        content="User prefers concise answers.",
        store=agent._memory_store,
    )
    result = json.loads(raw)
    assert result.get("success") is True, (
        f"memory tool handler must work with skip_memory=True + memory "
        f"toolset (#65429), got: {raw}"
    )
    assert "Memory is not available" not in raw

    # The write must actually persist to the profile-scoped memories dir.
    memory_md = tmp_path / "hm" / "memories" / "MEMORY.md"
    assert memory_md.exists()
    assert "User prefers concise answers." in memory_md.read_text()


def test_skip_memory_disabled_toolset_does_not_load_store(monkeypatch, tmp_path):
    """Cron shape: skip_memory=True, memory named in enabled AND disabled.

    #65429 must not load MEMORY.md just because the default cron toolset
    still lists memory while the denylist hides the tool.
    """
    home = tmp_path / "hm"
    monkeypatch.setenv("HERMES_HOME", str(home))
    mem_dir = home / "memories"
    mem_dir.mkdir(parents=True)
    secret = "cron-should-never-see-this-memory"
    (mem_dir / "MEMORY.md").write_text(secret + "\n")
    (mem_dir / "USER.md").write_text("cron-should-never-see-this-profile\n")

    agent = _make_agent(
        monkeypatch,
        enabled_toolsets=["memory", "file"],
        disabled_toolsets=["memory"],
        skip_memory=True,
    )
    assert agent._memory_store is None
    assert agent._memory_manager is None
    assert agent._memory_enabled is False
    assert agent._user_profile_enabled is False

    from agent.system_prompt import build_system_prompt_parts

    parts = build_system_prompt_parts(agent)
    blob = " ".join(str(v) for v in parts.values())
    assert secret not in blob
    assert "cron-should-never-see-this-profile" not in blob
