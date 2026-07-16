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


def _make_agent(monkeypatch, enabled_toolsets=None, skip_memory=True):
    monkeypatch.setattr("run_agent.get_tool_definitions", lambda **kw: [])
    monkeypatch.setattr("run_agent.check_toolset_requirements", lambda: {})
    monkeypatch.setattr("run_agent.OpenAI", _FakeOpenAI)
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
    )


def test_skip_memory_with_memory_toolset_creates_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=True)
    assert agent._memory_store is not None, (
        "memory toolset enabled despite skip_memory=True must still build "
        "the built-in store (#65429)"
    )


def test_skip_memory_without_memory_toolset_has_no_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=None, skip_memory=True)
    assert agent._memory_store is None, (
        "flush/background agent with skip_memory and no memory toolset must "
        "have no built-in store"
    )


def test_memory_toolset_without_skip_memory_creates_store(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hm"))
    agent = _make_agent(monkeypatch, enabled_toolsets=["memory"], skip_memory=False)
    assert agent._memory_store is not None
