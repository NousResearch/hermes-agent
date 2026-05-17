"""Tests for agent/profile.py — AgentProfile + ContextVar isolation."""

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.profile import (
    AgentProfile,
    get_active_profile,
    set_active_profile,
    use_profile,
    load_agent_registry,
    DEFAULT_AGENT_ID,
)


class TestAgentProfile:
    def test_default_profile(self):
        p = AgentProfile()
        assert p.id == DEFAULT_AGENT_ID
        assert p.home_dir is None
        assert p.model is None

    def test_resolved_home_fallback(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        p = AgentProfile()
        assert p.resolved_home == tmp_path

    def test_resolved_home_explicit(self, tmp_path):
        p = AgentProfile(home_dir=tmp_path / "profiles" / "coder")
        assert p.resolved_home == tmp_path / "profiles" / "coder"

    def test_resolved_home_expands_tilde(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/testuser")
        p = AgentProfile(home_dir="~/.hermes/profiles/coder")
        assert p.resolved_home == Path("/home/testuser/.hermes/profiles/coder")

    def test_soul_md_path(self, tmp_path):
        p = AgentProfile(home_dir=tmp_path)
        assert p.soul_md_path == tmp_path / "SOUL.md"

    def test_memory_dir(self, tmp_path):
        p = AgentProfile(home_dir=tmp_path)
        assert p.memory_dir == tmp_path / "memories"

    def test_skills_dir(self, tmp_path):
        p = AgentProfile(home_dir=tmp_path)
        assert p.skills_dir == tmp_path / "skills"

    def test_sessions_path(self, tmp_path):
        p = AgentProfile(home_dir=tmp_path)
        assert p.sessions_path == tmp_path / "sessions.json"

    def test_config_overrides(self):
        p = AgentProfile(config_overrides={"foo": "bar"})
        assert p.config_overrides == {"foo": "bar"}


class TestContextVar:
    def test_get_active_profile_default_none(self):
        """No profile set → None."""
        assert get_active_profile() is None

    def test_set_active_profile(self):
        p = AgentProfile(id="coder")
        token = set_active_profile(p)
        assert get_active_profile() == p
        # Cleanup
        from agent.profile import _current_agent_profile
        _current_agent_profile.reset(token)

    def test_use_profile_context_manager(self):
        p = AgentProfile(id="coder")
        assert get_active_profile() is None
        with use_profile(p):
            assert get_active_profile() == p
        assert get_active_profile() is None

    def test_use_profile_none_is_noop(self):
        """Passing None to use_profile is a no-op."""
        with use_profile(None):
            assert get_active_profile() is None

    def test_use_profile_nested(self):
        outer = AgentProfile(id="outer")
        inner = AgentProfile(id="inner")
        with use_profile(outer):
            assert get_active_profile() == outer
            with use_profile(inner):
                assert get_active_profile() == inner
            assert get_active_profile() == outer
        assert get_active_profile() is None

    def test_use_profile_exception_cleanup(self):
        p = AgentProfile(id="coder")
        with pytest.raises(ValueError):
            with use_profile(p):
                assert get_active_profile() == p
                raise ValueError("boom")
        assert get_active_profile() is None


class TestContextVarAsyncIsolation:
    """ContextVar must propagate through await but NOT leak to sibling tasks."""

    @pytest.mark.asyncio
    async def test_async_propagation(self):
        p = AgentProfile(id="coder")
        with use_profile(p):
            # await should keep the profile
            await asyncio.sleep(0)
            assert get_active_profile() == p
        assert get_active_profile() is None

    @pytest.mark.asyncio
    async def test_gather_isolation(self):
        """asyncio.gather with copy_context preserves per-task profiles."""
        async def task_a():
            with use_profile(AgentProfile(id="a")):
                await asyncio.sleep(0.01)
                return get_active_profile().id if get_active_profile() else None

        async def task_b():
            with use_profile(AgentProfile(id="b")):
                await asyncio.sleep(0.01)
                return get_active_profile().id if get_active_profile() else None

        # Tasks spawned from clean context — each sets its own profile
        results = await asyncio.gather(task_a(), task_b())
        assert set(results) == {"a", "b"}

    @pytest.mark.asyncio
    async def test_sibling_tasks_dont_leak(self):
        """A task spawned before profile is set should not see the profile."""
        barrier = asyncio.Event()

        async def child():
            barrier.set()
            await asyncio.sleep(0.05)
            return get_active_profile()

        task = asyncio.create_task(child())
        await barrier.wait()

        with use_profile(AgentProfile(id="parent")):
            await asyncio.sleep(0.01)
            assert get_active_profile().id == "parent"

        result = await task
        # Child was created before profile was set → should not see it
        assert result is None


class TestLoadAgentRegistry:
    def test_empty_config_returns_main(self):
        registry = load_agent_registry(None)
        assert "main" in registry
        assert registry["main"].id == "main"
        assert registry["main"].home_dir is None

    def test_single_agent(self):
        class FakeConfig:
            agents = {
                "coder": {"model": "claude-opus", "provider": "anthropic"},
            }

        registry = load_agent_registry(FakeConfig())
        assert "main" in registry
        assert "coder" in registry
        assert registry["coder"].model == "claude-opus"
        assert registry["coder"].provider == "anthropic"

    def test_multiple_agents(self):
        class FakeConfig:
            agents = {
                "coder": {"model": "claude-opus"},
                "research": {"model": "claude-sonnet"},
            }

        registry = load_agent_registry(FakeConfig())
        assert len(registry) == 3  # main + coder + research
        assert registry["coder"].model == "claude-opus"
        assert registry["research"].model == "claude-sonnet"

    def test_home_dir_expansion(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/test")

        class FakeConfig:
            agents = {
                "coder": {"home_dir": "~/.hermes/profiles/coder"},
            }

        registry = load_agent_registry(FakeConfig())
        assert registry["coder"].resolved_home == Path("/home/test/.hermes/profiles/coder")

    def test_config_overrides(self):
        class FakeConfig:
            agents = {
                "coder": {"model": "claude-opus", "custom_key": "custom_value"},
            }

        registry = load_agent_registry(FakeConfig())
        assert registry["coder"].config_overrides == {"custom_key": "custom_value"}

    def test_non_dict_agent_ignored(self):
        class FakeConfig:
            agents = {
                "coder": "not-a-dict",
                "research": {"model": "claude-sonnet"},
            }

        registry = load_agent_registry(FakeConfig())
        assert "coder" not in registry
        assert "research" in registry

    def test_main_always_present(self):
        class FakeConfig:
            agents = {"coder": {"model": "claude-opus"}}

        registry = load_agent_registry(FakeConfig())
        assert "main" in registry
        assert registry["main"].id == "main"
        assert registry["main"].home_dir is None

    def test_main_can_be_overridden(self):
        class FakeConfig:
            agents = {"main": {"model": "claude-opus"}}

        registry = load_agent_registry(FakeConfig())
        assert registry["main"].model == "claude-opus"
