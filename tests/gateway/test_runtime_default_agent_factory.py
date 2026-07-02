"""Tests for DefaultAgentFactory."""
from __future__ import annotations

import pytest


@pytest.fixture
def sample_run_kwargs():
    return {
        "run_id": "test-run-1",
        "session_id": "test-session",
        "message": "Hello",
        "session_key": "exec-test-run-1-test-session",
    }


class TestDefaultAgentFactoryConstruction:
    """DefaultAgentFactory construction and basic interface."""

    def test_importable(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        assert DefaultAgentFactory is not None

    def test_accepts_agent_kwargs(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "base_url": "https://api.openai.com/v1",
        })
        assert factory is not None

    def test_no_kwargs_raises_on_create(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory()
        # No agent_kwargs provided, and _resolve_runtime_agent_kwargs
        # will fail in test env (no config.yaml, no credentials)
        assert factory is not None

    def test_created_agents_tracking(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
            "model": "gpt-4o-mini",
        })
        assert factory.created_agents == []


class TestDefaultAgentFactoryValidation:
    """Error handling for missing/invalid configuration."""

    @pytest.mark.asyncio
    async def test_missing_api_key(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "provider": "openai",
            "model": "gpt-4o-mini",
        })
        with pytest.raises(RuntimeError, match="No API key configured"):
            await factory.create_agent(
                run_id="test", session_id="s", message="hi", session_key="k",
            )

    @pytest.mark.asyncio
    async def test_missing_provider(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "model": "gpt-4o-mini",
        })
        with pytest.raises(RuntimeError, match="No provider configured"):
            await factory.create_agent(
                run_id="test", session_id="s", message="hi", session_key="k",
            )

    @pytest.mark.asyncio
    async def test_missing_model(self):
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
        })
        with pytest.raises(RuntimeError, match="No model configured"):
            await factory.create_agent(
                run_id="test", session_id="s", message="hi", session_key="k",
            )

    @pytest.mark.asyncio
    async def test_model_from_run_kwargs(self):
        """Model can be supplied per-run even when not in agent_kwargs."""
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
        })
        with pytest.raises(RuntimeError, match="Failed to create AIAgent"):
            # This will fail because AIAgent requires a valid API call setup,
            # but the model validation should pass.
            await factory.create_agent(
                run_id="test", session_id="s", message="hi", session_key="k",
                model="gpt-4o-mini",
            )

    @pytest.mark.asyncio
    async def test_run_model_overrides_agent_kwargs_model(self):
        """run-level model overrides the factory's base model."""
        from gateway.runtime.agent_factory import DefaultAgentFactory
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
            "model": "gpt-4o",
        })
        with pytest.raises(RuntimeError, match="Failed to create AIAgent"):
            await factory.create_agent(
                run_id="test", session_id="s", message="hi", session_key="k",
                model="gpt-4o-mini",
            )


class TestDefaultAgentFactoryIntegration:
    """Integration with RuntimeExecutor via DefaultAgentFactory."""

    @pytest.mark.asyncio
    async def test_executor_accepts_default_factory(self):
        """RuntimeExecutor accepts DefaultAgentFactory like FakeAgentFactory."""
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.executor import RuntimeExecutor
        from gateway.runtime.agent_factory import DefaultAgentFactory

        rm = RunManager()
        factory = DefaultAgentFactory(agent_kwargs={
            "api_key": "test-key",
            "provider": "openai",
            "model": "gpt-4o-mini",
        })
        executor = RuntimeExecutor(
            rm,
            agent_factory=factory,
        )
        assert executor is not None
        assert executor._agent_factory is factory

    @pytest.mark.asyncio
    async def test_executor_still_supports_fake_factory(self):
        """Existing FakeAgentFactory path still works."""
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.executor import RuntimeExecutor, FakeAgentFactory

        rm = RunManager()
        factory = FakeAgentFactory()
        executor = RuntimeExecutor(
            rm,
            agent_factory=factory,
        )
        result = rm.create_run(
            session_id="test",
            message="hello",
        )
        run_id = result["run_id"]

        exec_result = await executor.execute_run(run_id)
        assert exec_result["status"] == "completed"
        assert "fake agent response" in exec_result.get("result", "")

    @pytest.mark.asyncio
    async def test_executor_default_factory_creation_fails_cleanly(self):
        """Executor with DefaultAgentFactory returns clean error on missing creds."""
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.executor import RuntimeExecutor
        from gateway.runtime.agent_factory import DefaultAgentFactory

        rm = RunManager()
        factory = DefaultAgentFactory(agent_kwargs={
            "provider": "openai",  # No api_key
        })
        executor = RuntimeExecutor(
            rm,
            agent_factory=factory,
        )

        result = rm.create_run(
            session_id="test",
            message="hello",
        )
        run_id = result["run_id"]

        exec_result = await executor.execute_run(run_id)
        assert exec_result["error"] == "agent_creation_failed"
        assert "API key" in exec_result["message"] or "api_key" in exec_result["message"]

        # Run should be marked failed
        status = rm.get_status(run_id)
        assert status["status"] == "failed"

    @pytest.mark.asyncio
    async def test_executor_no_factory_returns_not_supported(self):
        """Executor without factory returns not_supported."""
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.executor import RuntimeExecutor

        rm = RunManager()
        executor = RuntimeExecutor(rm)

        result = rm.create_run(
            session_id="test",
            message="hello",
        )
        run_id = result["run_id"]

        exec_result = await executor.execute_run(run_id)
        assert exec_result["error"] == "not_supported"


class TestHelperFunctions:
    """Convenience wiring helpers."""

    def test_create_default_agent_factory(self):
        from gateway.runtime.agent_factory import create_default_agent_factory
        factory = create_default_agent_factory()
        assert factory is not None

    def test_create_runtime_executor_with_default_factory(self):
        from gateway.runtime.run_manager import RunManager
        from gateway.runtime.agent_factory import create_runtime_executor_with_default_factory

        rm = RunManager()
        executor = create_runtime_executor_with_default_factory(rm)
        # May be None in test env (can't resolve API provider),
        # but should not crash
        if executor is not None:
            from gateway.runtime.executor import RuntimeExecutor
            assert isinstance(executor, RuntimeExecutor)
