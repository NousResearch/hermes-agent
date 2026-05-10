"""Tests for agent/config.py — configuration dataclasses for AIAgent.

RED phase: These tests define the contract for the decomposed configuration
objects that replace the 60-parameter AIAgent constructor.

SOLID principles:
  - S (Single Responsibility): Each config class owns one concern
  - O (Open/Closed): Extensible via new fields without breaking consumers
  - L (Liskov): All configs are usable wherever their base type is expected
  - I (Interface Segregation): Consumers depend only on the config they need
  - D (Dependency Inversion): AIAgent depends on abstract config, not params

DRY: Validation, serialization, and default logic live once in each class.
"""

import pytest


# ---------------------------------------------------------------------------
# ProviderConfig
# ---------------------------------------------------------------------------

class TestProviderConfig:
    """ProviderConfig groups all provider/API routing parameters."""

    def test_create_with_defaults(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig()
        assert cfg.base_url == ""
        assert cfg.api_key is None
        assert cfg.provider == ""
        assert cfg.model == ""
        assert cfg.api_mode == "chat_completions"
        assert cfg.max_iterations == 90

    def test_create_with_all_fields(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-test",
            provider="openrouter",
            model="anthropic/claude-sonnet-4",
            api_mode="chat_completions",
            max_iterations=50,
            fallback_model={"model": "backup-model"},
            providers_allowed=["anthropic"],
            providers_ignored=["cohere"],
            providers_order=["anthropic", "google"],
            provider_sort="throughput",
            openrouter_min_coding_score=0.8,
            credential_pool=None,
            service_tier="auto",
        )
        assert cfg.base_url == "https://openrouter.ai/api/v1"
        assert cfg.api_key == "sk-test"
        assert cfg.provider == "openrouter"
        assert cfg.fallback_model == {"model": "backup-model"}
        assert cfg.max_iterations == 50

    def test_provider_normalized_to_lowercase(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(provider="OpenRouter")
        assert cfg.provider == "openrouter"

    def test_api_mode_auto_detection_anthropic_url(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(base_url="https://api.anthropic.com")
        assert cfg.api_mode == "anthropic_messages"

    def test_api_mode_auto_detection_bedrock_url(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(base_url="https://bedrock-runtime.us-east-1.amazonaws.com")
        assert cfg.api_mode == "bedrock_converse"

    def test_api_mode_explicit_overrides_detection(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(
            base_url="https://api.anthropic.com",
            api_mode="chat_completions",
        )
        assert cfg.api_mode == "chat_completions"

    def test_to_dict_roundtrip(self):
        from agent.config import ProviderConfig
        cfg = ProviderConfig(base_url="http://localhost:8080", model="test-model")
        d = cfg.to_dict()
        assert d["base_url"] == "http://localhost:8080"
        assert d["model"] == "test-model"
        cfg2 = ProviderConfig.from_dict(d)
        assert cfg2.base_url == cfg.base_url
        assert cfg2.model == cfg.model

    def test_immutability_idempotent(self):
        """Replacing the same value twice does not raise (idempotent replace)."""
        from agent.config import ProviderConfig
        cfg = ProviderConfig(model="a")
        cfg2 = cfg.replace(model="b")
        assert cfg2.model == "b"
        assert cfg.model == "a"  # original unchanged


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

class TestSessionConfig:
    """SessionConfig groups session identity and platform context."""

    def test_create_with_defaults(self):
        from agent.config import SessionConfig
        cfg = SessionConfig()
        assert cfg.session_id is None
        assert cfg.platform is None
        assert cfg.user_id is None
        assert cfg.chat_id is None

    def test_create_with_platform_context(self):
        from agent.config import SessionConfig
        cfg = SessionConfig(
            session_id="abc-123",
            platform="telegram",
            user_id="456",
            user_name="alice",
            chat_id="chat-789",
            chat_name="general",
            chat_type="group",
            thread_id="thread-001",
            gateway_session_key="agent:main:telegram:dm:123",
            parent_session_id="parent-abc",
        )
        assert cfg.platform == "telegram"
        assert cfg.user_name == "alice"
        assert cfg.gateway_session_key == "agent:main:telegram:dm:123"

    def test_to_dict_roundtrip(self):
        from agent.config import SessionConfig
        cfg = SessionConfig(platform="cli", session_id="s1")
        d = cfg.to_dict()
        cfg2 = SessionConfig.from_dict(d)
        assert cfg2.platform == "cli"
        assert cfg2.session_id == "s1"

    def test_generate_session_id_if_missing(self):
        from agent.config import SessionConfig
        cfg = SessionConfig()
        sid = cfg.effective_session_id
        assert sid is not None
        assert len(sid) > 0


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------

class TestBudgetConfig:
    """BudgetConfig groups iteration/token budget and trajectory settings."""

    def test_create_with_defaults(self):
        from agent.config import BudgetConfig
        cfg = BudgetConfig()
        assert cfg.max_iterations == 90
        assert cfg.save_trajectories is False
        assert cfg.max_tokens is None

    def test_create_custom(self):
        from agent.config import BudgetConfig
        cfg = BudgetConfig(
            max_iterations=30,
            save_trajectories=True,
            max_tokens=4096,
            reasoning_config={"effort": "high"},
        )
        assert cfg.max_iterations == 30
        assert cfg.save_trajectories is True
        assert cfg.max_tokens == 4096
        assert cfg.reasoning_config == {"effort": "high"}

    def test_to_dict_roundtrip(self):
        from agent.config import BudgetConfig
        cfg = BudgetConfig(max_iterations=10)
        d = cfg.to_dict()
        cfg2 = BudgetConfig.from_dict(d)
        assert cfg2.max_iterations == 10

    def test_build_iteration_budget(self):
        """BudgetConfig can produce an IterationBudget for the loop."""
        from agent.config import BudgetConfig
        cfg = BudgetConfig(max_iterations=50)
        budget = cfg.to_iteration_budget()
        assert budget.max_total == 50
        assert budget.remaining == 50


# ---------------------------------------------------------------------------
# CallbackConfig
# ---------------------------------------------------------------------------

class TestCallbackConfig:
    """CallbackConfig groups all callback hooks into one object."""

    def test_create_with_defaults(self):
        from agent.config import CallbackConfig
        cfg = CallbackConfig()
        assert cfg.tool_progress_callback is None
        assert cfg.stream_delta_callback is None
        assert cfg.clarify_callback is None

    def test_create_with_callbacks(self):
        from agent.config import CallbackConfig
        progress = lambda name, args: None
        delta = lambda text: None
        clarify = lambda q, c: "answer"

        cfg = CallbackConfig(
            tool_progress_callback=progress,
            stream_delta_callback=delta,
            clarify_callback=clarify,
        )
        assert cfg.tool_progress_callback is progress
        assert cfg.stream_delta_callback is delta
        assert cfg.clarify_callback is clarify

    def test_to_dict_excludes_callables(self):
        """Callables can't be serialized; to_dict skips them."""
        from agent.config import CallbackConfig
        cfg = CallbackConfig(tool_progress_callback=lambda: None)
        d = cfg.to_dict()
        assert "tool_progress_callback" not in d


# ---------------------------------------------------------------------------
# Integration: AIAgent accepts config objects
# ---------------------------------------------------------------------------

class TestAIAgentConfigIntegration:
    """Verify AIAgent can be constructed from config dataclasses.

    We patch the provider resolution to avoid needing real API keys.
    """

    @pytest.fixture(autouse=True)
    def _patch_provider(self, monkeypatch):
        """Prevent AIAgent from requiring a real LLM provider."""
        # Set a dummy API key so the provider resolution guard passes.
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-dummy")

    def test_agent_accepts_provider_config(self):
        from agent.config import ProviderConfig
        from run_agent import AIAgent
        pc = ProviderConfig(
            model="test-model",
            provider="openrouter",
            api_key="sk-test-dummy",
            base_url="https://openrouter.ai/api/v1",
        )
        agent = AIAgent(provider_config=pc, skip_context_files=True, skip_memory=True)
        assert agent.model == "test-model"
        assert agent.provider == "openrouter"

    def test_agent_accepts_session_config(self):
        from agent.config import SessionConfig
        from run_agent import AIAgent
        sc = SessionConfig(platform="telegram", session_id="s-123")
        agent = AIAgent(
            session_config=sc,
            api_key="sk-test-dummy",
            provider="openrouter",
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.platform == "telegram"
        assert agent.session_id == "s-123"

    def test_agent_accepts_budget_config(self):
        from agent.config import BudgetConfig
        from run_agent import AIAgent
        bc = BudgetConfig(max_iterations=42)
        agent = AIAgent(
            budget_config=bc,
            api_key="sk-test-dummy",
            provider="openrouter",
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.max_iterations == 42
        assert agent.iteration_budget.max_total == 42

    def test_agent_accepts_all_configs(self):
        from agent.config import ProviderConfig, SessionConfig, BudgetConfig, CallbackConfig
        from run_agent import AIAgent
        pc = ProviderConfig(model="m1", api_key="sk-test-dummy", provider="openrouter")
        sc = SessionConfig(platform="cli")
        bc = BudgetConfig(max_iterations=10)
        cc = CallbackConfig()
        agent = AIAgent(
            provider_config=pc,
            session_config=sc,
            budget_config=bc,
            callback_config=cc,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.model == "m1"
        assert agent.platform == "cli"
        assert agent.max_iterations == 10

    def test_backward_compat_positional_params_still_work(self):
        """Old-style construction must still work (backward compatibility)."""
        from run_agent import AIAgent
        agent = AIAgent(
            model="test-model",
            api_key="sk-test-dummy",
            provider="openrouter",
            max_iterations=30,
            platform="discord",
            session_id="legacy-123",
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.model == "test-model"
        assert agent.provider == "openrouter"
        assert agent.max_iterations == 30
        assert agent.platform == "discord"

    def test_config_objects_override_positional_params(self):
        """When both config objects and positional params are given,
        config objects take precedence."""
        from agent.config import ProviderConfig
        from run_agent import AIAgent
        pc = ProviderConfig(model="from-config", api_key="sk-test-dummy", provider="openrouter")
        agent = AIAgent(
            model="from-positional",
            provider_config=pc,
            skip_context_files=True,
            skip_memory=True,
        )
        assert agent.model == "from-config"
