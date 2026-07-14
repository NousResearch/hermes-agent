"""Unit + init isolation tests for memory_provider_mode (off|tools|full).

Covers the PR #18565 salvage design:
- Built-in MEMORY.md/USER.md remain gated by skip_memory
- External providers can opt in independently via memory_provider_mode
- tools mode loads provider tools without auto prompt/prefetch/sync/session retain
- full mode is the interactive default when skip_memory=False
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from agent.memory_provider_mode import (
    VALID_MEMORY_PROVIDER_MODES,
    normalize_job_memory_provider,
    provider_lifecycle_enabled,
    provider_tools_enabled,
    resolve_memory_provider_mode,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestResolveMemoryProviderMode:
    def test_default_skip_memory_false_is_full(self):
        assert resolve_memory_provider_mode(skip_memory=False) == "full"

    def test_default_skip_memory_true_is_off(self):
        assert resolve_memory_provider_mode(skip_memory=True) == "off"

    def test_explicit_mode_wins_over_skip_memory(self):
        assert (
            resolve_memory_provider_mode(
                skip_memory=True, memory_provider_mode="tools"
            )
            == "tools"
        )
        assert (
            resolve_memory_provider_mode(
                skip_memory=True, memory_provider_mode="full"
            )
            == "full"
        )
        assert (
            resolve_memory_provider_mode(
                skip_memory=False, memory_provider_mode="off"
            )
            == "off"
        )

    def test_explicit_mode_wins_over_legacy_skip_memory_provider(self):
        assert (
            resolve_memory_provider_mode(
                skip_memory=True,
                memory_provider_mode="full",
                skip_memory_provider=True,
            )
            == "full"
        )

    def test_legacy_skip_memory_provider_true_is_off(self):
        assert (
            resolve_memory_provider_mode(
                skip_memory=False, skip_memory_provider=True
            )
            == "off"
        )

    def test_legacy_skip_memory_provider_false_is_tools(self):
        # Safer than historical global-on: enable tools without auto-sync.
        assert (
            resolve_memory_provider_mode(
                skip_memory=True, skip_memory_provider=False
            )
            == "tools"
        )

    def test_blank_explicit_mode_falls_through(self):
        assert (
            resolve_memory_provider_mode(
                skip_memory=True, memory_provider_mode="  "
            )
            == "off"
        )
        assert (
            resolve_memory_provider_mode(
                skip_memory=False, memory_provider_mode=""
            )
            == "full"
        )

    def test_case_insensitive_mode(self):
        assert (
            resolve_memory_provider_mode(memory_provider_mode="Tools") == "tools"
        )
        assert (
            resolve_memory_provider_mode(memory_provider_mode="FULL") == "full"
        )

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid memory_provider_mode"):
            resolve_memory_provider_mode(memory_provider_mode="half")


class TestNormalizeJobMemoryProvider:
    def test_none_and_blank_become_none(self):
        assert normalize_job_memory_provider(None) is None
        assert normalize_job_memory_provider("") is None
        assert normalize_job_memory_provider("   ") is None

    def test_valid_modes(self):
        assert normalize_job_memory_provider("off") == "off"
        assert normalize_job_memory_provider("tools") == "tools"
        assert normalize_job_memory_provider("FULL") == "full"

    def test_bools_rejected(self):
        with pytest.raises(ValueError, match="memory_provider must be"):
            normalize_job_memory_provider(True)
        with pytest.raises(ValueError, match="memory_provider must be"):
            normalize_job_memory_provider(False)

    def test_invalid_string_rejected(self):
        with pytest.raises(ValueError, match="Invalid memory_provider"):
            normalize_job_memory_provider("sometimes")


class TestProviderFlagHelpers:
    def test_lifecycle_only_full(self):
        assert provider_lifecycle_enabled("full") is True
        assert provider_lifecycle_enabled("tools") is False
        assert provider_lifecycle_enabled("off") is False

    def test_tools_enabled_for_tools_and_full(self):
        assert provider_tools_enabled("tools") is True
        assert provider_tools_enabled("full") is True
        assert provider_tools_enabled("off") is False

    def test_valid_modes_set(self):
        assert VALID_MEMORY_PROVIDER_MODES == frozenset({"off", "tools", "full"})


# ---------------------------------------------------------------------------
# AIAgent init isolation
# ---------------------------------------------------------------------------


class RecordingMemoryProvider:
    name = "recording"

    def __init__(self):
        self.init_kwargs = None
        self.init_session_id = None
        self.sync_calls = []
        self.prefetch_calls = []
        self.session_end_calls = []

    def is_available(self):
        return True

    def initialize(self, session_id, **kwargs):
        self.init_session_id = session_id
        self.init_kwargs = dict(kwargs)

    def get_tool_schemas(self):
        return [{"name": "recording_recall", "description": "test"}]

    def sync_turn(self, *args, **kwargs):
        self.sync_calls.append((args, kwargs))

    def prefetch(self, query, **kwargs):
        self.prefetch_calls.append(query)
        return ""

    def on_session_end(self, messages):
        self.session_end_calls.append(messages)

    def shutdown(self):
        pass

    def system_prompt_block(self):
        return "EXTERNAL MEMORY BLOCK"


@contextlib.contextmanager
def _agent_patches(cfg, provider=None):
    """Common patches for constructing AIAgent without network/tool discovery."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("hermes_cli.config.load_config", return_value=cfg))
        stack.enter_context(
            patch("plugins.memory.load_memory_provider", return_value=provider)
        )
        stack.enter_context(
            patch("agent.model_metadata.get_model_context_length", return_value=204_800)
        )
        stack.enter_context(patch("run_agent.get_tool_definitions", return_value=[]))
        stack.enter_context(patch("run_agent.check_toolset_requirements", return_value={}))
        stack.enter_context(patch("run_agent.OpenAI"))
        yield


class TestAIAgentMemoryProviderModeIsolation:
    def test_skip_memory_true_default_mode_off_no_provider(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording", "memory_enabled": True}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

        assert agent._memory_provider_mode == "off"
        assert agent._memory_manager is None
        assert agent._memory_provider_auto_sync is False
        assert agent._memory_provider_prefetch is False
        assert agent._memory_provider_prompt_context is False
        # Built-in memory stays off under skip_memory=True
        assert getattr(agent, "_memory_store", None) in (None, False) or not getattr(
            agent, "_memory_enabled", False
        )

    def test_tools_mode_loads_provider_without_lifecycle(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording", "memory_enabled": True}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                memory_provider_mode="tools",
                session_id="cron-sess",
                platform="cron",
            )

        assert agent._memory_provider_mode == "tools"
        assert agent._memory_manager is not None
        assert agent._memory_provider_auto_sync is False
        assert agent._memory_provider_prefetch is False
        assert agent._memory_provider_prompt_context is False
        # Built-in local memory still skipped (cron safety)
        assert not getattr(agent, "_memory_enabled", False)
        assert provider.init_session_id == "cron-sess"

    def test_full_mode_with_skip_memory_still_skips_builtin(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording", "memory_enabled": True}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                memory_provider_mode="full",
            )

        assert agent._memory_provider_mode == "full"
        assert agent._memory_manager is not None
        assert agent._memory_provider_auto_sync is True
        assert agent._memory_provider_prefetch is True
        assert agent._memory_provider_prompt_context is True
        assert not getattr(agent, "_memory_enabled", False)

    def test_default_interactive_is_full_when_skip_memory_false(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
            )

        assert agent._memory_provider_mode == "full"
        assert agent._memory_manager is not None
        assert agent._memory_provider_auto_sync is True

    def test_legacy_skip_memory_provider_false_maps_to_tools(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                skip_memory_provider=False,
            )

        assert agent._memory_provider_mode == "tools"
        assert agent._memory_manager is not None
        assert agent._memory_provider_auto_sync is False

    def test_invalid_mode_defaults_to_off(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                memory_provider_mode="bogus",
            )

        assert agent._memory_provider_mode == "off"
        assert agent._memory_manager is None

    def test_tools_mode_skips_auto_sync_and_session_end(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                memory_provider_mode="tools",
                session_id="s1",
            )

        # sync_all path is on the manager; spy at manager level
        mm = agent._memory_manager
        assert mm is not None
        mm.sync_all = MagicMock()
        mm.queue_prefetch_all = MagicMock()
        mm.on_session_end = MagicMock()
        mm.shutdown_all = MagicMock()

        agent._sync_external_memory_for_turn(
            original_user_message="hello",
            final_response="world",
            interrupted=False,
        )
        mm.sync_all.assert_not_called()
        mm.queue_prefetch_all.assert_not_called()

        agent.shutdown_memory_provider(messages=[{"role": "user", "content": "x"}])
        mm.on_session_end.assert_not_called()
        mm.shutdown_all.assert_called_once()

    def test_full_mode_runs_auto_sync(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                memory_provider_mode="full",
                session_id="s2",
            )

        mm = agent._memory_manager
        mm.sync_all = MagicMock()
        mm.queue_prefetch_all = MagicMock()

        agent._sync_external_memory_for_turn(
            original_user_message="hello",
            final_response="world",
            interrupted=False,
        )
        mm.sync_all.assert_called_once()
        mm.queue_prefetch_all.assert_called_once()

    def test_tools_mode_omits_provider_system_prompt_block(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                memory_provider_mode="tools",
            )

        from agent.system_prompt import build_system_prompt_parts

        parts = build_system_prompt_parts(agent)
        volatile = parts.get("volatile") or ""
        assert "EXTERNAL MEMORY BLOCK" not in volatile

    def test_full_mode_includes_provider_system_prompt_block(self):
        provider = RecordingMemoryProvider()
        cfg = {"memory": {"provider": "recording"}, "agent": {}}
        with _agent_patches(cfg, provider):
            from run_agent import AIAgent

            agent = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=False,
                memory_provider_mode="full",
            )

        from agent.system_prompt import build_system_prompt_parts

        parts = build_system_prompt_parts(agent)
        volatile = parts.get("volatile") or ""
        assert "EXTERNAL MEMORY BLOCK" in volatile
