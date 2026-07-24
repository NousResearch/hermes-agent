"""Tests for memory_mode parameter (replaces skip_memory bool).

Covers:
- AIAgent creation with all three modes (full, on_demand, off)
- Backward compat for deprecated skip_memory parameter (via init_agent)
- Delegate child agent memory_mode default and toolset injection
- System prompt volatile-tier gating based on _memory_mode
"""

from __future__ import annotations

import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────


def _write_config(tmp_path: Path, body: str = "") -> None:
    """Write a minimal config.yaml into the temp directory."""
    (tmp_path / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    """Create an AIAgent with sensible test defaults.

    Uses the same pattern as test_non_stream_stale_timeout.py.
    """
    from run_agent import AIAgent

    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


def _make_mock_parent(depth=0):
    """Create a mock parent agent for _build_child_agent tests.

    Mirrors the pattern in tests/tools/test_delegate.py.
    """
    import threading

    parent = MagicMock()
    parent.base_url = "https://openrouter.ai/api/v1"
    parent.api_key = "***"
    parent.provider = "openrouter"
    parent.api_mode = "chat_completions"
    parent.model = "anthropic/claude-sonnet-4"
    parent.platform = "cli"
    parent.providers_allowed = None
    parent.providers_ignored = None
    parent.providers_order = None
    parent.provider_sort = None
    parent._session_db = None
    parent._delegate_depth = depth
    parent._active_children = []
    parent._active_children_lock = threading.Lock()
    parent._print_fn = None
    parent.tool_progress_callback = None
    parent.thinking_callback = None
    parent.enabled_toolsets = None
    parent.valid_tool_names = []
    parent.disabled_toolsets = []
    parent.prefill_messages = None
    parent.fallback_model = None
    parent.max_tokens = None
    parent.reasoning_config = None
    parent._subagent_id = None
    parent._delegate_role = None
    parent.session_id = None
    return parent


# ── AIAgent creation tests ─────────────────────────────────────────────────


class TestMemoryModeCreation:
    """AIAgent creation with the memory_mode parameter."""

    def test_memory_mode_full_default(self, monkeypatch, tmp_path):
        """AIAgent() defaults _memory_mode to 'full'."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(tmp_path)

        agent = _make_agent(tmp_path)
        assert agent._memory_mode == "full"

    def test_memory_mode_on_demand(self, monkeypatch, tmp_path):
        """AIAgent(memory_mode='on_demand') sets _memory_mode == 'on_demand'."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(tmp_path)

        agent = _make_agent(tmp_path, memory_mode="on_demand")
        assert agent._memory_mode == "on_demand"

    def test_memory_mode_off_store_is_none(self, monkeypatch, tmp_path):
        """AIAgent(memory_mode='off') keeps _memory_store=None even with memory enabled."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(
            tmp_path,
            """\
memory:
  memory_enabled: true
""",
        )

        agent = _make_agent(tmp_path, memory_mode="off")
        assert agent._memory_store is None

    def test_memory_mode_off_memory_disabled(self, monkeypatch, tmp_path):
        """AIAgent(memory_mode='off') sets _memory_enabled False."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(
            tmp_path,
            """\
memory:
  memory_enabled: true
""",
        )

        agent = _make_agent(tmp_path, memory_mode="off")
        assert agent._memory_enabled is False


# ── Backward compat tests — call init_agent directly ────────────────────────
# skip_memory was removed from AIAgent.__init__ signature on this branch;
# the deprecation logic lives in init_agent.  We test it via init_agent
# directly to reach the backward-compat path.


class TestSkipMemoryBackwardCompat:
    """Deprecated skip_memory parameter backward compatibility (init_agent)."""

    def test_skip_memory_true_warns_and_maps_to_off(self, monkeypatch, tmp_path):
        """skip_memory=True emits DeprecationWarning and sets memory_mode='off'."""
        from run_agent import AIAgent
        from agent.agent_init import init_agent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(tmp_path)

        agent = AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            api_key="sk-dummy",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            platform="cli",
            memory_mode="full",
        )

        # Re-init with skip_memory=True to hit the backward compat path
        agent._memory_mode = None  # reset to detect the change
        with pytest.warns(DeprecationWarning, match="skip_memory"):
            init_agent(
                agent,
                base_url=agent.base_url,
                api_key=agent.api_key,
                provider=agent.provider,
                api_mode=agent.api_mode,
                model=agent.model,
                quiet_mode=True,
                skip_context_files=True,
                platform="cli",
                skip_memory=True,
            )

        assert agent._memory_mode == "off"

    def test_skip_memory_false_no_warning(self, monkeypatch, tmp_path):
        """skip_memory=False emits no DeprecationWarning and leaves memory_mode='full'."""
        from run_agent import AIAgent
        from agent.agent_init import init_agent

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        _write_config(tmp_path)

        agent = AIAgent(
            model="gpt-5.5",
            provider="openai-codex",
            api_key="sk-dummy",
            base_url="https://chatgpt.com/backend-api/codex",
            quiet_mode=True,
            skip_context_files=True,
            platform="cli",
            memory_mode="full",
        )

        agent._memory_mode = None
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            init_agent(
                agent,
                base_url=agent.base_url,
                api_key=agent.api_key,
                provider=agent.provider,
                api_mode=agent.api_mode,
                model=agent.model,
                quiet_mode=True,
                skip_context_files=True,
                platform="cli",
                skip_memory=False,
            )

        deprecation = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        assert len(deprecation) == 0
        assert agent._memory_mode == "full"


# ── Delegate child agent tests ─────────────────────────────────────────────


class TestDelegateChildAgent:
    """_build_child_agent memory_mode default and toolset injection."""

    def test_child_default_memory_mode_on_demand(self):
        """_build_child_agent defaults memory_mode to 'on_demand'."""
        from tools.delegate_tool import _build_child_agent

        parent = _make_mock_parent()

        with patch("tools.delegate_tool._load_config", return_value={}):
            with patch(
                "tools.delegate_tool._build_child_system_prompt", return_value=""
            ):
                child = _build_child_agent(
                    task_index=0,
                    goal="Test goal",
                    context="",
                    toolsets=None,
                    model="anthropic/claude-sonnet-4",
                    max_iterations=4,
                    task_count=1,
                    parent_agent=parent,
                )

        # AIAgent stores the value as _memory_mode (underscore prefix)
        assert child._memory_mode == "on_demand"

    def test_child_gets_memory_toolset_when_mode_not_off(self):
        """Child enabled_toolsets include 'memory' when memory_mode != 'off'."""
        from tools.delegate_tool import _build_child_agent

        parent = _make_mock_parent()

        with patch("tools.delegate_tool._load_config", return_value={}):
            with patch(
                "tools.delegate_tool._build_child_system_prompt", return_value=""
            ):
                child = _build_child_agent(
                    task_index=0,
                    goal="Test goal",
                    context="",
                    toolsets=["terminal", "file"],
                    model="anthropic/claude-sonnet-4",
                    max_iterations=4,
                    task_count=1,
                    parent_agent=parent,
                    memory_mode="on_demand",
                )

        assert "memory" in child.enabled_toolsets

    def test_child_no_memory_toolset_when_mode_off(self):
        """Child enabled_toolsets exclude 'memory' when memory_mode='off'."""
        from tools.delegate_tool import _build_child_agent

        parent = _make_mock_parent()

        with patch("tools.delegate_tool._load_config", return_value={}):
            with patch(
                "tools.delegate_tool._build_child_system_prompt", return_value=""
            ):
                child = _build_child_agent(
                    task_index=0,
                    goal="Test goal",
                    context="",
                    toolsets=["terminal", "file"],
                    model="anthropic/claude-sonnet-4",
                    max_iterations=4,
                    task_count=1,
                    parent_agent=parent,
                    memory_mode="off",
                )

        assert "memory" not in child.enabled_toolsets

    def test_child_gets_memory_toolset_with_full_mode(self):
        """Child enabled_toolsets include 'memory' when memory_mode='full'."""
        from tools.delegate_tool import _build_child_agent

        parent = _make_mock_parent()

        with patch("tools.delegate_tool._load_config", return_value={}):
            with patch(
                "tools.delegate_tool._build_child_system_prompt", return_value=""
            ):
                child = _build_child_agent(
                    task_index=0,
                    goal="Test goal",
                    context="",
                    toolsets=["terminal", "file"],
                    model="anthropic/claude-sonnet-4",
                    max_iterations=4,
                    task_count=1,
                    parent_agent=parent,
                    memory_mode="full",
                )

        assert "memory" in child.enabled_toolsets


# ── System prompt gating test ──────────────────────────────────────────────


class TestSystemPromptMemoryMode:
    """System prompt volatile tier gated by _memory_mode."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_minimal_agent(memory_mode: str) -> MagicMock:
        """Return a MagicMock agent wired for build_system_prompt_parts."""
        agent = MagicMock()
        agent._memory_store = MagicMock()
        agent._memory_enabled = True
        agent._user_profile_enabled = False
        agent._memory_mode = memory_mode

        # Keep stable / context tiers quiet
        agent.valid_tool_names = []
        agent.model = "gpt-5.5"
        agent.platform = "cli"
        agent.provider = "openai-codex"
        agent._tool_use_enforcement = False
        agent._task_completion_guidance = False
        agent._parallel_tool_call_guidance = False
        agent._environment_probe = False
        agent.load_soul_identity = False
        agent.skip_context_files = True
        agent.context_compressor = None
        agent._kanban_worker_guidance = None
        agent._memory_manager = None
        agent.pass_session_id = False
        agent.session_id = None
        return agent

    @staticmethod
    def _call_build_with_patches(agent: MagicMock):
        """Call build_system_prompt_parts with necessary dependency patches."""
        from agent.system_prompt import build_system_prompt_parts

        with patch("agent.system_prompt._ra") as mock_ra:
            mock_r = MagicMock()
            mock_r.load_soul_md.return_value = None
            mock_r.build_nous_subscription_prompt.return_value = ""
            mock_r.build_environment_hints.return_value = ""
            mock_r.build_context_files_prompt.return_value = ""
            mock_r.get_toolset_for_tool.return_value = None
            mock_r.build_skills_system_prompt.return_value = ""
            mock_ra.return_value = mock_r

            with patch(
                "agent.system_prompt.get_hermes_home", return_value="/tmp/hermes"
            ):
                with patch(
                    "agent.system_prompt.resolve_context_cwd", return_value="/tmp"
                ):
                    # _resolve_active_profile_name is imported locally inside
                    # build_system_prompt_parts — patch its origin module.
                    with patch(
                        "agent.file_safety._resolve_active_profile_name",
                        return_value="default",
                    ):
                        return build_system_prompt_parts(agent)

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_on_demand_skips_memory_blocks(self):
        """build_system_prompt_parts with _memory_mode='on_demand' omits memory."""
        agent = self._make_minimal_agent("on_demand")
        # Set a recognizable return so we can assert it's absent
        agent._memory_store.format_for_system_prompt.return_value = (
            "SHOULD_NOT_APPEAR"
        )
        parts = self._call_build_with_patches(agent)
        # The volatile tier still contains the timestamp line, but must NOT
        # contain the memory block.
        assert "SHOULD_NOT_APPEAR" not in parts["volatile"]
        # Sanity check: the timestamp line *is* present.
        assert "Conversation started:" in parts["volatile"]

    def test_full_mode_includes_memory_blocks(self):
        """build_system_prompt_parts with _memory_mode='full' includes memory."""
        agent = self._make_minimal_agent("full")
        agent._memory_store.format_for_system_prompt.return_value = (
            "MEMORY CONTENT HERE"
        )
        parts = self._call_build_with_patches(agent)
        assert "MEMORY CONTENT HERE" in parts["volatile"]
