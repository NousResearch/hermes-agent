"""Unit tests for run_agent.py (AIAgent) — build_api_kwargs, system-prompt building, tool-use enforcement, max-tokens.

Split out of the former monolithic ``tests/run_agent/test_run_agent.py`` (which
outgrew the per-file CI wall-clock cap). Shared fixtures live in ``conftest.py``;
mock-builders in ``_run_agent_helpers.py``.
"""

import json
from unittest.mock import MagicMock, patch
import run_agent
from run_agent import AIAgent
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY

from tests.run_agent._run_agent_helpers import (
    _make_tool_defs,
)


class TestBuildSystemPrompt:
    def test_always_has_identity(self, agent):
        prompt = agent._build_system_prompt()
        assert DEFAULT_AGENT_IDENTITY in prompt

    def test_can_use_soul_identity_even_when_context_files_are_skipped(self):
        with (
            patch("run_agent.get_tool_definitions", return_value=_make_tool_defs("terminal")),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch("run_agent.load_soul_md", return_value="SOUL IDENTITY"),
        ):
            agent = AIAgent(
                api_key="test-k...7890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                load_soul_identity=True,
                skip_memory=True,
            )
            prompt = agent._build_system_prompt()

        assert "SOUL IDENTITY" in prompt
        assert DEFAULT_AGENT_IDENTITY not in prompt

    def test_includes_system_message(self, agent):
        prompt = agent._build_system_prompt(system_message="Custom instruction")
        assert "Custom instruction" in prompt

    def test_memory_guidance_when_memory_tool_loaded(self, agent_with_memory_tool):
        from agent.prompt_builder import MEMORY_GUIDANCE

        prompt = agent_with_memory_tool._build_system_prompt()
        assert MEMORY_GUIDANCE in prompt

    def test_no_memory_guidance_without_tool(self, agent):
        from agent.prompt_builder import MEMORY_GUIDANCE

        prompt = agent._build_system_prompt()
        assert MEMORY_GUIDANCE not in prompt

    def test_includes_datetime(self, agent):
        prompt = agent._build_system_prompt()
        # Should contain current date info like "Conversation started:"
        assert "Conversation started:" in prompt

    def test_datetime_is_date_only_not_minute_precision(self, agent):
        """Timestamp must be date-only (no HH:MM) so the system prompt
        stays byte-stable for the full day. Minute precision invalidates
        prefix-cache KV on every rebuild path (compression, fresh-agent
        gateway turns, session resume without a stored prompt)."""
        prompt = agent._build_system_prompt()
        # Find the line and strip it for inspection
        for line in prompt.splitlines():
            if line.startswith("Conversation started:"):
                # Must NOT contain AM/PM indicator (minute precision had %I:%M %p)
                assert " AM" not in line and " PM" not in line, (
                    f"Timestamp line has time-of-day, breaks daily cache stability: {line!r}"
                )
                # Must NOT contain a colon followed by two digits (HH:MM pattern)
                import re as _re
                assert not _re.search(r":\d{2}", line), (
                    f"Timestamp line has HH:MM, breaks daily cache stability: {line!r}"
                )
                break
        else:
            assert False, "Expected a 'Conversation started:' line in the system prompt"

    def test_includes_nous_subscription_prompt(self, agent, monkeypatch):
        monkeypatch.setattr(run_agent, "build_nous_subscription_prompt", lambda tool_names: "NOUS SUBSCRIPTION BLOCK")
        prompt = agent._build_system_prompt()
        assert "NOUS SUBSCRIPTION BLOCK" in prompt

    def test_skills_prompt_derives_available_toolsets_from_loaded_tools(self):
        tools = _make_tool_defs("web_search", "skills_list", "skill_view", "skill_manage")
        toolset_map = {
            "web_search": "web",
            "skills_list": "skills",
            "skill_view": "skills",
            "skill_manage": "skills",
        }

        with (
            patch("run_agent.get_tool_definitions", return_value=tools),
            patch(
                "run_agent.check_toolset_requirements",
                side_effect=AssertionError("should not re-check toolset requirements"),
            ),
            patch("run_agent.get_toolset_for_tool", create=True, side_effect=toolset_map.get),
            patch("run_agent.build_skills_system_prompt", return_value="SKILLS_PROMPT") as mock_skills,
            patch("run_agent.OpenAI"),
        ):
            agent = AIAgent(
                api_key="test-k...7890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )

            prompt = agent._build_system_prompt()

        assert "SKILLS_PROMPT" in prompt
        assert mock_skills.call_args.kwargs["available_tools"] == set(toolset_map)
        assert mock_skills.call_args.kwargs["available_toolsets"] == {"web", "skills"}


class TestToolUseEnforcementConfig:
    """Tests for the agent.tool_use_enforcement config option."""

    def _make_agent(self, model="openai/gpt-4.1", tool_use_enforcement="auto"):
        """Create an agent with tools and a specific enforcement config."""
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("terminal", "web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"tool_use_enforcement": tool_use_enforcement}},
            ),
        ):
            a = AIAgent(
                model=model,
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            return a

    def test_auto_injects_for_gpt(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_injects_for_codex(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/codex-mini", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_skips_for_claude(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_auto_injects_for_grok(self):
        """xAI Grok / xai-oauth models hit the same enforcement path as GPT."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="x-ai/grok-4.3", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_injects_for_qwen(self):
        """Qwen models default to chatty/hallucinatory tool use without enforcement."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="qwen/qwen-plus", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_injects_for_deepseek(self):
        """DeepSeek models default to chatty/hallucinatory tool use without enforcement."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="deepseek/deepseek-r1", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_auto_injects_execution_guidance_for_grok(self):
        """Grok also gets OPENAI_MODEL_EXECUTION_GUIDANCE (verification,
        mandatory_tool_use, act_dont_ask). Same failure modes as GPT in
        practice — claims completion without tool calls, suggests workarounds
        instead of using existing tools.
        """
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(model="x-ai/grok-4.3", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert OPENAI_MODEL_EXECUTION_GUIDANCE in prompt

    def test_auto_injects_execution_guidance_for_xai_oauth_model(self):
        """xai-oauth bare model names (no slash) also match the grok pattern."""
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(model="grok-4.3", tool_use_enforcement="auto")
        prompt = agent._build_system_prompt()
        assert OPENAI_MODEL_EXECUTION_GUIDANCE in prompt

    def test_auto_does_not_inject_execution_guidance_for_claude(self):
        """Sanity: execution guidance stays off for non-targeted families."""
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(
            model="anthropic/claude-sonnet-4", tool_use_enforcement="auto"
        )
        prompt = agent._build_system_prompt()
        assert OPENAI_MODEL_EXECUTION_GUIDANCE not in prompt

    def test_true_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement=True)
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_string_true_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-sonnet-4", tool_use_enforcement="true")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_always_forces_for_all_models(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="deepseek/deepseek-r1", tool_use_enforcement="always")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_false_disables_for_gpt(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement=False)
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_string_false_disables(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(model="openai/gpt-4.1", tool_use_enforcement="off")
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_custom_list_matches(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="deepseek/deepseek-r1",
            tool_use_enforcement=["deepseek", "gemini"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_custom_list_no_match(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="anthropic/claude-sonnet-4",
            tool_use_enforcement=["deepseek", "gemini"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt

    def test_custom_list_case_insensitive(self):
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        agent = self._make_agent(
            model="openai/GPT-4.1",
            tool_use_enforcement=["GPT", "Codex"],
        )
        prompt = agent._build_system_prompt()
        assert TOOL_USE_ENFORCEMENT_GUIDANCE in prompt

    def test_no_tools_never_injects(self):
        """Even with enforcement=true, no injection when agent has no tools."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"tool_use_enforcement": True}},
            ),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
            a.client = MagicMock()
            prompt = a._build_system_prompt()
            assert TOOL_USE_ENFORCEMENT_GUIDANCE not in prompt


class TestTaskCompletionGuidance:
    """Tests for the universal task-completion / no-fabrication guidance
    (config.yaml ``agent.task_completion_guidance``).

    Unlike tool_use_enforcement, this block is model-family-agnostic — it
    targets cross-model failure modes (stopping after a stub; fabricating
    output when blocked) and should appear for every model by default."""

    def _make_agent(self, model="anthropic/claude-opus-4.8",
                    task_completion_guidance=True, **extra_cfg):
        agent_cfg = {"task_completion_guidance": task_completion_guidance}
        agent_cfg.update(extra_cfg)
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("terminal", "web_search"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": agent_cfg},
            ),
        ):
            a = AIAgent(
                model=model,
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            return a

    def test_default_injects_for_claude(self):
        """The block must reach Claude by default — that's the
        primary motivating model family."""
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        agent = self._make_agent(model="anthropic/claude-opus-4.8")
        prompt = agent._build_system_prompt()
        assert TASK_COMPLETION_GUIDANCE in prompt

    def test_default_injects_for_deepseek(self):
        """And for DeepSeek — the other model that failed the Sarasota
        real-estate task by fabricating output."""
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        agent = self._make_agent(model="deepseek/deepseek-v4-flash")
        prompt = agent._build_system_prompt()
        assert TASK_COMPLETION_GUIDANCE in prompt

    def test_default_injects_for_gpt(self):
        """Also reaches model families that already get enforcement —
        it's additive, not exclusive."""
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        agent = self._make_agent(model="openai/gpt-5.4")
        prompt = agent._build_system_prompt()
        assert TASK_COMPLETION_GUIDANCE in prompt

    def test_false_disables(self):
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        agent = self._make_agent(
            model="anthropic/claude-opus-4.8", task_completion_guidance=False
        )
        prompt = agent._build_system_prompt()
        assert TASK_COMPLETION_GUIDANCE not in prompt

    def test_no_tools_no_injection(self):
        """Same gate as tool_use_enforcement — no tools means no guidance.
        The guidance refers to ``tool calls`` and ``tool output``; without
        tools it would be advice for a capability the agent doesn't have."""
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        with (
            patch("run_agent.get_tool_definitions", return_value=[]),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"task_completion_guidance": True}},
            ),
        ):
            a = AIAgent(
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                enabled_toolsets=[],
            )
            a.client = MagicMock()
            assert TASK_COMPLETION_GUIDANCE not in a._build_system_prompt()


class TestEnvironmentProbeIntegration:
    """Tests for the local Python toolchain probe wiring (config.yaml
    ``agent.environment_probe``).  The probe itself is unit-tested in
    tests/tools/test_env_probe.py; this class confirms it lands in the
    system prompt when enabled and stays out when disabled."""

    def _make_agent(self, model="anthropic/claude-opus-4.8",
                    environment_probe=True):
        with (
            patch(
                "run_agent.get_tool_definitions",
                return_value=_make_tool_defs("terminal"),
            ),
            patch("run_agent.check_toolset_requirements", return_value={}),
            patch("run_agent.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"environment_probe": environment_probe}},
            ),
        ):
            a = AIAgent(
                model=model,
                api_key="test-key-1234567890",
                base_url="https://openrouter.ai/api/v1",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
            )
            a.client = MagicMock()
            return a

    def test_probe_appears_when_problem_detected(self, monkeypatch):
        """When the probe finds something off, the line lands in the prompt."""
        from tools import env_probe
        env_probe._reset_cache_for_tests()
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: {"python3": "3.11.15"}.get(b))
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which",
                            lambda name: None if name == "uv" else "/usr/bin/" + name)

        agent = self._make_agent(environment_probe=True)
        prompt = agent._build_system_prompt()
        assert "Python toolchain:" in prompt
        assert "3.11.15" in prompt

    def test_probe_silent_on_clean_env(self, monkeypatch):
        """Clean environment → probe emits nothing → no line in prompt."""
        from tools import env_probe
        env_probe._reset_cache_for_tests()
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: "3.13.3" if b == "python3" else None)
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: True)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: False)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.13")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        agent = self._make_agent(environment_probe=True)
        prompt = agent._build_system_prompt()
        assert "Python toolchain:" not in prompt

    def test_probe_disabled_by_config(self, monkeypatch):
        """Even with detectable problems, the probe stays out when disabled."""
        from tools import env_probe
        env_probe._reset_cache_for_tests()
        monkeypatch.setattr(env_probe, "_python_version_of",
                            lambda b: {"python3": "3.11.15"}.get(b))
        monkeypatch.setattr(env_probe, "_has_pip_module", lambda b: False)
        monkeypatch.setattr(env_probe, "_detect_pep668", lambda b: True)
        monkeypatch.setattr(env_probe, "_pip_python_version", lambda: "3.12")
        monkeypatch.setattr(env_probe.shutil, "which", lambda name: None)

        agent = self._make_agent(environment_probe=False)
        prompt = agent._build_system_prompt()
        assert "Python toolchain:" not in prompt


class TestInvalidateSystemPrompt:
    def test_clears_cache(self, agent):
        agent._cached_system_prompt = "cached value"
        agent._invalidate_system_prompt()
        assert agent._cached_system_prompt is None

    def test_reloads_memory_store(self, agent):
        mock_store = MagicMock()
        agent._memory_store = mock_store
        agent._cached_system_prompt = "cached"
        agent._invalidate_system_prompt()
        mock_store.load_from_disk.assert_called_once()


class TestBuildApiKwargs:
    def test_basic_kwargs(self, agent):
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["model"] == agent.model
        assert kwargs["messages"] is messages
        assert kwargs["timeout"] == 1800.0

    def test_public_moonshot_kimi_k2_5_omits_temperature(self, agent):
        """Kimi models should NOT have client-side temperature overrides.

        The Kimi gateway selects the correct temperature server-side.
        """
        agent.base_url = "https://api.moonshot.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-k2.5"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert "temperature" not in kwargs

    def test_public_moonshot_cn_kimi_k2_5_omits_temperature(self, agent):
        agent.base_url = "https://api.moonshot.cn/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-k2.5"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert "temperature" not in kwargs

    def test_kimi_coding_endpoint_omits_temperature(self, agent):
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-k2.5"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert "temperature" not in kwargs

    def test_kimi_coding_endpoint_sends_max_tokens_and_reasoning(self, agent):
        """Kimi endpoint sends max_tokens=32000. With no reasoning_config it
        defaults to the thinking toggle (xor contract: never paired with a
        top-level reasoning_effort)."""
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-for-coding"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["max_tokens"] == 32000
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kwargs

    def test_kimi_coding_endpoint_respects_custom_effort(self, agent):
        """reasoning_effort should reflect reasoning_config.effort when set."""
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-for-coding"
        agent.reasoning_config = {"enabled": True, "effort": "high"}
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["reasoning_effort"] == "high"

    def test_kimi_coding_endpoint_sends_thinking_extra_body(self, agent):
        """Kimi endpoint should send extra_body.thinking={"type":"enabled"}
        to activate reasoning mode, mirroring Kimi CLI's with_thinking()."""
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-for-coding"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}

    def test_kimi_coding_endpoint_disables_thinking(self, agent):
        """When reasoning_config.enabled=False, thinking should be disabled
        and reasoning_effort should be omitted entirely — mirroring Kimi
        CLI's with_thinking("off") which maps to reasoning_effort=None."""
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.kimi.com/coding/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-for-coding"
        agent.reasoning_config = {"enabled": False}
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["extra_body"]["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in kwargs

    def test_moonshot_endpoint_sends_max_tokens_and_reasoning(self, agent):
        """api.moonshot.ai should get the same Kimi-compatible params."""
        agent.provider = "kimi-coding"
        agent.base_url = "https://api.moonshot.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-k2.5"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["max_tokens"] == 32000
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kwargs

    def test_moonshot_cn_endpoint_sends_max_tokens_and_reasoning(self, agent):
        """api.moonshot.cn (China endpoint) should get the same params."""
        agent.provider = "kimi-coding-cn"
        agent.base_url = "https://api.moonshot.cn/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.model = "kimi-k2.5"
        messages = [{"role": "user", "content": "hi"}]

        kwargs = agent._build_api_kwargs(messages)

        assert kwargs["max_tokens"] == 32000
        assert kwargs["extra_body"]["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in kwargs

    def test_provider_preferences_injected(self, agent):
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.providers_allowed = ["Anthropic"]
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["provider"]["only"] == ["Anthropic"]

    def test_reasoning_config_default_openrouter(self, agent):
        """Default reasoning config for OpenRouter should be medium."""
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "anthropic/claude-sonnet-4-20250514"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        reasoning = kwargs["extra_body"]["reasoning"]
        assert reasoning["enabled"] is True
        assert reasoning["effort"] == "medium"

    def test_reasoning_config_custom(self, agent):
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "anthropic/claude-sonnet-4-20250514"
        agent.reasoning_config = {"enabled": False}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_reasoning_not_sent_for_unsupported_openrouter_model(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "minimax/minimax-m2.5"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_reasoning_sent_for_supported_openrouter_model(self, agent):
        agent.provider = "openrouter"
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "qwen/qwen3.5-plus-02-15"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    def test_reasoning_sent_for_nous_route(self, agent):
        agent.provider = "nous"
        agent.base_url = "https://inference-api.nousresearch.com/v1"
        agent.model = "minimax/minimax-m2.5"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["extra_body"]["reasoning"]["effort"] == "medium"

    def test_reasoning_sent_for_copilot_gpt5(self, agent):
        """Copilot/GitHub Models: GPT-5 reasoning goes in extra_body.reasoning."""
        from agent.transports import get_transport
        from providers import get_provider_profile

        transport = get_transport("chat_completions")
        profile = get_provider_profile("copilot")
        msgs = [{"role": "user", "content": "hi"}]
        kwargs = transport.build_kwargs(
            model="gpt-5.4",
            messages=msgs,
            tools=None,
            supports_reasoning=True,
            provider_profile=profile,
        )
        assert kwargs["extra_body"]["reasoning"] == {"effort": "medium"}

    def test_reasoning_xhigh_normalized_for_copilot(self, agent):
        """xhigh effort should normalize to high for Copilot GitHub Models."""
        from agent.transports import get_transport
        from providers import get_provider_profile

        transport = get_transport("chat_completions")
        profile = get_provider_profile("copilot")
        msgs = [{"role": "user", "content": "hi"}]
        kwargs = transport.build_kwargs(
            model="gpt-5.4",
            messages=msgs,
            tools=None,
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "xhigh"},
            provider_profile=profile,
        )
        assert kwargs["extra_body"]["reasoning"] == {"effort": "high"}

    def test_reasoning_omitted_for_non_reasoning_copilot_model(self, agent):
        agent.base_url = "https://api.githubcopilot.com"
        agent.model = "gpt-4.1"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" not in kwargs.get("extra_body", {})

    def test_max_tokens_injected(self, agent):
        agent.max_tokens = 4096
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["max_tokens"] == 4096


    def test_qwen_portal_formats_messages_and_metadata(self, agent):
        agent.provider = "qwen-oauth"
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.session_id = "sess-123"
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "assistant", "content": "Got it"},
            {"role": "user", "content": "hi"},
        ]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["metadata"]["sessionId"] == "sess-123"
        assert kwargs["extra_body"]["vl_high_resolution_images"] is True
        assert isinstance(kwargs["messages"][0]["content"], list)
        assert kwargs["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert kwargs["messages"][2]["content"][0]["text"] == "hi"

    def test_qwen_portal_normalizes_bare_string_content_parts(self, agent):
        agent.provider = "qwen-oauth"
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "system"}]},
            {"role": "user", "content": ["hello", {"type": "text", "text": "world"}]},
        ]
        kwargs = agent._build_api_kwargs(messages)
        user_content = kwargs["messages"][1]["content"]
        assert user_content[0] == {"type": "text", "text": "hello"}
        assert user_content[1] == {"type": "text", "text": "world"}

    def test_qwen_portal_no_system_message(self, agent):
        agent.provider = "qwen-oauth"
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        # Should not crash even without a system message
        assert kwargs["messages"][0]["content"][0]["text"] == "hi"
        assert "cache_control" not in kwargs["messages"][0]["content"][0]

    def test_qwen_portal_sends_explicit_max_tokens(self, agent):
        """When the user explicitly sets max_tokens, it should be sent to Qwen Portal."""
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.max_tokens = 4096
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["max_tokens"] == 4096

    def test_qwen_portal_default_max_tokens(self, agent):
        """When max_tokens is None, Qwen Portal gets a default of 65536
        to prevent reasoning models from exhausting their output budget."""
        agent.provider = "qwen-oauth"
        agent.base_url = "https://portal.qwen.ai/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.max_tokens = None
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["max_tokens"] == 65536

    def test_ollama_think_false_on_effort_none(self, agent):
        """Custom (Ollama) provider with effort=none should inject think=false."""
        agent.provider = "custom"
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.reasoning_config = {"effort": "none"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs.get("extra_body", {}).get("think") is False

    def test_ollama_think_false_on_enabled_false(self, agent):
        """Custom (Ollama) provider with enabled=false should inject think=false."""
        agent.provider = "custom"
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.reasoning_config = {"enabled": False}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs.get("extra_body", {}).get("think") is False

    def test_ollama_no_think_param_when_reasoning_enabled(self, agent):
        """Custom provider with reasoning enabled should NOT inject think=false."""
        agent.provider = "custom"
        agent.base_url = "http://localhost:11434/v1"
        agent._base_url_lower = agent.base_url.lower()
        agent.reasoning_config = {"enabled": True, "effort": "medium"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs.get("extra_body", {}).get("think") is None

    def test_non_custom_provider_unaffected(self, agent):
        """OpenRouter provider with effort=none should NOT inject think=false."""
        agent.provider = "openrouter"
        agent.model = "qwen/qwen3.5-plus-02-15"
        agent.reasoning_config = {"effort": "none"}
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs.get("extra_body", {}).get("think") is None


class TestFormatToolsForSystemMessage:
    def test_no_tools_returns_empty_array(self, agent):
        agent.tools = []
        assert agent._format_tools_for_system_message() == "[]"

    def test_formats_single_tool(self, agent):
        agent.tools = _make_tool_defs("web_search")
        result = agent._format_tools_for_system_message()
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "web_search"

    def test_formats_multiple_tools(self, agent):
        agent.tools = _make_tool_defs("web_search", "terminal", "read_file")
        result = agent._format_tools_for_system_message()
        parsed = json.loads(result)
        assert len(parsed) == 3
        names = {t["name"] for t in parsed}
        assert names == {"web_search", "terminal", "read_file"}


class TestMaxTokensParam:
    """Verify _max_tokens_param returns the correct key for each provider."""

    def test_returns_max_completion_tokens_for_direct_openai(self, agent):
        agent.base_url = "https://api.openai.com/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_tokens_for_openrouter(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_returns_max_tokens_for_local(self, agent):
        agent.base_url = "http://localhost:11434/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_not_tricked_by_openai_in_openrouter_url(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1/api.openai.com"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_returns_max_completion_tokens_for_azure(self, agent):
        """Azure OpenAI requires max_completion_tokens for gpt-5.x models."""
        agent.base_url = "https://my-resource.openai.azure.com/openai/v1"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_completion_tokens_for_github_copilot(self, agent):
        """GitHub Copilot's OpenAI-compatible API rejects max_tokens for newer models."""
        agent.base_url = "https://api.githubcopilot.com"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_completion_tokens_for_github_copilot_path(self, agent):
        """Detect Copilot by hostname even when the configured URL includes a path."""
        agent.base_url = "https://api.githubcopilot.com/chat/completions"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    # ── Model-name fallback for non-openai.com endpoints serving newer families ──

    def test_returns_max_completion_tokens_for_gpt5_on_custom_endpoint(self, agent):
        """Custom OpenAI-compatible endpoint serving gpt-5.x must also use
        max_completion_tokens — otherwise the server 400s on max_tokens."""
        agent.base_url = "https://my-gateway.example.com/v1"
        agent.model = "gpt-5.4"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_completion_tokens_for_gpt4o_on_openrouter(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "openai/gpt-4o-mini"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_completion_tokens_for_o1_on_custom_endpoint(self, agent):
        agent.base_url = "https://custom.example.com/v1"
        agent.model = "o1-preview"
        result = agent._max_tokens_param(4096)
        assert result == {"max_completion_tokens": 4096}

    def test_returns_max_tokens_for_classic_gpt4_on_openrouter(self, agent):
        """Classic gpt-4 (non-omni) still uses max_tokens. Don't over-match."""
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "openai/gpt-4-turbo"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}

    def test_returns_max_tokens_for_llama_on_local(self, agent):
        agent.base_url = "http://localhost:11434/v1"
        agent.model = "llama3"
        result = agent._max_tokens_param(4096)
        assert result == {"max_tokens": 4096}


class TestGpt5ApiModeRouting:
    """Verify provider-specific GPT-5 API-mode routing."""

    def test_azure_gpt5_stays_on_chat_completions(self, agent):
        """Azure serves gpt-5.x on /chat/completions — must not upgrade to codex_responses."""
        agent.base_url = "https://my-resource.openai.azure.com/openai/v1"
        agent.api_mode = "chat_completions"
        agent.model = "gpt-5.4-mini"
        # Mirror the routing logic from __init__
        if (
            agent.api_mode == "chat_completions"
            and not agent._is_azure_openai_url()
            and (
                agent._is_direct_openai_url()
                or agent._provider_model_requires_responses_api(
                    agent.model, provider=agent.provider,
                )
            )
        ):
            agent.api_mode = "codex_responses"
        assert agent.api_mode == "chat_completions"

    def test_non_azure_gpt5_upgrades_to_codex_responses(self, agent):
        """On api.openai.com, gpt-5.x must still upgrade to codex_responses."""
        agent.base_url = "https://api.openai.com/v1"
        agent.api_mode = "chat_completions"
        agent.model = "gpt-5.4-mini"
        if (
            agent.api_mode == "chat_completions"
            and not agent._is_azure_openai_url()
            and (
                agent._is_direct_openai_url()
                or agent._provider_model_requires_responses_api(
                    agent.model, provider=agent.provider,
                )
            )
        ):
            agent.api_mode = "codex_responses"
        assert agent.api_mode == "codex_responses"

    def test_nous_gpt5_stays_on_chat_completions(self, agent):
        """Nous serves gpt-5.x on /chat/completions — must not upgrade to codex_responses."""
        agent.provider = "nous"
        agent.base_url = "https://inference-api.nousresearch.com/v1"
        agent.api_mode = "chat_completions"
        agent.model = "openai/gpt-5.5"
        if (
            agent.api_mode == "chat_completions"
            and not agent._is_azure_openai_url()
            and (
                agent._is_direct_openai_url()
                or agent._provider_model_requires_responses_api(
                    agent.model, provider=agent.provider,
                )
            )
        ):
            agent.api_mode = "codex_responses"
        assert agent.api_mode == "chat_completions"

    def test_is_azure_openai_url_detection(self, agent):
        assert agent._is_azure_openai_url("https://foo.openai.azure.com/openai/v1") is True
        assert agent._is_azure_openai_url("https://api.openai.com/v1") is False
        assert agent._is_azure_openai_url("https://openrouter.ai/api/v1") is False
        # Path-embedded azure string should still detect — we're ~substring matching
        agent.base_url = "https://my-resource.openai.azure.com/openai/v1"
        assert agent._is_azure_openai_url() is True


class TestSystemPromptStability:
    """Verify that the system prompt stays stable across turns for cache hits."""

    def test_stored_prompt_reused_for_continuing_session(self, agent):
        """When conversation_history is non-empty and session DB has a stored
        prompt, it should be reused instead of rebuilding from disk."""
        stored = "You are helpful. [stored from turn 1]"
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"system_prompt": stored}
        agent._session_db = mock_db

        # Simulate a continuing session with history
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]

        # First call — _cached_system_prompt is None, history is non-empty
        agent._cached_system_prompt = None

        # Patch run_conversation internals to just test the system prompt logic.
        # We'll call the prompt caching block directly by simulating what
        # run_conversation does.
        conversation_history = history

        # The block under test (from run_conversation):
        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                try:
                    session_row = agent._session_db.get_session(agent.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt

        assert agent._cached_system_prompt == stored
        mock_db.get_session.assert_called_once_with(agent.session_id)

    def test_fresh_build_when_no_history(self, agent):
        """On the first turn (no history), system prompt should be built fresh."""
        mock_db = MagicMock()
        agent._session_db = mock_db

        agent._cached_system_prompt = None
        conversation_history = []

        # The block under test:
        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                session_row = agent._session_db.get_session(agent.session_id)
                if session_row:
                    stored_prompt = session_row.get("system_prompt") or None

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt
            else:
                agent._cached_system_prompt = agent._build_system_prompt()

        # Should have built fresh, not queried the DB
        mock_db.get_session.assert_not_called()
        assert agent._cached_system_prompt is not None
        assert "Hermes Agent" in agent._cached_system_prompt

    def test_fresh_build_when_db_has_no_prompt(self, agent):
        """If the session DB has no stored prompt, build fresh even with history."""
        mock_db = MagicMock()
        mock_db.get_session.return_value = {"system_prompt": ""}
        agent._session_db = mock_db

        agent._cached_system_prompt = None
        conversation_history = [{"role": "user", "content": "hi"}]

        if agent._cached_system_prompt is None:
            stored_prompt = None
            if conversation_history and agent._session_db:
                try:
                    session_row = agent._session_db.get_session(agent.session_id)
                    if session_row:
                        stored_prompt = session_row.get("system_prompt") or None
                except Exception:
                    pass

            if stored_prompt:
                agent._cached_system_prompt = stored_prompt
            else:
                agent._cached_system_prompt = agent._build_system_prompt()

        # Empty string is falsy, so should fall through to fresh build
        assert "Hermes Agent" in agent._cached_system_prompt


class TestBuildApiKwargsAnthropicMaxTokens:
    """Bug fix: max_tokens was always None for Anthropic mode, ignoring user config."""

    def test_max_tokens_passed_to_anthropic(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = 4096
        agent.reasoning_config = None

        with patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 4096}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            _, kwargs = mock_build.call_args
            if not kwargs:
                kwargs = dict(zip(
                    ["model", "messages", "tools", "max_tokens", "reasoning_config"],
                    mock_build.call_args[0],
                ))
            assert kwargs.get("max_tokens") == 4096 or mock_build.call_args[1].get("max_tokens") == 4096

    def test_max_tokens_none_when_unset(self, agent):
        agent.api_mode = "anthropic_messages"
        agent.max_tokens = None
        agent.reasoning_config = None

        with patch("agent.anthropic_adapter.build_anthropic_kwargs") as mock_build:
            mock_build.return_value = {"model": "claude-sonnet-4-20250514", "messages": [], "max_tokens": 16384}
            agent._build_api_kwargs([{"role": "user", "content": "test"}])
            call_args = mock_build.call_args
            # max_tokens should be None (let adapter use its default)
            if call_args[1]:
                assert call_args[1].get("max_tokens") is None
            else:
                assert call_args[0][3] is None
