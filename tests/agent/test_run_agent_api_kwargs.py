"""test_run_agent_api_kwargs: split from the former tests/agent/test_run_agent.py monolith (#99982)."""
import json
from unittest.mock import MagicMock, patch
from run_agent import AIAgent
from agent.prompt_builder import DEFAULT_AGENT_IDENTITY
from tests.agent._run_agent_helpers import _make_tool_defs


class TestBuildSystemPrompt:
    def test_always_has_identity(self, agent):
        prompt = agent._build_system_prompt()
        assert DEFAULT_AGENT_IDENTITY in prompt

    def test_can_use_soul_identity_even_when_context_files_are_skipped(self):
        with (
            patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("terminal")),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch("agent.prompt_builder.load_soul_md", return_value="SOUL IDENTITY"),
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


    def test_memory_guidance_when_memory_tool_loaded(self, agent_with_memory_tool):
        agent_with_memory_tool._memory_enabled = True
        prompt = agent_with_memory_tool._build_system_prompt()
        from agent.prompt_builder import build_memory_guidance
        assert build_memory_guidance(True, True, skill_manage_available=False) in prompt

    def test_no_memory_guidance_when_both_builtin_stores_disabled(
        self, agent_with_memory_tool
    ):
        """Guidance must follow the stores, not just the tool's presence.

        With both built-in stores off, ``agent_init`` never builds a
        ``MemoryStore``, so every memory call returns "Memory is not
        available" — telling the model to save facts there is a dead
        instruction paid for on every API call.
        """
        from agent.prompt_builder import MEMORY_GUIDANCE, USER_PROFILE_GUIDANCE

        agent_with_memory_tool._memory_enabled = False
        agent_with_memory_tool._user_profile_enabled = False
        prompt = agent_with_memory_tool._build_system_prompt()
        assert MEMORY_GUIDANCE not in prompt
        assert USER_PROFILE_GUIDANCE not in prompt

    def test_profile_guidance_when_only_user_profile_enabled(
        self, agent_with_memory_tool
    ):
        """USER.md alone gets the narrower profile-only guidance.

        The full MEMORY_GUIDANCE block instructs the model to save notes to a
        MEMORY.md store that does not exist in this configuration, so the
        profile-specific block is injected instead.
        """
        from agent.prompt_builder import MEMORY_GUIDANCE

        agent_with_memory_tool._memory_enabled = False
        agent_with_memory_tool._user_profile_enabled = True
        prompt = agent_with_memory_tool._build_system_prompt()
        assert MEMORY_GUIDANCE not in prompt
        from agent.prompt_builder import build_memory_guidance
        assert build_memory_guidance(False, True, skill_manage_available=False) in prompt


    def test_datetime_is_date_only_not_minute_precision(self, agent):
        """Timestamp must be date-only (no HH:MM) so the system prompt
        stays byte-stable for the full day. Minute precision invalidates
        prefix-cache KV on every rebuild path (compression, fresh-agent
        gateway turns, session resume without a stored prompt).

        The trailing zone parenthetical -- e.g. ``(America/New_York, EDT,
        UTC-04:00)`` -- is exempt from the HH:MM check: a UTC offset is not
        time-of-day and is constant for the whole day (it shifts only at a DST
        transition), so it does not affect cache stability.
        """
        prompt = agent._build_system_prompt()
        # Find the line and strip it for inspection
        for line in prompt.splitlines():
            if line.startswith("Conversation started:"):
                # Must NOT contain AM/PM indicator (minute precision had %I:%M %p)
                assert " AM" not in line and " PM" not in line, (
                    f"Timestamp line has time-of-day, breaks daily cache stability: {line!r}"
                )
                # Must NOT contain a colon followed by two digits (HH:MM pattern)
                # in the date portion, i.e. everything before the zone suffix.
                import re as _re
                date_part = line.split(" (")[0]
                assert not _re.search(r":\d{2}", date_part), (
                    f"Timestamp line has HH:MM, breaks daily cache stability: {line!r}"
                )
                break
        else:
            assert False, "Expected a 'Conversation started:' line in the system prompt"

    def test_datetime_includes_utc_offset(self, agent):
        """Timestamp must carry an explicit UTC offset.

        Tools that accept instants (e.g. nutrition/calendar MCP servers) reject
        naive datetimes and require an offset. With a bare date the model has to
        infer EST vs EDT on its own, which is a coin-flip near a DST boundary and
        silently writes records onto the wrong day when it guesses wrong.
        """
        prompt = agent._build_system_prompt()
        import re as _re
        for line in prompt.splitlines():
            if line.startswith("Conversation started:"):
                assert _re.search(r"UTC[+-]\d{2}:\d{2}", line), (
                    f"Timestamp line is missing a UTC offset: {line!r}"
                )
                break
        else:
            assert False, "Expected a 'Conversation started:' line in the system prompt"

    def test_datetime_line_is_stable_across_rebuilds(self, agent):
        """Two rebuilds within the same day must produce a byte-identical
        timestamp line, or the prefix cache is invalidated on every rebuild."""
        def _line(p):
            return next(ln for ln in p.splitlines()
                        if ln.startswith("Conversation started:"))
        assert _line(agent._build_system_prompt()) == _line(agent._build_system_prompt())

    def test_skills_prompt_derives_available_toolsets_from_loaded_tools(self):
        tools = _make_tool_defs("web_search", "skills_list", "skill_view", "skill_manage")
        toolset_map = {
            "web_search": "web",
            "skills_list": "skills",
            "skill_view": "skills",
            "skill_manage": "skills",
        }

        with (
            patch("model_tools.get_tool_definitions", return_value=tools),
            patch(
                "model_tools.check_toolset_requirements",
                side_effect=AssertionError("should not re-check toolset requirements"),
            ),
            patch("model_tools.get_toolset_for_tool", create=True, side_effect=toolset_map.get),
            patch("agent.prompt_builder.build_skills_system_prompt", return_value="SKILLS_PROMPT") as mock_skills,
            patch("agent.process_bootstrap.OpenAI"),
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


    def test_no_tools_never_injects(self):
        """Even with enforcement=true, no injection when agent has no tools."""
        from agent.prompt_builder import TOOL_USE_ENFORCEMENT_GUIDANCE
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"tool_use_enforcement": True}},
            ), patch(
                "hermes_cli.config.load_config_readonly",
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

class TestExecutionGuidanceConfig:
    """End-to-end tests for the agent.execution_guidance config option —
    from config.yaml through agent_init to the built system prompt."""

    def _make_agent(self, model="deepseek/deepseek-v4-pro", execution_guidance=None):
        agent_cfg = {"tool_use_enforcement": False}
        if execution_guidance is not None:
            agent_cfg["execution_guidance"] = execution_guidance
        with (
            patch(
                "model_tools.get_tool_definitions",
                return_value=_make_tool_defs("terminal", "web_search"),
            ),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": agent_cfg},
            ), patch(
                "hermes_cli.config.load_config_readonly",
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


    def test_config_false_suppresses(self):
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(
            model="deepseek/deepseek-v4-pro", execution_guidance=False
        )
        assert OPENAI_MODEL_EXECUTION_GUIDANCE not in agent._build_system_prompt()

    def test_config_list_matches(self):
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(
            model="moonshotai/kimi-k3", execution_guidance=["kimi"]
        )
        assert OPENAI_MODEL_EXECUTION_GUIDANCE in agent._build_system_prompt()

    def test_config_list_non_match_suppresses(self):
        from agent.prompt_builder import OPENAI_MODEL_EXECUTION_GUIDANCE
        agent = self._make_agent(
            model="openai/gpt-4.1", execution_guidance=["kimi"]
        )
        assert OPENAI_MODEL_EXECUTION_GUIDANCE not in agent._build_system_prompt()

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
                "model_tools.get_tool_definitions",
                return_value=_make_tool_defs("terminal", "web_search"),
            ),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": agent_cfg},
            ), patch(
                "hermes_cli.config.load_config_readonly",
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


    def test_no_tools_no_injection(self):
        """Same gate as tool_use_enforcement — no tools means no guidance.
        The guidance refers to ``tool calls`` and ``tool output``; without
        tools it would be advice for a capability the agent doesn't have."""
        from agent.prompt_builder import TASK_COMPLETION_GUIDANCE
        with (
            patch("model_tools.get_tool_definitions", return_value=[]),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"task_completion_guidance": True}},
            ), patch(
                "hermes_cli.config.load_config_readonly",
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
                "model_tools.get_tool_definitions",
                return_value=_make_tool_defs("terminal"),
            ),
            patch("model_tools.check_toolset_requirements", return_value={}),
            patch("agent.process_bootstrap.OpenAI"),
            patch(
                "hermes_cli.config.load_config",
                return_value={"agent": {"environment_probe": environment_probe}},
            ), patch(
                "hermes_cli.config.load_config_readonly",
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

class TestBuildApiKwargs:
    def test_basic_kwargs(self, agent):
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert kwargs["model"] == agent.model
        assert kwargs["messages"] is messages

    def test_explicit_request_local_tools_reach_native_transport(self, agent, monkeypatch):
        from agent.prompt_caching import build_prompt_cache_plan

        canonical_tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        plan = build_prompt_cache_plan(
            [{"role": "system", "content": "stable\nvolatile"}, {"role": "user", "content": "lookup"}],
            canonical_tools,
            native_anthropic=True,
            static_system_prefix="stable",
            direct_native_tool_cache=True,
        )
        transport = MagicMock()
        transport.build_kwargs.side_effect = lambda **kwargs: kwargs
        agent.api_mode = "anthropic_messages"
        agent.provider = "anthropic"
        agent.base_url = "https://api.anthropic.com"
        monkeypatch.setattr(agent, "_get_transport", lambda: transport)
        monkeypatch.setattr(agent, "_prepare_anthropic_messages_for_api", lambda messages: messages)

        kwargs = agent._build_api_kwargs(plan.messages, tools_for_api=plan.tools)

        assert "cache_control" not in canonical_tools[-1]
        assert kwargs["tools"][-1]["cache_control"] == {"type": "ephemeral"}

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


    def test_reasoning_not_sent_for_unsupported_openrouter_model(self, agent):
        agent.base_url = "https://openrouter.ai/api/v1"
        agent.model = "minimax/minimax-m2.5"
        messages = [{"role": "user", "content": "hi"}]
        kwargs = agent._build_api_kwargs(messages)
        assert "reasoning" not in kwargs.get("extra_body", {})


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


    def test_core_responses_preserves_supported_xhigh(self, agent, monkeypatch):
        """The core GitHub Responses path must preserve a supported xhigh."""
        monkeypatch.setattr(
            "hermes_cli.models.github_model_reasoning_efforts",
            lambda _model: ["none", "low", "medium", "high", "xhigh"],
        )
        agent.model = "gpt-5.5"
        agent.reasoning_config = {"enabled": True, "effort": "xhigh"}

        assert agent._github_models_reasoning_extra_body() == {"effort": "xhigh"}


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

class TestGpt5ApiModeRouting:
    """Verify provider-specific GPT-5 API-mode routing."""


    def test_nous_gpt5_stays_on_chat_completions(self, agent):
        """Nous serves gpt-5.x on /chat/completions — must not upgrade to codex_responses."""
        agent.provider = "nous"
        agent.base_url = "https://inference-api.nousresearch.com/v1"
        agent.api_mode = "chat_completions"
        agent.model = "openai/gpt-5.5"
        assert not agent._is_direct_openai_url()
        assert not AIAgent._provider_model_requires_responses_api(agent.model, provider=agent.provider)

    def test_is_azure_openai_url_detection(self, agent):
        assert agent._is_azure_openai_url("https://foo.openai.azure.com/openai/v1") is True
        assert agent._is_azure_openai_url("https://api.openai.com/v1") is False
        assert agent._is_azure_openai_url("https://openrouter.ai/api/v1") is False
        # Path-embedded azure string should still detect — we're ~substring matching
        agent.base_url = "https://my-resource.openai.azure.com/openai/v1"
        assert agent._is_azure_openai_url() is True

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
