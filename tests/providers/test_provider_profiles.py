"""Tests for the provider module registry and profiles."""

from providers import get_provider_profile, _REGISTRY
from providers.base import ProviderProfile, OMIT_TEMPERATURE


class TestRegistry:
    def test_discovery_populates_registry(self):
        p = get_provider_profile("nvidia")
        assert p is not None
        assert p.name == "nvidia"





class TestNvidiaProfile:
    def test_max_tokens(self):
        p = get_provider_profile("nvidia")
        assert p.default_max_tokens == 16384


    def test_base_url(self):
        p = get_provider_profile("nvidia")
        assert "nvidia.com" in p.base_url


    def test_prepare_messages_strips_tool_result_names(self):
        p = get_provider_profile("nvidia")
        msgs = [
            {"role": "user", "content": "run a command"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "name": "terminal",
                "tool_name": "terminal",
                "tool_call_id": "call_1",
                "content": "ok",
            },
        ]

        result = p.prepare_messages(msgs)

        assert "name" not in result[2]
        assert "tool_name" not in result[2]
        assert result[2] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "ok",
        }
        assert msgs[2]["name"] == "terminal"
        assert msgs[2]["tool_name"] == "terminal"

    def test_prepare_messages_passthrough_without_tool_result_names(self):
        p = get_provider_profile("nvidia")
        msgs = [{"role": "tool", "tool_call_id": "call_1", "content": "ok"}]
        assert p.prepare_messages(msgs) is msgs


class TestKimiProfile:
    def test_temperature_omit(self):
        p = get_provider_profile("kimi")
        assert p.fixed_temperature is OMIT_TEMPERATURE




    def test_thinking_enabled(self):
        # xor contract (fix ce4e74b3): an explicit recognized effort sends
        # reasoning_effort ONLY — never paired with extra_body.thinking.
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"})
        assert tl["reasoning_effort"] == "high"
        assert "thinking" not in eb





class TestOpenRouterProfile:
    def test_extra_body_with_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(provider_preferences={"allow": ["anthropic"]})
        assert body["provider"] == {"allow": ["anthropic"]}

    def test_sticky_session_id_normalizes_cron_timestamp(self):
        """Cron re-fires of the same job keep the same sticky routing key."""
        p = get_provider_profile("openrouter")
        first = p.build_extra_body(session_id="cron_job42_20260801_090000")
        second = p.build_extra_body(session_id="cron_job42_20260802_090000")
        assert first["session_id"] == "cron_job42"
        assert first["session_id"] == second["session_id"]





    def test_pareto_min_coding_score_emitted_for_pareto_model(self):
        """min_coding_score → plugins block when model is openrouter/pareto-code."""
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(
            model="openrouter/pareto-code",
            openrouter_min_coding_score=0.65,
        )
        assert body["plugins"] == [
            {"id": "pareto-router", "min_coding_score": 0.65}
        ]











    def test_grok_session_id_sets_cache_affinity_header(self):
        """OpenRouter + Grok model + session_id => x-grok-conv-id header."""
        p = get_provider_profile("openrouter")
        _, tl = p.build_api_kwargs_extras(
            model="x-ai/grok-4",
            session_id="sess-abc123",
        )
        assert tl["extra_headers"]["x-grok-conv-id"] == "sess-abc123"

    def test_grok_conv_id_normalizes_cron_timestamp(self):
        """Cron re-fires of the same job must pin to the same xAI backend,
        same as the body.session_id sticky key (#78941)."""
        p = get_provider_profile("openrouter")
        _, first = p.build_api_kwargs_extras(
            model="x-ai/grok-4", session_id="cron_job42_20260801_090000",
        )
        _, second = p.build_api_kwargs_extras(
            model="x-ai/grok-4", session_id="cron_job42_20260802_090000",
        )
        assert first["extra_headers"]["x-grok-conv-id"] == "cron_job42"
        assert (
            first["extra_headers"]["x-grok-conv-id"]
            == second["extra_headers"]["x-grok-conv-id"]
        )





    # --- reasoning-mandatory Anthropic effort → top-level verbosity (#43432) ---
    #
    # These models (Claude 4.6+ / fable / mythos-class) ignore
    # ``reasoning.effort`` and use adaptive thinking. OpenRouter honors the
    # requested effort on the top-level ``verbosity`` field instead (maps to
    # Anthropic ``output_config.effort``). The profile must route the existing
    # ``reasoning_config["effort"]`` there while still NEVER emitting a
    # ``reasoning`` field (which would 400 — see #42991). Gate every fixture on
    # the real predicate so this stays a behavior contract, not a name snapshot.

    @staticmethod
    def _is_mandatory(model):
        import inspect
        p = get_provider_profile("openrouter")
        mod = inspect.getmodule(type(p))
        return mod._anthropic_reasoning_is_mandatory(model)






    def test_mandatory_anthropic_verbosity_coexists_with_grok_header(self):
        """A reasoning-mandatory Anthropic model is never a Grok model, but the
        top-level dict must remain a single merged dict — verify the verbosity
        path doesn't clobber the extra_headers slot used by Grok affinity."""
        p = get_provider_profile("openrouter")
        # mandatory anthropic + effort → verbosity, no extra_headers
        _, tl = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
            model="anthropic/claude-fable-5",
        )
        assert tl == {"verbosity": "high"}


class TestNousProfile:
    def test_tags(self):
        from agent.portal_tags import nous_portal_tags
        p = get_provider_profile("nous")
        body = p.build_extra_body()
        assert body["tags"] == nous_portal_tags()

    def test_sticky_session_id_normalizes_cron_timestamp(self):
        """Cron re-fires of the same job keep the same sticky routing key."""
        p = get_provider_profile("nous")
        first = p.build_extra_body(session_id="cron_job42_20260801_090000")
        second = p.build_extra_body(session_id="cron_job42_20260802_090000")
        assert first["session_id"] == "cron_job42"
        assert first["session_id"] == second["session_id"]





    def test_auth_type(self):
        p = get_provider_profile("nous")
        assert p.auth_type == "oauth_device_code"




class TestQwenProfile:






    def test_prepare_messages_protects_nested_image_url_retry_mutation(self):
        qwen = get_provider_profile("qwen-oauth")
        image_url = {"url": "data:image/png;base64,original"}
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see image"},
                    {"type": "image_url", "image_url": image_url},
                ],
            },
        ]

        qwen_result = qwen.prepare_messages(msgs)

        assert qwen_result[1] is not msgs[1]
        assert qwen_result[1]["content"] is not msgs[1]["content"]
        assert qwen_result[1]["content"][1] is not msgs[1]["content"][1]
        assert qwen_result[1]["content"][1]["image_url"] is not image_url

        qwen_result[1]["content"][1]["image_url"]["url"] = (
            "data:image/png;base64,shrunk"
        )
        assert msgs[1]["content"][1]["image_url"]["url"] == (
            "data:image/png;base64,original"
        )

    def test_metadata_top_level(self):
        p = get_provider_profile("qwen-oauth")
        meta = {"sessionId": "s123", "promptId": "p456"}
        eb, tl = p.build_api_kwargs_extras(qwen_session_metadata=meta)
        assert tl["metadata"] == meta
        assert "metadata" not in eb

    def test_reasoning_disabled_unless_enabled_is_true(self):
        p = get_provider_profile("qwen-oauth")
        assert p is not None
        disabled_configs: tuple[object, ...] = (
            None,
            {},
            {"effort": "high"},
            {"enabled": False},
            {"enabled": "false"},
            {"enabled": 1},
            {"enabled": None},
            [],
            "malformed",
        )
        for config in disabled_configs:
            eb, _ = p.build_api_kwargs_extras(reasoning_config=config)
            assert eb == {"enable_thinking": False}, config
        enabled_with_effort = (
            {"enabled": True},
            {"enabled": True, "effort": "none"},
            {"enabled": True, "effort": " NONE "},
            {"enabled": True, "effort": "NoNe"},
            {"enabled": True, "effort": "high"},
            {"enabled": True, "effort": object()},
        )
        for config in enabled_with_effort:
            eb, _ = p.build_api_kwargs_extras(reasoning_config=config)
            assert eb == {"enable_thinking": True, "thinking_budget": 32768}, config
        eb, _ = p.build_api_kwargs_extras(reasoning_config={"enabled": "malformed"})
        assert eb == {"enable_thinking": False}

    def test_reasoning_is_enabled_with_bounded_budget_and_headroom(self):
        p = get_provider_profile("qwen-oauth")
        assert p is not None
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"}
        )
        assert eb == {"enable_thinking": True, "thinking_budget": 32768}
        assert p.default_max_tokens > eb["thinking_budget"]

    def test_metadata_is_preserved_with_reasoning_contract(self):
        p = get_provider_profile("qwen-oauth")
        assert p is not None
        meta = {"sessionId": "s123", "promptId": "p456"}
        eb, tl = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True}, qwen_session_metadata=meta
        )
        assert eb == {"enable_thinking": True, "thinking_budget": 32768}
        assert tl == {"metadata": meta}


class TestAlibabaRegionalAndTokenPlanProfiles:
    """#73265: the models.dev catalog advertises alibaba-cn /
    alibaba-token-plan(-cn) / alibaba-coding-plan-cn, but none were registered
    at runtime — `model.provider: alibaba-coding-plan-cn` failed with
    "Unknown provider" and users were forced onto the `custom` escape hatch.
    Profile names intentionally match the catalog keys exactly so model
    metadata lines up."""

    def test_alibaba_cn_registered(self):
        p = get_provider_profile("alibaba-cn")
        assert p is not None and p.name == "alibaba-cn"
        assert p.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert "DASHSCOPE_API_KEY" in p.env_vars
        assert "DASHSCOPE_CN_BASE_URL" in p.env_vars

    def test_alibaba_coding_plan_cn_registered(self):
        p = get_provider_profile("alibaba-coding-plan-cn")
        assert p is not None and p.name == "alibaba-coding-plan-cn"
        assert p.base_url == "https://coding.dashscope.aliyuncs.com/v1"
        assert "ALIBABA_CODING_PLAN_API_KEY" in p.env_vars

    def test_alibaba_token_plan_registered(self):
        p = get_provider_profile("alibaba-token-plan")
        assert p is not None and p.name == "alibaba-token-plan"
        assert p.base_url == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
        assert "ALIBABA_TOKEN_PLAN_API_KEY" in p.env_vars

    def test_alibaba_token_plan_cn_registered(self):
        p = get_provider_profile("alibaba-token-plan-cn")
        assert p is not None and p.name == "alibaba-token-plan-cn"
        assert p.base_url == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
        assert "ALIBABA_TOKEN_PLAN_API_KEY" in p.env_vars

    def test_cn_variants_resolve_in_auth_registry(self, monkeypatch):
        """The reporter's exact failure site: ``auth.resolve_provider()`` only
        consults PROVIDER_REGISTRY (auto-extended from provider profiles,
        hermes_cli/auth.py:461-490) and raised
        "Unknown provider 'alibaba-coding-plan-cn'" (hermes_cli/auth.py:1937)
        even though the models.dev catalog advertised the id — the
        resolve_provider_full() catalog chain covers only the CLI --provider
        path, not the credential/runtime path."""
        from hermes_cli.auth import PROVIDER_REGISTRY, resolve_provider
        monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-test")
        monkeypatch.setenv("ALIBABA_CODING_PLAN_API_KEY", "sk-test")
        monkeypatch.setenv("ALIBABA_TOKEN_PLAN_API_KEY", "sk-test")
        for pid in ("alibaba-cn", "alibaba-coding-plan-cn",
                    "alibaba-token-plan", "alibaba-token-plan-cn"):
            assert pid in PROVIDER_REGISTRY, f"{pid} missing from PROVIDER_REGISTRY"
            assert resolve_provider(pid) == pid
        assert (PROVIDER_REGISTRY["alibaba-token-plan-cn"].inference_base_url
                == "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")
