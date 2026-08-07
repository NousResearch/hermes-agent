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


class TestSystemPromptMode:
    """system_prompt_mode compatibility modes (#76783)."""

    @staticmethod
    def _profile(mode):
        return ProviderProfile(name="test-relay", system_prompt_mode=mode)

    def test_default_is_system_pass_through(self):
        p = ProviderProfile(name="plain")
        msgs = [{"role": "system", "content": "You are Hermes Agent."},
                {"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) is msgs

    def test_user_mode_moves_prompt_into_first_user_message(self):
        p = self._profile("user")
        runtime = "You are Hermes Agent, an intelligent AI assistant.\nLots of rules."
        msgs = [
            {"role": "system", "content": runtime},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        out = p.prepare_messages(msgs)
        # system keeps a short identity marker
        assert out[0]["role"] == "system"
        assert out[0]["content"] == "You are Hermes Agent, an intelligent AI assistant."
        # full prompt prepended to first user message
        assert out[1]["role"] == "user"
        assert "[Hermes runtime instructions]" in out[1]["content"]
        assert runtime in out[1]["content"]
        assert out[1]["content"].endswith("Hello")
        # input not mutated
        assert msgs[0]["content"] == runtime
        assert msgs[1]["content"] == "Hello"

    def test_user_mode_multimodal_user_content(self):
        p = self._profile("user")
        msgs = [
            {"role": "system", "content": "You are Hermes Agent."},
            {"role": "user",
             "content": [{"type": "text", "text": "describe"},
                         {"type": "image_url", "image_url": {"url": "data:x"}}]},
        ]
        out = p.prepare_messages(msgs)
        assert isinstance(out[1]["content"], list)
        assert out[1]["content"][0]["type"] == "text"
        assert "[Hermes runtime instructions]" in out[1]["content"][0]["text"]
        # original parts preserved in order after the prepended wrapper
        assert out[1]["content"][1] == {"type": "text", "text": "describe"}
        assert out[1]["content"][2]["type"] == "image_url"

    def test_user_mode_no_user_message_is_noop(self):
        p = self._profile("user")
        msgs = [{"role": "system", "content": "You are Hermes Agent."}]
        out = p.prepare_messages(msgs)
        assert out == msgs

    def test_developer_mode_swaps_role(self):
        p = self._profile("developer")
        msgs = [{"role": "system", "content": "You are Hermes Agent."},
                {"role": "user", "content": "hi"}]
        out = p.prepare_messages(msgs)
        assert out[0]["role"] == "developer"
        assert out[0]["content"] == "You are Hermes Agent."
        assert msgs[0]["role"] == "system"  # input not mutated

    def test_user_mode_requires_leading_system_message(self):
        p = self._profile("user")
        msgs = [{"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) == msgs

    def test_empty_system_content_is_noop(self):
        p = self._profile("user")
        msgs = [{"role": "system", "content": ""}, {"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) == msgs




