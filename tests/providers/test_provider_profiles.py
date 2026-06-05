"""Tests for the provider module registry and profiles."""

from providers import get_provider_profile, _REGISTRY
from providers.base import ProviderProfile, OMIT_TEMPERATURE


class TestRegistry:
    def test_discovery_populates_registry(self):
        p = get_provider_profile("nvidia")
        assert p is not None
        assert p.name == "nvidia"

    def test_alias_lookup(self):
        assert get_provider_profile("kimi").name == "kimi-coding"
        assert get_provider_profile("moonshot").name == "kimi-coding"
        assert get_provider_profile("kimi-coding-cn").name == "kimi-coding-cn"
        assert get_provider_profile("or").name == "openrouter"
        assert get_provider_profile("nous-portal").name == "nous"
        assert get_provider_profile("qwen").name == "qwen-oauth"
        assert get_provider_profile("qwen-portal").name == "qwen-oauth"

    def test_unknown_provider_returns_none(self):
        assert get_provider_profile("nonexistent-provider") is None

    def test_all_providers_have_name(self):
        get_provider_profile("nvidia")  # trigger discovery
        for name, profile in _REGISTRY.items():
            assert profile.name == name


class TestNvidiaProfile:
    def test_max_tokens(self):
        p = get_provider_profile("nvidia")
        assert p.default_max_tokens == 16384

    def test_no_special_temperature(self):
        p = get_provider_profile("nvidia")
        assert p.fixed_temperature is None

    def test_base_url(self):
        p = get_provider_profile("nvidia")
        assert "nvidia.com" in p.base_url

    def test_billing_header_not_profile_wide(self):
        p = get_provider_profile("nvidia")
        assert p.default_headers == {}


class TestKimiProfile:
    def test_temperature_omit(self):
        p = get_provider_profile("kimi")
        assert p.fixed_temperature is OMIT_TEMPERATURE

    def test_max_tokens(self):
        p = get_provider_profile("kimi")
        assert p.default_max_tokens == 32000

    def test_cn_separate_profile(self):
        p = get_provider_profile("kimi-coding-cn")
        assert p.name == "kimi-coding-cn"
        assert p.env_vars == ("KIMI_CN_API_KEY",)
        assert "moonshot.cn" in p.base_url

    def test_cn_not_alias_of_kimi(self):
        kimi = get_provider_profile("kimi-coding")
        cn = get_provider_profile("kimi-coding-cn")
        assert kimi is not cn
        assert kimi.base_url != cn.base_url

    def test_thinking_enabled(self):
        # xor contract (fix ce4e74b3): an explicit recognized effort sends
        # reasoning_effort ONLY — never paired with extra_body.thinking.
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": True, "effort": "high"})
        assert tl["reasoning_effort"] == "high"
        assert "thinking" not in eb

    def test_thinking_disabled(self):
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": False})
        assert eb["thinking"] == {"type": "disabled"}
        assert "reasoning_effort" not in tl

    def test_reasoning_effort_default(self):
        # enabled with no effort → thinking toggle only, no top-level effort.
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config={"enabled": True})
        assert eb["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in tl

    def test_no_config_defaults(self):
        # No reasoning_config → thinking on, server picks depth; no effort.
        p = get_provider_profile("kimi")
        eb, tl = p.build_api_kwargs_extras(reasoning_config=None)
        assert eb["thinking"] == {"type": "enabled"}
        assert "reasoning_effort" not in tl


class TestOpenRouterProfile:
    def test_extra_body_with_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(provider_preferences={"allow": ["anthropic"]})
        assert body["provider"] == {"allow": ["anthropic"]}

    def test_extra_body_session_id(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(session_id="test-session-123")
        assert body["session_id"] == "test-session-123"

    def test_extra_body_no_prefs(self):
        p = get_provider_profile("openrouter")
        body = p.build_extra_body()
        assert body == {}

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

    def test_pareto_score_ignored_for_other_models(self):
        """Score has no effect on any other model — plugins block must not appear."""
        p = get_provider_profile("openrouter")
        body = p.build_extra_body(
            model="anthropic/claude-sonnet-4.6",
            openrouter_min_coding_score=0.65,
        )
        assert "plugins" not in body

    def test_pareto_score_unset_omits_plugins(self):
        """Empty/None score → no plugins block (router uses its omission default)."""
        p = get_provider_profile("openrouter")
        for unset in (None, ""):
            body = p.build_extra_body(
                model="openrouter/pareto-code",
                openrouter_min_coding_score=unset,
            )
            assert "plugins" not in body, f"unset={unset!r}"

    def test_pareto_score_out_of_range_dropped(self):
        """Invalid scores are silently dropped — never forwarded to OR."""
        p = get_provider_profile("openrouter")
        for bad in (1.5, -0.1, "not-a-number"):
            body = p.build_extra_body(
                model="openrouter/pareto-code",
                openrouter_min_coding_score=bad,
            )
            assert "plugins" not in body, f"bad={bad!r}"

    def test_reasoning_full_config(self):
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "high"},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": True, "effort": "high"}

    def test_reasoning_disabled_still_passes(self):
        """OpenRouter passes disabled reasoning through (unlike Nous)."""
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": False}

    def test_reasoning_disable_omitted_for_mandatory_anthropic(self):
        """Reasoning-mandatory Anthropic models (4.6+/fable) reject any disable
        form: OpenRouter translates ``reasoning: {enabled: false}`` into
        Anthropic's ``thinking: {type: disabled}``, which 400s. The profile must
        omit ``reasoning`` so the model falls back to adaptive thinking instead.
        """
        p = get_provider_profile("openrouter")
        for model in (
            "anthropic/claude-fable-5",          # new named model
            "anthropic/claude-some-future-7",    # unknown → default mandatory
            "anthropic/claude-opus-4.8",
            "anthropic/claude-opus-4.6",
        ):
            for cfg in ({"enabled": False}, {"effort": "none"}):
                eb, _ = p.build_api_kwargs_extras(
                    reasoning_config=cfg,
                    supports_reasoning=True,
                    model=model,
                )
                assert "reasoning" not in eb, (model, cfg, eb)

    def test_reasoning_disable_kept_for_legacy_anthropic(self):
        """Older Anthropic models still accept an explicit disable form, so the
        profile must keep forwarding it."""
        p = get_provider_profile("openrouter")
        for model in (
            "anthropic/claude-3.7-sonnet",
            "anthropic/claude-opus-4.5",
            "anthropic/claude-sonnet-4.5",
        ):
            eb, _ = p.build_api_kwargs_extras(
                reasoning_config={"enabled": False},
                supports_reasoning=True,
                model=model,
            )
            assert eb["reasoning"] == {"enabled": False}, (model, eb)

    def test_reasoning_disable_kept_for_non_anthropic(self):
        """Non-Anthropic models (DeepSeek, Qwen, …) disable reasoning fine; the
        Anthropic-mandatory guard must not touch them."""
        p = get_provider_profile("openrouter")
        for model in ("deepseek/deepseek-chat", "qwen/qwen3-max", "openai/gpt-5.4"):
            eb, _ = p.build_api_kwargs_extras(
                reasoning_config={"enabled": False},
                supports_reasoning=True,
                model=model,
            )
            assert eb["reasoning"] == {"enabled": False}, (model, eb)

    def test_reasoning_omitted_for_mandatory_anthropic_even_when_enabled(self):
        """Reasoning-mandatory Anthropic models (4.6+/fable) use adaptive
        thinking — OpenRouter ignores reasoning.effort for them, and sending any
        reasoning field makes OpenRouter emit thinking.type.disabled on
        tool-continuation turns (whose assistant tool_calls carry no thinking
        block), 400ing every turn after the first tool call. The profile must
        omit reasoning entirely so the model defaults to adaptive.
        """
        p = get_provider_profile("openrouter")
        for cfg in (
            {"enabled": True, "effort": "medium"},
            {"enabled": True, "effort": "xhigh"},
            {"effort": "high"},
            {"enabled": True},
        ):
            eb, _ = p.build_api_kwargs_extras(
                reasoning_config=cfg,
                supports_reasoning=True,
                model="anthropic/claude-fable-5",
            )
            assert "reasoning" not in eb, (cfg, eb)

    def test_default_reasoning(self):
        p = get_provider_profile("openrouter")
        eb, _ = p.build_api_kwargs_extras(supports_reasoning=True)
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_grok_session_id_sets_cache_affinity_header(self):
        """OpenRouter + Grok model + session_id => x-grok-conv-id header."""
        p = get_provider_profile("openrouter")
        _, tl = p.build_api_kwargs_extras(
            model="x-ai/grok-4",
            session_id="sess-abc123",
        )
        assert tl["extra_headers"]["x-grok-conv-id"] == "sess-abc123"

    def test_grok_xai_prefix_also_supported(self):
        """xai/ prefix (without dash) should also get the header."""
        p = get_provider_profile("openrouter")
        _, tl = p.build_api_kwargs_extras(
            model="xai/grok-3",
            session_id="sess-xyz",
        )
        assert tl["extra_headers"]["x-grok-conv-id"] == "sess-xyz"

    def test_non_grok_model_no_affinity_header(self):
        """OpenRouter + non-Grok model => no x-grok-conv-id header."""
        p = get_provider_profile("openrouter")
        _, tl = p.build_api_kwargs_extras(
            model="anthropic/claude-sonnet-4.6",
            session_id="sess-abc123",
        )
        assert "extra_headers" not in tl
        assert "x-grok-conv-id" not in tl

    def test_grok_without_session_id_no_header(self):
        """Grok model but no session_id => no header (nothing to pin)."""
        p = get_provider_profile("openrouter")
        _, tl = p.build_api_kwargs_extras(model="x-ai/grok-4")
        assert "extra_headers" not in tl

    def test_grok_reasoning_and_header_together(self):
        """Reasoning extra_body and Grok header should coexist."""
        p = get_provider_profile("openrouter")
        eb, tl = p.build_api_kwargs_extras(
            model="x-ai/grok-4",
            session_id="sess-123",
            supports_reasoning=True,
            reasoning_config={"enabled": True, "effort": "high"},
        )
        assert eb["reasoning"] == {"enabled": True, "effort": "high"}
        assert tl["extra_headers"]["x-grok-conv-id"] == "sess-123"

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

    def test_mandatory_anthropic_effort_routes_to_verbosity(self):
        """effort set + reasoning enabled → top-level verbosity == effort,
        and NO reasoning field in extra_body.

        Covers the full real config range produced by
        ``hermes_constants.parse_reasoning_effort`` —
        ``VALID_REASONING_EFFORTS = (minimal, low, medium, high, xhigh)``.
        """
        p = get_provider_profile("openrouter")
        model = "anthropic/claude-fable-5"
        assert self._is_mandatory(model)  # fixture really is mandatory
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            eb, tl = p.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                supports_reasoning=True,
                model=model,
            )
            assert tl["verbosity"] == effort, (effort, tl)
            assert "reasoning" not in eb, (effort, eb)

    def test_mandatory_anthropic_effort_without_enabled_key_routes(self):
        """effort present without an explicit ``enabled`` key still routes to
        verbosity (enabled defaults to True)."""
        p = get_provider_profile("openrouter")
        eb, tl = p.build_api_kwargs_extras(
            reasoning_config={"effort": "xhigh"},
            supports_reasoning=True,
            model="anthropic/claude-fable-5",
        )
        assert tl["verbosity"] == "xhigh"
        assert "reasoning" not in eb

    def test_mandatory_anthropic_verbosity_is_value_agnostic_passthrough(self):
        """The mapping passes the effort value through verbatim — it must NOT
        clamp or whitelist. ``xhigh`` is a real config value; ``max`` is not
        producible by ``parse_reasoning_effort`` today but OpenRouter accepts it
        for Claude (live-proven in #43432), so a forward value must survive
        rather than be silently dropped. The OpenAI SDK type only literals
        ``low|medium|high`` but it's a TypedDict (no runtime validation), so the
        extended scale reaches the wire untouched."""
        p = get_provider_profile("openrouter")
        for effort in ("xhigh", "max"):
            _, tl = p.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": effort},
                supports_reasoning=True,
                model="anthropic/claude-fable-5",
            )
            assert tl["verbosity"] == effort

    def test_mandatory_anthropic_no_verbosity_when_effort_absent(self):
        """No effort / none / disabled → no verbosity emitted, so the model
        keeps its own adaptive default. Still no reasoning field."""
        p = get_provider_profile("openrouter")
        model = "anthropic/claude-fable-5"
        for cfg in (
            None,
            {},
            {"enabled": True},
            {"effort": "none"},
            {"enabled": True, "effort": "none"},
            {"enabled": False, "effort": "high"},  # explicitly disabled wins
        ):
            eb, tl = p.build_api_kwargs_extras(
                reasoning_config=cfg,
                supports_reasoning=True,
                model=model,
            )
            assert "verbosity" not in tl, (cfg, tl)
            assert "reasoning" not in eb, (cfg, eb)

    def test_non_mandatory_reasoning_model_unchanged_no_verbosity(self):
        """Non-mandatory reasoning models (DeepSeek, Qwen, GPT) keep getting
        ``reasoning`` in extra_body and never get a ``verbosity`` field — the
        new path must not touch them."""
        p = get_provider_profile("openrouter")
        for model in ("deepseek/deepseek-chat", "qwen/qwen3-max", "openai/gpt-5.4"):
            assert not self._is_mandatory(model)  # fixture really is non-mandatory
            eb, tl = p.build_api_kwargs_extras(
                reasoning_config={"enabled": True, "effort": "high"},
                supports_reasoning=True,
                model=model,
            )
            assert eb["reasoning"] == {"enabled": True, "effort": "high"}, (model, eb)
            assert "verbosity" not in tl, (model, tl)

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

    def test_auth_type(self):
        p = get_provider_profile("nous")
        assert p.auth_type == "oauth_device_code"

    def test_reasoning_enabled(self):
        p = get_provider_profile("nous")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": True, "effort": "medium"},
            supports_reasoning=True,
        )
        assert eb["reasoning"] == {"enabled": True, "effort": "medium"}

    def test_reasoning_omitted_when_disabled(self):
        p = get_provider_profile("nous")
        eb, _ = p.build_api_kwargs_extras(
            reasoning_config={"enabled": False},
            supports_reasoning=True,
        )
        assert "reasoning" not in eb


class TestQwenProfile:
    def test_max_tokens(self):
        p = get_provider_profile("qwen-oauth")
        assert p.default_max_tokens == 65536

    def test_auth_type(self):
        p = get_provider_profile("qwen-oauth")
        assert p.auth_type == "oauth_external"

    def test_extra_body_vl(self):
        p = get_provider_profile("qwen-oauth")
        body = p.build_extra_body()
        assert body["vl_high_resolution_images"] is True

    def test_prepare_messages_normalizes_content(self):
        p = get_provider_profile("qwen-oauth")
        msgs = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "hello"},
        ]
        result = p.prepare_messages(msgs)
        # System message: content normalized to list, cache_control on last part
        assert isinstance(result[0]["content"], list)
        assert result[0]["content"][-1].get("cache_control") == {"type": "ephemeral"}
        assert result[0]["content"][-1]["text"] == "Be helpful"
        # User message: content normalized to list
        assert isinstance(result[1]["content"], list)
        assert result[1]["content"][0]["text"] == "hello"

    def test_prepare_messages_copy_on_write(self):
        p = get_provider_profile("qwen-oauth")
        system_part = {"type": "text", "text": "Be helpful"}
        msgs = [
            {"role": "system", "content": [system_part]},
            {"role": "assistant", "content": [{"type": "text", "text": "unchanged"}]},
            {"role": "user", "content": ["hello"]},
        ]

        result = p.prepare_messages(msgs)

        assert result is not msgs
        assert result[0] is not msgs[0]
        assert result[0]["content"] is not msgs[0]["content"]
        assert result[0]["content"][0] is not system_part
        assert result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in system_part
        assert result[1] is msgs[1]
        assert result[2] is not msgs[2]
        assert result[2]["content"] == [{"type": "text", "text": "hello"}]
        assert msgs[2]["content"] == ["hello"]

    def test_prepare_messages_does_not_poison_strict_provider_history(self):
        qwen = get_provider_profile("qwen-oauth")
        msgs = [
            {"role": "system", "content": [{"type": "text", "text": "Be helpful"}]},
            {"role": "user", "content": "hello"},
        ]

        qwen_result = qwen.prepare_messages(msgs)

        assert qwen_result[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in msgs[0]["content"][0]
        assert msgs[1]["content"] == "hello"

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


class TestBaseProfile:
    def test_prepare_messages_passthrough(self):
        p = ProviderProfile(name="test")
        msgs = [{"role": "user", "content": "hi"}]
        assert p.prepare_messages(msgs) is msgs

    def test_build_extra_body_empty(self):
        p = ProviderProfile(name="test")
        assert p.build_extra_body() == {}

    def test_build_api_kwargs_extras_empty(self):
        p = ProviderProfile(name="test")
        eb, tl = p.build_api_kwargs_extras()
        assert eb == {}
        assert tl == {}
