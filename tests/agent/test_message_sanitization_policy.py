"""Tests for the single-owner call_id + reasoning_content policies.

Audit F4 consolidation: agent/message_sanitization.py now owns the
deterministic call_id synthesis, call_id coalescing/dedup, and the
reasoning_content strip-vs-repad provider-direction policy. These tests pin
the owner functions' behavior (including byte-exact hash outputs — they feed
prompt-cache keys) and verify the legacy entry points still delegate here.
"""

from types import SimpleNamespace

import pytest

from agent.message_sanitization import (
    apply_reasoning_content_policy,
    coalesce_tool_call_id,
    deterministic_call_id,
    matches_reasoning_echo_family,
    needs_reasoning_echo,
    reapply_reasoning_echo,
    reasoning_echo_family,
    uniquify_tool_call_ids,
)

# ---------------------------------------------------------------------------
# deterministic_call_id — byte-exact (prompt-cache keys)
# ---------------------------------------------------------------------------

class TestDeterministicCallId:
    def test_known_hash_outputs_are_stable(self):
        # Golden values: sha256(f"{fn}:{args}:{index}")[:12] prefixed call_.
        # Any change here invalidates users' prompt caches — do NOT update
        # these expectations without a migration plan.
        assert deterministic_call_id("terminal", '{"command":"ls"}', 0) == \
            "call_40ccaef54d02"
        assert deterministic_call_id("terminal", '{"command":"ls"}', 1) == \
            "call_567cb168d22d"
        assert deterministic_call_id("", "", 0) == "call_feda901d71ea"

    def test_index_disambiguates(self):
        assert deterministic_call_id("t", "{}", 0) != deterministic_call_id("t", "{}", 1)

    def test_surrogates_do_not_crash(self):
        out = deterministic_call_id("t", "bad \ud800 arg", 0)
        assert out.startswith("call_")

# ---------------------------------------------------------------------------
# coalesce_tool_call_id
# ---------------------------------------------------------------------------

class TestCoalesceToolCallId:
    def test_dict_call_id_wins_over_id(self):
        assert coalesce_tool_call_id({"call_id": "c", "id": "i"}) == "c"

    def test_dict_falls_back_to_id_and_strips(self):
        assert coalesce_tool_call_id({"id": " i "}) == "i"
        assert coalesce_tool_call_id({"call_id": "", "id": "i2"}) == "i2"

    def test_dict_empty(self):
        assert coalesce_tool_call_id({}) == ""

    def test_object_forms(self):
        assert coalesce_tool_call_id(SimpleNamespace(call_id="c", id="i")) == "c"
        assert coalesce_tool_call_id(SimpleNamespace(call_id=None, id=" i ")) == "i"
        assert coalesce_tool_call_id(SimpleNamespace(call_id=None, id=None)) == ""

# ---------------------------------------------------------------------------
# uniquify_tool_call_ids
# ---------------------------------------------------------------------------

class TestUniquifyToolCallIds:
    def test_no_duplicates_untouched(self):
        tcs = [
            {"id": "a", "function": {"name": "f", "arguments": "{}"}},
            {"id": "b", "function": {"name": "g", "arguments": "{}"}},
        ]
        out = uniquify_tool_call_ids(tcs)
        assert out is tcs
        assert [tc["id"] for tc in out] == ["a", "b"]

    def test_duplicate_gets_deterministic_suffix(self):
        tcs = [
            {"id": "x", "call_id": "x", "function": {"name": "f", "arguments": "{}"}},
            {"id": "x", "call_id": "x", "function": {"name": "g", "arguments": "{}"}},
            {"id": "x", "function": {"name": "h", "arguments": "{}"}},
        ]
        uniquify_tool_call_ids(tcs)
        assert tcs[0]["id"] == "x"
        assert tcs[1]["id"] == "x_d2"
        assert tcs[1]["call_id"] == "x_d2"
        assert tcs[2]["id"] == "x_d3"

    def test_composite_id_collides_on_call_half_and_preserves_item_half(self):
        tcs = [
            {"id": "call_y|fc_1", "function": {"name": "f", "arguments": "{}"}},
            {"id": "call_y|fc_2", "function": {"name": "g", "arguments": "{}"}},
        ]
        uniquify_tool_call_ids(tcs)
        assert tcs[0]["id"] == "call_y|fc_1"
        assert tcs[1]["id"] == "call_y_d2|fc_2"

    def test_suffix_collision_advances_counter(self):
        tcs = [
            {"id": "z", "function": {"name": "a", "arguments": "{}"}},
            {"id": "z_d2", "function": {"name": "b", "arguments": "{}"}},
            {"id": "z", "function": {"name": "c", "arguments": "{}"}},
        ]
        uniquify_tool_call_ids(tcs)
        assert tcs[2]["id"] == "z_d3"

    def test_blank_and_non_string_ids_skipped(self):
        tcs = [
            {"id": "", "function": {"name": "a", "arguments": "{}"}},
            {"id": None, "function": {"name": "b", "arguments": "{}"}},
            SimpleNamespace(id=42, call_id=None, function=None),
        ]
        uniquify_tool_call_ids(tcs)
        assert tcs[0]["id"] == ""
        assert tcs[1]["id"] is None

    def test_namespace_objects_mutated(self):
        tcs = [
            SimpleNamespace(id="n", call_id="n",
                            function=SimpleNamespace(name="a", arguments="{}")),
            SimpleNamespace(id="n", call_id="n",
                            function=SimpleNamespace(name="b", arguments="{}")),
        ]
        uniquify_tool_call_ids(tcs)
        assert tcs[1].id == "n_d2"
        assert tcs[1].call_id == "n_d2"

    def test_empty_and_none_inputs(self):
        assert uniquify_tool_call_ids([]) == []
        assert uniquify_tool_call_ids(None) is None

# ---------------------------------------------------------------------------
# reasoning_echo_family — the provider-direction table
# ---------------------------------------------------------------------------

class TestReasoningEchoFamily:
    @pytest.mark.parametrize("provider,model,base_url,family", [
        ("kimi-coding", None, "https://x", "kimi"),
        ("kimi-coding-cn", None, "https://x", "kimi"),
        ("custom", None, "https://api.kimi.com/v1", "kimi"),
        ("custom", None, "https://api.moonshot.ai/v1", "kimi"),
        ("custom", None, "https://api.moonshot.cn/v1", "kimi"),
        ("deepseek", "whatever", "https://x", "deepseek"),
        ("DeepSeek", "whatever", "https://x", "deepseek"),
        ("openrouter", "deepseek/deepseek-v3", "https://openrouter.ai", "deepseek"),
        ("custom", None, "https://api.deepseek.com", "deepseek"),
        ("xiaomi", None, "https://x", "mimo"),
        ("custom", "MiMo-7B", "https://x", "mimo"),
        ("custom", None, "https://api.xiaomimimo.com/v1", "mimo"),
        ("openai", "gpt-5", "https://api.openai.com/v1", None),
        ("mistral", "mistral-large", "https://api.mistral.ai/v1", None),
        (None, None, None, None),
    ])
    def test_table(self, provider, model, base_url, family):
        assert reasoning_echo_family(provider, model, base_url) == family
        assert needs_reasoning_echo(provider, model, base_url) is (family is not None)

    def test_kimi_provider_match_is_exact_not_lowered(self):
        # Original predicate compared the raw provider string against the
        # kimi-coding set; keep that semantic.
        assert matches_reasoning_echo_family("kimi", "KIMI-CODING", None, "https://x") is False

    def test_membership_is_per_family(self):
        # A deepseek model pointed at a kimi host matches both families
        # independently (the per-family predicates on AIAgent rely on this).
        assert matches_reasoning_echo_family(
            "kimi", "custom", "deepseek-chat", "https://api.kimi.com") is True
        assert matches_reasoning_echo_family(
            "deepseek", "custom", "deepseek-chat", "https://api.kimi.com") is True

    def test_unknown_family_raises(self):
        with pytest.raises(KeyError):
            matches_reasoning_echo_family("nope", "p", "m", "https://x")

# ---------------------------------------------------------------------------
# apply_reasoning_content_policy
# ---------------------------------------------------------------------------

class TestApplyReasoningContentPolicy:
    def test_non_assistant_untouched(self):
        api = {"role": "user", "content": "u", "reasoning_content": "keep"}
        apply_reasoning_content_policy(
            {"role": "user", "content": "u", "reasoning_content": "keep"}, api, True)
        assert api["reasoning_content"] == "keep"

    def test_require_side_preserves_existing(self):
        api = {"role": "assistant", "content": "x"}
        apply_reasoning_content_policy(
            {"role": "assistant", "content": "x", "reasoning_content": "thoughts"},
            api, True)
        assert api["reasoning_content"] == "thoughts"

    def test_require_side_upgrades_empty_string_to_space(self):
        api = {"role": "assistant", "content": "x", "reasoning_content": ""}
        apply_reasoning_content_policy(
            {"role": "assistant", "content": "x", "reasoning_content": ""}, api, True)
        assert api["reasoning_content"] == " "

    def test_strict_side_strips_existing(self):
        api = {"role": "assistant", "content": "x", "reasoning_content": " "}
        apply_reasoning_content_policy(
            {"role": "assistant", "content": "x", "reasoning_content": " "}, api, False)
        assert "reasoning_content" not in api

    def test_cross_provider_poisoned_history_pads_with_space(self):
        src = {"role": "assistant", "content": "x", "reasoning": "other-provider CoT",
               "tool_calls": [{"id": "c", "function": {"name": "t", "arguments": "{}"}}]}
        api = {"role": "assistant", "content": "x"}
        apply_reasoning_content_policy(src, api, True)
        assert api["reasoning_content"] == " "  # pad, never the foreign CoT

    def test_reasoning_promoted_only_on_require_side(self):
        src = {"role": "assistant", "content": "x", "reasoning": "healthy"}
        api = {"role": "assistant", "content": "x"}
        apply_reasoning_content_policy(src, api, True)
        assert api["reasoning_content"] == "healthy"
        api2 = {"role": "assistant", "content": "x", "reasoning_content": "stale"}
        apply_reasoning_content_policy(src, api2, False)
        assert "reasoning_content" not in api2

    def test_require_side_pads_bare_assistant_turn(self):
        api = {"role": "assistant", "content": "x"}
        apply_reasoning_content_policy({"role": "assistant", "content": "x"}, api, True)
        assert api["reasoning_content"] == " "

    def test_non_string_reasoning_content_removed(self):
        api = {"role": "assistant", "content": "x", "reasoning_content": None}
        apply_reasoning_content_policy(
            {"role": "assistant", "content": "x", "reasoning_content": None}, api, False)
        assert "reasoning_content" not in api

# ---------------------------------------------------------------------------
# reapply_reasoning_echo
# ---------------------------------------------------------------------------

class TestReapplyReasoningEcho:
    MSGS = [
        {"role": "assistant", "content": "a1", "reasoning_content": " "},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u"},
        {"role": "tool", "content": "t", "tool_call_id": "c"},
    ]

    def test_require_side_pads_missing_only(self):
        import copy
        msgs = copy.deepcopy(self.MSGS)
        assert reapply_reasoning_echo(msgs, True) == 1
        assert msgs[0]["reasoning_content"] == " "  # untouched
        assert msgs[1]["reasoning_content"] == " "  # padded
        assert "reasoning_content" not in msgs[2]

    def test_strict_side_strips_all(self):
        import copy
        msgs = copy.deepcopy(self.MSGS)
        assert reapply_reasoning_echo(msgs, False) == 1
        assert all("reasoning_content" not in m for m in msgs)

    def test_idempotent(self):
        import copy
        msgs = copy.deepcopy(self.MSGS)
        reapply_reasoning_echo(msgs, True)
        assert reapply_reasoning_echo(msgs, True) == 0
        reapply_reasoning_echo(msgs, False)
        assert reapply_reasoning_echo(msgs, False) == 0

# ---------------------------------------------------------------------------
# Per-provider reasoning_echo config opt-in — preserves reasoning_content
# on replay for custom providers / OpenAI-compatible gateways that proxy
# thinking-mode models but are not matched by the built-in host-based
# _REASONING_ECHO_RULES (DeepSeek / Kimi / MiMo).
#
# The flag is per-active-provider:
#   - Primary: read from ``model.reasoning_echo`` in config at init / switch_model
#   - Fallback: set by try_activate_fallback from the fallback entry's field
#   - Restore: restore_primary_runtime copies the snapshot saved by switch_model
#
# Unlike a global toggle, the flag travels with the active provider, so
# falling back to a strict provider (Mistral, Groq, Cerebras) correctly
# strips reasoning_content even when the primary had the flag enabled.
# ---------------------------------------------------------------------------

class TestPerProviderReasoningEcho:
    """Verify the per-provider reasoning_echo opt-in flag."""

    def _make_agent(self, reasoning_echo_flag=False, provider="custom",
                    model="my-model", base_url="https://gw.example.com/v1"):
        """Build a minimal AIAgent-shaped object without full init."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        agent.base_url = base_url
        agent._base_url_lower = base_url.lower()
        agent._thinking_pad_cache = None
        agent._reasoning_echo_flag = reasoning_echo_flag
        agent._needs_deepseek_tool_reasoning = lambda: False
        agent._needs_kimi_tool_reasoning = lambda: False
        agent._needs_mimo_tool_reasoning = lambda: False
        return agent

    def test_default_false_strips_for_custom_provider(self):
        """Default (flag=False): a custom gateway is NOT an echo family,
        so reasoning_content is stripped — historical behavior."""
        agent = self._make_agent(reasoning_echo_flag=False)
        assert agent._needs_thinking_reasoning_pad() is False
        assert agent._reasoning_echo_opt_in() is False

    def test_opt_in_preserves_for_custom_provider(self):
        """Flag on: even a non-echo-family custom gateway keeps
        reasoning_content on replay."""
        agent = self._make_agent(reasoning_echo_flag=True)
        assert agent._needs_thinking_reasoning_pad() is True
        assert agent._reasoning_echo_opt_in() is True

    def test_strict_fallback_strips_despite_primary_opt_in(self):
        """Primary has flag=True, fallback switches to a strict provider.
        The fallback reconciling path (reapply_reasoning_echo_for_provider)
        uses _needs_thinking_reasoning_pad which checks the *current*
        provider's flag — strict providers have flag=False, so the field
        is stripped (no HTTP 400)."""
        agent = self._make_agent(reasoning_echo_flag=True)  # primary opt-in
        # Simulate fallback to a strict provider
        agent.provider = "mistral"
        agent.model = "mistral-large"
        agent.base_url = "https://api.mistral.ai/v1"
        agent._base_url_lower = "https://api.mistral.ai/v1"
        agent._thinking_pad_cache = None  # invalidate per-provider cache
        agent._reasoning_echo_flag = False  # fallback entry has no opt-in
        assert agent._needs_thinking_reasoning_pad() is False
        # End-to-end: reapply_reasoning_echo strips the field
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        api_msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "thoughts"},
            {"role": "assistant", "content": "hi2", "reasoning_content": " "},
            {"role": "user", "content": "bye"},
        ]
        changed = reapply_reasoning_echo_for_provider(agent, api_msgs)
        assert changed == 2
        for m in api_msgs:
            if m["role"] == "assistant":
                assert "reasoning_content" not in m

    def test_fallback_opt_in_preserves_reasoning(self):
        """Fallback to a custom provider with reasoning_echo=True:
        the field is preserved/re-padded."""
        agent = self._make_agent(reasoning_echo_flag=False)  # primary no opt-in
        # Simulate fallback to a custom provider with opt-in
        agent.provider = "custom"
        agent.model = "t9s/kimi-k3"
        agent.base_url = "https://gw.example.com/v1"
        agent._base_url_lower = "https://gw.example.com/v1"
        agent._thinking_pad_cache = None
        agent._reasoning_echo_flag = True  # fallback entry has opt-in
        assert agent._needs_thinking_reasoning_pad() is True
        # End-to-end: reapply_reasoning_echo re-pads
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        api_msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},  # no reasoning_content
            {"role": "user", "content": "bye"},
        ]
        changed = reapply_reasoning_echo_for_provider(agent, api_msgs)
        assert changed == 1
        assert api_msgs[1].get("reasoning_content") == " "

    def test_restore_primary_reverts_flag(self):
        """After fallback, restore_primary_runtime reverts the flag
        from the snapshot saved by switch_model."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._reasoning_echo_flag = True  # primary had opt-in
        agent._fallback_activated = True
        agent._rate_limited_until = 0
        agent._primary_runtime = {
            "model": "glm-5.2",
            "provider": "custom",
            "requested_provider": "custom",
            "base_url": "https://gw.example.com/v1",
            "api_mode": "chat_completions",
            "api_key": "sk-test",
            "client_kwargs": {"api_key": "sk-test", "base_url": "https://gw.example.com/v1"},
            "use_prompt_caching": True,
            "use_native_cache_layout": False,
            "reasoning_echo_flag": True,  # snapshot saved by switch_model
        }
        agent._transport_cache = {}
        agent.client = None
        agent._client_kwargs = {"api_key": "sk-fallback", "base_url": "https://fallback.com/v1"}
        agent._use_prompt_caching = False
        agent._use_native_cache_layout = False
        agent.api_mode = "chat_completions"
        agent.api_key = "sk-fallback"
        agent.model = "fallback-model"
        agent.provider = "fallback-provider"
        agent.requested_provider = "fallback-provider"
        agent.base_url = "https://fallback.com/v1"
        agent.context_compressor = None
        agent._config_context_length = None

        # Simulate: fallback set the flag to False
        agent._reasoning_echo_flag = False

        from agent.agent_runtime_helpers import restore_primary_runtime
        restore_primary_runtime(agent)

        # Flag should be restored from snapshot
        assert agent._reasoning_echo_flag is True
        assert agent.model == "glm-5.2"


# ---------------------------------------------------------------------------
# ``model.reasoning_echo: never`` — opt a route out of echo-back.
#
# The family rules match DeepSeek/MiMo by *model name*, so any gateway serving
# ``deepseek-*`` is classified as an echo family even when its own backend does
# not enforce the echo (opencode.ai accepts a replayed assistant tool-call turn
# with the field absent). Echoing there only hands the model its own stale
# chain-of-thought back, which it then re-emits as a restatement of a past
# answer. ``never`` forces the strict strip side over every family rule.
# ---------------------------------------------------------------------------

class TestReasoningEchoNeverMode:
    """The explicit strip-side override, and the auto default it must not disturb."""

    def _make_agent(self, mode="", provider="opencode-go", model="deepseek-v4.1-flash",
                    base_url="https://opencode.ai/zen/go/v1"):
        """A gateway route whose model name matches the DeepSeek family rule."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent.provider = provider
        agent.model = model
        agent.base_url = base_url
        agent._base_url_lower = base_url.lower()
        agent._thinking_pad_cache = None
        agent._reasoning_echo_mode = mode
        agent._reasoning_echo_flag = mode == "always"
        return agent

    def test_gateway_deepseek_is_an_echo_family_without_the_override(self):
        """Baseline: the model-name rule classifies the gateway route as echo-back."""
        agent = self._make_agent(mode="")
        assert agent._reasoning_echo_forced_strip() is False
        assert agent._needs_thinking_reasoning_pad() is True

    def test_never_forces_strip_over_family_detection(self):
        """``never`` wins over the DeepSeek-by-model-name match."""
        agent = self._make_agent(mode="never")
        assert agent._reasoning_echo_forced_strip() is True
        assert agent._needs_thinking_reasoning_pad() is False

    def test_never_strips_even_when_the_family_predicate_says_echo(self):
        """The override is unconditional: a True family predicate cannot re-enable the echo."""
        agent = self._make_agent(mode="never")
        agent._needs_deepseek_tool_reasoning = lambda: True
        assert agent._needs_thinking_reasoning_pad() is False

    def test_never_strips_even_with_a_stale_opt_in_flag(self):
        """A leftover opt-in flag cannot out-vote ``never``."""
        agent = self._make_agent(mode="never")
        agent._reasoning_echo_flag = True
        assert agent._needs_thinking_reasoning_pad() is False

    def test_always_still_echoes_a_non_family_route(self):
        """``always`` keeps the existing opt-in path working."""
        agent = self._make_agent(mode="always", provider="custom", model="my-model",
                                 base_url="https://gw.example.com/v1")
        assert agent._needs_thinking_reasoning_pad() is True

    def test_override_strips_the_wire_payload_end_to_end(self):
        """End-to-end: the override reaches reapply_reasoning_echo_for_provider."""
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        agent = self._make_agent(mode="never")
        api_msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "stale chain-of-thought"},
            {"role": "user", "content": "bye"},
        ]
        assert reapply_reasoning_echo_for_provider(agent, api_msgs) == 1
        assert "reasoning_content" not in api_msgs[1]

    def test_auto_route_is_unchanged_by_the_override(self):
        """No override: a detected family keeps echoing — no behavior change for existing users."""
        agent = self._make_agent(mode="")
        agent._needs_deepseek_tool_reasoning = lambda: True
        assert agent._needs_thinking_reasoning_pad() is True


class TestReasoningEchoModeParsing:
    """``normalize_reasoning_echo_mode`` — the one parser for config keys and fallback entries."""

    @pytest.mark.parametrize("raw,expected", [
        ("never", "never"),
        ("NEVER", "never"),
        ("  never  ", "never"),
        ("always", "always"),
        (True, "always"),
        # False and "auto" stay auto: the boolean has always meant "no opt-in, defer to
        # detection", so an explicit False must NOT become a strip override.
        (False, None),
        ("auto", None),
        ("", None),
        ("garbage", None),
        (None, None),
        (1, None),
    ])
    def test_normalize(self, raw, expected):
        from agent.reasoning_params import normalize_reasoning_echo_mode
        assert normalize_reasoning_echo_mode(raw) == expected

    def test_sync_never_sets_mode_and_drops_the_pad_cache(self):
        """``_sync_reasoning_echo_from_config`` derives both carriers and invalidates the cache."""
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._thinking_pad_cache = ("stale-key", True)
        agent._read_reasoning_echo_from_config = lambda: "never"
        agent._sync_reasoning_echo_from_config()
        assert agent._reasoning_echo_mode == "never"
        assert agent._reasoning_echo_flag is False
        assert agent._thinking_pad_cache is None

    def test_sync_always_sets_the_flag(self):
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._thinking_pad_cache = None
        agent._read_reasoning_echo_from_config = lambda: "always"
        agent._sync_reasoning_echo_from_config()
        assert agent._reasoning_echo_mode == "always"
        assert agent._reasoning_echo_flag is True

    def test_sync_auto_clears_both_carriers(self):
        from run_agent import AIAgent
        agent = object.__new__(AIAgent)
        agent._thinking_pad_cache = None
        agent._reasoning_echo_mode = "never"
        agent._reasoning_echo_flag = True
        agent._read_reasoning_echo_from_config = lambda: None
        agent._sync_reasoning_echo_from_config()
        assert agent._reasoning_echo_mode == ""
        assert agent._reasoning_echo_flag is False


class TestStaleThinkingWireTruthHonoursOverride:
    """The preflight estimator and the compressor tail walk must agree, override included."""

    ROUTE = ("chat_completions", "opencode-go", "deepseek-v4.1-flash", "https://opencode.ai/zen/go/v1")

    def test_forced_strip_reports_nothing_stale_on_the_wire(self):
        from agent.message_sanitization import stale_thinking_reaches_wire
        assert stale_thinking_reaches_wire(*self.ROUTE) is True
        assert stale_thinking_reaches_wire(*self.ROUTE, forced_strip=True) is False

    def test_both_consumers_agree_under_never(self, monkeypatch):
        """Both sides read ``read_reasoning_echo_mode``, so a disagreement cannot open a compaction loop."""
        from agent import reasoning_params
        from agent.context_compressor import ContextCompressor
        from agent.turn_context import _agent_stale_thinking_on_wire

        monkeypatch.setattr(reasoning_params, "read_reasoning_echo_mode", lambda: "never")
        agent = SimpleNamespace(api_mode="chat_completions", provider="opencode-go",
                                model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1")
        assert _agent_stale_thinking_on_wire(agent) is False
        comp = object.__new__(ContextCompressor)
        comp.api_mode, comp.provider = "chat_completions", "opencode-go"
        comp.model, comp.base_url = "deepseek-v4.1-flash", "https://opencode.ai/zen/go/v1"
        assert comp._stale_thinking_on_wire() is False

    def test_both_consumers_agree_without_the_override(self, monkeypatch):
        from agent import reasoning_params
        from agent.context_compressor import ContextCompressor
        from agent.turn_context import _agent_stale_thinking_on_wire

        monkeypatch.setattr(reasoning_params, "read_reasoning_echo_mode", lambda: None)
        agent = SimpleNamespace(api_mode="chat_completions", provider="opencode-go",
                                model="deepseek-v4.1-flash", base_url="https://opencode.ai/zen/go/v1")
        assert _agent_stale_thinking_on_wire(agent) is True
        comp = object.__new__(ContextCompressor)
        comp.api_mode, comp.provider = "chat_completions", "opencode-go"
        comp.model, comp.base_url = "deepseek-v4.1-flash", "https://opencode.ai/zen/go/v1"
        assert comp._stale_thinking_on_wire() is True
