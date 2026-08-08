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
from run_agent import AIAgent


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

    def test_deterministic_across_calls(self):
        a = deterministic_call_id("web_search", '{"q":"x"}', 3)
        b = deterministic_call_id("web_search", '{"q":"x"}', 3)
        assert a == b
        assert a.startswith("call_")
        assert len(a) == len("call_") + 12

    def test_index_disambiguates(self):
        assert deterministic_call_id("t", "{}", 0) != deterministic_call_id("t", "{}", 1)

    def test_surrogates_do_not_crash(self):
        out = deterministic_call_id("t", "bad \ud800 arg", 0)
        assert out.startswith("call_")

    def test_codex_adapter_wrapper_delegates(self):
        from agent.codex_responses_adapter import _deterministic_call_id
        assert _deterministic_call_id("terminal", '{"command":"ls"}', 0) == \
            deterministic_call_id("terminal", '{"command":"ls"}', 0)

    def test_run_agent_static_delegates(self):
        from run_agent import AIAgent
        assert AIAgent._deterministic_call_id("terminal", '{"command":"ls"}', 0) == \
            deterministic_call_id("terminal", '{"command":"ls"}', 0)


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

    def test_run_agent_static_delegates(self):
        from run_agent import AIAgent
        tc = {"call_id": "c9", "id": "i9"}
        assert AIAgent._get_tool_call_id_static(tc) == coalesce_tool_call_id(tc)


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
# agent.reasoning_echo config toggle — preserve reasoning_content for
# OpenAI-compatible local backends (llama.cpp, vLLM, ...) whose KV cache
# keeps thinking tokens. When enabled, assistant turns replay with
# reasoning_content intact, so the server-side prompt-cache prefix survives
# across turns. Default disabled keeps strict-provider (Mistral, Groq, ...)
# behavior unchanged. The toggle widens ONLY the replay gate
# (_preserve_reasoning_echo); the fallback reconciling path still uses
# _needs_thinking_reasoning_pad, so a strict-provider fallback strips the
# field.
# ---------------------------------------------------------------------------

class TestReasoningEchoConfigToggle:
    def _make_agent(self, reasoning_echo, monkeypatch):
        """Build an AIAgent-shaped object without a full init."""
        import hermes_cli.config as cfg
        monkeypatch.setattr(
            cfg, "load_config_readonly",
            lambda: {"agent": {"reasoning_echo": reasoning_echo}})
        agent = object.__new__(AIAgent)
        agent.provider = "custom"
        agent.model = "qwen3.6-27b"
        agent.base_url = "http://127.0.0.1:8080/v1"
        agent._base_url_lower = "http://127.0.0.1:8080/v1"
        agent._thinking_pad_cache = None
        agent._preserve_reasoning_echo_cache = None
        agent._needs_deepseek_tool_reasoning = lambda: False
        agent._needs_kimi_tool_reasoning = lambda: False
        agent._needs_mimo_tool_reasoning = lambda: False
        return agent

    def test_disabled_default_matches_strict_provider(self, monkeypatch):
        # Default (false): a local llama.cpp endpoint is NOT an echo family,
        # so reasoning_content is stripped — historical behavior.
        agent = self._make_agent(False, monkeypatch)
        assert agent._needs_thinking_reasoning_pad() is False
        assert agent._preserve_reasoning_echo() is False

    def test_enabled_preserves_reasoning_for_local_backend(self, monkeypatch):
        # Toggle on: even a non-echo family (local llama.cpp) keeps
        # reasoning_content on replay, protecting the KV-cache prefix.
        agent = self._make_agent(True, monkeypatch)
        assert agent._needs_thinking_reasoning_pad() is False  # family unchanged
        assert agent._preserve_reasoning_echo() is True  # replay gate widened

    def test_enabled_does_not_break_echo_family(self, monkeypatch):
        # DeepSeek/Kimi still get echo-back regardless of the toggle.
        import hermes_cli.config as cfg
        monkeypatch.setattr(
            cfg, "load_config_readonly", lambda: {"agent": {"reasoning_echo": False}})
        agent = object.__new__(AIAgent)
        agent.provider = "deepseek"
        agent.model = "deepseek-v4-pro"
        agent.base_url = "https://api.deepseek.com/v1"
        agent._base_url_lower = "https://api.deepseek.com/v1"
        agent._thinking_pad_cache = None
        agent._preserve_reasoning_echo_cache = None
        agent._needs_deepseek_tool_reasoning = lambda: True
        agent._needs_kimi_tool_reasoning = lambda: False
        agent._needs_mimo_tool_reasoning = lambda: False
        assert agent._needs_thinking_reasoning_pad() is True

    def test_strict_fallback_still_strips_despite_toggle(self, monkeypatch):
        # Sweeper-requested scenario: toggle enabled on the primary local
        # endpoint, then a fallback to a strict provider. The fallback
        # reconciling path (reapply_reasoning_echo_for_provider) uses only
        # _needs_thinking_reasoning_pad — family-based — so the strict
        # provider never receives reasoning_content (no HTTP 400).
        agent = self._make_agent(True, monkeypatch)  # toggle on
        # Fallback switches to a strict provider: Mistral.
        agent.provider = "mistral"
        agent.model = "mistral-large"
        agent.base_url = "https://api.mistral.ai/v1"
        agent._base_url_lower = "https://api.mistral.ai/v1"
        agent._thinking_pad_cache = None  # invalidate per-provider cache
        assert agent._needs_thinking_reasoning_pad() is False  # strict: strip
        assert agent._preserve_reasoning_echo() is True  # toggle still on
        # End-to-end fallback test: reapply_reasoning_echo_for_provider
        # must strip reasoning_content from all assistant turns when
        # falling back to a strict provider, even with the toggle on.
        from agent.agent_runtime_helpers import reapply_reasoning_echo_for_provider
        api_msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi", "reasoning_content": "thoughts"},
            {"role": "assistant", "content": "hi2", "reasoning_content": " "},
            {"role": "user", "content": "bye"},
        ]
        changed = reapply_reasoning_echo_for_provider(agent, api_msgs)
        assert changed == 2  # both assistant turns modified
        for m in api_msgs:
            if m["role"] == "assistant":
                assert "reasoning_content" not in m, \
                    "reapply_reasoning_echo_for_provider failed to strip for Mistral"
