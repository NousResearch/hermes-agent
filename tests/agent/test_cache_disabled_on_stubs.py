"""Regression for #76085: prompt_caching.cache_ttl off on stub policy paths.

Blank SimpleNamespace stubs used by MoA decoration and auxiliary/MoA
plan_cache_sections_for_destination never carried ``_cache_disabled``, so
``anthropic_prompt_cache_policy`` re-enabled cache_control markers even when
the operator set ``prompt_caching.cache_ttl: false``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch


def _has_cache_control(obj) -> bool:
    if isinstance(obj, dict):
        if "cache_control" in obj:
            return True
        return any(_has_cache_control(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_cache_control(v) for v in obj)
    return False


class TestPromptCachingDisabledFromConfig:
    def test_off_values(self):
        from agent.agent_runtime_helpers import prompt_caching_disabled_from_config

        for ttl in (False, None, "off", "false", "disabled", "no", "none", "OFF"):
            with patch(
                "hermes_cli.config.load_config_readonly",
                return_value={"prompt_caching": {"cache_ttl": ttl}},
            ):
                assert prompt_caching_disabled_from_config() is True, ttl

    def test_enabled_values(self):
        from agent.agent_runtime_helpers import prompt_caching_disabled_from_config

        for ttl in ("5m", "1h"):
            with patch(
                "hermes_cli.config.load_config_readonly",
                return_value={"prompt_caching": {"cache_ttl": ttl}},
            ):
                assert prompt_caching_disabled_from_config() is False, ttl

    def test_shared_predicate_matches_agent_init_semantics(self):
        """agent_init and the stub paths must share one disable predicate.

        Unknown values keep caching enabled (default TTL), exactly like the
        historical inline detection in agent_init (#76085 drift guard).
        """
        from agent.agent_runtime_helpers import cache_ttl_means_disabled

        for ttl in (False, None, "off", "false", "disabled", "no", "none", "OFF"):
            assert cache_ttl_means_disabled(ttl) is True, ttl
        for ttl in ("5m", "1h", "2h", 5, True, "weird"):
            assert cache_ttl_means_disabled(ttl) is False, ttl


class TestPlanCacheSectionsHonorsDisable:
    def test_explicit_cache_disabled_strips_markers(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "search",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        out_msgs, out_tools = plan_cache_sections_for_destination(
            messages,
            tools,
            provider="anthropic",
            base_url="https://api.anthropic.com",
            api_mode="anthropic_messages",
            model="claude-opus-4.8",
            cache_disabled=True,
        )
        assert not _has_cache_control(out_msgs)
        assert not _has_cache_control(out_tools)

    def test_config_off_without_explicit_flag(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
        ]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": "off"}},
        ):
            out_msgs, out_tools = plan_cache_sections_for_destination(
                messages,
                None,
                provider="anthropic",
                base_url="https://api.anthropic.com",
                api_mode="anthropic_messages",
                model="claude-opus-4.8",
            )
        assert not _has_cache_control(out_msgs)
        assert out_tools is None or not _has_cache_control(out_tools)

    def test_enabled_still_adds_markers_on_native_anthropic(self):
        from agent.agent_runtime_helpers import plan_cache_sections_for_destination

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
            {"role": "user", "content": "again"},
        ]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": "5m"}},
        ):
            out_msgs, _ = plan_cache_sections_for_destination(
                messages,
                None,
                provider="anthropic",
                base_url="https://api.anthropic.com",
                api_mode="anthropic_messages",
                model="claude-opus-4.8",
                cache_disabled=False,
            )
        assert _has_cache_control(out_msgs), (
            "With caching enabled, native Anthropic destinations must still "
            "receive cache_control breakpoints."
        )


class TestMoASlotDecorationHonorsDisable:
    def test_maybe_apply_skips_markers_when_disabled(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4.8",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }
        out = _maybe_apply_moa_cache_control(
            messages, runtime, cache_disabled=True
        )
        assert not _has_cache_control(out)
        # Inputs must not be mutated.
        assert not _has_cache_control(messages)

    def test_maybe_apply_config_off(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
        runtime = {
            "provider": "anthropic",
            "model": "claude-opus-4.8",
            "base_url": "",
            "api_mode": "anthropic_messages",
        }
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": False}},
        ):
            out = _maybe_apply_moa_cache_control(messages, runtime)
        assert not _has_cache_control(out)


class TestPreparedAggregatorNoAgentConfigOff:
    """Prepared-aggregator facades from ``MoAChatCompletions.__new__`` have
    no ``_agent``. The planner must not raise and must honor config-off.
    """

    def test_prepared_aggregator_without_agent_honors_config_off(self):
        import copy

        from agent import moa_loop

        calls = []
        with (
            patch.object(
                moa_loop,
                "call_llm",
                side_effect=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=[]),
                        finish_reason="stop",
                    )],
                    usage=None,
                    model="fake",
                ),
            ),
            patch.object(
                moa_loop,
                "_slot_runtime",
                return_value={
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "base_url": "https://api.anthropic.com",
                    "api_mode": "anthropic_messages",
                },
            ),
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value={"prompt_caching": {"cache_ttl": "off"}},
            ),
        ):
            completions = moa_loop.MoAChatCompletions.__new__(
                moa_loop.MoAChatCompletions
            )
            # No _agent attribute — the regression surface from review.
            completions._pending_trace = None
            prepared = {
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "lookup"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{
                            "id": "lookup",
                            "function": {"name": "lookup", "arguments": "{}"},
                        }],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "lookup",
                        "content": "result",
                    },
                ],
                "guidance": None,
                "aggregator": {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                },
                "aggregator_temperature": None,
            }
            tools = [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }]
            canonical_tools = copy.deepcopy(tools)

            completions._call_prepared_aggregator(prepared, {"tools": tools})

        assert calls, "prepared aggregator must still call the LLM"
        assert not _has_cache_control(calls[0].get("tools") or []), (
            "config cache_ttl=off with no live _agent must not inject "
            "cache_control (planner config fallback, not forced False)."
        )
        assert not _has_cache_control(calls[0].get("messages") or [])
        assert tools == canonical_tools


class TestBlankCachePolicyStubFactory:
    def test_factory_sets_cache_disabled_from_config(self):
        from agent.agent_runtime_helpers import blank_cache_policy_stub

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": "off"}},
        ):
            stub = blank_cache_policy_stub()
        assert stub._cache_disabled is True

    def test_factory_honors_explicit_false(self):
        from agent.agent_runtime_helpers import blank_cache_policy_stub

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": "off"}},
        ):
            stub = blank_cache_policy_stub(False)
        assert stub._cache_disabled is False


class TestOneShotSynthesisAgentDisable:
    """aggregate_moa_context must pin agent._cache_disabled onto decoration
    so the one-shot synthesis path cannot re-enable markers mid-session.
    """

    def test_synthesis_untouched_when_agent_disables_cache(self):
        from agent import moa_loop

        calls = []
        with (
            patch.object(
                moa_loop,
                "call_llm",
                side_effect=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(
                    choices=[SimpleNamespace(
                        message=SimpleNamespace(content="synth", tool_calls=[]),
                        finish_reason="stop",
                    )],
                    usage=None,
                    model="fake",
                ),
            ),
            patch.object(
                moa_loop,
                "_run_references_parallel",
                return_value=[("advisor-a", "advice from a", None)],
            ),
            patch.object(
                moa_loop,
                "_slot_runtime",
                return_value={
                    "provider": "anthropic",
                    "model": "claude-opus-4.8",
                    "base_url": "",
                    "api_mode": "anthropic_messages",
                },
            ),
            # Config would enable caching; agent snapshot must win.
            patch(
                "hermes_cli.config.load_config_readonly",
                return_value={"prompt_caching": {"cache_ttl": "5m"}},
            ),
        ):
            moa_loop.aggregate_moa_context(
                user_prompt="what should I do next?",
                api_messages=[{"role": "user", "content": "help me plan"}],
                reference_models=[{"provider": "openrouter", "model": "openai/gpt-5.5"}],
                aggregator={"provider": "anthropic", "model": "claude-opus-4.8"},
                agent=SimpleNamespace(_cache_disabled=True),
            )

        assert calls, "synthesis must still call the LLM"
        synth_msgs = calls[0].get("messages") or []
        assert not _has_cache_control(synth_msgs), (
            "agent._cache_disabled must keep the one-shot synthesis "
            "message undecorated even on a cache-honoring route"
        )


class TestAdvisorRuntimeDisable:
    def test_maybe_apply_honors_runtime_cache_disabled_snapshot(self):
        from agent.moa_loop import _maybe_apply_moa_cache_control

        messages = [
            {"role": "system", "content": "advisor"},
            {"role": "user", "content": "review"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "again"},
        ]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"prompt_caching": {"cache_ttl": "5m"}},
        ):
            out = _maybe_apply_moa_cache_control(
                messages,
                {
                    "provider": "anthropic",
                    "model": "claude-sonnet-4-6",
                    "base_url": "https://api.anthropic.com",
                    "api_mode": "anthropic_messages",
                    "_cache_disabled": True,
                },
            )
        assert not _has_cache_control(out)
        # Inputs must not be mutated.
        assert not _has_cache_control(messages)
