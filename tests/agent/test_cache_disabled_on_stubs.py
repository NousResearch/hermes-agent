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
