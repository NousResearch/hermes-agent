"""Phase 1 RED contract for provider_routing.models.<model-id>.

These tests exercise the existing provider-preference and transport seams. They intentionally run
against the frozen Kensei baseline before the upstream per-model overlay is implemented.
"""
from types import SimpleNamespace

import pytest

from agent import chat_completion_helpers as cch
from agent.profile_runtime_scope import profile_runtime_scope
from agent.transports.chat_completions import ChatCompletionsTransport
from hermes_constants import get_hermes_home


def _agent(model, **flat):
    base = dict(
        model=model,
        provider="openrouter",
        providers_allowed=None,
        providers_ignored=None,
        providers_order=None,
        provider_sort="price",
        provider_require_parameters=False,
        provider_data_collection=None,
    )
    base.update(flat)
    return SimpleNamespace(**base)


@pytest.fixture
def routing_cfg(monkeypatch):
    cfg = {
        "provider_routing": {
            "sort": "price",
            "models": {
                "openai/gpt-6-astra": {"only": ["openai"]},
                "anthropic/claude-fable-5.1": {
                    "only": ["anthropic"],
                    "sort": "throughput",
                },
                "qwen/qwen3-coder": {
                    "ignore": ["slow-provider"],
                    "order": ["fast-provider"],
                    "require_parameters": True,
                    "data_collection": "deny",
                },
            },
        }
    }
    import hermes_cli.config as config_mod

    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    return cfg


def test_per_model_entry_overlays_flat_routing_for_current_model(routing_cfg):
    assert cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra")) == {
        "only": ["openai"],
        "sort": "price",
    }
    assert cch._provider_preferences_for_agent(_agent("anthropic/claude-fable-5.1")) == {
        "only": ["anthropic"],
        "sort": "throughput",
    }
    assert cch._provider_preferences_for_agent(_agent("moonshotai/kimi-k2.6")) == {
        "sort": "price",
    }


def test_per_model_overlay_is_partial_and_preserves_unset_flat_values(routing_cfg):
    agent = _agent(
        "qwen/qwen3-coder",
        providers_allowed=["flat-provider"],
        providers_ignored=["flat-ignore"],
        providers_order=["flat-order"],
        provider_sort="latency",
        provider_require_parameters=False,
        provider_data_collection="allow",
    )
    assert cch._provider_preferences_for_agent(agent) == {
        "only": ["flat-provider"],
        "ignore": ["slow-provider"],
        "order": ["fast-provider"],
        "sort": "latency",
        "require_parameters": True,
        "data_collection": "deny",
    }


def test_model_matching_tolerates_openrouter_prefix_and_spelling_variants(routing_cfg):
    agent = _agent("openrouter/openai/gpt-6-astra", providers_allowed=["parent-provider"])
    assert cch._provider_preferences_for_agent(agent)["only"] == ["openai"]
    agent.model = "claude-fable-5-1"
    assert cch._provider_preferences_for_agent(agent)["only"] == ["anthropic"]


def test_delegated_target_profile_routing_does_not_leak_parent_model_rules(monkeypatch):
    import hermes_cli.config as config_mod

    monkeypatch.setattr(
        config_mod,
        "load_config_readonly",
        lambda: {"provider_routing": {"models": {"shared/model": {"only": ["target"]}}}},
    )
    parent = _agent("shared/model", providers_allowed=["parent"])
    assert cch._provider_preferences_for_agent(parent)["only"] == ["target"]


def test_target_profile_scope_selects_target_config_for_same_model(monkeypatch, tmp_path):
    import hermes_cli.config as config_mod

    target_home = tmp_path / "target-profile"
    parent_home = get_hermes_home()

    def config_for_active_home():
        if get_hermes_home() == target_home:
            return {"provider_routing": {"models": {"shared/model": {"only": ["target"]}}}}
        return {"provider_routing": {"models": {"shared/model": {"only": ["parent"]}}}}

    monkeypatch.setattr(config_mod, "load_config_readonly", config_for_active_home)
    agent = _agent("shared/model", providers_allowed=["flat-parent"])
    assert cch._provider_preferences_for_agent(agent)["only"] == ["parent"]
    with profile_runtime_scope(target_home, {"TARGET_SECRET": "target"}, hydrate_secrets=False):
        assert get_hermes_home() == target_home
        assert cch._provider_preferences_for_agent(agent)["only"] == ["target"]
    assert get_hermes_home() == parent_home


def test_fallback_model_change_re_resolves_model_specific_overlay(routing_cfg):
    agent = _agent("openai/gpt-6-astra")
    assert cch._provider_preferences_for_agent(agent)["only"] == ["openai"]
    agent.model = "anthropic/claude-fable-5.1"
    assert cch._provider_preferences_for_agent(agent)["only"] == ["anthropic"]


def test_openrouter_payload_contains_provider_routing_only_for_openrouter():
    transport = ChatCompletionsTransport()
    openrouter = transport.build_kwargs(
        "openai/gpt-6-astra",
        [],
        is_openrouter=True,
        provider_preferences={"only": ["openai"]},
    )
    direct = transport.build_kwargs(
        "openai/gpt-6-astra",
        [],
        is_openrouter=False,
        provider_preferences={"only": ["openai"]},
    )
    assert openrouter["extra_body"]["provider"] == {"only": ["openai"]}
    assert "provider" not in direct.get("extra_body", {})


def test_provider_routing_stays_separate_from_cross_provider_fallback():
    prefs = cch._provider_preferences_for_agent(_agent("openai/gpt-6-astra"))
    assert "fallback_providers" not in prefs
    assert "provider" not in prefs


def test_direct_and_nous_portal_requests_do_not_receive_openrouter_preferences():
    transport = ChatCompletionsTransport()
    for provider in ("anthropic", "nous", "nous-portal", "nousresearch"):
        kwargs = transport.build_kwargs(
            "openai/gpt-6-astra",
            [],
            provider_name=provider,
            is_openrouter=False,
            provider_preferences={"only": ["openai"]},
        )
        assert "provider" not in kwargs.get("extra_body", {})
