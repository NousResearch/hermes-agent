"""Built-in / first-class provider ``providers.<name>.extra_body`` resolution.

Addresses hermes-sweeper review on #21554:
- Resolve once at agent setup into ``request_overrides`` (not per transport call)
- Canonicalize aliases (``dashscope`` → ``alibaba``) so either config key works
- Preserve named custom-endpoint behavior (URL-bearing providers entries)
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.agent_init import (
    _builtin_provider_extra_body_for_agent,
    _merge_custom_provider_extra_body,
    _provider_lookup_keys,
)


def test_provider_lookup_keys_prefer_session_then_canonical_then_aliases():
    keys = _provider_lookup_keys("dashscope")
    assert keys[0] == "dashscope"
    assert "alibaba" in keys
    # Other documented aliases should be present after canonical
    assert "alibaba-cloud" in keys or "qwen-dashscope" in keys


def test_provider_lookup_keys_skip_custom():
    assert _provider_lookup_keys("custom") == []
    assert _provider_lookup_keys("custom:foo") == []


def test_builtin_extra_body_via_alias_key():
    """Config under providers.dashscope applies when session provider is alibaba."""
    got = _builtin_provider_extra_body_for_agent(
        provider="alibaba",
        providers_cfg={
            "dashscope": {"extra_body": {"enable_thinking": False}},
        },
    )
    assert got == {"enable_thinking": False}


def test_builtin_extra_body_via_canonical_key_when_session_is_alias():
    got = _builtin_provider_extra_body_for_agent(
        provider="dashscope",
        providers_cfg={
            "alibaba": {"extra_body": {"enable_thinking": False}},
        },
    )
    assert got == {"enable_thinking": False}


def test_exact_session_key_wins_over_canonical():
    """If both alias and canonical keys exist, session string match wins."""
    got = _builtin_provider_extra_body_for_agent(
        provider="dashscope",
        providers_cfg={
            "dashscope": {"extra_body": {"source": "alias"}},
            "alibaba": {"extra_body": {"source": "canonical"}},
        },
    )
    assert got == {"source": "alias"}


def test_url_bearing_providers_entry_skipped_by_builtin_path():
    """Named custom endpoints keep the custom path; builtin must not steal them."""
    got = _builtin_provider_extra_body_for_agent(
        provider="alibaba",
        providers_cfg={
            "dashscope": {
                "api": "https://example.test/v1",
                "extra_body": {"enable_thinking": False},
            },
        },
    )
    assert got is None


def test_merge_builtin_into_request_overrides():
    agent = SimpleNamespace(
        provider="alibaba",
        model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(
        agent,
        custom_providers=[],
        agent_cfg={
            "providers": {
                "dashscope": {"extra_body": {"enable_thinking": False}},
            }
        },
    )
    assert agent.request_overrides == {
        "extra_body": {"enable_thinking": False},
    }


def test_merge_caller_extra_body_wins_over_builtin():
    agent = SimpleNamespace(
        provider="alibaba",
        model="qwen-plus",
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        request_overrides={
            "extra_body": {"enable_thinking": True, "caller_only": 1},
        },
    )
    _merge_custom_provider_extra_body(
        agent,
        custom_providers=[],
        agent_cfg={
            "providers": {
                "alibaba": {
                    "extra_body": {
                        "enable_thinking": False,
                        "from_config": True,
                    }
                }
            }
        },
    )
    assert agent.request_overrides["extra_body"] == {
        "enable_thinking": True,  # caller wins
        "from_config": True,
        "caller_only": 1,
    }


def test_custom_endpoint_path_still_preferred_over_builtin():
    """When provider is custom, builtin lookup must not run."""
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={},
    )
    _merge_custom_provider_extra_body(
        agent,
        custom_providers=[
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {"from_custom": True},
            }
        ],
        agent_cfg={
            "providers": {
                # Would match if wrongly applied to custom sessions
                "alibaba": {"extra_body": {"from_builtin": True}},
            }
        },
    )
    assert agent.request_overrides == {"extra_body": {"from_custom": True}}
