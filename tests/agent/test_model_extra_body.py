"""Top-level ``model.extra_body`` must reach the request for built-in providers.

Regression for GitHub #336: a user setting ``model.extra_body`` (e.g.
``frequency_penalty``/``presence_penalty``) in config.yaml had it silently
dropped for first-class providers like DeepSeek — only ``custom_providers``
entries carried an ``extra_body`` through. ``_merge_model_extra_body`` delivers
it via ``request_overrides['extra_body']``, the same channel the
chat-completions transport merges last (so it wins over a provider profile's
own keys).
"""

from types import SimpleNamespace

from agent.agent_init import (
    _merge_custom_provider_extra_body,
    _merge_model_extra_body,
)


def test_model_extra_body_merges_into_request_overrides():
    # Built-in provider (DeepSeek), no custom provider involved.
    agent = SimpleNamespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        request_overrides={},
    )

    _merge_model_extra_body(agent, {"extra_body": {"frequency_penalty": 0.15}})

    assert agent.request_overrides == {"extra_body": {"frequency_penalty": 0.15}}


def test_model_extra_body_preserves_caller_override():
    agent = SimpleNamespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        request_overrides={"extra_body": {"frequency_penalty": 0.5}},
    )

    _merge_model_extra_body(
        agent,
        {"extra_body": {"frequency_penalty": 0.15, "presence_penalty": 0.2}},
    )

    # Caller wins on conflict; new sibling keys still merge in.
    assert agent.request_overrides["extra_body"] == {
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
    }


def test_custom_provider_wins_over_model_extra_body():
    # Run the two merges in the same order init_agent does: custom first, then
    # model.extra_body. Precedence must be caller > custom_providers > model.
    agent = SimpleNamespace(
        provider="custom",
        model="google/gemma-4-31b-it",
        base_url="https://example.test/v1",
        request_overrides={},
    )

    _merge_custom_provider_extra_body(
        agent,
        [
            {
                "name": "gemma",
                "base_url": "https://example.test/v1",
                "model": "google/gemma-4-31b-it",
                "extra_body": {"frequency_penalty": 0.9},
            }
        ],
    )
    _merge_model_extra_body(
        agent,
        {"extra_body": {"frequency_penalty": 0.15, "top_p": 0.8}},
    )

    assert agent.request_overrides["extra_body"] == {
        "frequency_penalty": 0.9,  # custom_providers wins
        "top_p": 0.8,              # model-only key merges in
    }


def test_model_extra_body_noop_on_missing_or_invalid():
    base = {"service_tier": "priority"}
    for cfg in ({}, {"extra_body": {}}, {"extra_body": None}, "not-a-dict", None):
        agent = SimpleNamespace(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com/v1",
            request_overrides=dict(base),
        )
        _merge_model_extra_body(agent, cfg)
        assert agent.request_overrides == base


def test_model_extra_body_does_not_alias_config():
    # Mutating request_overrides must not write back into the loaded config dict.
    model_cfg = {"extra_body": {"frequency_penalty": 0.15}}
    agent = SimpleNamespace(
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com/v1",
        request_overrides={},
    )

    _merge_model_extra_body(agent, model_cfg)
    agent.request_overrides["extra_body"]["frequency_penalty"] = 0.99

    assert model_cfg["extra_body"]["frequency_penalty"] == 0.15
