"""Pure OpenCode provider identity and routing projections."""

import pytest

from providers import (
    normalize_opencode_base_url,
    normalize_opencode_model_id,
    opencode_provider_family,
)


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("opencode-go", "opencode-go"), ("custom:opencode-go-bridge", "opencode-go"),
     ("OPENCODE-ZEN", "opencode-zen"), ("openrouter", None), ("", None)],
)
def test_opencode_provider_family(provider, expected):
    assert opencode_provider_family(provider) == expected


@pytest.mark.parametrize(
    ("provider", "model", "expected"),
    [("opencode-go", "opencode-go/kimi-k2", "kimi-k2"),
     ("custom:opencode-go-bridge", "opencode-go/kimi-k2", "kimi-k2"),
     ("opencode-zen", "opencode-zen/big-pickle", "big-pickle"),
     ("openrouter", "openrouter/foo", "openrouter/foo")],
)
def test_normalize_opencode_model_id(provider, model, expected):
    assert normalize_opencode_model_id(provider, model) == expected


def test_normalize_opencode_base_url_delegates_shared_policy():
    assert normalize_opencode_base_url(
        "opencode-go", "chat_completions", "https://opencode.ai/zen/v1"
    ) == "https://opencode.ai/zen/go/v1"
