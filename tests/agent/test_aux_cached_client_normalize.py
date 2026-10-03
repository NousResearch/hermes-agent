"""``_get_cached_client`` must return the caller's model normalized for the target provider.

``resolve_provider_client`` normalizes the model it resolves, but the cached-client layer returned
the caller's raw override (caller-wins) on both the build and the cache-hit path, so a dotted
Anthropic id reached the wire un-dashed and 404'd.
"""
from unittest.mock import patch

import pytest

import agent.auxiliary_client as ac


class _Client:
    base_url = "https://api.anthropic.com"


def _fake_resolve(provider, model, *args, **kwargs):
    return _Client(), ac._normalize_resolved_model(model, ac._normalize_aux_provider(provider))


@pytest.fixture(autouse=True)
def _clean_cache():
    ac._client_cache.clear()
    yield
    ac._client_cache.clear()


@pytest.mark.parametrize("provider", ["anthropic", "claude"])
def test_caller_model_is_normalized_on_build_and_cache_hit(provider):
    with patch.object(ac, "resolve_provider_client", _fake_resolve):
        _, built = ac._get_cached_client(provider, "claude-haiku-4.5")
        _, hit = ac._get_cached_client(provider, "claude-haiku-4.5")
    # The resolver's own normalization of the same id, so the two layers cannot disagree.
    expected = ac._normalize_resolved_model("claude-haiku-4.5", "anthropic")
    assert expected == "claude-haiku-4-5"  # non-vacuity: the normalizer really rewrites this id
    assert built == expected
    assert hit == expected


def test_unrecognized_provider_keeps_the_raw_caller_model():
    with patch.object(ac, "resolve_provider_client", _fake_resolve):
        _, model = ac._get_cached_client("custom", "my-model.v2")
    assert model == "my-model.v2"
