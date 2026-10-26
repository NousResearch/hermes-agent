"""Sakana Fugu context window: 1,000,000, not the 256K probe-down fallback.

``sakana/fugu-ultra`` was curated into the model picker by merged #56617
(hermes_cli/models_catalog_static.py:42) but never got a catalog entry, so it resolved through
step 9 of ``get_model_context_length`` — ``DEFAULT_FALLBACK_CONTEXT`` (256,000) plus a
"Could not determine context length" warning — a ~4x under-report that also caps compression.

Sources, fetched keyless on 2026-09-13 and recorded in the PR body:
  * https://openrouter.ai/api/v1/models  -> ``data[].context_length``
  * https://inference-api.nousresearch.com/v1/models -> ``data[].context_length``
Both report 1,000,000 for sakana/fugu-ultra, sakana/fugu-ultra-v2 and sakana/fugu-max, and
262,144 for the unrelated sakana/sakana-namazu.

``provider="custom"`` keeps every assertion on the hardcoded catalog (step 8): it skips the
OpenRouter metadata fetch an unqualified call would make, so these tests need no network.
"""
import pytest

from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _PRE_CATALOG_STALE_KEYS,
    _catalog_key_matches,
    _stale_pre_catalog_cache_entry,
    get_model_context_length,
)


def _ctx(model: str) -> int:
    return get_model_context_length(model, provider="custom")


@pytest.mark.parametrize("model", [
    "sakana/fugu-ultra",      # the id curated into the picker by #56617
    "sakana/fugu-ultra-v2",
    "sakana/fugu-max",
    "fugu-ultra",             # bare slug (no publisher prefix)
])
def test_fugu_slugs_resolve_to_the_catalog_1m(model):
    assert _ctx(model) == 1_000_000
    assert _ctx(model) != DEFAULT_FALLBACK_CONTEXT


def test_fugu_key_does_not_swallow_sakana_namazu():
    """The other Sakana model is 262,144 on OpenRouter. A "sakana" prefix would have claimed it for
    1M; "fugu" must not match it, so it keeps falling through to the 256K fallback (a separate,
    pre-existing gap this PR deliberately does not touch)."""
    assert _ctx("sakana/sakana-namazu") == DEFAULT_FALLBACK_CONTEXT
    assert _ctx("sakana/sakana-namazu") != 1_000_000


def test_fugu_is_registered_as_a_pre_catalog_stale_key():
    """The block's rule: list a key whose value is strictly above every shorter matching key and the
    256K fallback, so a window persisted by a pre-entry build is re-resolved instead of pinned.

    ``fugu`` qualifies — 1,000,000 with no shorter matching key — and needs it, because every build
    before this entry cached exactly the 256,000 fallback for ``sakana/fugu-ultra``.
    """
    assert "fugu" in _PRE_CATALOG_STALE_KEYS
    assert DEFAULT_CONTEXT_LENGTHS["fugu"] > DEFAULT_FALLBACK_CONTEXT
    matching = [k for k in DEFAULT_CONTEXT_LENGTHS if _catalog_key_matches(k, "sakana/fugu-ultra")]
    assert matching == ["fugu"], f"another catalog key also matches fugu-ultra: {matching}"

    assert _stale_pre_catalog_cache_entry("sakana/fugu-ultra", DEFAULT_FALLBACK_CONTEXT) is True
    # A cache entry already at the real window is not "stale".
    assert _stale_pre_catalog_cache_entry("sakana/fugu-ultra", 1_000_000) is False
    # ...and the invalidation does not spill onto the sibling model.
    assert _stale_pre_catalog_cache_entry("sakana/sakana-namazu", DEFAULT_FALLBACK_CONTEXT) is False
