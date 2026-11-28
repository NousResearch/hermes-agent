"""Nemotron context windows: the paid ids are 262,144 on both live catalogs, and the 1M windows
belong to the ``:free`` promo slugs only — so the ``nemotron`` catch-all must not under-report the
family 2x, and ``nemotron-3.5-lightning`` must not over-report the paid id 4x (same bug class as
the GLM catalog fix in #106976).

Sources, fetched keyless on 2026-09-13 and recorded in the PR body:
  * https://openrouter.ai/api/v1/models  -> ``data[].context_length``
  * https://inference-api.nousresearch.com/v1/models -> ``data[].context_length``

``provider="custom"`` keeps every assertion on the hardcoded catalog (step 8 of
``get_model_context_length``): it skips the OpenRouter metadata fetch that an unqualified call
would make, so these tests need no network.
"""
import pytest

from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    DEFAULT_FALLBACK_CONTEXT,
    _PRE_CATALOG_STALE_KEYS,
    get_model_context_length,
)


def _ctx(model: str) -> int:
    return get_model_context_length(model, provider="custom")


@pytest.mark.parametrize("model", [
    # OpenRouter context_length == Nous Portal context_length == 262144 for all of these.
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-super-120b-a12b:free",   # the :free twin is 262144 too, not 1M
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3-nano-30b-a3b",
    "nvidia/nemotron-3.5-lightning",
    "nvidia/nemotron-3.5-lightning-30b-a3b",    # the NIM id in models_catalog_static.py:184
])
def test_paid_nemotron_ids_resolve_to_the_catalog_262k(model):
    """131,072 under-reported every paid Nemotron by 2x; 1,000,000 over-reported Lightning by ~4x."""
    assert _ctx(model) == 262144


@pytest.mark.parametrize("model", [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-lightning:free",
])
def test_free_promo_slugs_keep_the_1m_window(model):
    """Only the ``:free`` promo ids report 1,000,000 (OpenRouter); the paid twins report 262,144."""
    assert _ctx(model) == 1_000_000


def test_content_safety_stays_narrower_than_the_family():
    """OpenRouter reports 131,072 for nvidia/nemotron-3.5-content-safety — below the new 262,144
    catch-all, so it needs its own key or raising the family would start over-reporting it."""
    assert _ctx("nvidia/nemotron-3.5-content-safety") == 131072
    assert _ctx("nvidia/nemotron-3.5-content-safety:free") == 131072


def test_nano_omni_stays_narrower_than_the_family():
    """OpenRouter reports 256,000 for nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free."""
    assert _ctx("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free") == 256000
    assert _ctx("nvidia/nemotron-3-nano-omni-30b-a3b-reasoning") == 256000


@pytest.mark.parametrize("model", ["nemotron-3.5-lightning-free", "nemotron-3-ultra-free"])
def test_opencode_hyphenated_free_slugs_take_the_paid_window(model):
    """``_normalize_model_version`` maps only ``.`` -> ``-``, so a ``:free`` key can never match
    OpenCode Zen's hyphenated ``-free`` slugs (models_catalog_static.py:232, 241).

    OpenCode's ``GET /zen/v1/models`` publishes no ``context_length``, so there is nothing to pin
    those two to; they deliberately fall to the paid 262,144 — under-reporting, never over-reporting,
    which is the safe direction for a compression threshold. Before this change
    ``nemotron-3.5-lightning-free`` inherited the 1,000,000 Lightning key.
    """
    assert _ctx(model) == 262144


def test_family_catch_all_is_not_treated_as_a_pre_catalog_stale_key():
    """``_stale_pre_catalog_cache_entry`` discards a cached window <= ``max(shorter matches, 256K)``.

    ``nemotron`` at 262,144 sits only 6,144 above ``DEFAULT_FALLBACK_CONTEXT`` and has no shorter
    matching key, so listing it would also throw away a legitimate 256,000 probe result for any
    nemotron-named relay. Unlike ``fugu``/``grok-4.3``/``minimax-m3``, it does not clear that bar
    with room to spare, so it stays out.
    """
    assert DEFAULT_CONTEXT_LENGTHS["nemotron"] == 262144
    assert DEFAULT_CONTEXT_LENGTHS["nemotron"] - DEFAULT_FALLBACK_CONTEXT == 6144
    assert "nemotron" not in _PRE_CATALOG_STALE_KEYS
