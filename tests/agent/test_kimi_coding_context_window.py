"""Kimi coding-plan context windows (#126224): ``kimi-for-coding`` is 1 Mi — the ``kimi`` family
catch-all's 256K is a stale underreport for it — while the ``-highspeed`` SKU is genuinely 256K.
Covers the endpoint-scoped window, the cold-catalog fallback, the persisted-cache repair, and the
renamed models.dev provider slugs the catalog lookup goes through."""

from agent import models_dev
from agent.model_metadata import (
    DEFAULT_CONTEXT_LENGTHS,
    _validate_cached_context_length,
    get_model_context_length,
)

CANONICAL_ENDPOINT = "https://api.kimi.com/coding/v1"


def test_kimi_for_coding_resolves_to_1m_while_highspeed_stays_256k():
    """The reported repro: the coding-plan default model must resolve to its real 1 Mi window on
    the canonical endpoint, and the genuinely-smaller highspeed SKU must not inherit it."""
    assert get_model_context_length(
        "kimi-for-coding", base_url=CANONICAL_ENDPOINT, provider="kimi-coding",
    ) == 1_048_576
    assert get_model_context_length(
        "kimi-for-coding-highspeed", base_url=CANONICAL_ENDPOINT, provider="kimi-coding",
    ) == 262_144


def test_kimi_for_coding_window_survives_a_cold_catalog():
    """With no endpoint metadata and no reachable catalog (offline/cold cache), the static table
    must still report the real window instead of the family catch-all."""
    assert get_model_context_length("kimi-for-coding", provider="kimi-coding") == 1_048_576
    assert DEFAULT_CONTEXT_LENGTHS["kimi-for-coding"] > DEFAULT_CONTEXT_LENGTHS["kimi"]


def test_stale_persisted_256k_window_is_dropped_for_reresolution():
    """A 256K window persisted by pre-fix builds must not pin the model forever at step 1."""
    assert _validate_cached_context_length("kimi-for-coding", CANONICAL_ENDPOINT, 262_144) is None


def test_catalog_lookup_goes_through_the_renamed_provider_slugs(monkeypatch):
    """models.dev renamed the coding-plan providers (kimi-code-plan-global/-cn); the Hermes
    provider ids must reach the new slugs or catalog metadata silently misses."""
    registry = {
        "kimi-code-plan-global": {
            "kimi-for-coding": {"limit": {"context": 1_048_576}},
            "kimi-for-coding-highspeed": {"limit": {"context": 262_144}},
        },
        "kimi-code-plan-cn": {"kimi-for-coding": {"limit": {"context": 1_048_576}}},
        "moonshotai": {"kimi-k3": {"limit": {"context": 1_048_576}}},
    }
    monkeypatch.setattr(models_dev, "_load_model_overrides", lambda: {})
    monkeypatch.setattr(models_dev, "_registry_models", lambda mdev_id, **k: registry.get(mdev_id))
    assert models_dev.lookup_models_dev_context("kimi-coding", "kimi-for-coding") == 1_048_576
    assert models_dev.lookup_models_dev_context("kimi-coding-cn", "kimi-for-coding") == 1_048_576
    assert models_dev.lookup_models_dev_context("moonshot", "kimi-k3") == 1_048_576
    # The highspeed SKU keeps its own (smaller) catalog window.
    assert models_dev.lookup_models_dev_context("kimi-coding", "kimi-for-coding-highspeed") == 262_144
