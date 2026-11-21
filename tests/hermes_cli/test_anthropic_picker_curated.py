"""The Anthropic model picker lists the account's live catalog, in Anthropic's order.

Bug: the picker hid the account's own models.
    ``_anthropic_catalog`` put the whole curated ``_PROVIDER_MODELS["anthropic"]`` block first and
    appended live-only models after it. ``_fetch_anthropic_models`` also re-sorted the live list
    into opus, sonnet, haiku tiers, so non-tier families such as mythos and fable came last.
    Telegram pages the picker 8 rows at a time. On a Claude login, the newest entitled models
    (``claude-sonnet-5-5``, ``claude-mythos-5-2``) were pushed to pages 2-3, behind curated ids the
    account's listing does not contain.

    ``/v1/models`` is the per-credential entitlement list, ordered newest first, so the picker now
    shows it verbatim. The curated list is kept only as the fallback for when the live fetch fails.
    A typed alias that the listing omits is still soft-accepted by ``_validate_anthropic``.
"""

from unittest.mock import patch

from hermes_cli import models as M


def test_picker_lists_exactly_the_live_catalog_in_api_order():
    live = ["claude-mythos-5-2", "claude-sonnet-5-5", "claude-opus-4-8", "claude-future-9-99"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert result == live


def test_curated_ids_missing_from_the_live_catalog_are_not_listed():
    live = ["claude-sonnet-5-5"]
    with patch.object(M, "_fetch_anthropic_models", return_value=live):
        result = M.provider_model_ids("anthropic")

    assert result == live
    assert not set(M._PROVIDER_MODELS["anthropic"]) - set(live) & set(result)


def test_anthropic_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim, flagged as a placeholder."""
    with patch.object(M, "_fetch_anthropic_models", return_value=None):
        result = M.provider_model_ids("anthropic")

    assert result == list(M._PROVIDER_MODELS["anthropic"])
    assert isinstance(M._anthropic_catalog("anthropic", False), M.CuratedFallbackModels)


def test_live_fetch_keeps_anthropics_newest_first_order():
    page = {
        "data": [{"id": "claude-mythos-5-2"}, {"id": "claude-haiku-5-5"},
                 {"id": "claude-sonnet-5-5"}, {"id": "claude-opus-5-5"}, {"id": "claude-mythos-5-2"}],
        "has_more": False,
    }
    with patch.object(M, "_get_json", return_value=page):
        got = M._fetch_anthropic_models(api_key="sk-ant-api03-test")

    assert got == ["claude-mythos-5-2", "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]


class TestColdPickerReadRefreshesInBackground:
    """A non-blocking read of a cold or credential-mismatched Anthropic row never waits on the network
    (#114215): it serves what it has and a background refresh warms the next read."""

    @staticmethod
    def _read(cache, fp="fp-now", live=("claude-mythos-5-2", "claude-sonnet-5-5"), provider="anthropic"):
        with patch.object(M, "_load_provider_models_cache", return_value=cache), \
             patch.object(M, "_credential_fingerprint", return_value=fp), \
             patch.object(M, "_spawn_swr_refresh") as spawn, \
             patch.object(M, "_store_cache_entry") as store, \
             patch.object(M, "provider_model_ids", return_value=list(live)) as fetch:
            out = M.cached_provider_model_ids(provider, non_blocking=True)
        return out, fetch, store, spawn

    def test_empty_cache_refreshes_in_background(self):
        out, fetch, store, spawn = self._read({})

        assert out == []
        fetch.assert_not_called()
        store.assert_not_called()
        spawn.assert_called_once_with("anthropic")

    def test_refreshed_credentials_refresh_in_background(self):
        cache = {"anthropic": {"fp": "fp-before-refresh", "at": M.time.time() - 30, "models": ["claude-opus-4-8"]}}

        out, fetch, _store, spawn = self._read(cache)

        assert out == []
        fetch.assert_not_called()
        spawn.assert_called_once_with("anthropic")

    def test_expired_curated_fallback_row_refreshes_in_background(self):
        cache = {"anthropic": {"fp": "fp-now", "at": M.time.time() - M._PROVIDER_MODELS_FALLBACK_TTL - 5,
                               "models": list(M._PROVIDER_MODELS["anthropic"]), "fallback": True}}

        out, fetch, _store, spawn = self._read(cache)

        assert out == list(M._PROVIDER_MODELS["anthropic"])
        fetch.assert_not_called()
        spawn.assert_called_once_with("anthropic")

    def test_same_credentials_row_is_served_without_waiting(self):
        cache = {"anthropic": {"fp": "fp-now", "at": M.time.time() - 2 * M._PROVIDER_MODELS_CACHE_TTL,
                               "models": ["claude-mythos-5-2"]}}

        out, fetch, _store, spawn = self._read(cache)

        assert out == ["claude-mythos-5-2"]
        fetch.assert_not_called()
        spawn.assert_called_once_with("anthropic")

    def test_other_providers_keep_the_never_block_contract(self):
        out, fetch, _store, spawn = self._read({}, provider="openrouter", live=("m1",))

        assert out == []
        fetch.assert_not_called()
        spawn.assert_called_once_with("openrouter")
