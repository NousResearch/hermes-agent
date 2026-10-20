"""Regression tests for the MiniMax model-picker merging live and curated catalogs.

Bug — the ``/model`` picker for the ``minimax-oauth`` (and sibling) provider
    served only the static ``_PROVIDER_MODELS["minimax-oauth"]`` curated
    list, never probing the live Anthropic-compatible ``/v1/models`` endpoint.
    The endpoint's listing often lags freshly-shipped SKUs (a model can be
    invocable for weeks before the listing updates), so the picker missed
    every new rollout until a Hermes release added it to the static table.
    The picker now probes the live endpoint and merges its result with the
    curated table — curated entries first, live-only models appended,
    deduped — mirroring the Anthropic picker pattern (#121387 follow-up,
    pattern established by ``test_anthropic_picker_curated.py``).
"""

from unittest.mock import patch

from hermes_cli import models as M


def test_minimax_oauth_merge_dedupes_overlap_and_appends_live_only():
    """Models in both lists appear once; live-only models are appended."""
    live = [
        "MiniMax-M2",          # overlaps curated
        "MiniMax-M2.7",        # overlaps curated
        "MiniMax-M9-Future",   # live-only, not curated
    ]
    with patch.object(M, "_fetch_minimax_models", return_value=live):
        result = M.provider_model_ids("minimax-oauth")

    # No duplicates introduced by the merge.
    assert result.count("MiniMax-M2") == 1
    # Live-only entry is preserved (discovery still works for unknown models).
    assert "MiniMax-M9-Future" in result
    # Curated entries lead, live-only trails.
    curated = list(M._PROVIDER_MODELS["minimax-oauth"])
    assert result[: len(curated)] == curated and result[-1] == "MiniMax-M9-Future"


def test_minimax_oauth_falls_back_to_curated_when_live_unavailable():
    """No creds / live failure -> curated list verbatim (alias still present)."""
    with patch.object(M, "_fetch_minimax_models", return_value=None):
        result = M.provider_model_ids("minimax-oauth")

    assert result == list(M._PROVIDER_MODELS["minimax-oauth"])


def test_minimax_oauth_registered_in_fetcher_table():
    """The MiniMax provider family must be wired into the catalog fetcher
    registry so ``provider_model_ids`` actually calls the live fetcher.
    Without this registration, the picker degrades to the static curated
    list and the live discovery is dead code."""
    for name in ("minimax-oauth", "minimax", "minimax-cn"):
        assert M._PROVIDER_CATALOG_FETCHERS.get(name) is M._minimax_oauth_catalog


def test_minimax_oauth_registered_in_relay_aware_set():
    """``_RELAY_AWARE_CATALOG_FETCHERS`` must include the MiniMax provider
    family — otherwise the "configured base_url is terminal" interception
    in ``provider_model_ids`` short-circuits to the static catalog and
    never reaches the live fetcher (#121387)."""
    for name in ("minimax-oauth", "minimax", "minimax-cn"):
        assert name in M._RELAY_AWARE_CATALOG_FETCHERS


def test_minimax_oauth_relay_serves_live_only_no_curated_merge():
    """A user-installed relay (``base_url`` != vendor canonical) must NOT
    be merged with the vendor curated list — the relay user sees only
    the relay's catalog (#121387: "configured base_url is terminal for
    live catalog egress")."""
    fake_relay_models = ["relay-only-model-a", "relay-only-model-b"]
    with patch.object(M, "_configured_relay_base_url", return_value="https://my-relay.example.com/anthropic"), \
         patch.object(M, "_fetch_minimax_models", return_value=fake_relay_models) as fetch_mock, \
         patch.object(M, "_merge_unique") as merge_mock:
        result = M._minimax_oauth_catalog("minimax-oauth", force_refresh=True)

    # Relay catalog returned verbatim, with no curated merge.
    assert result == fake_relay_models
    # And the fetcher was called against the relay URL, not the vendor default.
    assert fetch_mock.call_args.kwargs["base_url"] == "https://my-relay.example.com/anthropic"
    # Defense-in-depth: the curated merge path was not exercised on a relay.
    merge_mock.assert_not_called()


def test_minimax_oauth_vendor_host_merges_curated_with_live():
    """When the configured base_url is the vendor canonical (or absent),
    the picker merges live with curated — curated first, live-only
    appended — so freshly-shipped aliases (reachable but not yet
    enumerated in /v1/models) remain visible."""
    curated = list(M._PROVIDER_MODELS["minimax-oauth"])
    live = [curated[0], "MiniMax-NEW-LIVE-SKU"]
    with patch.object(M, "_configured_relay_base_url", return_value=""), \
         patch.object(M, "_fetch_minimax_models", return_value=live):
        result = M._minimax_oauth_catalog("minimax-oauth", force_refresh=True)

    assert result[: len(curated)] == curated
    assert "MiniMax-NEW-LIVE-SKU" in result
    # No duplicates where live and curated overlap.
    assert result.count(curated[0]) == 1
