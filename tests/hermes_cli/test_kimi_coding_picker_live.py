"""kimi-coding /model picker merges the live coding-plan catalog with the curated list.

The kimi-coding endpoint (``GET {base}/models`` on the coding plan API) is the
authoritative catalog for the subscription surface: it serves the K3 generation
(``k3.1-pro``, ``k3.1-pro-highspeed``, ``k3``, ``k3-256k``) that the curated
static list (a K2-era snapshot) does not contain. Every real picker surface
resolves through ``provider_model_ids`` -> the provider's catalog fetcher, so
the merge must happen there: live rows lead, curated-only additions follow,
and when the fetch declines (no credentials / outage) the caller falls through
to the existing curated static path — never an empty picker.
"""
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from hermes_cli.models import provider_model_ids

KIMI_CREDS = {
    "provider": "kimi-coding",
    "api_key": "***",
    "base_url": "https://api.kimi.com/coding",
}

# Verified against the coding-plan endpoint (2026-11-21): all six ids, in order.
KIMI_CODING_LIVE_IDS = [
    "kimi-for-coding", "kimi-for-coding-highspeed",
    "k3", "k3-256k", "k3.1-pro", "k3.1-pro-highspeed",
]


@pytest.fixture()
def hermetic_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so the picker cannot read a real disk cache row."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "kimi-coding-home"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    yield home
    reset_hermes_home_override(token)


@contextmanager
def kimi_coding_live_listing(ids):
    """Patch the credential resolver and the /v1/models probe."""
    with patch(
        "hermes_cli.auth.resolve_api_key_provider_credentials",
        return_value=dict(KIMI_CREDS),
    ), patch("hermes_cli.models.fetch_api_models", return_value=list(ids)):
        yield


class TestKimiCodingPickerMergesLiveWithCurated:
    def test_live_response_surfaces_k3_generation_models(self, hermetic_home):
        """The reported repro: subscription endpoint up, k3.1-pro missing from /model."""
        with kimi_coding_live_listing(KIMI_CODING_LIVE_IDS):
            ids = provider_model_ids("kimi-coding")
        for required in ("k3.1-pro", "k3.1-pro-highspeed", "k3", "k3-256k"):
            assert required in ids, f"{required} missing although the live coding-plan catalog serves it"
        # Live rows lead, so the current generation tops the picker.
        assert ids[:2] == ["kimi-for-coding", "kimi-for-coding-highspeed"]

    def test_merge_keeps_curated_floor_for_live_omissions(self, hermetic_home):
        """Live /models omits the legacy K2 ids; curated additions must not disappear."""
        with kimi_coding_live_listing(["k3.1-pro"]):
            ids = provider_model_ids("kimi-coding")
        assert "k3.1-pro" in ids
        for legacy in ("kimi-k2.6", "kimi-for-coding", "kimi-k2.7-code"):
            assert legacy in ids, f"curated-only model {legacy} dropped when live fetch succeeded"

    def test_fetch_decline_falls_through_to_curated_static(self, hermetic_home):
        """No credentials / outage: the curated list serves as before (never an empty picker)."""
        with patch(
            "hermes_cli.auth.resolve_api_key_provider_credentials",
            return_value={"api_key": "", "base_url": ""},
        ), patch("hermes_cli.models.fetch_api_models") as fetch:
            ids = provider_model_ids("kimi-coding")
        fetch.assert_not_called()
        assert "kimi-for-coding" in ids
        assert "kimi-k2.6" in ids
