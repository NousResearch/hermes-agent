"""The MiniMax model picker must read the vendor's live catalog.

MiniMax publishes its catalog under ``<base>/v1/models``; the generic
``{base_url}/models`` probe answers 404. The profile therefore never returned a
catalog, ``merge_profile_catalog`` degraded every picker surface to the curated
floor, and a model MiniMax had shipped stayed invisible — with no failure
reported anywhere (#142648).

Every real picker surface resolves through ``provider_model_ids`` ->
``_profile_live_catalog`` -> ``ProviderProfile.fetch_models``, so the live rows
and the curated floor must meet there.
"""

import json
import urllib.error
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from hermes_cli.models import provider_model_ids

MINIMAX_CREDS = {
    "provider": "minimax",
    "api_key": "sk-minimax-test",
    "base_url": "https://api.minimax.io/anthropic",
}

# The vendor's live catalog (verified against https://api.minimax.io/v1/models): it lists
# the -highspeed variants the curated floor does not, and omits some models the floor keeps.
LIVE_MODEL_IDS = [
    "MiniMax-M3.1-Flash", "MiniMax-M3",
    "MiniMax-M2.7", "MiniMax-M2.7-highspeed",
    "MiniMax-M2.5", "MiniMax-M2.5-highspeed",
    "MiniMax-M2.1", "MiniMax-M2.1-highspeed",
    "MiniMax-M2",
]


class _FakeCatalogResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def hermetic_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME so the picker cannot read a real config or cache row."""
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = tmp_path / "minimax-home"
    home.mkdir()
    token = set_hermes_home_override(str(home))
    yield home
    reset_hermes_home_override(token)


@contextmanager
def minimax_catalog(probed, model_ids=None):
    """Serve the vendor catalog; record every URL the profile probes."""
    rows = LIVE_MODEL_IDS if model_ids is None else list(model_ids)

    def fake_open(request, **_kwargs):
        url = request.full_url
        probed.append(url)
        if url.endswith("/v1/models"):
            return _FakeCatalogResponse({"data": [{"id": mid} for mid in rows]})
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)

    with patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value=dict(MINIMAX_CREDS)), \
            patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open):
        yield


class TestMinimaxPickerReadsTheLiveCatalog:
    def test_catalog_is_probed_on_the_v1_route(self, hermetic_home):
        """The reported repro: the probe URL 404s, so the live catalog is never read."""
        probed: list[str] = []
        with minimax_catalog(probed):
            provider_model_ids("minimax")
        assert probed == ["https://api.minimax.io/anthropic/v1/models"], (
            f"minimax catalog probed the wrong URL: {probed}"
        )

    def test_every_live_model_is_offered(self, hermetic_home):
        """The vendor's catalog is wider than the curated floor — it must reach the picker."""
        with minimax_catalog([]):
            ids = provider_model_ids("minimax")

        for model_id in LIVE_MODEL_IDS:
            assert model_id in ids, f"minimax picker dropped the live catalog model {model_id}"

    def test_a_partial_live_catalog_keeps_the_curated_floor(self, hermetic_home):
        """A live catalog that omits curated models must not shrink the picker."""
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        partial = ["MiniMax-M3.1-Flash", "MiniMax-M3"]
        with minimax_catalog([], model_ids=partial):
            ids = provider_model_ids("minimax")

        for model_id in partial:
            assert model_id in ids, f"minimax picker dropped the live catalog model {model_id}"
        curated_only = [m for m in _PROVIDER_MODELS["minimax"] if m not in partial]
        assert curated_only, "the curated floor is empty — the merge contract has nothing to prove"
        for model_id in curated_only:
            assert model_id in ids, f"minimax picker dropped the curated model {model_id}"
        assert len(ids) == len({m.lower() for m in ids}), "duplicate rows in the merged picker"

    def test_unreachable_catalog_still_serves_the_curated_floor(self, hermetic_home):
        """An outage degrades to the curated list, not to an empty picker."""
        from hermes_cli.models_catalog_static import _PROVIDER_MODELS

        def fake_open(request, **_kwargs):
            raise urllib.error.URLError("unreachable")

        with patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value=dict(MINIMAX_CREDS)), \
                patch("hermes_cli.urllib_security.open_credentialed_url", side_effect=fake_open):
            ids = provider_model_ids("minimax")

        assert list(ids) == list(_PROVIDER_MODELS["minimax"])
