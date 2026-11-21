"""MiniMax live model discovery probes the vendor's real catalog endpoint (#152406).

The generic ``{base_url}/models`` probe resolves to ``api.minimax.io/anthropic/models`` — a 404 on
both vendor hosts (verified: the path does not exist) — so live discovery silently returned nothing
and every new MiniMax release stayed invisible in the picker until hand-added to the static list.
The profiles pin ``models_url`` at the endpoint the vendor actually serves
(``/anthropic/v1/models`` — 401 unauthenticated, 200 with a real key, per the report's live probes).
"""

import json

import pytest

from providers import get_provider_profile

_VENDOR_CATALOG = {
    "data": [
        {"id": "MiniMax-M3.1-Flash"}, {"id": "MiniMax-M3"}, {"id": "MiniMax-M2.7"},
        {"id": "MiniMax-M2.7-highspeed"}, {"id": "MiniMax-M2.5"}, {"id": "MiniMax-M2.5-highspeed"},
        {"id": "MiniMax-M2.1"}, {"id": "MiniMax-M2.1-highspeed"}, {"id": "MiniMax-M2"},
    ]
}


@pytest.mark.parametrize(("provider", "expected_url"), [
    ("minimax", "https://api.minimax.io/anthropic/v1/models"),
    ("minimax-cn", "https://api.minimaxi.com/anthropic/v1/models"),
])
def test_live_catalog_probes_the_vendor_v1_endpoint(monkeypatch, provider, expected_url):
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps(_VENDOR_CATALOG).encode()

    def _open(req, timeout=0):
        seen["url"] = req.full_url
        return _Response()

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _open)

    models = get_provider_profile(provider).fetch_models(api_key="probe-key", timeout=1.5)

    assert seen["url"] == expected_url, \
        "the probe must hit the vendor catalog, not the 404ing {base}/models fallback"
    assert models[:1] == ["MiniMax-M3.1-Flash"]


def test_custom_base_still_probes_the_custom_endpoint(monkeypatch):
    """A user-configured proxy base keeps today's behavior: probe {custom_base}/models, never the
    vendor URL (an absolute models_url must not shadow a caller's explicit custom base)."""
    seen = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return json.dumps({"data": [{"id": "proxy/model"}]}).encode()

    def _open(req, timeout=0):
        seen["url"] = req.full_url
        return _Response()

    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", _open)

    profile = get_provider_profile("minimax")
    models = profile.fetch_models(base_url="https://proxy.internal/anthropic", api_key="k", timeout=1.5)

    assert seen["url"] == "https://proxy.internal/anthropic/models"
    assert models == ["proxy/model"]
