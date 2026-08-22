"""Live keyless model catalog from a llama-swap endpoint.

The llamacpp setup is a custom_providers entry with NO declared models
list and NO api key: the picker must discover the catalog live from the
endpoint's /v1/models (llama-swap serves its full configured catalog
there, keyless), and a refresh must track swap-config adds/removes.

Two properties guard that flow:

1. The /models probe itself is keyless - no Authorization header is sent
   when no api key is configured.
2. Keyless discovery serves the live catalog on the picker row, not the
   (absent) config-declared subset.

Self-pin protection - a persisted catalog reading back as a user
allowlist and suppressing every future re-probe - is upstream's
``models_discovered`` flag and is covered by
``test_auto_saved_catalog_round_trips_without_pinning`` in
``tests/hermes_cli/test_model_switch_custom_providers.py``.
"""

import json
import urllib.request

import hermes_cli.providers as providers_mod
from hermes_cli.model_switch import list_authenticated_providers
from hermes_cli.models import fetch_api_models


class _FakeResp:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_keyless_models_probe_sends_no_auth_header(monkeypatch):
    seen: list[urllib.request.Request] = []

    def _fake_urlopen(req, timeout=None):
        seen.append(req)
        return _FakeResp({"data": [{"id": "qwen38-27b-mtp-q8"},
                                   {"id": "gemma-4-e4b-q4"}]})

    monkeypatch.setattr(
        "hermes_cli.models._urlopen_model_catalog_request", _fake_urlopen
    )

    models = fetch_api_models("", "http://192.0.2.21:8080/v1")

    assert models == ["qwen38-27b-mtp-q8", "gemma-4-e4b-q4"]
    assert seen, "no request issued"
    for req in seen:
        assert not req.has_header("Authorization")


def test_keyless_discovery_serves_live_catalog(monkeypatch):
    catalog = ["qwen38-27b-mtp-q8", "gemma-4-e4b-q4"]
    monkeypatch.setattr("agent.models_dev.fetch_models_dev", lambda: {})
    monkeypatch.setattr(providers_mod, "HERMES_OVERLAYS", {})
    monkeypatch.setattr(
        "hermes_cli.models.cached_fetch_api_models",
        lambda api_key, api_url, **kw: list(catalog),
    )
    monkeypatch.setattr(
        "hermes_cli.model_switch._save_discovered_models_to_config",
        lambda api_url, model_ids, **kw: None,
    )
    rows = list_authenticated_providers(
        current_provider="llamacpp",
        current_model="qwen38-27b-mtp-q8",
        user_providers={},
        custom_providers=[
            {"name": "llamacpp", "base_url": "http://192.0.2.21:8099/v1"}
        ],
        probe_custom_providers=True,
    )

    row = next(r for r in rows if r["slug"] == "custom:llamacpp")
    assert row["models"] == catalog
