"""Prime's endpoint owns model availability across setup and the picker."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock
from urllib.error import URLError

import pytest

from providers import get_provider_profile


@pytest.fixture
def prime_catalog(monkeypatch):
    from agent import models_dev
    from hermes_cli import models

    profile = get_provider_profile("prime")
    monkeypatch.setenv("PRIME_API_KEY", "prime-test-key")
    monkeypatch.setattr(profile, "fallback_models", ("example/curated-only",))
    monkeypatch.setattr(models, "_api_key_credentials", lambda *_: ("prime-test-key", profile.base_url))
    monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **_: {
        "prime": {"models": {"example/cached-only": {"tool_call": True}}},
    })
    return profile


def catalog_flows(profile):
    from hermes_cli import model_setup_flows as flows, models

    return (
        flows._api_key_provider_model_list(
            profile.name, SimpleNamespace(name=profile.display_name),
            "prime-test-key", "PRIME_API_KEY", profile.base_url,
        ),
        models.provider_model_ids(profile.name, force_refresh=True),
    )


@pytest.mark.parametrize("empty", (False, True))
def test_live_catalog_replaces_cached_and_curated_models(prime_catalog, monkeypatch, empty):
    modalities = {"input": ["text"], "output": ["text"]}
    rows = [
        {"id": "example/hosted", "serving": "hosted", "supported_parameters": ["tools"],
         "specs": {"modalities": modalities}},
        {"id": "example/gateway", "serving": "gateway", "supported_parameters": ["tools"],
         "specs": {"modalities": modalities}},
        {"id": "example/no-tools", "supported_parameters": [], "specs": {"modalities": modalities}},
    ]
    response = MagicMock()
    response.__enter__.return_value.read.return_value = json.dumps({"data": [] if empty else rows}).encode()
    opener = MagicMock(return_value=response)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)

    expected = [] if empty else ["example/hosted", "example/gateway"]
    setup, picker = catalog_flows(prime_catalog)
    assert setup == picker == expected
    for call in opener.call_args_list:
        assert call.args[0].full_url == prime_catalog.base_url + "/models"
        assert call.args[0].get_header("Authorization") is None
    assert opener.call_count == 2


@pytest.mark.parametrize("cached", (True, False))
@pytest.mark.parametrize("failure", ("network", "malformed"))
def test_discovery_failure_uses_cached_then_curated_models(prime_catalog, monkeypatch, cached, failure):
    from agent import models_dev
    from hermes_cli.models import CuratedFallbackModels

    if not cached:
        monkeypatch.setattr(models_dev, "fetch_models_dev", lambda **_: {})
    response = MagicMock()
    response.__enter__.return_value.read.return_value = b'{"error": "catalog unavailable"}'
    opener = MagicMock(side_effect=URLError("offline")) if failure == "network" else MagicMock(return_value=response)
    monkeypatch.setattr("hermes_cli.urllib_security.open_credentialed_url", opener)

    expected = ["example/cached-only"] if cached else list(prime_catalog.fallback_models)
    setup, picker = catalog_flows(prime_catalog)
    assert setup == picker == expected
    # Outage placeholders must not overwrite a previously cached live catalog.
    assert isinstance(setup, CuratedFallbackModels)
    assert isinstance(picker, CuratedFallbackModels)
