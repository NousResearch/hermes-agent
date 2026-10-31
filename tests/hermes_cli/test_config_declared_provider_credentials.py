"""Regression for #145452: model-catalog probes must use a config-declared provider's own
credential and endpoint.

A ``providers.<id>`` block that is not in ``PROVIDER_REGISTRY`` is invisible to the registry-only
credential resolver, so catalog helpers probed it keyless (or with an ambient ``OPENAI_API_KEY``)
and without its base URL. ``${env:VAR}`` secret refs must also be expanded before they reach an
``Authorization`` header.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock

import hermes_yaml as yaml
import pytest

from agent import auxiliary_client as aux
from hermes_cli import models as models_mod
from hermes_cli import models_pricing
from hermes_cli.auth import AuthError, resolve_api_key_provider_credentials

BASE_URL = "https://proxy.example.test/v1"
PROVIDER = "my-proxy"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return home


@pytest.fixture(autouse=True)
def _clear_catalog_caches():
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()
    yield
    models_pricing._pricing_cache.clear()
    models_pricing._pricing_cache_retry_after.clear()


def _write_provider(home, entry: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "my-combo", "provider": PROVIDER, "base_url": BASE_URL},
        "providers": {PROVIDER: {"name": "my proxy", "base_url": BASE_URL, **entry}},
    }), encoding="utf-8")


def test_inline_api_key_env_ref_is_expanded(hermes_home, monkeypatch):
    """``api_key: ${env:VAR}`` must never reach a probe as the literal ref."""
    monkeypatch.setenv("PROBE_KEY", "sk-declared-inline")
    _write_provider(hermes_home, {"api_key": "${env:PROBE_KEY}"})

    creds = resolve_api_key_provider_credentials(PROVIDER)

    assert creds["api_key"] == "sk-declared-inline"
    assert creds["base_url"] == BASE_URL


@pytest.mark.parametrize("field", ["key_env", "api_key_env"])
def test_key_env_pointer_is_resolved(hermes_home, monkeypatch, field):
    """``key_env`` and its documented ``api_key_env`` alias both name the credential."""
    monkeypatch.setenv("PROBE_KEY", "sk-declared-env")
    _write_provider(hermes_home, {field: "PROBE_KEY"})

    creds = resolve_api_key_provider_credentials(PROVIDER)

    assert creds["api_key"] == "sk-declared-env"
    assert creds["source"] == "PROBE_KEY"


def test_key_cmd_token_is_materialized(hermes_home, monkeypatch, tmp_path):
    """A ``key_cmd`` helper resolves to a probe-ready token through the same resolver."""
    from agent.command_token_source import materialize_probe_api_key

    helper = tmp_path / "mint_key.py"
    helper.write_text("print('sk-declared-cmd')\n", encoding="utf-8")
    _write_provider(hermes_home, {"key_cmd": f'"{sys.executable}" "{helper}"'})

    creds = resolve_api_key_provider_credentials(PROVIDER)

    assert materialize_probe_api_key(creds["api_key"]) == "sk-declared-cmd"


def test_api_field_is_the_base_url(hermes_home):
    """The ``api`` spelling (legacy named-endpoint field) is honoured like ``base_url``."""
    (hermes_home / "config.yaml").write_text(yaml.safe_dump({
        "providers": {PROVIDER: {"name": "my proxy", "api": "https://api.example.test/v1"}},
    }), encoding="utf-8")

    creds = resolve_api_key_provider_credentials(PROVIDER)

    assert creds["base_url"] == "https://api.example.test/v1"


def test_unknown_provider_still_raises(hermes_home):
    """Only a real config entry widens the resolver; an unknown name stays an error."""
    _write_provider(hermes_home, {"key_env": "PROBE_KEY"})

    with pytest.raises(AuthError):
        resolve_api_key_provider_credentials("not-configured-anywhere")


def test_fast_model_catalog_sends_the_declared_credential(hermes_home, monkeypatch):
    """``_fast_model_from_catalog`` probes the config-declared endpoint with its own key —
    never an unexpanded ref — and reads the catalog from its declared base URL."""
    monkeypatch.setenv("PROBE_KEY", "sk-declared-catalog")
    _write_provider(hermes_home, {"api_key": "${env:PROBE_KEY}"})
    seen: list[str | None] = []

    def _fake_urlopen(req, timeout=8.0):
        seen.append(req.get_header("Authorization"))
        payload = {
            "data": [{
                "id": "proxy-small-mini",
                "pricing": {"prompt": "0.000001", "completion": "0.000002"},
            }],
        }
        resp = MagicMock()
        resp.read.return_value = json.dumps(payload).encode()
        resp.__enter__ = lambda self: self
        resp.__exit__ = lambda *args: False
        return resp

    monkeypatch.setattr(models_mod, "_urlopen_model_catalog_request", _fake_urlopen)

    assert aux._fast_model_from_catalog(PROVIDER) == "proxy-small-mini"
    assert seen == ["Bearer sk-declared-catalog"]
