"""A configured ``providers.llamacpp`` endpoint wins over the managed llama.cpp runtime.

``provider: llamacpp`` with ``providers.llamacpp.api`` pointing at a llama-server on another
host used to raise "The local model server is turned off": the managed-runtime rung ran before
the named-provider lookup, and only the ``--base-url`` flag counted as an explicit endpoint. A
configured endpoint is the user pointing at a specific server, exactly like the flag, so it
takes the same precedence. With no configured endpoint the managed runtime path is unchanged.
"""

from __future__ import annotations

import pytest

from hermes_cli import runtime_provider as rp

CONFIGURED_URL = "http://10.0.0.21:8080/v1"
MANAGED_URL = "http://127.0.0.1:8080/v1"


@pytest.fixture
def scratch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    return tmp_path / ".hermes"


def _configured(extra=None):
    config = {"local_runtime": {"enabled": False},
              "providers": {"llamacpp": {"api": CONFIGURED_URL}}}
    if extra:
        config["providers"]["llamacpp"].update(extra)
    return config


def _patch_endpoint(monkeypatch, endpoint):
    monkeypatch.setattr("hermes_cli.local_runtime.endpoint.resolve_llamacpp_endpoint",
                        lambda *a, **k: endpoint)


def test_configured_endpoint_resolves_without_a_managed_server(scratch_home, monkeypatch):
    _patch_endpoint(monkeypatch, None)
    monkeypatch.setattr("hermes_cli.config.load_config", _configured)

    result = rp._resolve_named_custom_runtime(requested_provider="llamacpp")

    assert result["provider"] == "custom"
    assert result["base_url"] == CONFIGURED_URL
    assert result["requested_provider"] == "llamacpp"
    assert result["source"] == "custom_provider:llamacpp"


def test_configured_endpoint_wins_over_a_running_managed_server(scratch_home, monkeypatch):
    _patch_endpoint(monkeypatch, {"base_url": MANAGED_URL, "api_key": ""})
    monkeypatch.setattr("hermes_cli.config.load_config", _configured)

    result = rp._resolve_named_custom_runtime(requested_provider="llamacpp")

    assert result["base_url"] == CONFIGURED_URL
    assert result["source"] == "custom_provider:llamacpp"


def test_explicit_base_url_still_wins_over_the_configured_endpoint(scratch_home, monkeypatch):
    _patch_endpoint(monkeypatch, None)
    monkeypatch.setattr("hermes_cli.config.load_config", _configured)

    result = rp._resolve_named_custom_runtime(requested_provider="llamacpp",
                                              explicit_base_url="http://127.0.0.1:9999/v1")

    assert result["base_url"] == "http://127.0.0.1:9999/v1"


@pytest.mark.parametrize("config", [
    {"local_runtime": {"enabled": False}},
    {"local_runtime": {"enabled": False}, "providers": {}},
    {"local_runtime": {"enabled": False}, "providers": {"llamacpp": {"default_model": "m"}}},
    {"local_runtime": {"enabled": False}, "providers": {"other": {"api": CONFIGURED_URL}}},
])
def test_no_configured_endpoint_keeps_the_managed_runtime_path(scratch_home, monkeypatch, config):
    _patch_endpoint(monkeypatch, None)
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: config)

    with pytest.raises(ValueError, match="turned off"):
        rp._resolve_named_custom_runtime(requested_provider="llamacpp")


def test_no_configured_endpoint_still_uses_a_running_managed_server(scratch_home, monkeypatch):
    _patch_endpoint(monkeypatch, {"base_url": MANAGED_URL, "api_key": ""})
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"local_runtime": {"enabled": True}})

    result = rp._resolve_named_custom_runtime(requested_provider="llamacpp")

    assert result["base_url"] == MANAGED_URL
    assert result["source"] == "local-runtime"


def test_config_file_with_bare_llamacpp_provider_resolves_end_to_end(scratch_home, monkeypatch):
    scratch_home.mkdir()
    (scratch_home / "config.yaml").write_text(
        "model:\n  default: m\n  provider: llamacpp\n"
        f"providers:\n  llamacpp:\n    api: {CONFIGURED_URL}\n")
    _patch_endpoint(monkeypatch, None)

    result = rp.resolve_runtime_provider()

    assert result["provider"] == "custom"
    assert result["base_url"] == CONFIGURED_URL
    assert result["requested_provider"] == "llamacpp"
