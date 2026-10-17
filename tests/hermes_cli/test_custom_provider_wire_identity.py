"""Custom auth belongs to an exact endpoint or its known OpenCode OpenAI-wire URL."""

import pytest

from agent.credential_pool import (
    PooledCredential, credential_pool_entry_serves_endpoint,
    credential_pool_matches_provider, custom_provider_pool_key_candidates,
)
from hermes_cli.config import atomic_config_write, get_custom_provider_extra_headers


@pytest.mark.parametrize("configured, runtime_url, accepted", [
    ("https://opencode.ai/zen", "https://opencode.ai/zen/v1", True),
    ("https://opencode.ai/zen/go?tenant=fixture", "https://opencode.ai/zen/go/v1?tenant=fixture", True),
    ("https://edge.opencode.ai/zen/go", "https://edge.opencode.ai/zen/go/v1", True),
    ("https://opencode.ai:8443/zen/go", "https://opencode.ai:8443/zen/go/v1", True),
    ("https://relay.example.test/zen/go", "https://relay.example.test/zen/go/v1", False),
    ("https://opencode.ai/zen/go/v1", "https://opencode.ai/zen/go", False),
    ("https://relayopencode.ai/zen/go", "https://relayopencode.ai/zen/go/v1", False),
    ("https://opencode.ai/zen/go?tenant=one", "https://opencode.ai/zen/go/v1?tenant=two", False),
])
def test_only_forward_opencode_wire_derivation_keeps_auth(
    tmp_path, monkeypatch, configured, runtime_url, accepted,
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config = {"providers": {"fixture": {
        "base_url": configured, "extra_headers": {"X-Owner": "fixture"},
    }}}
    atomic_config_write(tmp_path / "config.yaml", config)
    assert get_custom_provider_extra_headers(runtime_url, config=config) == (
        {"X-Owner": "fixture"} if accepted else {})
    assert custom_provider_pool_key_candidates(runtime_url) == (
        ["fixture", "custom:fixture"] if accepted else [])
    assert credential_pool_matches_provider("custom:fixture", "fixture", base_url=runtime_url) is accepted


@pytest.mark.parametrize("configured", [
    "https://opencode.ai/zen/go//",
    "https://open\ncode.ai/zen/go",
    "https://opencode.ai/zen/go?",
])
def test_wire_derivation_does_not_repair_distinct_route_identity(configured):
    entries = [{"name": "fixture", "base_url": configured, "extra_headers": {"X-Owner": "fixture"}}]
    assert get_custom_provider_extra_headers(
        "https://opencode.ai/zen/go/v1", custom_providers=entries,
    ) == {}


@pytest.mark.parametrize("base_url", [
    "https://relay.example.test/zen/go/v1",
    "https://opencode.ai.attacker.test/zen/go/v1",
    "https://opencode.ai/zen/v1",
    "https://opencode.ai/zen/go/v2",
    "https://opencode.ai/zen/go/v1/other",
    "https://opencode.ai/zen/go/v1?other=1",
    "https://opencode.ai:8443/zen/go/v1",
    "http://opencode.ai/zen/go/v1",
    "https://fixture@opencode.ai/zen/go/v1",
])
def test_nearby_routes_do_not_borrow_custom_auth(tmp_path, monkeypatch, base_url):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    configured = "https://opencode.ai/zen/go"
    config = {"providers": {"fixture": {
        "base_url": configured, "api_mode": "chat_completions",
        "extra_headers": {"X-Relay-Access": "fixture-access"},
    }}}
    atomic_config_write(tmp_path / "config.yaml", config)
    assert get_custom_provider_extra_headers(base_url, config=config) == {}
    assert custom_provider_pool_key_candidates(base_url) == []
    assert not credential_pool_matches_provider("custom:fixture", "fixture", base_url=base_url)
    assert not credential_pool_matches_provider("fixture", "custom", base_url=base_url)
    row = PooledCredential(
        provider="custom:fixture", id="fixture", label="fixture", auth_type="api_key",
        priority=0, source="manual:fixture", access_token="fixture-key", base_url=configured,
    )
    assert not credential_pool_entry_serves_endpoint(row, base_url)


def test_exact_route_owns_auth_before_a_derived_route(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw_url, wire_url = "https://opencode.ai/zen/go", "https://opencode.ai/zen/go/v1"
    config = {"providers": {
        "raw": {"base_url": raw_url, "extra_headers": {"X-Owner": "raw"}},
        "wire": {"base_url": wire_url, "extra_headers": {"X-Owner": "wire"}},
    }}
    atomic_config_write(tmp_path / "config.yaml", config)
    assert get_custom_provider_extra_headers(wire_url, config=config) == {"X-Owner": "wire"}
    assert custom_provider_pool_key_candidates(wire_url) == ["wire", "custom:wire"]
    assert custom_provider_pool_key_candidates(wire_url, "raw") == ["raw", "custom:raw"]
    assert not credential_pool_matches_provider("custom:raw", "wire", base_url=wire_url)


def test_ambiguous_derived_routes_do_not_choose_an_auth_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    raw_url, wire_url = "https://opencode.ai/zen/go", "https://opencode.ai/zen/go/v1"
    config = {"providers": {
        "first": {"base_url": raw_url, "extra_headers": {"X-Owner": "first"}},
        "second": {"base_url": raw_url, "extra_headers": {"X-Owner": "second"}},
    }}
    atomic_config_write(tmp_path / "config.yaml", config)
    assert get_custom_provider_extra_headers(wire_url, config=config) == {}
    assert custom_provider_pool_key_candidates(wire_url) == []
