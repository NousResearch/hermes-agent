"""``GET /api/model/fallback`` exposes the ordered top-level ``fallback_providers`` chain (#122572).

The dashboard Models page had no view of the primary failover chain: it could only be
edited by hand in config.yaml. This read endpoint is the display-only data provider for
that section, so the three properties the UI depends on are pinned here:

  * the chain is EXPOSED (route registered + entries carry provider/model/base_url),
  * the chain is ORDERED (failover order == config order),
  * masked values STAY MASKED (an inline ``api_key`` never appears in the payload).
"""

from __future__ import annotations

import json
from unittest.mock import patch

from hermes_cli.web_routers.models import get_fallback_providers

CHAIN = [
    # Deliberately ordered; order is the contract (first entry = first failover).
    {"provider": "openrouter", "model": "google/gemini-2.5-flash",
     "api_key": "sk-or-supersecret-value-1234567890"},
    {"provider": "anthropic", "model": "claude-haiku-4-5",
     "base_url": "https://api.anthropic.com/v1"},
    {"provider": "custom", "model": "glm-local", "base_url": "http://127.0.0.1:8080/v1",
     "key_env": "LOCAL_GATEWAY_KEY"},
]


def test_fallback_route_registered():
    """The endpoint exists at a stable path the frontend can call."""
    from hermes_cli.web_routers.models import router

    paths = [r.path for r in router.routes]
    assert "/api/model/fallback" in paths, (
        "GET /api/model/fallback is not registered; the Models page has no "
        "fallback_providers data provider"
    )


def test_fallback_chain_exposed_and_ordered():
    """Entries come back in config order with their display fields intact."""
    with (
        patch("hermes_cli.config.load_config",
              return_value={"fallback_providers": list(CHAIN)}),
        patch("hermes_cli.web_server_profiles._profile_scope"),
    ):
        resp = get_fallback_providers()

    chain = resp["chain"]
    assert [(e["provider"], e["model"]) for e in chain] == [
        ("openrouter", "google/gemini-2.5-flash"),
        ("anthropic", "claude-haiku-4-5"),
        ("custom", "glm-local"),
    ], "fallback chain must preserve config.yaml failover order"
    # Optional fields survive for display.
    assert chain[1]["base_url"] == "https://api.anthropic.com/v1"
    assert chain[2]["base_url"] == "http://127.0.0.1:8080/v1"
    assert chain[2]["key_env"] == "LOCAL_GATEWAY_KEY"


def test_fallback_inline_api_key_stays_masked():
    """A plaintext ``api_key`` on an entry must never leave the endpoint raw.

    ``load_config()`` env-expands entries, so even ``api_key: ${MY_KEY}`` arrives
    here as a live secret -- the endpoint must mask whatever form it sees.
    """
    with (
        patch("hermes_cli.config.load_config",
              return_value={"fallback_providers": list(CHAIN)}),
        patch("hermes_cli.web_server_profiles._profile_scope"),
    ):
        resp = get_fallback_providers()

    wire = json.dumps(resp)
    assert "sk-or-supersecret-value-1234567890" not in wire, (
        "GET /api/model/fallback leaked a plaintext api_key"
    )
    entry = resp["chain"][0]
    assert entry["api_key_preview"], "inline api_key should surface as a masked preview"
    assert "supersecret" not in entry["api_key_preview"]


def test_fallback_key_env_shows_env_reference_not_secret():
    """``key_env`` entries show the variable reference, never a resolved secret."""
    with (
        patch("hermes_cli.config.load_config",
              return_value={"fallback_providers": list(CHAIN)}),
        patch("hermes_cli.web_server_profiles._profile_scope"),
    ):
        resp = get_fallback_providers()

    entry = resp["chain"][2]
    assert entry["key_env"] == "LOCAL_GATEWAY_KEY"
    assert entry["api_key_preview"] == "${LOCAL_GATEWAY_KEY}"


def test_fallback_empty_config_yields_empty_chain():
    """A config with no chain renders as an empty list, not an error."""
    with (
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.web_server_profiles._profile_scope"),
    ):
        resp = get_fallback_providers()

    assert resp == {"chain": []}
