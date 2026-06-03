"""Integration: the real Hindsight provider through the Desktop config surface.

Hindsight is the framework's forcing function — it's the only provider whose
schema uses ``when``-gated conditional fields (api_url/api_key repeated per
mode). Verifies the generic layer enriches its real schema and that the
conditional gating resolves per mode.
"""

from __future__ import annotations

import pytest

from hermes_cli.memory_provider_surface import enrich_schema, field_visible


def _hindsight_schema():
    from plugins.memory import load_memory_provider

    provider = load_memory_provider("hindsight")
    if provider is None:
        pytest.skip("hindsight provider not loadable")
    return provider.get_config_schema()


def test_hindsight_schema_enriches_without_error():
    rows = enrich_schema(_hindsight_schema())
    assert rows, "hindsight should declare config fields"
    # every enriched field has a valid kind
    assert all(r["kind"] in {"text", "secret", "select", "bool", "number"} for r in rows)
    # the mode field is a select
    mode = next(r for r in rows if r["key"] == "mode")
    assert mode["kind"] == "select"
    assert {"cloud", "local_external"}.issubset({o["value"] for o in mode["options"]})


def test_hindsight_api_url_is_mode_gated():
    rows = enrich_schema(_hindsight_schema())
    api_urls = [r for r in rows if r["key"] == "api_url"]
    # declared more than once, each gated by a different mode
    assert len(api_urls) >= 2
    assert all("when" in r for r in api_urls)

    cloud_vals = {"mode": "cloud"}
    visible_for_cloud = [r for r in api_urls if field_visible(r, cloud_vals)]
    # exactly one api_url row is visible for a given mode
    assert len(visible_for_cloud) == 1
    assert visible_for_cloud[0]["when"]["mode"] == "cloud"


def test_hindsight_secret_fields_carry_env_key():
    rows = enrich_schema(_hindsight_schema())
    secrets = [r for r in rows if r["kind"] == "secret"]
    assert secrets, "hindsight has at least one secret (api_key)"
    assert all(r.get("env_key") for r in secrets)
