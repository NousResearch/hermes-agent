"""Tests for the Honcho provider's Desktop config surface override.

Verifies Honcho routes read_config through its own host-block resolution
(not the generic conventional reader), never leaks the api_key, and exposes
safe/advanced tiers via the enriched schema.
"""

from __future__ import annotations

import json

import pytest


def _load_honcho():
    from plugins.memory import load_memory_provider

    provider = load_memory_provider("honcho")
    if provider is None:
        pytest.skip("honcho provider not loadable in this environment")
    return provider


def test_schema_tiers_and_kinds():
    schema = _load_honcho().get_config_schema()
    by_key = {f["key"]: f for f in schema}
    assert by_key["api_key"]["kind"] == "secret"
    assert by_key["api_key"]["tier"] == "safe"
    assert by_key["baseUrl"]["tier"] == "safe"
    assert by_key["workspace"]["tier"] == "safe"
    # behavior knobs are grouped advanced (editable, not locked)
    assert by_key["recallMode"]["tier"] == "advanced"
    assert by_key["dialecticCadence"]["tier"] == "advanced"


def test_read_config_reads_host_block(tmp_path, monkeypatch):
    """read_config must resolve hosts.<host>.apiKey + root baseUrl/workspace,
    which the generic conventional reader (which reads <home>/honcho/config.json,
    flat keys) cannot do."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HONCHO_API_KEY", raising=False)
    # Honcho's native file: host-keyed apiKey + root-level baseUrl/workspace
    (tmp_path / "honcho.json").write_text(json.dumps({
        "hosts": {"hermes": {"apiKey": "hk-secret"}},
        "baseUrl": "https://self.hosted",
        "workspace": "myspace",
        "recallMode": "tools",
    }), encoding="utf-8")

    state = _load_honcho().read_config(str(tmp_path))
    # secret resolved but never returned
    assert state["api_key"]["value"] == ""
    assert state["api_key"]["is_set"] is True
    assert state["baseUrl"]["value"] == "https://self.hosted"
    assert state["workspace"]["value"] == "myspace"
    assert state["recallMode"]["value"] == "tools"
    assert "hk-secret" not in json.dumps(state)


def test_read_config_env_fallback_for_key(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HONCHO_API_KEY", "env-key")
    state = _load_honcho().read_config(str(tmp_path))
    assert state["api_key"]["is_set"] is True
    assert state["api_key"]["value"] == ""


def test_desktop_surface_masks_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HONCHO_API_KEY", "env-key")
    from hermes_cli.memory_provider_surface import build_surface

    surface = build_surface(_load_honcho(), str(tmp_path))
    assert surface["name"] == "honcho"
    assert "env-key" not in json.dumps(surface)
    fields = {f["key"]: f for f in surface["fields"]}
    assert fields["api_key"]["is_set"] is True
    assert fields["api_key"]["value"] == ""
