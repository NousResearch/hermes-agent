"""Tests for optional Google Maps Platform configuration in the maps skill."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "productivity"
    / "maps"
    / "scripts"
    / "maps_client.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("maps_client_google_maps_test", SCRIPT_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_google_maps_capability_absent_for_missing_or_blank_key(monkeypatch):
    mod = load_module()

    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)
    assert mod.google_maps_api_key_available() is False
    assert mod.google_maps_available() is False
    assert mod._get_google_maps_api_key() is None

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "   ")
    assert mod.google_maps_api_key_available() is False
    assert mod.google_maps_available() is False
    assert mod._get_google_maps_api_key() is None


def test_google_maps_capability_detects_key_without_emitting_it(monkeypatch, capsys):
    mod = load_module()
    secret = "gmaps-secret-must-not-leak"
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", f"  {secret}  ")

    assert mod.google_maps_api_key_available() is True
    assert mod.google_maps_available() is True
    assert mod._get_google_maps_api_key() == secret

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err

    public_payload = json.dumps(
        {
            "available": mod.google_maps_api_key_available(),
            "alias_available": mod.google_maps_available(),
        }
    )
    assert secret not in public_payload


def test_no_key_osm_search_behavior_unchanged(monkeypatch):
    mod = load_module()
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    captured = {}

    def fake_http_get(url, params=None, retries=mod.MAX_RETRIES, silent=False):
        captured["url"] = url
        captured["params"] = params
        return [
            {
                "display_name": "Amsterdam, North Holland, Netherlands",
                "lat": "52.3728",
                "lon": "4.8936",
                "type": "city",
                "importance": 0.9,
                "address": {"city": "Amsterdam", "country": "Netherlands"},
            }
        ]

    monkeypatch.setattr(mod, "http_get", fake_http_get)

    result = mod.nominatim_search("Amsterdam", limit=1)

    assert captured["url"] == mod.NOMINATIM_SEARCH
    assert captured["params"] == {
        "q": "Amsterdam",
        "format": "json",
        "limit": 1,
        "addressdetails": 1,
    }
    assert result[0]["display_name"] == "Amsterdam, North Holland, Netherlands"
    assert "GOOGLE_MAPS_API_KEY" not in json.dumps(result)
