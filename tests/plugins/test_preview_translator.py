"""Tests for plugins/preview-translator/dashboard/plugin_api.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _load_plugin():
    plugin_file = Path(__file__).resolve().parents[2] / "plugins" / "preview-translator" / "dashboard" / "plugin_api.py"
    spec = importlib.util.spec_from_file_location("hermes_dashboard_plugin_preview_translator_test", plugin_file)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def api():
    return _load_plugin()


@pytest.fixture
def client(api):
    app = FastAPI()
    app.include_router(api.router, prefix="/api/plugins/preview-translator")
    return TestClient(app, raise_server_exceptions=False)


def test_extract_json_array_plain_and_fenced(api):
    assert api._extract_json_array('["a", "b"]') == ["a", "b"]
    assert api._extract_json_array('```json\n["a"]\n```') == ["a"]
    assert api._extract_json_array('Here you go: ["ok"]') == ["ok"]
    assert api._extract_json_array('{"translations": ["x"]}') == ["x"]


def test_extract_json_array_rejects_garbage(api):
    with pytest.raises(ValueError, match="JSON array"):
        api._extract_json_array("not json at all")


def test_parse_model(api):
    assert api._parse_model("auto", None) == (None, None)
    assert api._parse_model("gemini/gemini-flash-latest", None) == ("gemini", "gemini-flash-latest")
    assert api._parse_model("gpt-4o-mini", "openai") == ("openai", "gpt-4o-mini")
    assert api._parse_model("gpt-4o-mini", None) == (None, "gpt-4o-mini")


def test_models_endpoint_always_includes_auto(client):
    r = client.get("/api/plugins/preview-translator/models")
    assert r.status_code == 200
    ids = [m["id"] for m in r.json()["models"]]
    assert ids[0] == "auto"


def test_translate_empty(client):
    r = client.post("/api/plugins/preview-translator/translate", json={"texts": []})
    assert r.status_code == 200
    assert r.json() == {"translations": [], "model_used": "auto"}


def test_translate_rejects_too_many(client, api):
    r = client.post(
        "/api/plugins/preview-translator/translate",
        json={"texts": ["x"] * (api.MAX_TEXTS + 1)},
    )
    assert r.status_code == 400


def test_translate_dedups_and_maps_back(api, monkeypatch):
    seen = []

    def fake_unique(texts, target_lang, provider, model):
        seen.append(list(texts))
        return [f"{t}-{target_lang}" for t in texts]

    monkeypatch.setattr(api, "_translate_unique", fake_unique)
    out = api._translate_all(["Hi", "Hi", "Bye"], "English", None, None)
    assert seen == [["Hi", "Bye"]]
    assert out == ["Hi-English", "Hi-English", "Bye-English"]


def test_translate_endpoint_uses_llm_helper(client, api, monkeypatch):
    monkeypatch.setattr(api, "_translate_all", lambda texts, lang, provider, model: [t.upper() for t in texts])
    r = client.post(
        "/api/plugins/preview-translator/translate",
        json={"texts": ["hola"], "target_lang": "English", "model": "auto"},
    )
    assert r.status_code == 200
    assert r.json()["translations"] == ["HOLA"]


def test_translate_endpoint_surfaces_llm_errors(client, api, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("provider 429")

    monkeypatch.setattr(api, "_translate_all", boom)
    r = client.post("/api/plugins/preview-translator/translate", json={"texts": ["hola"]})
    assert r.status_code == 500


def test_message_text_joins_content_parts(api):
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=[{"text": "a"}, {"text": "b"}]))]
    )
    assert api._message_text(response) == "ab"
