"""Ollama /api/tags quantization_level → inventory map, without changing list[str] parse."""

from hermes_cli import inventory as inv
from hermes_cli.models_local import (
    _parse_ollama_quantization,
    _parse_ollama_tags,
    ollama_local_quantization_map,
)


def _tags_payload():
    return {
        "models": [
            {"name": "qwen3.5:9b", "details": {"quantization_level": "Q4_K_M"}},
            {"name": "qwen3.5:9b-q8", "model": "qwen3.5:9b-q8", "details": {"quantization_level": "Q8_0"}},
            {"name": "cloud-ish", "details": {}},
        ]
    }


def test_parse_ollama_quantization_map_from_tags_payload():
    payload = _tags_payload()
    quant = _parse_ollama_quantization(payload)
    assert quant["qwen3.5:9b"] == "Q4_K_M"
    assert quant["qwen3.5:9b-q8"] == "Q8_0"
    assert "cloud-ish" not in quant


def test_parse_ollama_tags_still_returns_model_id_list():
    assert _parse_ollama_tags(_tags_payload()) == ["qwen3.5:9b", "qwen3.5:9b-q8", "cloud-ish"]


def test_parse_ollama_quantization_fail_open_missing_details():
    assert _parse_ollama_quantization({"models": [{"name": "qwen3.5:9b"}]}) == {}
    assert _parse_ollama_quantization({"models": [{"name": "qwen3.5:9b", "details": {}}]}) == {}
    assert _parse_ollama_quantization(
        {"models": [{"name": "qwen3.5:9b", "details": {"quantization_level": ""}}]}
    ) == {}
    assert _parse_ollama_quantization(
        {"models": [{"name": "qwen3.5:9b", "details": {"quantization_level": 4}}]}
    ) == {}


def test_quantization_map_getter_empty_on_cache_miss():
    assert ollama_local_quantization_map("http://127.0.0.1:19998") == {}


def test_apply_quantization_attaches_local_rows_skips_cloud(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.models_local.ollama_local_quantization_map",
        lambda base_url=None, headers=None: {"qwen3.5:9b": "Q4_K_M"},
    )
    rows = [
        {"slug": "ollama", "models": ["qwen3.5:9b"]},
        {
            "slug": "custom",
            "api_url": "http://127.0.0.1:11434",
            "is_user_defined": True,
            "models": ["qwen3.5:9b"],
        },
        {"slug": "openai", "models": ["gpt-5.5"]},
        {"slug": "ollama-cloud", "models": ["cloud-ish"]},
    ]
    inv._apply_quantization(rows)
    assert rows[0]["quantization"]["qwen3.5:9b"] == "Q4_K_M"
    assert rows[1]["quantization"]["qwen3.5:9b"] == "Q4_K_M"
    assert not rows[2].get("quantization")
    assert not rows[3].get("quantization")
