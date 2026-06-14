from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent.local_secretary.llama_contract import (
    MIN_CONTEXT_SIZE,
    TARGET_CONTEXT_SIZE,
    extract_context_size,
    extract_model_ids,
    looks_like_plaintext_tool_call,
    run_llama_contract_checks,
    validate_config_context_length,
)


def test_extract_context_size_from_default_generation_settings():
    payload = {"default_generation_settings": {"n_ctx": 65536}}
    assert extract_context_size(payload) == 65536


def test_extract_model_ids_accepts_dicts_and_strings():
    payload = {"data": [{"id": "qwen35-9b-secretary"}, "hermes3-8b-fallback"]}
    assert extract_model_ids(payload) == [
        "qwen35-9b-secretary",
        "hermes3-8b-fallback",
    ]


def test_validate_config_context_length_minimum():
    ok, err = validate_config_context_length(65536)
    assert ok is True
    assert err is None

    ok, err = validate_config_context_length(32000)
    assert ok is False
    assert "32000" in (err or "")


def test_plaintext_tool_call_detection():
    assert looks_like_plaintext_tool_call("Calling tool get_weather now") is True
    assert looks_like_plaintext_tool_call('{"tool_calls":[]}') is False


def test_run_llama_contract_checks_happy_path():
    models_payload = {"data": [{"id": "qwen35-9b-secretary"}]}
    props_payload = {"default_generation_settings": {"n_ctx": 65536}}
    chat_payload = {"choices": [{"message": {"content": "pong"}}]}
    tool_payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        }
                    ]
                }
            }
        ]
    }

    def fake_get(url: str, timeout: float = 15.0):
        if url.endswith("/v1/models"):
            return 200, models_payload
        if url.endswith("/props"):
            return 200, props_payload
        raise AssertionError(url)

    def fake_post(url: str, payload: dict, timeout: float = 120.0):
        if "tools" in payload:
            return 200, tool_payload
        return 200, chat_payload

    with patch("agent.local_secretary.llama_contract._get_json", side_effect=fake_get), patch(
        "agent.local_secretary.llama_contract._post_json", side_effect=fake_post
    ):
        result = run_llama_contract_checks("http://127.0.0.1:8080")

    assert result["ok"] is True
    assert result["checks"]["context_size"]["n_ctx"] == 65536


def test_run_llama_contract_checks_flags_low_context():
    with patch(
        "agent.local_secretary.llama_contract._get_json",
        side_effect=[
            (200, {"data": [{"id": "small"}]}),
            (200, {"default_generation_settings": {"n_ctx": 8192}}),
        ],
    ), patch(
        "agent.local_secretary.llama_contract._post_json",
        return_value=(200, {"choices": [{"message": {"content": "pong"}}]}),
    ):
        result = run_llama_contract_checks("http://127.0.0.1:8080")

    assert result["ok"] is False
    assert result["checks"]["context_size"]["ok"] is False


def test_local_secretary_example_config_context():
    repo_root = Path(__file__).resolve().parents[2]
    example = repo_root / "config" / "local-secretary.example.yaml"
    data = yaml.safe_load(example.read_text(encoding="utf-8"))
    ctx = data["model"]["context_length"]
    ok, err = validate_config_context_length(ctx)
    assert ok, err
    assert ctx >= MIN_CONTEXT_SIZE
    assert ctx == TARGET_CONTEXT_SIZE


@pytest.mark.integration
def test_live_llama_server_contract_if_running():
    result = run_llama_contract_checks("http://127.0.0.1:8080")
    if not result["checks"].get("models", {}).get("ok"):
        pytest.skip("llama.cpp server not running on :8080")
    assert result["ok"] is True
