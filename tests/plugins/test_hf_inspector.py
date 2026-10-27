"""Tests for plugins.hf_inspector — Hugging Face Model & Quant Explorer."""

import json
import urllib.error
from unittest.mock import MagicMock, patch

import pytest
from plugins.hf_inspector.tools import (
    _format_params,
    _format_size,
    handle_hf_inspect_model,
    handle_hf_list_quants,
)
from plugins.hf_inspector import register


def test_format_helpers():
    assert _format_params(8_030_000_000) == "8.0B"
    assert _format_params(70_000_000_000) == "70.0B"
    assert _format_params(350_000_000) == "350.0M"
    assert _format_params(None) == "Unknown"

    assert "GB" in _format_size(4_500_000_000)
    assert "MB" in _format_size(45_000_000)
    assert _format_size(None) == "Unknown size"


def test_inspect_model_empty_input():
    res = handle_hf_inspect_model("")
    assert "Error: model_id is required" in res


@patch("plugins.hf_inspector.tools._fetch_hf_api")
def test_inspect_model_success(mock_fetch):
    mock_fetch.return_value = {
        "id": "NousResearch/Hermes-3-Llama-3.1-8B",
        "pipeline_tag": "text-generation",
        "downloads": 250000,
        "likes": 1200,
        "gated": False,
        "tags": ["license:apache-2.0", "conversational"],
        "safetensors": {"total": 8030000000},
        "config": {
            "architectures": ["LlamaForCausalLM"],
            "max_position_embeddings": 131072,
        },
    }

    res = handle_hf_inspect_model("NousResearch/Hermes-3-Llama-3.1-8B")
    assert "NousResearch/Hermes-3-Llama-3.1-8B" in res
    assert "Parameters: 8.0B" in res
    assert "Context Length: 131,072 tokens" in res
    assert "LlamaForCausalLM" in res
    assert "apache-2.0" in res
    assert "Public" in res


@patch("plugins.hf_inspector.tools._fetch_hf_api")
def test_inspect_model_404_not_found(mock_fetch):
    mock_fetch.side_effect = urllib.error.HTTPError(
        url="https://huggingface.co/api/models/fake",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )
    res = handle_hf_inspect_model("fake/non-existent-model")
    assert "not found (404)" in res


@patch("plugins.hf_inspector.tools._fetch_hf_api")
def test_inspect_model_gated(mock_fetch):
    mock_fetch.side_effect = urllib.error.HTTPError(
        url="https://huggingface.co/api/models/gated",
        code=403,
        msg="Forbidden",
        hdrs={},
        fp=None,
    )
    res = handle_hf_inspect_model("meta-llama/Meta-Llama-3.1-8B")
    assert "gated or private (HTTP 403)" in res


@patch("plugins.hf_inspector.tools._fetch_hf_api")
def test_list_quants_success(mock_fetch):
    mock_fetch.return_value = {
        "siblings": [
            {"rfilename": "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf", "size": 4920000000},
            {"rfilename": "Hermes-3-Llama-3.1-8B-Q8_0.gguf", "size": 8540000000},
            {"rfilename": "README.md", "size": 5000},
        ]
    }

    res = handle_hf_list_quants("NousResearch/Hermes-3-Llama-3.1-8B-GGUF")
    assert "Quantized Files in NousResearch/Hermes-3-Llama-3.1-8B-GGUF" in res
    assert "Hermes-3-Llama-3.1-8B-Q4_K_M.gguf" in res
    assert "Hermes-3-Llama-3.1-8B-Q8_0.gguf" in res
    assert "README.md" not in res


@patch("plugins.hf_inspector.tools._fetch_hf_api")
def test_list_quants_none_found(mock_fetch):
    mock_fetch.return_value = {
        "siblings": [
            {"rfilename": "config.json", "size": 1000},
            {"rfilename": "model.safetensors", "size": 5000000000},
        ]
    }

    res = handle_hf_list_quants("NousResearch/Hermes-3-Llama-3.1-8B")
    assert "No GGUF, AWQ, or GPTQ quantized files found" in res


def test_plugin_registration():
    mock_ctx = MagicMock()
    register(mock_ctx)
    assert mock_ctx.register_tool.call_count == 2
    registered_names = [call.kwargs["name"] for call in mock_ctx.register_tool.call_args_list]
    assert "hf_inspect_model" in registered_names
    assert "hf_list_quants" in registered_names
