import json

import pytest

from hermes_cli import models
from hermes_cli import models_local


MODEL = "publisher/model"
BASE_URL = "http://127.0.0.1:1234/v1"


class _JsonResponse:
    def __init__(self, payload):
        self._body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._body


def _catalog(*, loaded_context=None, maximum=262_144):
    loaded_instances = []
    if loaded_context is not None:
        loaded_instances.append({
            "id": f"{MODEL}:active",
            "config": {"context_length": loaded_context},
        })
    return [{
        "key": MODEL,
        "max_context_length": maximum,
        "loaded_instances": loaded_instances,
    }]


def _capture_load(monkeypatch, response_payload):
    requests = []

    def fake_open(request, *, timeout):
        requests.append((request, timeout, json.loads(request.data.decode())))
        return _JsonResponse(response_payload)

    monkeypatch.setattr(models, "_urlopen_model_catalog_request", fake_open)
    return requests






def test_missing_echo_refreshes_loaded_state(monkeypatch):
    catalogs = iter([_catalog(), _catalog(loaded_context=88_000)])
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **_kwargs: next(catalogs)
    )
    _capture_load(monkeypatch, {"status": "loaded"})

    result = models_local.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=100_000
    )

    assert result == 88_000


def test_jit_load_without_explicit_context_reads_runtime_from_loaded_instance(monkeypatch):
    catalogs = iter([_catalog(), _catalog(loaded_context=132_096)])
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models", lambda **_kwargs: next(catalogs))
    requests = _capture_load(monkeypatch, {"status": "loaded"})

    result = models_local.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=None
    )

    assert result == 132_096
    assert "context_length" not in requests[0][2]


def test_explicit_override_above_known_maximum_rejects_even_when_loaded(monkeypatch):
    monkeypatch.setattr(models_local, "_lmstudio_fetch_raw_models",
        lambda **_kwargs: _catalog(loaded_context=64_000, maximum=128_000),
    )
    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        lambda *_args, **_kwargs: pytest.fail("invalid override must not be posted"),
    )

    result = models_local.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=256_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is False
    assert result.rejected is True


def test_loaded_instance_alias_resolves_parent_capabilities(monkeypatch):
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [{
            "key": MODEL,
            "loaded_instances": [{"id": "loaded-instance-alias", "config": {"context_length": 98_304}}],
            "capabilities": {"reasoning": {"allowed_options": ["off", "low"]}},
        }],
    )

    assert models_local.lmstudio_model_reasoning_options(
        "loaded-instance-alias", BASE_URL, api_key=""
    ) == ["off", "low"]
    assert models_local.ensure_lmstudio_model_loaded(
        "loaded-instance-alias", BASE_URL, api_key="", target_context_length=None
    ) == 98_304


def test_loaded_instance_alias_wins_duplicate_slug(monkeypatch):
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [
            {"key": "unsloth/qwen3.8-27b", "loaded_instances": []},
            {
                "key": "qwen/qwen3.8-27b",
                "loaded_instances": [{"id": "qwen/qwen3.8-27b:2", "config": {"context_length": 98_304}}],
                "capabilities": {"reasoning": {"allowed_options": ["off", "low"]}},
            },
        ],
    )

    assert models_local.lmstudio_model_reasoning_options(
        "qwen3.8-27b", BASE_URL, api_key=""
    ) == ["off", "low"]
    assert models_local.ensure_lmstudio_model_loaded(
        "qwen3.8-27b", BASE_URL, api_key="", target_context_length=None
    ) == 98_304


def test_loaded_instance_alias_selects_matching_context(monkeypatch):
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [{
            "key": "qwen/qwen3.8-27b",
            "loaded_instances": [
                {"id": "qwen/qwen3.8-27b", "config": {"context_length": 131072}},
                {"id": "qwen/qwen3.8-27b:2", "config": {"context_length": 98_304}},
            ],
        }],
    )

    assert models_local.ensure_lmstudio_model_loaded(
        "qwen/qwen3.8-27b:2", BASE_URL, api_key="", target_context_length=None
    ) == 98_304


def test_short_alias_cold_load_uses_canonical_model_key(monkeypatch):
    requests = _capture_load(monkeypatch, {"load_config": {"context_length": 98_304}})
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [{"key": "qwen/qwen3.8-27b", "loaded_instances": []}],
    )

    assert models_local.ensure_lmstudio_model_loaded(
        "qwen3.8-27b", BASE_URL, api_key="", target_context_length=None
    ) == 98_304
    assert requests[0][2]["model"] == "qwen/qwen3.8-27b"


def test_exact_instance_with_invalid_context_does_not_use_sibling(monkeypatch):
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [{
            "key": "qwen/qwen3.8-27b",
            "loaded_instances": [
                {"id": "qwen/qwen3.8-27b:2", "config": {"context_length": True}},
                {"id": "qwen/qwen3.8-27b", "config": {"context_length": 131072}},
            ],
        }],
    )

    assert models_local.ensure_lmstudio_model_loaded(
        "qwen/qwen3.8-27b:2", BASE_URL, api_key="", target_context_length=None
    ) is None


def test_duplicate_loaded_slugs_use_the_same_first_match_policy(monkeypatch):
    monkeypatch.setattr(
        models_local,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: [
            {"key": "publisher/qwen3.8-27b", "loaded_instances": [
                {"id": "publisher/qwen3.8-27b", "config": {"context_length": 65536}},
            ]},
            {"key": "other/qwen3.8-27b", "loaded_instances": [
                {"id": "other/qwen3.8-27b", "config": {"context_length": 131072}},
            ]},
        ],
    )

    assert models_local.ensure_lmstudio_model_loaded(
        "qwen3.8-27b", BASE_URL, api_key="", target_context_length=None
    ) == 65536
