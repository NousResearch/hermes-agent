import json

import pytest

from hermes_cli import models


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
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: next(catalogs)
    )
    _capture_load(monkeypatch, {"status": "loaded"})

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=100_000
    )

    assert result == 88_000


def test_load_without_explicit_context_reads_runtime_from_loaded_instance(monkeypatch):
    """When loaded without explicit context, read runtime from now-loaded instance.
    
    Regression test: LM Studio's load response omits `load_config` when no 
    explicit context_length is provided, so we must re-fetch the models list
    to discover the actual runtime context that was applied.
    """
    catalogs = iter([
        _catalog(),  # Before load (no loaded instance)
        _catalog(loaded_context=132_096),  # After load with LM Studio's default
    ])
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: next(catalogs)
    )
    _capture_load(monkeypatch, {"status": "loaded"})

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=None  # No explicit context
    )

    assert result == 132_096


def test_explicit_override_above_known_maximum_rejects_even_when_loaded(monkeypatch):
    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: _catalog(loaded_context=64_000, maximum=128_000),
    )
    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        lambda *_args, **_kwargs: pytest.fail("invalid override must not be posted"),
    )

    result = models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=256_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is False
    assert result.rejected is True
