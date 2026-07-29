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


def test_loaded_64k_runtime_is_preserved_without_post(monkeypatch):
    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: _catalog(loaded_context=64_000),
    )
    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        lambda *_args, **_kwargs: pytest.fail("loaded model must not be reloaded"),
    )

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=None
    )

    assert result == 64_000


def test_unloaded_no_override_omits_context_and_requests_echo(monkeypatch):
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: _catalog()
    )
    requests = _capture_load(monkeypatch, {
        "load_config": {"context_length": 96_000},
    })

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=None
    )

    assert result == 96_000
    assert requests[0][2] == {"model": MODEL, "echo_load_config": True}


@pytest.mark.parametrize("requested_context", [32_000, 100_000])
def test_unloaded_explicit_override_sends_exact_context(monkeypatch, requested_context):
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: _catalog()
    )
    requests = _capture_load(monkeypatch, {
        "load_config": {"context_length": requested_context},
    })

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=requested_context
    )

    assert result == requested_context
    assert requests[0][2] == {
        "model": MODEL,
        "context_length": requested_context,
        "echo_load_config": True,
    }


def test_echoed_applied_context_wins_over_requested_context(monkeypatch):
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: _catalog()
    )
    _capture_load(monkeypatch, {"load_config": {"context_length": 96_000}})

    result = models.ensure_lmstudio_model_loaded(
        MODEL, BASE_URL, api_key="", target_context_length=100_000
    )

    assert result == 96_000


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


def test_successful_load_without_verifiable_context_returns_unknown(monkeypatch):
    catalogs = iter([_catalog(), _catalog()])
    monkeypatch.setattr(
        models, "_lmstudio_fetch_raw_models", lambda **_kwargs: next(catalogs)
    )
    _capture_load(monkeypatch, {"status": "loaded"})

    result = models.ensure_lmstudio_model_loaded(
        MODEL,
        BASE_URL,
        api_key="",
        target_context_length=100_000,
        return_load_result=True,
    )

    assert result.context_length is None
    assert result.load_attempted is True
    assert result.rejected is False


def test_explicit_override_above_known_maximum_rejects_without_post(monkeypatch):
    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: _catalog(maximum=128_000),
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
