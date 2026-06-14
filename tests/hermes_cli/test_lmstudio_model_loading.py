"""Regression tests for LM Studio loaded-instance management."""

from __future__ import annotations

import json

from hermes_cli import models


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return b"{}"


def test_lmstudio_target_alone_with_sufficient_context_is_noop(monkeypatch):
    raw_models = [
        {
            "id": "target-model",
            "type": "llm",
            "max_context_length": 131072,
            "loaded_instances": [
                {
                    "id": "target-instance",
                    "config": {"context_length": 131072},
                }
            ],
        }
    ]

    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: raw_models,
    )

    def unexpected_request(*_args, **_kwargs):
        raise AssertionError("clean LM Studio state must not unload or reload")

    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        unexpected_request,
    )

    loaded = models.ensure_lmstudio_model_loaded(
        "target-model",
        "http://127.0.0.1:12345",
        api_key="lm-secret",
        target_context_length=64000,
    )

    assert loaded == 131072


def test_lmstudio_dirty_state_unloads_all_then_reloads_target(monkeypatch):
    raw_models = [
        {
            "id": "target-model",
            "type": "llm",
            "max_context_length": 131072,
            "loaded_instances": [
                {
                    "id": "target-instance",
                    "config": {"context_length": 131072},
                }
            ],
        },
        {
            "id": "competing-model",
            "type": "llm",
            "max_context_length": 131072,
            "loaded_instances": [
                {
                    "id": "competing-instance",
                    "config": {"context_length": 131072},
                }
            ],
        },
    ]
    requests = []

    monkeypatch.setattr(
        models,
        "_lmstudio_fetch_raw_models",
        lambda **_kwargs: raw_models,
    )

    def fake_open(request, *, timeout):
        requests.append(
            (
                request.get_method(),
                request.full_url,
                json.loads((request.data or b"{}").decode("utf-8")),
                timeout,
            )
        )
        return _Response()

    monkeypatch.setattr(
        models,
        "_urlopen_model_catalog_request",
        fake_open,
    )

    loaded = models.ensure_lmstudio_model_loaded(
        "target-model",
        "http://127.0.0.1:12345",
        api_key="lm-secret",
        target_context_length=64000,
        timeout=15,
    )

    assert loaded == 64000
    assert requests == [
        (
            "POST",
            "http://127.0.0.1:12345/api/v1/models/unload",
            {"instance_id": "target-instance"},
            15,
        ),
        (
            "POST",
            "http://127.0.0.1:12345/api/v1/models/unload",
            {"instance_id": "competing-instance"},
            15,
        ),
        (
            "POST",
            "http://127.0.0.1:12345/api/v1/models/load",
            {"model": "target-model", "context_length": 64000},
            15,
        ),
    ]
