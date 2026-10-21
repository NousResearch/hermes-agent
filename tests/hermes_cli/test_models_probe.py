"""The ``/models`` discovery probe sends the credential the route's inference client sends."""

from __future__ import annotations

import pytest

from hermes_cli import models


@pytest.fixture
def sent(monkeypatch):
    def get_json(url, timeout, headers=None, **kwargs):
        sent.append({k.lower(): v for k, v in (headers or {}).items()})
        return {"data": [{"id": "m"}]}

    sent = []
    monkeypatch.setattr(models, "_get_json", get_json)
    return sent


def test_bearer_anthropic_host_probe_sends_bearer_and_keeps_anthropic_version(sent):
    models.probe_api_models("k", "https://api.minimax.io/anthropic", api_mode="anthropic_messages")
    assert sent[0]["authorization"] == "Bearer k" and "x-api-key" not in sent[0]
    assert sent[0]["anthropic-version"] == "2023-06-01"


def test_other_anthropic_host_probe_keeps_x_api_key(sent):
    models.probe_api_models("k", "https://anthropic-compatible.invalid", api_mode="anthropic_messages")
    assert sent[0]["x-api-key"] == "k" and "authorization" not in sent[0]
    assert sent[0]["anthropic-version"] == "2023-06-01"


def test_command_key_source_is_minted_before_the_probe(sent, monkeypatch):
    from agent import command_token_source

    monkeypatch.setattr(command_token_source, "_mint", lambda command, label: ("minted", 3600))
    source = command_token_source.build_command_token_provider("helper", "gateway")
    models.probe_api_models(source, "https://gateway.invalid/v1")
    assert sent[0]["authorization"] == "Bearer minted"
