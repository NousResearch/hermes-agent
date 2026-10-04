"""Final-wire HTTPX physical admission (cells 1-4, 7, 13).

Inert MockTransport only. No provider network, no 180s waits, no credentials.
Physical boundary is selected transport.handle_request after SDK merge.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from openai import OpenAI

from agent.conversation_compression import ProviderBoundRequestOverLimit
from agent.final_wire_admission import (
    COVERED_MAIN,
    NOT_COVERED_AUXILIARY,
    FinalAttemptIdentity,
    ProviderBoundUnsupportedAccounting,
    bind_attempt_identity,
    build_covered_keepalive_http_client,
    unwrap_local_refusal,
    wrap_httpx_client_transports,
)


WINDOW = 1000


def _chat_ok_json():
    return {
        "id": "inert",
        "object": "chat.completion",
        "created": 0,
        "model": "test/model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "inert"},
                "finish_reason": "stop",
            }
        ],
    }


class CountingTransport(httpx.BaseTransport):
    def __init__(self, inner: httpx.BaseTransport):
        self.inner = inner
        self.calls = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.inner.handle_request(request)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if callable(close):
            close()


def _mock_inner(handler=None):
    def receive(request):
        if handler is not None:
            return handler(request)
        return httpx.Response(200, json=_chat_ok_json())

    return httpx.MockTransport(receive)


def _identity(**overrides):
    values = dict(
        purpose=COVERED_MAIN,
        family="chat_completions",
        model="test/model",
        endpoint="https://inert.invalid/v1/chat/completions",
        window=WINDOW,
        correlation_id="attempt-1",
    )
    values.update(overrides)
    return FinalAttemptIdentity(**values)


def _openai_over_client(transport, *, max_retries=0):
    return OpenAI(
        api_key="inert-not-a-secret",
        base_url="https://inert.invalid/v1",
        max_retries=max_retries,
        http_client=httpx.Client(transport=transport),
    )


def test_covered_httpx_without_attempt_identity_fails_closed():
    inner = CountingTransport(_mock_inner())
    client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(client, covered=True)
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises((ProviderBoundUnsupportedAccounting, ProviderBoundRequestOverLimit)):
        client.post("https://inert.invalid/v1/chat/completions", json=body)
    assert inner.calls == []
    client.close()


def test_safe_final_httpx_delegate_exactly_once():
    inner = CountingTransport(_mock_inner())
    client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(client, covered=True)
    body = {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]}
    with bind_attempt_identity(_identity()):
        response = client.post("https://inert.invalid/v1/chat/completions", json=body)
    assert response.status_code == 200
    assert len(inner.calls) == 1
    client.close()


def test_over_limit_final_httpx_delegate_zero():
    inner = CountingTransport(_mock_inner())
    client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(client, covered=True)
    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "b" * 4400}],
    }
    with bind_attempt_identity(_identity()):
        with pytest.raises(ProviderBoundRequestOverLimit):
            client.post("https://inert.invalid/v1/chat/completions", json=body)
    assert inner.calls == []
    client.close()


def test_all_selected_httpx_mounts_and_default_transport_are_guarded():
    https_inner = CountingTransport(_mock_inner())
    http_inner = CountingTransport(_mock_inner())
    default_inner = CountingTransport(_mock_inner())
    client = httpx.Client(
        transport=default_inner,
        mounts={"http://": http_inner, "https://": https_inner},
    )
    wrap_httpx_client_transports(client, covered=True)
    over = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "b" * 4400}],
    }
    with bind_attempt_identity(_identity()):
        with pytest.raises(ProviderBoundRequestOverLimit):
            client.post("https://inert.invalid/v1/chat/completions", json=over)
        with pytest.raises(ProviderBoundRequestOverLimit):
            client.post("http://inert.invalid/v1/chat/completions", json=over)
    assert https_inner.calls == []
    assert http_inner.calls == []
    assert default_inner.calls == []
    client.close()


def test_covered_factory_none_cannot_bypass_guard():
    with patch("httpx.Client", side_effect=RuntimeError("construction failed")):
        with pytest.raises((ProviderBoundUnsupportedAccounting, RuntimeError)):
            client = build_covered_keepalive_http_client("https://inert.invalid/v1")
            assert client is not None


def test_covered_none_fallback_is_not_silent():
    result = build_covered_keepalive_http_client("https://inert.invalid/v1")
    assert result is not None
    assert isinstance(result, httpx.Client)
    transports = [result._transport, *list(result._mounts.values())]
    guarded = []
    for t in transports:
        if t is None:
            continue
        if getattr(type(t), "__module__", "").endswith("final_wire_admission"):
            guarded.append(t)
    assert guarded, "COVERED_MAIN keepalive client must wrap selected transports"
    result.close()


def test_sdk_extra_body_replacement_counted_once_at_httpx(monkeypatch):
    inner_calls = []

    def receive(request):
        inner_calls.append(json.loads(request.content))
        return httpx.Response(200, json=_chat_ok_json())

    inner = CountingTransport(httpx.MockTransport(receive))
    http_client = httpx.Client(transport=inner)
    wrap_httpx_client_transports(http_client, covered=True)
    sdk = OpenAI(
        api_key="inert-not-a-secret",
        base_url="https://inert.invalid/v1",
        max_retries=0,
        http_client=http_client,
    )
    payload = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_body": {"messages": [{"role": "user", "content": "b" * 4400}]},
    }
    with bind_attempt_identity(_identity()):
        try:
            sdk.chat.completions.create(**payload)
        except Exception as exc:
            refusal = unwrap_local_refusal(exc)
            assert isinstance(refusal, ProviderBoundRequestOverLimit), exc
        else:
            raise AssertionError("over-limit extra_body replacement was admitted")
    sdk.close()
    assert inner.calls == []
    assert inner_calls == []


def test_classified_transport_metadata_excluded_from_pressure():
    from agent.final_wire_admission import estimate_final_httpx_pressure

    body = {
        "model": "test/model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.2,
        "stream": False,
        "store": True,
    }
    pressure = estimate_final_httpx_pressure(
        body,
        identity=_identity(),
        headers={"Authorization": "Bearer secret-token", "Content-Type": "application/json"},
    )
    tiny = estimate_final_httpx_pressure(
        {"model": "test/model", "messages": [{"role": "user", "content": "hi"}]},
        identity=_identity(),
        headers={"Content-Type": "application/json"},
    )
    assert pressure == tiny
    assert pressure < 50


def test_auxiliary_purpose_is_not_covered_and_does_not_require_identity():
    from agent.process_bootstrap import build_keepalive_http_client

    aux = build_keepalive_http_client("https://inert.invalid/v1")
    assert aux is None or isinstance(aux, httpx.Client)
    if aux is not None:
        aux.close()
    assert NOT_COVERED_AUXILIARY != COVERED_MAIN
