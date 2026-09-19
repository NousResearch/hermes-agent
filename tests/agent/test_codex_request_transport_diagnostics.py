"""Diagnostics for Codex Responses transport failures."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from agent.codex_runtime import _codex_request_failure_details, run_codex_stream


def test_transport_failure_without_attached_request_reports_unknown_size():
    error = httpx.RemoteProtocolError("connection closed")

    request_body_bytes, exception_chain = _codex_request_failure_details(error)

    assert request_body_bytes is None
    assert exception_chain == "RemoteProtocolError"


def test_transport_failure_logs_exact_request_bytes_and_class_chain(caplog):
    request_content = b'{"input":"payload"}'
    request = httpx.Request(
        "POST",
        "https://example.invalid/responses",
        content=request_content,
    )
    transport_error = httpx.RemoteProtocolError(
        "server disconnected without sending a response",
        request=request,
    )
    connection_error = APIConnectionError(request=request)
    connection_error.__cause__ = transport_error

    class FailingResponses:
        def create(self, **_kwargs):
            raise connection_error

    client = SimpleNamespace(responses=FailingResponses())
    agent = SimpleNamespace(
        _interrupt_requested=False,
        _current_api_request_id="request-id",
        _fallback_index=0,
        is_subagent=False,
        model="gpt-5.6-sol",
        provider="openai-codex",
        session_id="",
        _client_log_context=lambda: "",
    )

    with caplog.at_level(logging.WARNING, logger="agent.codex_runtime"):
        with pytest.raises(APIConnectionError):
            run_codex_stream(agent, {"model": "gpt-5.6-sol"}, client=client)

    message = caplog.messages[-1]
    assert f"serialized_request_body_bytes={len(request_content)}" in message
    assert "stream_opened=false" in message
    assert "exception_chain=APIConnectionError <- RemoteProtocolError" in message
    assert "attempt=2/2" in message
    assert "payload" not in message
    assert request_content.decode() not in message
    assert "example.invalid" not in message
