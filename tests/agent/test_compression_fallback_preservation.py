"""A final fallback error must not weaken conversation preservation (#72637)."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from agent import auxiliary_client as aux
from agent.context_compressor import ContextCompressor
from agent.conversation_compression_diagnostics import _emit_compression_auth_hint


@pytest.fixture
def fallback_case(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("OPENROUTER_API_KEY", "primary-key")
    (tmp_path / "config.yaml").write_text(
        "auxiliary:\n  transient_retries: 0\n  compression:\n    provider: openrouter\n    model: main-B\n"
        "    fallback_chain:\n      - provider: custom\n        model: main-B\n"
        "        base_url: https://fallback.example/v1\n        api_key: fallback-key\n"
    )
    wire = []
    statuses = {"openrouter.ai": 401, "fallback.example": 429}

    def handle(request):
        payload = json.loads(request.content)
        wire.append((request.url.host, payload["model"]))
        status = statuses[request.url.host]
        if status == "network":
            raise httpx.ConnectError("connection refused", request=request)
        if status == "timeout":
            raise httpx.ReadTimeout("request timed out", request=request)
        if status == "invalid":
            # Exercise a non-streaming endpoint's malformed HTTP-200 response.
            if payload.get("stream"):
                return httpx.Response(400, request=request, json={"error": {"message": "stream is not supported"}})
            return httpx.Response(200, request=request, json={
                "id": "bad", "object": "chat.completion", "created": 1, "model": "main-B", "choices": [],
            })
        if status == "overloaded":
            return httpx.Response(429, request=request, json={"error": {"message": "provider overloaded"}})
        return httpx.Response(status, request=request, json={
            "error": {"message": "invalid credential" if status == 401 else "rate limit exceeded"},
        })

    class TransportClient(OpenAI):
        def __init__(self, **kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(handle))
            kwargs["max_retries"] = 0
            super().__init__(**kwargs)

    monkeypatch.setattr(aux, "OpenAI", TransportClient)
    aux._client_cache.clear()
    comp = ContextCompressor(
        model="main-B", provider="custom", base_url="https://main.example/v1", api_key="main-key",
        quiet_mode=True, protect_first_n=2, protect_last_n=2, abort_on_summary_failure=False,
        config_context_length=256_000,
    )
    messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}: " + "detail " * 200}
                for i in range(20)]
    try:
        yield SimpleNamespace(compressor=comp, messages=messages, wire=wire, statuses=statuses)
    finally:
        aux._client_cache.clear()


@pytest.mark.parametrize("primary_status,fallback_status", [
    (401, 429), (429, 401), ("network", 429), ("timeout", 429), (429, "network"),
    ("invalid", 429), ("overloaded", 429), (429, "invalid"), (429, "overloaded"),
])
def test_exhausted_fallback_preserves_transcript_and_reports_final_route(
    fallback_case, primary_status, fallback_status,
):
    case = fallback_case
    case.statuses.update({"openrouter.ai": primary_status, "fallback.example": fallback_status})
    comp = case.compressor
    compressed = comp.compress(case.messages, current_tokens=120_000, force=True)
    assert list(dict.fromkeys(case.wire)) == [("openrouter.ai", "main-B"), ("fallback.example", "main-B")]
    assert compressed == case.messages
    assert comp._last_compress_aborted
    assert comp.compression_count == 0
    expected_class = {401: "auth", 429: "other", "network": "network", "invalid": "other", "overloaded": "other"}[fallback_status]
    assert comp._last_attempt_failure_class == expected_class
    warnings = []
    _emit_compression_auth_hint(SimpleNamespace(context_compressor=comp, _emit_warning=warnings.append))
    rendered = "\n".join(warnings)
    assert "fallback.example" in rendered and "main-B" in rendered
    assert "openrouter.ai" not in rendered
    if fallback_status == 429:
        assert "auth/permission" not in rendered
        assert "could not be reached" not in rendered
    if primary_status in ("network", "timeout"):
        assert comp._last_summary_network_failure


@pytest.mark.parametrize("primary_status,fallback_status", [
    ("overloaded", 429), (429, "overloaded"), ("overloaded", "overloaded"),
])
def test_exhausted_fallback_keeps_overload_grace_and_bounded_escalation(
    fallback_case, primary_status, fallback_status,
):
    case = fallback_case
    case.statuses.update({"openrouter.ai": primary_status, "fallback.example": fallback_status})
    comp = case.compressor
    for attempt in range(1, 4):
        # Retry both routes as their transient quarantine expires between attempts.
        aux._reset_aux_unhealthy_cache()
        case.wire.clear()
        compressed = comp.compress(case.messages, current_tokens=120_000, force=True)
        assert list(dict.fromkeys(case.wire)) == [("openrouter.ai", "main-B"), ("fallback.example", "main-B")]
        if attempt < 3:
            assert compressed == case.messages
            assert comp._last_compress_aborted
            assert comp.compression_count == 0
            assert comp._consecutive_overload_aborts == attempt
        else:
            assert compressed != case.messages
            assert not comp._last_compress_aborted
            assert comp._last_summary_fallback_used
            assert comp._last_summary_dropped_count > 0
            assert comp._last_compression_telemetry["failure_class"] == "summary_overload_degraded"
