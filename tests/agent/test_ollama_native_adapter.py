"""Tests for agent/ollama_native_adapter.py — the native Ollama /api/chat adapter.

Covers request/response/stream translation, the probe-based Ollama detection,
and the OpenAI-SDK-shaped client facade. Network is faked with
httpx.MockTransport + unittest.mock (no extra test dependency).
"""

import json
from unittest.mock import patch

import httpx
import pytest

from agent.ollama_native_adapter import (
    _OLLAMA_PROBE_CACHE,
    OllamaAPIError,
    OllamaNativeClient,
    build_ollama_request,
    is_native_ollama_base_url,
    native_root,
    translate_ollama_response,
    translate_stream_line,
)


def _mock_client(handler):
    """A real OllamaNativeClient wired to an httpx.MockTransport request handler."""
    return httpx.Client(transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _clear_probe_caches():
    """Both probes memoize process-globally (endpoint detection per root, thinking
    capability per (base_url, model)); clear them around every test so cases don't
    leak cached verdicts into each other — including when an assertion fails
    mid-test (an inline trailing clear() would be skipped)."""
    from agent.ollama_native_adapter import _THINKING_CAP_CACHE

    _OLLAMA_PROBE_CACHE.clear()
    _THINKING_CAP_CACHE.clear()
    yield
    _OLLAMA_PROBE_CACHE.clear()
    _THINKING_CAP_CACHE.clear()


# ── request construction ──────────────────────────────────────────────────────


def test_num_ctx_preserved_in_native_payload():
    req = build_ollama_request(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        options={"num_ctx": 64000},
    )
    assert req["options"]["num_ctx"] == 64000
    assert req["stream"] is False


def test_sampling_merges_without_clobbering_options():
    req = build_ollama_request(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.5,
        top_p=0.9,
        max_tokens=128,
        seed=7,
        stop="STOP",
        options={"num_ctx": 8192, "temperature": 0.1},
    )
    opts = req["options"]
    assert opts["num_ctx"] == 8192
    assert opts["temperature"] == 0.1  # explicit option wins over the 0.5 arg
    assert opts["top_p"] == 0.9
    assert opts["num_predict"] == 128  # max_tokens -> num_predict
    assert opts["seed"] == 7
    assert opts["stop"] == ["STOP"]


def test_assistant_tool_call_args_string_to_object_and_tool_role():
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "function": {"name": "lookup", "arguments": '{"q": "cats"}'},
                    }
                ],
            },
            {"role": "tool", "name": "lookup", "content": "result"},
        ],
    )
    assert req["messages"][0]["tool_calls"][0]["function"]["arguments"] == {"q": "cats"}
    assert req["messages"][1]["tool_name"] == "lookup"


def test_multimodal_image_data_url_extracted():
    req = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,QUJD"},
                    },
                ],
            }
        ],
    )
    assert req["messages"][0]["content"] == "what is this"
    assert req["messages"][0]["images"] == ["QUJD"]


# ── response + stream + embeddings translation ────────────────────────────────


def test_response_content_usage_and_finish_reason():
    resp = translate_ollama_response(
        {
            "model": "m",
            "message": {"role": "assistant", "content": "hello"},
            "done_reason": "stop",
            "prompt_eval_count": 10,
            "eval_count": 5,
        },
        model="m",
    )
    assert resp.choices[0].message.content == "hello"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage.prompt_tokens == 10
    assert resp.usage.completion_tokens == 5
    assert resp.usage.total_tokens == 15


def test_response_tool_calls_object_to_string():
    resp = translate_ollama_response(
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            }
        },
        model="m",
    )
    msg = resp.choices[0].message
    assert resp.choices[0].finish_reason == "tool_calls"
    assert msg.content is None
    assert msg.tool_calls[0].function.name == "f"
    assert json.loads(msg.tool_calls[0].function.arguments) == {"a": 1}


def test_stream_content_tool_call_and_done():
    tool_idx = {"n": 0}
    c1 = translate_stream_line({"message": {"content": "hi"}}, "m", tool_idx)
    assert c1[0].choices[0].delta.content == "hi"

    c2 = translate_stream_line(
        {
            "message": {
                "tool_calls": [{"function": {"name": "f", "arguments": {"x": 1}}}]
            }
        },
        "m",
        tool_idx,
    )
    tc = c2[0].choices[0].delta.tool_calls[0]
    assert tc.function.name == "f"
    assert json.loads(tc.function.arguments) == {"x": 1}
    assert tc.index == 0
    assert tool_idx["n"] == 1

    # A tool call was streamed, so the terminal chunk's finish_reason must be
    # "tool_calls" even though Ollama reports done_reason="stop".
    c3 = translate_stream_line(
        {"done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2},
        "m",
        tool_idx,
    )
    assert c3[-1].choices[0].finish_reason == "tool_calls"
    assert c3[-1].usage.prompt_tokens == 3


def test_stream_done_without_tool_calls_is_stop():
    # Pure content stream (fresh counter, no tool calls) → finish_reason "stop".
    chunks = translate_stream_line(
        {"done": True, "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 1},
        "m",
        {"n": 0},
    )
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].usage.completion_tokens == 1


def test_stream_error_frame_raises_with_message():
    # Ollama reports mid-stream failures (runner OOM) as an in-band
    # {"error": "<str>"} frame on the already-200 stream — it must surface the
    # actual message, not silently end the stream looking like a clean close.
    with pytest.raises(OllamaAPIError) as exc_info:
        translate_stream_line(
            {"error": "an error was encountered while running the model: CUDA out of memory"},
            "m",
            {"n": 0},
        )
    assert "CUDA out of memory" in str(exc_info.value)


def test_stream_transport_errors_propagate_unwrapped():
    # Transport failures must keep their concrete httpx types: the streaming
    # worker's mid-stream reconnect classifies on isinstance(e, httpx.ConnectError/
    # RemoteProtocolError/...) — a wrapped OllamaAPIError would disable reconnects.
    def handler(request):
        raise httpx.ConnectError("peer went away")

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    with pytest.raises(httpx.ConnectError):
        stream = client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], stream=True
        )
        list(stream)


def test_remote_image_url_degrades_to_text_marker():
    # Native /api/chat only accepts base64 images; a remote http(s) URL can't be
    # forwarded. It must degrade visibly (text marker), never vanish silently.
    body = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this? "},
                    {"type": "image_url", "image_url": {"url": "https://x.test/cat.png"}},
                ],
            }
        ],
    )
    msg = body["messages"][0]
    assert "[Image: https://x.test/cat.png]" in msg["content"]
    assert not msg.get("images")


def test_default_query_attached_to_requests():
    # agent_init moves a query-bearing base_url's params into default_query (URL-token
    # auth, api-version); the native client must send them on every request like the
    # stock OpenAI client does.
    captured = {}

    def handler(request):
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(
        base_url="http://h:11434",
        default_query={"access_token": "sekrit"},
        http_client=_mock_client(handler),
    )
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert captured["params"] == {"access_token": "sekrit"}


# ── detection ─────────────────────────────────────────────────────────────────


def test_native_root_strips_v1():
    assert native_root("http://h:11434/v1") == "http://h:11434"
    assert native_root("http://h:11434/v1/") == "http://h:11434"
    assert native_root("http://h:11434/") == "http://h:11434"


def test_detection_is_noop_for_empty_base_url():
    _OLLAMA_PROBE_CACHE.clear()
    # An empty base_url short-circuits to False without touching the network.
    with patch("agent.ollama_native_adapter.httpx.get") as get:
        assert is_native_ollama_base_url("") is False
        assert get.call_count == 0


def test_detection_positive_for_ollama():
    # Detection is probe-only (no feature flag): a positive /api/version identifies
    # Ollama and routes it to native mode, mirroring the Gemini adapter's always-on
    # selection.
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"version": "0.30.0"})
    with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
        assert is_native_ollama_base_url("http://h:11434/v1") is True


def test_detection_negative_for_non_ollama():
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(404)
    with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
        # A non-Ollama "custom" endpoint (vLLM/llama.cpp/LM Studio) is left on /v1.
        assert is_native_ollama_base_url("http://other:8000/v1") is False


def test_detection_rejects_200_without_version_field():
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"status": "ok"})
    with patch("agent.ollama_native_adapter.httpx.get", return_value=resp):
        assert is_native_ollama_base_url("http://h:11434") is False


# ── client facade (httpx.MockTransport) ───────────────────────────────────────


def test_chat_completion_posts_num_ctx_to_api_chat():
    captured = {}

    def handler(request):
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": "gemma4:e2b",
                "message": {"role": "assistant", "content": "hello"},
                "done_reason": "stop",
                "prompt_eval_count": 11,
                "eval_count": 3,
            },
        )

    client = OllamaNativeClient(
        base_url="http://h:11434/v1", http_client=_mock_client(handler)
    )
    result = client.chat.completions.create(
        model="gemma4:e2b",
        messages=[{"role": "user", "content": "hi"}],
        extra_body={"options": {"num_ctx": 64000}},
    )
    assert captured["url"].endswith("/api/chat")
    assert captured["body"]["options"]["num_ctx"] == 64000
    assert result.choices[0].message.content == "hello"
    assert result.usage.prompt_tokens == 11


def test_streaming_yields_openai_chunks():
    ndjson = (
        '{"message":{"content":"he"}}\n'
        '{"message":{"content":"llo"}}\n'
        '{"done":true,"done_reason":"stop","prompt_eval_count":5,"eval_count":2}\n'
    )
    client = OllamaNativeClient(
        base_url="http://h:11434",
        http_client=_mock_client(lambda request: httpx.Response(200, text=ndjson)),
    )
    chunks = list(
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "hi"}], stream=True
        )
    )
    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


def test_headers_drop_non_string_sentinels():
    class _Sentinel:
        pass

    headers = OllamaNativeClient(
        base_url="http://h:11434",
        default_headers={
            "X-Real": "keep",
            "OpenAI-Organization": _Sentinel(),
            "X-None": None,
        },
    )._headers()
    assert headers["X-Real"] == "keep"
    assert "OpenAI-Organization" not in headers
    assert "X-None" not in headers
    assert all(isinstance(v, (str, bytes)) for v in headers.values())


def test_version_probe_sends_auth_headers_and_verify():
    # Behind an authenticated reverse proxy / private CA, an unconfigured probe
    # would 401 or SSLError and native routing would silently never engage — the
    # probe must carry the caller's auth material and TLS trust.
    _OLLAMA_PROBE_CACHE.clear()
    resp = httpx.Response(200, json={"version": "0.32.5"})
    with patch("agent.ollama_native_adapter.httpx.get", return_value=resp) as get:
        assert (
            is_native_ollama_base_url(
                "http://h:11434/v1",
                api_key="proxy-secret",
                default_headers={"CF-Access-Client-Id": "cid"},
                default_query={"access_token": "tok"},
                verify="/etc/ssl/corp-ca.pem",
            )
            is True
        )
    kwargs = get.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer proxy-secret"
    assert kwargs["headers"]["CF-Access-Client-Id"] == "cid"
    assert kwargs["params"] == {"access_token": "tok"}
    assert kwargs["verify"] == "/etc/ssl/corp-ca.pem"
    _OLLAMA_PROBE_CACHE.clear()


def test_probe_default_headers_override_api_key_authorization():
    # _headers() lets default_headers override the api_key-derived Authorization,
    # so the probe must too — a placeholder api_key with real gateway auth in
    # extra_headers must probe with the credentials real requests will send.
    resp = httpx.Response(200, json={"version": "0.32.5"})
    with patch("agent.ollama_native_adapter.httpx.get", return_value=resp) as get:
        assert (
            is_native_ollama_base_url(
                "http://h:11434/v1",
                api_key="not-needed",
                default_headers={"Authorization": "Bearer real-gateway-token"},
            )
            is True
        )
    assert get.call_args.kwargs["headers"]["Authorization"] == "Bearer real-gateway-token"


def test_uppercase_data_scheme_still_decodes():
    # RFC 2397 scheme names are case-insensitive: an uppercase DATA: URL must be
    # decoded as an image (and a malformed uppercase one dropped), never routed
    # to the remote-URL marker path where its base64 would leak into the prompt.
    body = build_ollama_request(
        model="m",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "DATA:image/png;base64,QUJD"}},
                    {"type": "image_url", "image_url": {"url": "Data:image/png;base64" + "A" * 500}},
                ],
            }
        ],
    )
    msg = body["messages"][0]
    assert msg.get("images") == ["QUJD"]
    assert "[Image:" not in msg["content"] and "A" * 100 not in msg["content"]


def test_http_error_empty_error_string_keeps_raw_body():
    # {"error": ""} must not produce a bare-trailing-colon message — keep the
    # raw body as the diagnostic.
    from agent.ollama_native_adapter import ollama_http_error

    resp = httpx.Response(500, content=b'{"error": ""}')
    msg = str(ollama_http_error(resp))
    assert not msg.rstrip().endswith(":")
    assert '{"error": ""}' in msg


def _custom_profile():
    """Resolve the registered "custom" provider profile (the one create_openai_client's
    routing pairs with this adapter), via the same discovery path production uses."""
    import model_tools  # noqa: F401  (import triggers plugin discovery/registration)
    import providers

    profile = providers.get_provider_profile("custom")
    assert profile is not None, "custom provider profile must be registered"
    return profile


def _capture_client(capture, *, thinking_capable=True):
    caps = ["completion", "thinking"] if thinking_capable else ["completion"]

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": caps})
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    return OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))


def test_profile_reasoning_level_reaches_native_think():
    # The custom profile emits an enabled reasoning level as TOP-LEVEL
    # reasoning_effort (the field /v1 honors), not extra_body.think. Feed the
    # profile's actual output through the native client and assert the level
    # reaches the wire as `think` — the full profile-to-request path, not a
    # hand-built extra_body.
    profile = _custom_profile()
    extra_body, top_level = profile.build_api_kwargs_extras(
        reasoning_config={"enabled": True, "effort": "high"}, ollama_num_ctx=8192
    )
    assert top_level.get("reasoning_effort") == "high"  # the profile's contract
    assert "think" not in extra_body

    capture = {}
    client = _capture_client(capture)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], extra_body=extra_body, **top_level
    )
    assert capture["body"]["think"] == "high"
    assert capture["body"]["options"]["num_ctx"] == 8192


def test_profile_reasoning_disabled_reaches_native_think_false():
    # Disabled reasoning: the profile pairs reasoning_effort="none" with
    # extra_body.think=False — the wire request must carry think=False.
    profile = _custom_profile()
    extra_body, top_level = profile.build_api_kwargs_extras(reasoning_config={"effort": "none"})
    capture = {}
    client = _capture_client(capture)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], extra_body=extra_body, **top_level
    )
    assert capture["body"]["think"] is False


def test_explicit_think_wins_over_reasoning_effort():
    # An explicit extra_body.think takes precedence over the top-level field.
    capture = {}
    client = _capture_client(capture)
    client.chat.completions.create(
        model="m",
        messages=[{"role": "user", "content": "x"}],
        reasoning_effort="high",
        extra_body={"think": "low"},
    )
    assert capture["body"]["think"] == "low"


def test_version_probe_positive_expires_and_redetects(monkeypatch):
    """Positives are bounded by _VERSION_PROBE_TTL (the model_metadata precedent):
    a server swap on the same port re-detects within the TTL instead of hard-404ing
    every turn until process restart."""
    import time as real_time
    from types import SimpleNamespace

    from agent import ollama_native_adapter as adapter

    _OLLAMA_PROBE_CACHE.clear()
    now = {"t": 5000.0}
    monkeypatch.setattr(
        adapter, "time", SimpleNamespace(monotonic=lambda: now["t"], time=real_time.time)
    )
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"version": "0.32.5"})  # Ollama
        return httpx.Response(404)  # server swapped (vLLM on the same port)

    with patch("agent.ollama_native_adapter.httpx.get", side_effect=fake_get):
        assert is_native_ollama_base_url("http://h:11434/v1") is True
        assert is_native_ollama_base_url("http://h:11434/v1") is True  # cached
        assert calls["n"] == 1
        now["t"] += adapter._VERSION_PROBE_TTL + 1
        assert is_native_ollama_base_url("http://h:11434/v1") is False  # re-detected
        assert calls["n"] == 2
    _OLLAMA_PROBE_CACHE.clear()


# ── thinking-capability gate (/api/show) ──────────────────────────────────────


def _cap_handler(caps, capture):
    """Route /api/show (returns given capabilities) and /api/chat (captures body)."""

    def handler(request):
        if request.url.path == "/api/show":
            body = {"capabilities": caps} if caps is not None else {}
            return httpx.Response(200, json=body)
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    return handler


def _fresh_cap_client(caps, capture):
    return OllamaNativeClient(
        base_url="http://h:11434", http_client=_mock_client(_cap_handler(caps, capture))
    )


def test_think_dropped_for_non_thinking_model():
    """Newer Ollama 400s on `think` for models without the capability — never send it."""
    capture = {}
    client = _fresh_cap_client(["completion", "tools"], capture)
    client.chat.completions.create(
        model="llama3.2", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    assert "think" not in capture["body"]


def test_think_kept_for_thinking_model():
    capture = {}
    client = _fresh_cap_client(["completion", "tools", "thinking"], capture)
    client.chat.completions.create(
        model="qwen3.5", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    assert capture["body"]["think"] is False


def test_think_string_level_passes_through():
    """Newer Ollama accepts think as a string level ("low"/"medium"/"high"/"max");
    the adapter forwards the value verbatim (no bool coercion)."""
    capture = {}
    client = _fresh_cap_client(["completion", "thinking"], capture)
    client.chat.completions.create(
        model="gpt-oss", messages=[{"role": "user", "content": "x"}], extra_body={"think": "high"}
    )
    assert capture["body"]["think"] == "high"


def test_think_kept_when_capabilities_absent():
    """Pre-capability Ollama (no `capabilities` key) accepts think for every model."""
    capture = {}
    client = _fresh_cap_client(None, capture)
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": True}
    )
    assert capture["body"]["think"] is True


def test_no_probe_when_think_not_requested():
    seen = {"show": 0}

    def handler(request):
        if request.url.path == "/api/show":
            seen["show"] += 1
            return httpx.Response(200, json={"capabilities": []})
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
    assert seen["show"] == 0  # zero overhead unless `think` is present


def test_think_kept_when_probe_fails_and_failure_is_ttl_cached():
    """Fail-open: uncertainty never suppresses think — and the failure is TTL-cached
    so a flaky /api/show can't add a round trip to every chat call."""
    seen = {"show": 0}
    capture = {}

    def handler(request):
        if request.url.path == "/api/show":
            seen["show"] += 1
            raise httpx.ConnectError("down")
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
        )
    assert capture["body"]["think"] is False
    assert seen["show"] == 1


def test_probe_result_is_cached_per_model():
    seen = {"show": 0}

    def handler(request):
        if request.url.path == "/api/show":
            seen["show"] += 1
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
        )
    assert seen["show"] == 1  # definitive answer memoized


# ── error-body parsing + parallel-frame tool calls ────────────────────────────


def test_error_body_message_is_parsed():
    """Ollama reports failures as {"error": "<str>"} — surface the message itself."""

    def handler(request):
        return httpx.Response(400, json={"error": "model does not support thinking"})

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    try:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        raise AssertionError("expected OllamaAPIError")
    except OllamaAPIError as e:
        assert e.status_code == 400
        assert "model does not support thinking" in str(e)
        assert '{"error"' not in str(e)  # message, not the raw JSON blob


def test_parallel_tool_calls_in_separate_frames_get_distinct_indices():
    """Ollama can emit parallel calls in separate NDJSON frames with no ids; a
    per-frame index would merge them into one unparseable call. The counter is
    stream-global, so each call gets a distinct index and synthesized id."""
    tool_idx = {"n": 0}
    f1 = translate_stream_line(
        {"message": {"tool_calls": [{"function": {"name": "a", "arguments": {}}}]}}, "m", tool_idx
    )
    f2 = translate_stream_line(
        {"message": {"tool_calls": [{"function": {"name": "b", "arguments": {}}}]}}, "m", tool_idx
    )
    tc1 = f1[0].choices[0].delta.tool_calls[0]
    tc2 = f2[0].choices[0].delta.tool_calls[0]
    assert (tc1.index, tc2.index) == (0, 1)
    assert tc1.id and tc2.id and tc1.id != tc2.id


def test_error_body_non_json_falls_back_to_raw_text():
    """A plain-text error (overloaded proxy, HTML gateway page) must surface its
    raw text, not "<unreadable>"."""

    def handler(request):
        return httpx.Response(502, text="upstream proxy exploded")

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    try:
        client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}])
        raise AssertionError("expected OllamaAPIError")
    except OllamaAPIError as e:
        assert e.status_code == 502
        assert "upstream proxy exploded" in str(e)
        assert "<unreadable>" not in str(e)


def test_probe_sends_auth_headers():
    """Behind an authenticated reverse proxy an unauthenticated probe would 401
    and silently disable the gate — the probe must send the client's headers."""
    seen = {}

    def handler(request):
        if request.url.path == "/api/show":
            seen["auth"] = request.headers.get("Authorization")
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(
        base_url="http://h:11434", api_key="proxy-secret", http_client=_mock_client(handler)
    )
    client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False}
    )
    assert seen["auth"] == "Bearer proxy-secret"


def test_think_kept_on_non_200_probe_and_bounded():
    """A proxy that 404s /api/show while /api/chat works must not disable chat —
    think is forwarded (fail-open) and the failure is TTL-cached, not re-probed
    on every request."""
    seen = {"show": 0}
    capture = {}

    def handler(request):
        if request.url.path == "/api/show":
            seen["show"] += 1
            return httpx.Response(404)
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    for _ in range(3):
        client.chat.completions.create(
            model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": True}
        )
    assert capture["body"]["think"] is True
    assert seen["show"] == 1


def test_failure_ttl_expires_then_recovers_to_definitive(monkeypatch):
    """The load-bearing TTL contract: a cached failure is re-probed after the TTL
    lapses, and a now-healthy server yields a definitive answer that is memoized
    (and, for a non-thinking model, finally suppresses `think`)."""
    from types import SimpleNamespace

    from agent import ollama_native_adapter as adapter

    import time as real_time

    now = {"t": 1000.0}
    monkeypatch.setattr(
        adapter, "time", SimpleNamespace(monotonic=lambda: now["t"], time=real_time.time)
    )

    seen = {"show": 0}
    capture = {}

    def handler(request):
        if request.url.path == "/api/show":
            seen["show"] += 1
            if seen["show"] == 1:
                raise httpx.ConnectError("down")
            return httpx.Response(200, json={"capabilities": ["completion"]})
        capture["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "ok"}, "done_reason": "stop"}
        )

    client = OllamaNativeClient(base_url="http://h:11434", http_client=_mock_client(handler))
    kw = dict(model="m", messages=[{"role": "user", "content": "x"}], extra_body={"think": False})

    client.chat.completions.create(**kw)  # probe fails -> fail-open, TTL-cached
    assert capture["body"]["think"] is False
    client.chat.completions.create(**kw)  # within TTL -> cached, no re-probe
    assert seen["show"] == 1

    now["t"] += adapter._NEG_PROBE_TTL + 1  # TTL lapses
    client.chat.completions.create(**kw)  # re-probe -> definitive: no thinking cap
    assert seen["show"] == 2
    assert "think" not in capture["body"]
    assert adapter._THINKING_CAP_CACHE[("http://h:11434", "m")] == (False, None)

    client.chat.completions.create(**kw)  # memoized -> no further probes
    assert seen["show"] == 2
