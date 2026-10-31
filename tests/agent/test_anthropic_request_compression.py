"""Native Anthropic request gzip preserves payloads, SDK defaults and raw fallback."""

import gzip
import json
import re
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from agent import anthropic_adapter as adapter
from agent import anthropic_request_compression as compression
from agent.azure_identity_adapter import build_bearer_http_client

NATIVE = "https://api.anthropic.com/v1/messages"
_REAL_PADDING = compression._padding
PAYLOAD = {"text": "x" * 5000}
RAW = json.dumps(PAYLOAD).encode()
MESSAGE = {
    "id": "msg_test", "type": "message", "role": "assistant", "model": "test",
    "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn",
    "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
}


@pytest.fixture(autouse=True)
def isolated_compression(monkeypatch):
    monkeypatch.setattr(compression, "_latched_off", None)
    monkeypatch.setattr(compression, "_padding", lambda: b"\n \t")
    monkeypatch.setattr(adapter, "_anthropic_sdk", anthropic)
    monkeypatch.setattr(adapter, "_claude_code_version_cache", "0.0.0")
    for name in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "ANTHROPIC_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


def _client(handler):
    return compression.gzip_httpx_client_cls()(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("size", [4096, 8192])
def test_native_messages_are_gzip_and_lossless(size):
    payload = {"text": "x" * (size - 12)}
    raw = json.dumps(payload).encode()
    assert len(raw) == size

    def handler(request):
        assert request.headers["content-encoding"] == "gzip"
        assert int(request.headers["content-length"]) == len(request.content)
        assert gzip.decompress(request.content) == raw + b"\n \t"
        assert json.loads(gzip.decompress(request.content)) == payload
        assert request.extensions["test"] == "preserved"
        return httpx.Response(200)

    with _client(handler) as client:
        client.post(NATIVE + "?beta=true", content=raw, extensions={"test": "preserved"})


def test_wire_body_is_one_member_with_zero_mtime_and_whitespace_pad(monkeypatch):
    monkeypatch.setattr(compression, "_padding", _REAL_PADDING)
    bodies = []

    def handler(request):
        bodies.append(request.content)
        return httpx.Response(200)

    with _client(handler) as client:
        for _ in range(8):
            client.post(NATIVE, content=RAW)
    for body in bodies:
        assert body[:4] == b"\x1f\x8b\x08\x00" and body[4:8] == b"\x00\x00\x00\x00"  # gzip, no flags, mtime=0
        plain = gzip.decompress(body)
        assert plain.startswith(RAW) and re.fullmatch(rb"\n[ \t]{0,256}", plain[len(RAW):])
        assert json.loads(plain) == PAYLOAD
    assert len({len(gzip.decompress(body)) for body in bodies}) > 1  # the pad length varies per request


@pytest.mark.parametrize("method,url,headers,raw", [
    ("POST", NATIVE, {}, b"x" * 4095),
    ("GET", NATIVE, {}, RAW),
    ("POST", "https://api.anthropic.com/v1/models", {}, RAW),
    ("POST", NATIVE, {"Content-Encoding": "identity"}, RAW),
    ("POST", "https://proxy.example/v1/messages", {}, RAW),
    ("POST", "https://api.anthropic.com:8443/v1/messages", {}, RAW),
])
def test_ineligible_requests_stay_raw(method, url, headers, raw):
    def handler(request):
        assert request.content == raw
        assert request.headers.get("content-encoding") == headers.get("Content-Encoding")
        return httpx.Response(200)

    with _client(handler) as client:
        client.request(method, url, headers=headers, content=raw)


@pytest.mark.parametrize("sdk_client", [False, True])
@pytest.mark.parametrize("mounted", [False, True])
def test_compression_follows_selected_transport(sdk_client, mounted):
    captured = []

    def handler(request):
        captured.append(request)
        if mounted:
            assert "content-encoding" not in request.headers
            assert request.content == RAW
        else:
            assert request.headers["content-encoding"] == "gzip"
            assert gzip.decompress(request.content) == RAW + b"\n \t"
        return httpx.Response(200)

    def default_handler(request):
        assert not mounted, "mounted route must bypass the default transport"
        return handler(request)

    cls = compression.anthropic_gzip_client_cls(anthropic) if sdk_client else compression.gzip_httpx_client_cls()
    mounts = {"https://api.anthropic.com": httpx.MockTransport(handler)} if mounted else {}
    with cls(transport=httpx.MockTransport(default_handler), mounts=mounts) as client:
        assert client.send(httpx.Request("POST", NATIVE, content=RAW)).status_code == 200
    assert len(captured) == 1


@pytest.mark.parametrize("base_url", [None, "", "https://api.anthropic.com"])
def test_native_builder_uses_sdk_gzip_client(base_url):
    with adapter.build_anthropic_client("sk-ant-api03-x", base_url) as client:
        assert isinstance(client._client, compression.anthropic_gzip_client_cls(anthropic))
        assert client.with_options(timeout=5)._client is client._client


@pytest.mark.parametrize("base_url", [
    "https://proxy.example/anthropic",
    "https://resource.services.ai.azure.com/anthropic",
    "https://api.minimax.io/anthropic",
    "https://api.kimi.com/coding",
])
@pytest.mark.parametrize("callable_key", [False, True])
def test_other_builders_stay_raw(base_url, callable_key):
    key = (lambda: "sk-ant-api03-x") if callable_key else "sk-ant-api03-x"
    with adapter.build_anthropic_client(key, base_url) as client:
        assert not isinstance(client._client, compression._GzipRequestSend)


@pytest.mark.parametrize("callable_key", [False, True])
@pytest.mark.parametrize("no_proxy", [None, "*.anthropic.com", "api.anthropic.com"])
def test_builder_env_proxy_routing_controls_gzip(monkeypatch, callable_key, no_proxy):
    monkeypatch.setenv("HTTPS_PROXY", "http://p:1")
    if no_proxy is not None:
        monkeypatch.setenv("NO_PROXY", no_proxy)
    captured = []

    def handle_request(transport, request):
        captured.append(request)
        if no_proxy == "api.anthropic.com":
            assert request.headers["content-encoding"] == "gzip"
            assert gzip.decompress(request.content) == RAW + b"\n \t"
        else:
            assert "content-encoding" not in request.headers
            assert request.content == RAW
        return httpx.Response(200)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    key = (lambda: "sk-ant-api03-x") if callable_key else "sk-ant-api03-x"
    with adapter.build_anthropic_client(key) as client:
        assert client._client.send(httpx.Request("POST", NATIVE, content=RAW)).status_code == 200
    assert len(captured) == 1


@pytest.mark.parametrize("callable_key", [False, True])
def test_builder_system_proxy_sends_raw(monkeypatch, callable_key):
    import urllib.request

    import anthropic._utils._httpx as sdk_httpx
    import httpx._utils as httpx_utils

    def system_proxies():
        return {"https": "http://sys:8080"}

    monkeypatch.setattr(urllib.request, "getproxies", system_proxies)
    # Both clients bind getproxies at import time; patch those bindings too.
    monkeypatch.setattr(sdk_httpx, "getproxies", system_proxies)
    monkeypatch.setattr(httpx_utils, "getproxies", system_proxies)
    captured = []

    def handle_request(transport, request):
        captured.append(request)
        assert "content-encoding" not in request.headers
        assert request.content == RAW
        return httpx.Response(200)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    key = (lambda: "sk-ant-api03-x") if callable_key else "sk-ant-api03-x"
    with adapter.build_anthropic_client(key) as client:
        assert client._client.send(httpx.Request("POST", NATIVE, content=RAW)).status_code == 200
    assert len(captured) == 1


def test_custom_ca_disables_builder_gzip(monkeypatch):
    import ssl

    monkeypatch.setenv("SSL_CERT_FILE", ssl.get_default_verify_paths().cafile)
    with adapter.build_anthropic_client("sk-ant-api03-x") as client:
        assert not isinstance(client._client, compression._GzipRequestSend)


@pytest.mark.parametrize("name", ["SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"])
def test_custom_ca_gate_reads_environment_at_call_time(monkeypatch, name):
    assert compression.native_gzip_allowed()
    monkeypatch.setenv(name, "custom-ca")
    assert not compression.native_gzip_allowed()
    monkeypatch.setenv(name, "")
    assert compression.native_gzip_allowed()


def test_no_proxy_keeps_gzip_and_sdk_proxy_mounts(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:8888")
    monkeypatch.setenv("NO_PROXY", "api.anthropic.com")
    with adapter.build_anthropic_client("sk-ant-api03-x") as client, anthropic.DefaultHttpxClient() as plain:
        assert isinstance(client._client, compression.anthropic_gzip_client_cls(anthropic))
        assert {key.pattern for key in client._client._mounts} == {key.pattern for key in plain._mounts}


def _sse_body():
    events = [
        ("message_start", {"message": {**MESSAGE, "content": [], "stop_reason": None}}),
        ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "ok"}}),
        ("content_block_stop", {"index": 0}),
        ("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {}),
    ]
    return "".join(f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n" for kind, data in events).encode()


@pytest.mark.parametrize("mode", ["create", "stream", "create_stream"])
def test_sdk_request_modes_compress_on_the_wire(mode):
    captured = []
    messages = [{"role": "user", "content": "x" * 5000}]

    def handler(request):
        captured.append(request)
        assert request.headers["content-encoding"] == "gzip"
        body = json.loads(gzip.decompress(request.content))
        assert body["messages"] == messages
        assert body.get("stream", False) is (mode != "create")
        if mode == "create":
            return httpx.Response(200, json=MESSAGE)
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=_sse_body())

    cls = compression.anthropic_gzip_client_cls(anthropic)
    with anthropic.Anthropic(api_key="sk-ant-api03-x", max_retries=0,
                             http_client=cls(transport=httpx.MockTransport(handler))) as client:
        kwargs = {"model": "test", "max_tokens": 8, "messages": messages}
        if mode == "create":
            assert client.messages.create(**kwargs).content[0].text == "ok"
        elif mode == "stream":
            with client.messages.stream(**kwargs) as stream:
                assert stream.get_final_message().content[0].text == "ok"
        else:
            with client.messages.create(**kwargs, stream=True) as stream:
                assert [event.type for event in stream][-1] == "message_stop"
    assert len(captured) == 1


def test_rejection_resends_original_once_and_latches_for_other_clients(caplog):
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(415 if len(captured) == 1 else 200)

    original = httpx.Request("POST", NATIVE, content=RAW)
    with _client(handler) as client:
        assert client.send(original).status_code == 200
    assert len(captured) == 2
    assert captured[0].headers["content-encoding"] == "gzip"
    assert captured[1] is original
    assert captured[1].content == RAW
    assert "content-encoding" not in captured[1].headers
    with _client(handler) as other_client:
        other_client.post(NATIVE, content=RAW)
    assert len(captured) == 3
    assert captured[2].content == RAW
    assert "content-encoding" not in captured[2].headers
    assert compression._latched_off == "non_anthropic_rejection"
    compression._latch("unsupported_encoding", 415)
    assert caplog.messages == ["Anthropic request gzip disabled: non_anthropic_rejection (HTTP 415)"]


def test_rejected_raw_resend_is_returned_without_another_attempt():
    captured = []

    def handler(request):
        captured.append(request)
        return httpx.Response(415)

    with _client(handler) as client:
        assert client.post(NATIVE, content=RAW).status_code == 415
    assert [r.headers.get("content-encoding") for r in captured] == ["gzip", None]
    assert compression._latched_off == "non_anthropic_rejection"


@pytest.mark.parametrize("status,headers", [
    (400, {"request-id": "req_x"}), (403, {"cf-ray": "ray-x"}), (500, {}),
])
def test_genuine_errors_are_returned_unchanged(status, headers):
    response = httpx.Response(status, headers=headers, json={"error": {"message": "ordinary API error"}})
    captured = []

    def handler(request):
        captured.append(request)
        return response

    with _client(handler) as client:
        assert client.post(NATIVE, content=RAW) is response
    assert len(captured) == 1
    assert compression._latched_off is None


@pytest.mark.parametrize("status,headers,body,kind", [
    (400, {"request-id": "req_x"},
     b'{"type":"error","error":{"type":"invalid_request_error","message":"The request body is not valid JSON: gzip"}}',
     "invalid_json_at_origin"),
    (400, {"cf-ray": "ray-x"},
     b'{"type":"error","error":{"type":"invalid_request_error",'
     b'"message":"Failed to parse request: could not parse request body as JSON"}}',
     "invalid_json_at_origin"),
    (415, {"request-id": "req_x"}, b'unsupported', "unsupported_encoding"),
])
def test_encoding_rejections_resend_raw(status, headers, body, kind):
    captured = []

    def handler(request):
        captured.append(request)
        if len(captured) == 1:
            return httpx.Response(status, headers=headers, content=body)
        assert request.content == RAW
        assert "content-encoding" not in request.headers
        return httpx.Response(200)

    with _client(handler) as client:
        assert client.post(NATIVE, content=RAW).status_code == 200
    assert len(captured) == 2
    assert compression._latched_off == kind


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("error", [
    "bad json: invalid character 'x' looking for beginning of value",
    "bad json: unexpected EOF",
    "bad json: unexpected end of JSON input",
    "bad json: EOF",
    "bad json: json: cannot unmarshal string into Go value",
    "bad json: unexpected EOF trailing text",
    "bad json: gzip",
])
def test_bad_json_rejection_is_anchored(wrapped, error):
    # A raw body is trimmed at the end only, as Claude Code's trimEnd() does; a wrapped error is not.
    body = json.dumps({"reason": "bad_json", "error": error}).encode() if wrapped else f"{error} \t\n".encode()
    rejected = httpx.Response(400, headers={"request-id": "req_x"}, content=body)
    captured = []
    should_resend = error in {
        "bad json: invalid character 'x' looking for beginning of value",
        "bad json: unexpected EOF", "bad json: unexpected end of JSON input", "bad json: EOF",
    }

    def handler(request):
        captured.append(request)
        return rejected if len(captured) == 1 else httpx.Response(200)

    with _client(handler) as client:
        response = client.post(NATIVE, content=RAW)
    assert len(captured) == (2 if should_resend else 1)
    if should_resend:
        assert response.status_code == 200
        assert captured[1].content == RAW
        assert "content-encoding" not in captured[1].headers
        assert compression._latched_off == "invalid_json_at_origin"
    else:
        assert response is rejected
        assert compression._latched_off is None


# Claude Code 2.1.302 runs its regex on a raw body's trimEnd() and on a bad_json error untrimmed, and
# reads a message only from a full Anthropic error envelope.
@pytest.mark.parametrize("body,rejected", [
    ("bad json: EOF\ufeff\u00a0\n", True),  # trimEnd() strips U+FEFF and NBSP
    (" bad json: EOF", False),  # no leading trim
    ("bad json: EOF\x1c", False),  # Python's rstrip() would strip U+001C; trimEnd() does not
    (json.dumps({"reason": "bad_json", "error": "bad json: EOF\n"}), False),
    (json.dumps({"reason": "bad_json", "error": " bad json: EOF"}), False),
    ('{"error":{"message":"The request body is not valid JSON: gzip"}}', False),  # no envelope type
    ('{"type":"error","error":{"message":"The request body is not valid JSON: gzip"}}', False),
    ('{"type":"error","error":{"type":"invalid_request_error","message":"The request body is not valid JSON"}}', True),
])
def test_invalid_json_classifier_matches_claude_code(body, rejected):
    assert compression._says_invalid_json(body) is rejected


@pytest.mark.parametrize("error", [
    {
        "type": "invalid_request_error",
        "message": 'Unsupported Content-Encoding "zstd": send the request body gzip-compressed or uncompressed.',
        "details": {"error_code": "request_body_encoding_unsupported"},
    },
    {"message": "encoding refused", "details": {"error_code": "request_body_encoding_unsupported"}},
    {"message": 'Unsupported Content-Encoding "gzip"'},
])
def test_origin_400_unsupported_encoding_resends_and_latches(error):
    body = json.dumps({"type": "error", "error": error}).encode()
    captured = []

    def handler(request):
        captured.append(request)
        if len(captured) == 1:
            assert request.headers["content-encoding"] == "gzip"
            return httpx.Response(400, headers={"request-id": "req_x"}, content=body)
        assert request.content == RAW
        assert "content-encoding" not in request.headers
        return httpx.Response(200)

    with _client(handler) as client:
        assert client.post(NATIVE, content=RAW).status_code == 200
        assert len(captured) == 2
        assert compression._latched_off == "unsupported_encoding"
        assert client.post(NATIVE, content=RAW).status_code == 200
    assert len(captured) == 3


def test_rejected_stream_is_read_and_closed_before_resend():
    events = []

    class RejectedStream(httpx.SyncByteStream):
        def __iter__(self):
            events.append("read")
            yield b"unsupported"

        def close(self):
            events.append("closed")

    rejected = httpx.Response(415, stream=RejectedStream())

    def handler(request):
        if not events:
            return rejected
        assert events == ["read", "closed"]
        assert rejected.is_closed
        assert request.content == RAW
        return httpx.Response(200)

    with _client(handler) as client:
        with client.stream("POST", NATIVE, content=RAW) as response:
            assert response.status_code == 200
    assert events == ["read", "closed"]


@pytest.mark.parametrize("as_api_key", [False, True])
def test_bearer_hook_mints_auth_on_compressed_request(as_api_key):
    def handler(request):
        assert request.headers["content-encoding"] == "gzip"
        assert json.loads(gzip.decompress(request.content)) == PAYLOAD
        header = "x-api-key" if as_api_key else "authorization"
        assert request.headers[header] == ("fixture-token" if as_api_key else "Bearer fixture-token")
        return httpx.Response(200)

    provider = MagicMock(return_value="fixture-token")
    with build_bearer_http_client(provider, sends_as_api_key=lambda _: as_api_key,
                                 client_cls=compression.gzip_httpx_client_cls(),
                                 transport=httpx.MockTransport(handler)) as client:
        client.post(NATIVE, content=RAW)
    provider.assert_called_once_with()


def test_native_callable_builder_uses_gzip_client():
    with adapter.build_anthropic_client(lambda: "sk-ant-api03-x") as client:
        assert isinstance(client._client, compression.gzip_httpx_client_cls())


def test_network_errors_propagate_without_resend():
    captured = []

    def handler(request):
        captured.append(request)
        raise httpx.ReadError("fixture reset")

    with _client(handler) as client, pytest.raises(httpx.ReadError, match="fixture reset"):
        client.post(NATIVE, content=RAW)
    assert len(captured) == 1
    assert compression._latched_off is None


def test_mock_sdk_is_not_subclassed():
    assert compression.anthropic_gzip_client_cls(MagicMock()) is None
