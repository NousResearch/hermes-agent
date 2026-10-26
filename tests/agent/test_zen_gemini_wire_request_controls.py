"""Native Gemini request controls on the OpenCode Zen relay: headers, query, timeout, key advice.

Follow-up to the Zen native-route work: the Go-side implementation of the same relay wire (#139539)
found these defects in the shared ``GeminiNativeClient`` and fixed them there. They are not Go
specific — the relay path in this branch has the same request construction — so the contracts are
pinned here for the Zen route: caller headers must not duplicate on a case difference, repeated and
blank endpoint query options must survive, a configured timeout must reach the request even when a
keepalive ``http_client`` is injected, and a relay error must not carry Google credential advice.
"""

import json

import httpx
import pytest

from agent.gemini_native_adapter import GeminiAPIError, GeminiNativeClient, gemini_http_error
from agent.opencode_affinity import merge_session_affinity_headers

ZEN_BASE = "https://opencode.ai/zen/v1"
GOOGLE_BASE = "https://generativelanguage.googleapis.com"
_OK_PAYLOAD = {"candidates": [{"content": {"parts": [{"text": "OK"}]}, "finishReason": "STOP"}]}


def _capture(requests, payload, *, status=200, stream=False):
    """Transport handler recording every request, answering with ``payload``.

    ``HTTPTransport.handle_request`` is replaced on the class, so it arrives bound.
    """

    def respond(_transport, request):
        requests.append(request)
        if status != 200:
            return httpx.Response(status, json=payload)
        if stream:
            return httpx.Response(200, headers={"content-type": "text/event-stream"},
                                  text=f"data: {json.dumps(payload)}\n\n")
        return httpx.Response(200, json=payload)

    return respond


def _call(client, *, extra_headers=None, stream=False):
    result = client.chat.completions.create(
        model="gemini-4-argon", messages=[{"role": "user", "content": "Reply OK"}],
        stream=stream, extra_headers=extra_headers,
    )
    if stream:
        list(result)
    return result


def test_caller_header_casing_does_not_duplicate_headers_on_the_wire(monkeypatch):
    requests = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, _OK_PAYLOAD))
    client = GeminiNativeClient(api_key="zen-key", base_url=ZEN_BASE,
                                default_headers={"accept": "application/json",
                                                 "x-opencode-session": "runtime"})
    try:
        _call(client, extra_headers={"Accept": "application/x-caller",
                                    "X-Opencode-Session": "caller"})
    finally:
        client.close()
    sent = requests[0].headers
    assert sent.get_list("accept") == ["application/x-caller"]
    assert sent.get_list("x-opencode-session") == ["caller"]


def test_streaming_caller_header_casing_still_pins_sse_exactly_once(monkeypatch):
    requests = []
    payload = {**_OK_PAYLOAD, "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, payload, stream=True))
    client = GeminiNativeClient(api_key="zen-key", base_url=ZEN_BASE)
    try:
        _call(client, extra_headers={"accept": "application/json", "X-Opencode-Session": "caller"},
              stream=True)
    finally:
        client.close()
    sent = requests[0].headers
    assert sent.get_list("accept") == ["text/event-stream"]
    assert sent.get_list("x-opencode-session") == ["caller"]


def test_affinity_merge_keeps_one_session_header_whatever_the_caller_casing():
    kwargs = {"extra_headers": {"X-Opencode-Session": "pinned-by-caller"}}
    merged = merge_session_affinity_headers(kwargs, "opencode-zen", ZEN_BASE, "conversation-1")["extra_headers"]
    session_keys = [key for key in merged if key.lower() == "x-opencode-session"]
    assert session_keys == ["X-Opencode-Session"]
    assert merged["X-Opencode-Session"] == "pinned-by-caller"


def test_repeated_and_blank_endpoint_query_options_survive(monkeypatch):
    requests = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, _OK_PAYLOAD))
    client = GeminiNativeClient(api_key="zen-key", base_url=f"{ZEN_BASE}?tag=a&tag=b&route=&tenant=primary")
    try:
        _call(client)
    finally:
        client.close()
    params = requests[0].url.params
    assert params.get_list("tag") == ["a", "b"]
    assert params.get_list("route") == [""]
    assert params["tenant"] == "primary"
    assert requests[0].url.path.endswith("/models/gemini-4-argon:generateContent")


def test_client_query_defaults_override_the_endpoint_value(monkeypatch):
    requests = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, _OK_PAYLOAD))
    client = GeminiNativeClient(api_key="zen-key", base_url=f"{ZEN_BASE}?tenant=stale&tag=a",
                                default_query={"tenant": "primary", "tag": ["c", "d"]})
    try:
        _call(client)
    finally:
        client.close()
    params = requests[0].url.params
    assert params["tenant"] == "primary"
    assert params.get_list("tag") == ["c", "d"]


def test_streaming_owns_alt_without_dropping_the_endpoint_query(monkeypatch):
    requests = []
    payload = {**_OK_PAYLOAD, "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, payload, stream=True))
    client = GeminiNativeClient(api_key="zen-key", base_url=f"{ZEN_BASE}?alt=json&tag=a")
    try:
        _call(client, stream=True)
    finally:
        client.close()
    params = requests[0].url.params
    assert params.get_list("alt") == ["sse"]
    assert params.get_list("tag") == ["a"]


def test_configured_timeout_reaches_the_request_with_an_injected_http_client():
    requests = []
    injected = httpx.Client(timeout=httpx.Timeout(99.0), transport=httpx.MockTransport(
        lambda request: requests.append(request) or httpx.Response(200, json=_OK_PAYLOAD)))
    client = GeminiNativeClient(api_key="zen-key", base_url=ZEN_BASE, timeout=httpx.Timeout(7.0),
                                http_client=injected)
    try:
        _call(client)
    finally:
        client.close()
    assert requests[0].extensions["timeout"]["read"] == 7.0
    assert requests[0].extensions["timeout"]["connect"] == 7.0


def test_adapter_default_read_timeout_still_applies_to_long_decodes(monkeypatch):
    requests = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _capture(requests, _OK_PAYLOAD))
    client = GeminiNativeClient(api_key="zen-key", base_url=ZEN_BASE)
    try:
        _call(client)
    finally:
        client.close()
    assert requests[0].extensions["timeout"]["read"] == 1200.0


_QUOTA_BODY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                         "message": "You exceeded your current quota: free_tier requests per day"}}


@pytest.mark.parametrize("base_url, guided", [(GOOGLE_BASE, True), (ZEN_BASE, False)])
def test_relay_errors_omit_google_credential_guidance(base_url, guided):
    transport = httpx.MockTransport(lambda request: httpx.Response(429, json=_QUOTA_BODY))
    client = GeminiNativeClient(api_key="zen-key", base_url=base_url, http_client=httpx.Client(transport=transport))
    try:
        with pytest.raises(GeminiAPIError) as excinfo:
            _call(client)
    finally:
        client.close()
    # The guidance text itself is unconditional for a Google endpoint: only the relay skips it.
    guided_message = str(gemini_http_error(httpx.Response(429, json=_QUOTA_BODY), base_url=GOOGLE_BASE))
    assert "aistudio.google.com/apikey" in guided_message
    assert ("aistudio.google.com/apikey" in str(excinfo.value)) is guided
