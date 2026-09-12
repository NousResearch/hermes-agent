"""Wire-level guard for #106495 — the OpenCode Zen FREE tier must leave Hermes carrying the
OpenCode client's request fingerprint.

The free tier of the Zen relay (``https://opencode.ai/zen/v1``, served anonymously) rate-limits
HTTP 429 ``FreeUsageLimitError`` any request that carries the canonical Hermes attribution set
(``User-Agent: HermesAgent/<v>`` + ``HTTP-Referer: https://hermes-agent.nousresearch.com`` +
``X-Title: Hermes Agent``), while the very same model answers 200 when the request looks like the
OpenCode client (``User-Agent: opencode/0.20.5``, ``X-Session-ID: <uuid>``,
``HTTP-Referer: https://opencode.ai/``, ``X-Title: opencode``).

Evidence shape: a local HTTP mock IS the provider base_url, so every assertion below runs against
the header set the OpenAI SDK actually puts on the wire — not against the kwargs Hermes intended
to pass. Both keyless routes are covered (a free model selected under ``opencode-zen``/
``opencode-go``, which ``opencode_zen_free_runtime`` heals to the Zen relay, and the keyless
``opencode-free`` provider), and the keyed Zen/Go chains are pinned as unchanged so the
free-tier fingerprint can never leak into them.
"""

from __future__ import annotations

import http.server
import json
import threading
from contextlib import contextmanager

import httpx
import pytest

from agent.agent_runtime_helpers import create_openai_client
from hermes_cli.models import OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER

# The fingerprint the OpenCode client itself sends and the relay's free tier accepts.
OC_USER_AGENT = "opencode/0.20.5"
OC_REFERER = "https://opencode.ai/"
OC_TITLE = "opencode"

_PROXY_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
)


class _Recorder(http.server.BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802 — stdlib handler API
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b"{}"
        type(self).requests.append({
            "path": self.path,
            "headers": {k.lower(): v for k, v in self.headers.items()},
            "body": json.loads(body or b"{}"),
        })
        payload = json.dumps({
            "id": "chatcmpl-mock", "object": "chat.completion", "created": 0,
            "model": "big-pickle",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "pong"},
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep pytest output clean
        pass


@contextmanager
def _mock_zen_relay():
    """Local stand-in for the Zen relay base_url; records every incoming header set."""
    _Recorder.requests = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _Agent:
    """Minimal stand-in for the pieces ``create_openai_client`` reads off the agent."""

    def __init__(self, provider: str, api_key: str, base_url: str = ""):
        self.provider = provider
        self.api_key = api_key
        self.base_url = base_url
        self.model = "big-pickle"
        self.api_mode = "chat_completions"

    def _client_log_context(self):
        return {}

    def _build_keepalive_http_client(self, base_url, verify=True):
        return None


@pytest.fixture(autouse=True)
def _no_ambient_proxy(monkeypatch):
    """The box running the suite may export a proxy; the mock relay is on loopback."""
    for var in _PROXY_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


def _post(provider, api_key, base_url, *, default_headers=None, extra_headers=None):
    """Build a real client against the mock relay, POST once, return what the relay saw."""
    agent = _Agent(provider, api_key, base_url)
    kwargs = {
        "api_key": api_key,
        "base_url": base_url,
        "http_client": httpx.Client(trust_env=False),
    }
    if default_headers:
        kwargs["default_headers"] = dict(default_headers)
    client = create_openai_client(agent, kwargs, reason="test", shared=False)
    try:
        client.chat.completions.create(
            model="big-pickle",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
            extra_headers=extra_headers or None,
        )
    finally:
        client.close()
    assert _Recorder.requests, "the mock relay never saw a request"
    return _Recorder.requests[-1]["headers"]


def _profile_headers(provider):
    from providers import get_provider_profile

    profile = get_provider_profile(provider)
    assert profile is not None, f"provider {provider!r} is not registered"
    return dict(profile.default_headers or {})


def _assert_opencode_fingerprint(headers):
    assert headers.get("user-agent") == OC_USER_AGENT, headers
    assert headers.get("http-referer") == OC_REFERER, headers
    assert headers.get("x-title") == OC_TITLE, headers
    assert headers.get("x-session-id"), "free tier must identify a session to the relay"


def _assert_no_bearer(headers):
    auth = (headers.get("authorization") or "").lower()
    assert "bearer" not in auth, f"keyless free tier must not send a credential: {headers}"
    assert OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER not in auth


def _assert_hermes_attribution(headers):
    assert str(headers.get("user-agent", "")).startswith("HermesAgent/"), headers
    assert headers.get("x-title") == "Hermes Agent", headers
    assert "x-session-id" not in headers, "the free-tier fingerprint must not leak here"


# ---------------------------------------------------------------------------
# Keyless free tier — the #106495 repro
# ---------------------------------------------------------------------------


def test_zen_free_model_wire_headers_are_opencode_compatible():
    """A free slug under ``opencode-zen`` (the reported ``opencode-zen/big-pickle``) routes
    keylessly through the Zen relay and must carry the OpenCode fingerprint."""
    with _mock_zen_relay() as base_url:
        headers = _post("opencode-zen", OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, base_url)
    _assert_opencode_fingerprint(headers)
    _assert_no_bearer(headers)


def test_opencode_free_provider_wire_headers_are_opencode_compatible():
    with _mock_zen_relay() as base_url:
        headers = _post("opencode-free", OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, base_url)
    _assert_opencode_fingerprint(headers)
    _assert_no_bearer(headers)


def test_free_tier_keeps_101864_session_affinity_alongside_fingerprint():
    """#101864's ``x-opencode-session`` (the MissingSessionID fix) must survive on the same
    request as the free-tier fingerprint — the two header policies coexist."""
    with _mock_zen_relay() as base_url:
        headers = _post(
            "opencode-free", OPENCODE_ZEN_FREE_KEYLESS_PLACEHOLDER, base_url,
            extra_headers={"x-opencode-session": "sess-wire-1"},
        )
    _assert_opencode_fingerprint(headers)
    assert headers.get("x-opencode-session") == "sess-wire-1", headers


# ---------------------------------------------------------------------------
# Regression pins — keyed Zen / Go / other providers are untouched
# ---------------------------------------------------------------------------


def test_keyed_zen_still_identifies_as_hermes():
    with _mock_zen_relay() as base_url:
        headers = _post(
            "opencode-zen", "sk-zen-real", base_url,
            default_headers=_profile_headers("opencode-zen"),
        )
    _assert_hermes_attribution(headers)


def test_keyed_go_chain_still_identifies_as_hermes():
    with _mock_zen_relay() as base_url:
        headers = _post(
            "opencode-go", "sk-go-real", base_url,
            default_headers=_profile_headers("opencode-go"),
        )
    _assert_hermes_attribution(headers)


@pytest.mark.parametrize("provider", ["fireworks", "openrouter", "ai-gateway"])
def test_non_opencode_provider_still_identifies_as_hermes(provider):
    with _mock_zen_relay() as base_url:
        headers = _post(
            provider, "sk-other-real", base_url,
            default_headers=_profile_headers(provider),
        )
    # Only the opencode-free identity is allowed to say "opencode"; these providers must never
    # pick up the fingerprint nor lose the headers their own profile declares.
    assert headers.get("user-agent") != OC_USER_AGENT, headers
    assert headers.get("x-title") != OC_TITLE, headers
    assert "x-session-id" not in headers, headers


# ---------------------------------------------------------------------------
# Profile declarations (the "compatible headers profile" itself)
# ---------------------------------------------------------------------------


def test_keyed_profiles_declare_hermes_attribution():
    for provider in ("opencode-zen", "opencode-go"):
        headers = _profile_headers(provider)
        assert headers["User-Agent"].startswith("HermesAgent/"), (provider, headers)
        assert headers["X-Title"] == "Hermes Agent", (provider, headers)


def test_free_profile_declares_opencode_fingerprint():
    headers = _profile_headers("opencode-free")
    assert headers["Authorization"] == ""
    assert headers["User-Agent"] == OC_USER_AGENT, headers
    assert headers["HTTP-Referer"] == OC_REFERER, headers
    assert headers["X-Title"] == OC_TITLE, headers
    assert headers.get("X-Session-ID"), headers
