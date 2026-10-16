"""Real HTTPX redirect chains: hooks guard before transport or credential refresh."""

import asyncio

import httpx
import pytest

from agent import process_bootstrap as bootstrap
from agent.azure_identity_adapter import build_bearer_http_client


@pytest.mark.parametrize("policy", [None, False])
def test_disabled_guard_has_no_hooks(policy):
    assert bootstrap.provider_redirect_hooks(policy) == {}
    assert bootstrap.provider_redirect_hooks(policy, async_mode=True) == {}


@pytest.mark.parametrize("target", [
    "https://foreign.test/final", "http://provider.test/final",
    "https://provider.test:444/final", "https://provider.test:0/final",
])
@pytest.mark.parametrize("async_mode", [False, True])
def test_cross_origin_blocked_before_transport(target, async_mode):
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(307, headers={"location": target})

    kwargs = dict(transport=httpx.MockTransport(respond), follow_redirects=True,
                  event_hooks=bootstrap.provider_redirect_hooks(True, async_mode=async_mode))
    if async_mode:
        async def run():
            async with httpx.AsyncClient(**kwargs) as client:
                with pytest.raises(httpx.LocalProtocolError, match="Cross-origin"):
                    await client.post("https://provider.test/start", content=b"secret body")
        asyncio.run(run())
    else:
        with httpx.Client(**kwargs) as client:
            with pytest.raises(httpx.LocalProtocolError, match="Cross-origin"):
                client.post("https://provider.test/start", content=b"secret body")
    assert len(seen) == 1
    assert seen[0].url.host == "provider.test"


@pytest.mark.parametrize("status", [307, 308])
@pytest.mark.parametrize("async_mode", [False, True])
def test_multihop_replays_body_and_guards_initial_origin(status, async_mode):
    seen = []

    def respond(request):
        seen.append((str(request.url), request.method, request.content))
        targets = {"/start": "https://PROVIDER.test:443/middle", "/middle": "/final"}
        if request.url.path in targets:
            return httpx.Response(status, headers={"location": targets[request.url.path]})
        return httpx.Response(200)

    kwargs = dict(transport=httpx.MockTransport(respond), follow_redirects=True,
                  event_hooks=bootstrap.provider_redirect_hooks(True, async_mode=async_mode))
    if async_mode:
        async def run():
            async with httpx.AsyncClient(**kwargs) as client:
                return await client.post("https://provider.test/start", content=b"private JSON")
        response = asyncio.run(run())
    else:
        with httpx.Client(**kwargs) as client:
            response = client.post("https://provider.test/start", content=b"private JSON")
    assert response.status_code == 200
    assert len(response.history) == 2
    assert [url for url, _, _ in seen] == [f"https://provider.test/{path}" for path in ("start", "middle", "final")]
    assert all((method, body) == ("POST", b"private JSON") for _, method, body in seen)


@pytest.mark.parametrize("async_mode", [False, True])
def test_later_cross_origin_hop_and_redirect_loops(async_mode):
    seen = []

    def respond(request):
        seen.append(str(request.url))
        target = "/middle" if request.url.path == "/start" else "https://foreign.test/end"
        return httpx.Response(308, headers={"location": target})

    async def run_async():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), follow_redirects=True,
                                     event_hooks=bootstrap.provider_redirect_hooks(True, async_mode=True)) as client:
            with pytest.raises(httpx.LocalProtocolError):
                await client.get("https://provider.test/start")
            client._transport = httpx.MockTransport(lambda request: httpx.Response(307, headers={"location": "/loop"}))
            client.max_redirects = 2
            with pytest.raises(httpx.TooManyRedirects):
                await client.get("https://provider.test/loop")

    if async_mode:
        asyncio.run(run_async())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True,
                          event_hooks=bootstrap.provider_redirect_hooks(True)) as client:
            with pytest.raises(httpx.LocalProtocolError):
                client.get("https://provider.test/start")
            client._transport = httpx.MockTransport(lambda request: httpx.Response(307, headers={"location": "/loop"}))
            client.max_redirects = 2
            with pytest.raises(httpx.TooManyRedirects):
                client.get("https://provider.test/loop")
    assert seen == ["https://provider.test/start", "https://provider.test/middle"]


def test_request_origins_are_independent_on_one_client():
    seen = []

    def respond(request):
        seen.append(str(request.url))
        return httpx.Response(200 if request.url.path == "/end" else 307, headers={"location": "/end"})

    with httpx.Client(transport=httpx.MockTransport(respond), follow_redirects=True,
                      event_hooks=bootstrap.provider_redirect_hooks(True)) as client:
        for host in ("first.test", "second.test"):
            assert client.get(f"https://{host}/start").status_code == 200
    assert len(seen) == 4


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("proxy", [None, "http://proxy.test:8080"])
def test_keepalive_builders_guard_direct_shared_and_proxy(monkeypatch, async_mode, proxy):
    import httpx._client
    from agent import codex_headers

    seen, compression = [], []

    def respond(request):
        seen.append((str(request.url), request.content))
        return httpx.Response(307, headers={"location": "https://foreign.test/end"})

    def compress(request):
        compression.append(str(request.url))

    async def async_compress(request):
        compress(request)

    monkeypatch.setattr(bootstrap, "_get_proxy_for_base_url", lambda base_url: proxy)
    monkeypatch.setattr(codex_headers, "codex_request_compression_hooks", lambda *a, **k: {
        "request": [async_compress if async_mode else compress]})
    transport_name = "AsyncHTTPTransport" if async_mode else "HTTPTransport"
    monkeypatch.setattr(httpx, transport_name, lambda **kwargs: httpx.MockTransport(respond))
    monkeypatch.setattr(httpx._client, transport_name, lambda **kwargs: httpx.MockTransport(respond))
    bootstrap.close_shared_transports()
    client = bootstrap.build_keepalive_http_client("https://configured.test", async_mode=async_mode, follow_redirects=True)
    assert client is not None
    assert client.follow_redirects is True
    try:
        if async_mode:
            async def run():
                with pytest.raises(httpx.LocalProtocolError):
                    await client.post("https://actual.test/start", content=b"private")
                await client.aclose()
            asyncio.run(run())
        else:
            with client, pytest.raises(httpx.LocalProtocolError):
                client.post("https://actual.test/start", content=b"private")
    finally:
        bootstrap.close_shared_transports()
    assert seen == [("https://actual.test/start", b"private")]
    assert compression == ["https://actual.test/start"]


def test_opt_in_builder_failure_cannot_fall_back(monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("transport construction failed")

    monkeypatch.setattr(bootstrap, "_get_proxy_for_base_url", lambda base_url: None)
    monkeypatch.setattr(httpx, "HTTPTransport", fail)
    bootstrap.close_shared_transports()
    with pytest.raises(RuntimeError, match="construction failed"):
        bootstrap.build_keepalive_http_client(follow_redirects=True)
    assert bootstrap.build_keepalive_http_client() is None


def test_bearer_hooks_compose_and_block_before_minting():
    minted, seen, callbacks = [], [], []
    hooks = bootstrap.provider_redirect_hooks(True)
    hooks["request"].append(lambda request: callbacks.append("request"))
    hooks["response"] = [lambda response: callbacks.append("response")]
    original_requests = list(hooks["request"])

    def token():
        minted.append("mint")
        return "private-token"

    def respond(request):
        seen.append((str(request.url), request.headers.get("authorization"), request.headers.get("x-api-key")))
        target = "/middle" if request.url.path == "/start" else "https://foreign.test/end"
        return httpx.Response(307, headers={"location": target})

    with build_bearer_http_client(token, follow_redirects=True, event_hooks=hooks,
                                  transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.LocalProtocolError):
            client.post("https://provider.test/start", content=b"private")
    assert hooks["request"] == original_requests
    assert minted == ["mint", "mint"]
    assert callbacks == ["request", "response", "request", "response"]
    assert seen == [(f"https://provider.test/{path}", "Bearer private-token", None) for path in ("start", "middle")]


@pytest.mark.parametrize("policy", [None, False, True])
@pytest.mark.parametrize("callable_key", [False, True])
def test_native_anthropic_builder_policy_and_sdk_defaults(monkeypatch, policy, callable_key):
    import anthropic
    from agent import anthropic_adapter

    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)
    key = (lambda: "sk-ant-api03-private") if callable_key else "sk-ant-api03-private"
    client = anthropic_adapter.build_anthropic_client(key, follow_redirects=policy)
    try:
        expected = not callable_key if policy is None else policy
        assert client._client.follow_redirects is expected
        assert bool(client._client.event_hooks["request"]) is (callable_key or policy is True)
        if not callable_key and policy is not None:
            with anthropic.DefaultHttpxClient() as default:
                assert client._client._transport._pool._max_connections == default._transport._pool._max_connections
                assert client._client._transport._pool._max_keepalive_connections == default._transport._pool._max_keepalive_connections
        assert client.timeout.connect == 10
        assert client.timeout.read == 900
    finally:
        client.close()


@pytest.mark.parametrize("callable_key", [False, True])
@pytest.mark.parametrize("policy", [False, True])
def test_native_anthropic_actual_message_redirect(monkeypatch, callable_key, policy):
    import anthropic
    import httpx._client
    from agent import anthropic_adapter

    seen, minted = [], []

    def respond(request):
        seen.append((str(request.url), request.headers.get("x-api-key"), request.content))
        return httpx.Response(307, headers={"location": "https://foreign.test/v1/messages"})

    def token():
        minted.append("mint")
        return "sk-ant-api03-private"

    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)
    monkeypatch.setattr(httpx.Client, "_init_transport", lambda *args, **kwargs: httpx.MockTransport(respond))
    monkeypatch.setattr(httpx.Client, "_init_proxy_transport", lambda *args, **kwargs: httpx.MockTransport(respond))
    # Native SDK constructs its keepalive transport through a separately imported alias.
    monkeypatch.setitem(anthropic.DefaultHttpxClient.__init__.__globals__, "HTTPTransport",
                        lambda **kwargs: httpx.MockTransport(respond))
    client = anthropic_adapter.build_anthropic_client(token if callable_key else "sk-ant-api03-private",
                                                     follow_redirects=policy)
    mints_at_construction = len(minted)
    try:
        with pytest.raises(anthropic.APIConnectionError if policy else anthropic.APIStatusError):
            client.messages.create(model="claude-test", max_tokens=1,
                                   messages=[{"role": "user", "content": "private prompt"}])
    finally:
        client.close()
    assert len(seen) == 1
    assert seen[0][0] == "https://api.anthropic.com/v1/messages"
    assert seen[0][1] == "sk-ant-api03-private"
    assert b"private prompt" in seen[0][2]
    if callable_key:
        # Credential-route classification may sample the key before construction;
        # the blocked hop must not sample it again.
        assert len(minted) == mints_at_construction + 1


def test_native_anthropic_opt_in_construction_failure_propagates(monkeypatch):
    import anthropic
    from agent import anthropic_adapter

    monkeypatch.setattr(anthropic_adapter, "_anthropic_sdk", anthropic)

    def fail(**kwargs):
        raise RuntimeError("guarded client unavailable")

    monkeypatch.setattr(anthropic, "DefaultHttpxClient", fail)
    with pytest.raises(RuntimeError, match="guarded client unavailable"):
        anthropic_adapter.build_anthropic_client("sk-ant-api03-private", follow_redirects=True)
