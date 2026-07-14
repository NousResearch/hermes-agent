"""Loopback proxy protocol/security tests.

A fake ChatGPT upstream and the broker app both bind ephemeral 127.0.0.1
ports; an aiohttp client plays the Hermes gateway. Synthetic tokens only —
no real Keychain, OAuth, or non-loopback network.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
from aiohttp import web
import pytest

from agent.oauth_broker.account_slot import AccountSlot
from agent.oauth_broker.models import GrantStoreError, OAuthGrant
from agent.oauth_broker.proxy import (
    DEFAULT_UPSTREAM_ORIGIN,
    _SESSION_FACTORY_KEY,
    create_proxy_app,
    validate_upstream_origin,
)

LOCAL_KEY = "synthetic-local-client-key"
UPSTREAM_ACCESS_A = "synthetic-upstream-access-A"
UPSTREAM_ACCESS_A2 = "synthetic-upstream-access-A2"
UPSTREAM_REFRESH_A = "synthetic-upstream-refresh-A"
ACCOUNT_ID_A = "acct-synthetic-a"

SSE_CHUNKS = [
    b'event: response.created\ndata: {"type":"response.created"}\n\n',
    b'event: response.output_item.added\ndata: {"type":"response.output_item.added","item":{"type":"function_call","name":"get_weather"}}\n\n',
    b'event: response.output_text.delta\ndata: {"delta":"hel"}\n\n',
    b'event: response.output_text.delta\ndata: {"delta":"lo"}\n\n',
    b'event: response.completed\ndata: {"type":"response.completed"}\n\n',
    b"data: [DONE]\n\n",
]


class FakeGrantStore:
    def __init__(self, grants=None):
        self.grants = dict(grants or {})

    def load(self, alias):
        try:
            return self.grants[alias]
        except KeyError:
            raise GrantStoreError(
                alias=alias, category="not_found", detail="no grant provisioned"
            ) from None

    def replace(self, alias, grant):
        self.grants[alias] = grant

    def delete(self, alias):
        self.grants.pop(alias, None)


class FakeUpstream:
    """Synthetic ChatGPT backend recording every request it receives."""

    def __init__(self):
        self.requests = []
        self.responses_script = []  # queued behaviors for /codex/responses
        self.usage_payload = {
            "plan_type": "synthetic-plan",
            "rate_limit": {
                "primary_window": {"used_percent": 12.5, "reset_at": 4102444800},
            },
        }
        self.client_disconnected = asyncio.Event()
        self.streaming_started = asyncio.Event()
        self._runner = None
        self.origin = None

    async def _record(self, request):
        body = await request.read()
        self.requests.append(
            {
                "method": request.method,
                "path": request.path,
                "query": dict(request.query),
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "body": body,
            }
        )

    async def _handle_responses(self, request):
        await self._record(request)
        behavior = (
            self.responses_script.pop(0) if self.responses_script else ("sse",)
        )
        kind = behavior[0]
        if kind == "status":
            _, status, headers, body = behavior
            return web.Response(
                status=status,
                headers=headers,
                body=body,
                content_type="application/json",
            )
        if kind == "sse-infinite":
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            self.streaming_started.set()
            try:
                while True:
                    await resp.write(b"data: synthetic-tick\n\n")
                    await asyncio.sleep(0.01)
            except (ConnectionResetError, asyncio.CancelledError):
                self.client_disconnected.set()
                raise
        if kind == "sse-truncated":
            resp = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await resp.prepare(request)
            await resp.write(b"data: synthetic-partial-event\n\n")
            # Abort before the terminating HTTP chunk. The broker must not
            # turn this upstream truncation into a clean downstream EOF.
            request.transport.abort()
            return resp
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "synthetic-upstream-rid",
            },
        )
        await resp.prepare(request)
        for chunk in SSE_CHUNKS:
            await resp.write(chunk)
            await asyncio.sleep(0)
        await resp.write_eof()
        return resp

    async def _handle_usage(self, request):
        await self._record(request)
        return web.json_response(self.usage_payload)

    async def start(self):
        app = web.Application()
        app.router.add_post("/backend-api/codex/responses", self._handle_responses)
        app.router.add_get("/backend-api/wham/usage", self._handle_usage)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        host, port = self._runner.addresses[0][:2]
        self.origin = f"http://{host}:{port}"

    async def stop(self):
        if self._runner is not None:
            await self._runner.cleanup()


def make_slot(
    tmp_path,
    alias="A",
    *,
    access=UPSTREAM_ACCESS_A,
    account_id=ACCOUNT_ID_A,
    refresh_result=None,
    grant_present=True,
):
    grants = {}
    if grant_present:
        grants[alias] = OAuthGrant(
            access_token=access,
            refresh_token=UPSTREAM_REFRESH_A,
            expires_at=time.time() + 3600,
            account_id=account_id,
        )
    refresh_calls = []

    def refresh_fn(access_token, refresh_token):
        refresh_calls.append((access_token, refresh_token))
        if isinstance(refresh_result, Exception):
            raise refresh_result
        return refresh_result or {
            "access_token": UPSTREAM_ACCESS_A2,
            "refresh_token": "synthetic-upstream-refresh-A2",
            "expires_at": time.time() + 7200,
        }

    slot = AccountSlot(
        alias,
        grant_store=FakeGrantStore(grants),
        state_dir=tmp_path / f"state-{alias}",
        refresh_fn=refresh_fn,
    )
    return slot, refresh_calls


@asynccontextmanager
async def broker_env(tmp_path, *, slots=None, local_key=LOCAL_KEY, **app_kwargs):
    upstream = FakeUpstream()
    await upstream.start()
    made_calls = None
    if slots is None:
        slot, made_calls = make_slot(tmp_path)
        slots = {"A": slot}
    app = create_proxy_app(
        slots=slots,
        local_key=local_key,
        upstream_origin=upstream.origin,
        allow_test_upstream=True,
        **app_kwargs,
    )
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    host, port = runner.addresses[0][:2]
    base = f"http://{host}:{port}"
    session = aiohttp.ClientSession()
    try:
        yield upstream, base, session, made_calls
    finally:
        await session.close()
        await runner.cleanup()
        await upstream.stop()


AUTH = {"Authorization": f"Bearer {LOCAL_KEY}"}


# ---------------------------------------------------------------------------
# Local bearer and whitelist
# ---------------------------------------------------------------------------


def test_missing_local_key_rejected(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses", data=b"{}"
            )
            body = await resp.text()
            assert resp.status == 401
            assert LOCAL_KEY not in body
            assert upstream.requests == []

    asyncio.run(scenario())


def test_wrong_local_key_rejected(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers={"Authorization": "Bearer synthetic-wrong-key"},
            )
            assert resp.status == 401
            assert upstream.requests == []

    asyncio.run(scenario())


def test_unknown_alias_is_hidden_until_local_key_is_authenticated(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            unauthenticated = await session.post(
                f"{base}/accounts/D/backend-api/codex/responses",
                data=b"{}",
            )
            assert unauthenticated.status == 401

            authenticated = await session.post(
                f"{base}/accounts/D/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert authenticated.status == 404
            assert upstream.requests == []

    asyncio.run(scenario())


def test_non_whitelisted_paths_are_404(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            for path in (
                "/accounts/A/backend-api/codex/other",
                "/accounts/A/backend-api/codex",
                "/v1/chat/completions",
                "/accounts/A/backend-api/wham/usage/extra",
            ):
                resp = await session.get(f"{base}{path}", headers=AUTH)
                assert resp.status == 404, path
            assert upstream.requests == []

    asyncio.run(scenario())


def test_wrong_method_on_whitelisted_path_is_405(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.get(
                f"{base}/accounts/A/backend-api/codex/responses", headers=AUTH
            )
            assert resp.status == 405
            resp = await session.head(
                f"{base}/accounts/A/backend-api/wham/usage", headers=AUTH
            )
            assert resp.status == 405
            assert upstream.requests == []

    asyncio.run(scenario())


def test_local_bearer_uses_constant_time_comparison():
    import agent.oauth_broker.proxy as proxy_mod

    source = Path(proxy_mod.__file__).read_text(encoding="utf-8")
    assert "compare_digest" in source


def test_create_app_rejects_empty_local_key(tmp_path):
    slot, _ = make_slot(tmp_path)
    with pytest.raises(ValueError):
        create_proxy_app(
            slots={"A": slot},
            local_key="",
            upstream_origin="http://127.0.0.1:9",
        )


# ---------------------------------------------------------------------------
# Upstream origin gating
# ---------------------------------------------------------------------------


def test_upstream_origin_validation():
    assert validate_upstream_origin(DEFAULT_UPSTREAM_ORIGIN) == DEFAULT_UPSTREAM_ORIGIN
    validate_upstream_origin(
        "http://127.0.0.1:17999", allow_test_upstream=True
    )
    for bad in (
        "http://192.168.1.5:8080",
        "https://evil.example.com",
        "http://localhost:8080",  # only the literal loopback IP is accepted
        "https://chatgpt.com/extra/path",
        "https://chatgpt.com/",
        "https://chatgpt.com:",
        "https://CHATGPT.COM",
        "chatgpt.com",
        "",
    ):
        with pytest.raises(ValueError):
            validate_upstream_origin(bad)


def test_loopback_upstream_requires_explicit_test_only_gate(tmp_path):
    loopback = "http://127.0.0.1:17999"
    with pytest.raises(ValueError):
        validate_upstream_origin(loopback)
    assert (
        validate_upstream_origin(loopback, allow_test_upstream=True)
        == loopback
    )

    slot, _ = make_slot(tmp_path)
    with pytest.raises(ValueError):
        create_proxy_app(
            slots={"A": slot},
            local_key=LOCAL_KEY,
            upstream_origin=loopback,
        )


@pytest.mark.parametrize(
    "bad", ["http://127.0.0.1", "http://127.0.0.1:0", "http://127.0.0.1:65536"]
)
def test_test_only_loopback_origin_requires_explicit_valid_port(bad):
    with pytest.raises(ValueError):
        validate_upstream_origin(bad, allow_test_upstream=True)


@pytest.mark.parametrize(
    "bad",
    [
        "https://user@chatgpt.com",  # userinfo
        "https://user:pw@chatgpt.com",
        "https://chatgpt.com:8443",  # non-default port
        "https://chatgpt.com:443",  # explicit port — canonical form only
        "http://chatgpt.com",  # wrong scheme for the production host
        "https://www.chatgpt.com",  # host variant
        "https://chat.chatgpt.com",
        "https://chatgpt.com.evil.example",  # suffix trick
        "https://xchatgpt.com",
        "https://chatgpt.com?x=1",  # query on origin
        "https://chatgpt.com#frag",  # fragment on origin
        "http://127.0.0.1.evil.example:8080",  # loopback-lookalike host
        "http://[::1]:8080",  # only the 127.0.0.1 literal test origin
        "http://user@127.0.0.1:8080",  # userinfo on the test origin
        "https://127.0.0.1:8443",  # loopback test origin must be http
        "http://127.0.0.2:8080",  # non-canonical loopback address
    ],
)
def test_upstream_origin_validation_hardened_rejections(bad):
    with pytest.raises(ValueError):
        validate_upstream_origin(bad)


# ---------------------------------------------------------------------------
# Forwarded-header minimal allowlist
# ---------------------------------------------------------------------------


def test_forwarded_headers_use_documented_minimal_allowlist():
    """The allowlist is the exact set the Codex transport actually needs:
    accept/user-agent (account_usage.py, auxiliary_client.py), content-type
    (Responses POST body), originator (Cloudflare gate, auxiliary_client.py),
    openai-beta (OpenAI SDK), session_id + x-client-request-id
    (agent/transports/codex.py cache-scope routing)."""
    from agent.oauth_broker.proxy import FORWARD_REQUEST_HEADER_ALLOWLIST

    assert FORWARD_REQUEST_HEADER_ALLOWLIST == frozenset(
        {
            "accept",
            "content-type",
            "user-agent",
            "originator",
            "openai-beta",
            "session_id",
            "x-client-request-id",
        }
    )


def test_unlisted_request_headers_are_never_forwarded(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers={
                    **AUTH,
                    "Accept": "text/event-stream",
                    "Content-Type": "application/json",
                    "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
                    "originator": "codex_cli_rs",
                    "OpenAI-Beta": "responses=synthetic",
                    "session_id": "synthetic-session-1",
                    "x-client-request-id": "synthetic-session-1",
                    # None of the following may cross the broker:
                    "baggage": "synthetic-baggage",
                    "traceparent": "00-synthetic",
                    "x-stainless-os": "MacOS",
                    "x-forwarded-for": "203.0.113.7",
                    "x-attacker-header": "synthetic-injection",
                    "Referer": "http://127.0.0.1/synthetic",
                },
            )
            assert resp.status == 200
            await resp.read()
            [seen] = upstream.requests
            headers = seen["headers"]
            assert headers["accept"] == "text/event-stream"
            assert headers["content-type"] == "application/json"
            assert headers["user-agent"] == "codex_cli_rs/0.0.0 (Hermes Agent)"
            assert headers["originator"] == "codex_cli_rs"
            assert headers["openai-beta"] == "responses=synthetic"
            assert headers["session_id"] == "synthetic-session-1"
            assert headers["x-client-request-id"] == "synthetic-session-1"
            for banned in (
                "baggage",
                "traceparent",
                "x-stainless-os",
                "x-forwarded-for",
                "x-attacker-header",
                "referer",
            ):
                assert banned not in headers

    asyncio.run(scenario())


def test_default_upstream_session_disables_cookies_and_decompression(tmp_path):
    async def scenario():
        slot, _ = make_slot(tmp_path)
        app = create_proxy_app(slots={"A": slot}, local_key=LOCAL_KEY)
        session = app[_SESSION_FACTORY_KEY]()
        try:
            assert isinstance(session.cookie_jar, aiohttp.DummyCookieJar)
            assert session.auto_decompress is False
        finally:
            await session.close()

    asyncio.run(scenario())


def test_insecure_custom_upstream_session_is_rejected_and_closed(tmp_path):
    async def scenario():
        slot, _ = make_slot(tmp_path)
        insecure_session = None

        def insecure_factory():
            nonlocal insecure_session
            insecure_session = aiohttp.ClientSession(
                cookie_jar=aiohttp.CookieJar(unsafe=True),
                auto_decompress=True,
            )
            return insecure_session

        app = create_proxy_app(
            slots={"A": slot},
            local_key=LOCAL_KEY,
            client_session_factory=insecure_factory,
        )
        runner = web.AppRunner(app)
        with pytest.raises(RuntimeError, match="DummyCookieJar"):
            await runner.setup()
        assert insecure_session is not None
        assert insecure_session.closed

    asyncio.run(scenario())


def test_create_app_rejects_arbitrary_upstream(tmp_path):
    slot, _ = make_slot(tmp_path)
    with pytest.raises(ValueError):
        create_proxy_app(
            slots={"A": slot},
            local_key=LOCAL_KEY,
            upstream_origin="https://attacker.example.com",
        )


# ---------------------------------------------------------------------------
# Header replacement and byte preservation
# ---------------------------------------------------------------------------


def test_header_replacement_and_request_byte_preservation(tmp_path):
    request_body = json.dumps({"model": "gpt-5.3-codex", "stream": True}).encode()

    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=request_body,
                headers={
                    **AUTH,
                    "Content-Type": "application/json",
                    "originator": "codex_cli_rs",
                    "x-client-request-id": "synthetic-rid-1",
                    "Cookie": "synthetic-cookie=1",
                    "ChatGPT-Account-Id": "attacker-supplied-account",
                    "Proxy-Authorization": "Basic synthetic",
                },
            )
            assert resp.status == 200
            await resp.read()
            [seen] = upstream.requests
            headers = seen["headers"]
            assert seen["body"] == request_body
            assert headers["authorization"] == f"Bearer {UPSTREAM_ACCESS_A}"
            assert headers["chatgpt-account-id"] == ACCOUNT_ID_A
            assert headers["originator"] == "codex_cli_rs"
            assert headers["x-client-request-id"] == "synthetic-rid-1"
            assert "cookie" not in headers
            assert "proxy-authorization" not in headers
            assert LOCAL_KEY not in json.dumps(list(headers.values()))

    asyncio.run(scenario())


def test_usage_route_forwards_query_and_returns_payload(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.get(
                f"{base}/accounts/A/backend-api/wham/usage?window=weekly",
                headers=AUTH,
            )
            assert resp.status == 200
            payload = await resp.json()
            assert payload["plan_type"] == "synthetic-plan"
            [seen] = upstream.requests
            assert seen["path"] == "/backend-api/wham/usage"
            assert seen["query"] == {"window": "weekly"}
            assert seen["headers"]["authorization"] == f"Bearer {UPSTREAM_ACCESS_A}"

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# SSE transparency
# ---------------------------------------------------------------------------


def test_sse_stream_preserves_byte_order_and_final_event(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 200
            assert resp.headers["Content-Type"].startswith("text/event-stream")
            assert resp.headers["x-request-id"] == "synthetic-upstream-rid"
            body = await resp.read()
            assert body == b"".join(SSE_CHUNKS)
            assert body.endswith(b"data: [DONE]\n\n")
            # Tool-call event bytes are untouched.
            assert b'"name":"get_weather"' in body

    asyncio.run(scenario())


def test_client_cancellation_propagates_to_upstream(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            upstream.responses_script.append(("sse-infinite",))
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            await asyncio.wait_for(upstream.streaming_started.wait(), timeout=2)
            await resp.content.read(10)
            resp.close()
            await asyncio.wait_for(
                upstream.client_disconnected.wait(), timeout=2
            )

    asyncio.run(scenario())


def test_truncated_upstream_sse_aborts_downstream_instead_of_clean_eof(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            upstream.responses_script.append(("sse-truncated",))
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 200
            with pytest.raises(aiohttp.ClientError):
                await resp.read()

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# 429 and 401 semantics
# ---------------------------------------------------------------------------


def test_429_passes_through_with_reset_headers_and_no_refresh(tmp_path):
    limit_body = json.dumps(
        {"detail": "rate limited", "resets_in_seconds": 600}
    ).encode()

    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, calls):
            upstream.responses_script.append(
                (
                    "status",
                    429,
                    {
                        "Retry-After": "600",
                        "x-ratelimit-remaining": "0",
                        "x-codex-primary-reset-after-seconds": "600",
                    },
                    limit_body,
                )
            )
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 429
            assert resp.headers["Retry-After"] == "600"
            assert resp.headers["x-ratelimit-remaining"] == "0"
            assert resp.headers["x-codex-primary-reset-after-seconds"] == "600"
            assert await resp.read() == limit_body
            assert calls == []  # broker never account-switches or refreshes on 429
            assert len(upstream.requests) == 1

    asyncio.run(scenario())


def test_gzip_429_preserves_wire_bytes_and_encoding_header(tmp_path):
    plain_body = b'{"detail":"synthetic compressed rate limit"}'
    wire_body = gzip.compress(plain_body)

    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, _session, calls):
            upstream.responses_script.append(
                (
                    "status",
                    429,
                    {
                        "Retry-After": "600",
                        "Content-Encoding": "gzip",
                    },
                    wire_body,
                )
            )
            async with aiohttp.ClientSession(auto_decompress=False) as raw_session:
                resp = await raw_session.post(
                    f"{base}/accounts/A/backend-api/codex/responses",
                    data=b"{}",
                    headers=AUTH,
                )
                assert resp.status == 429
                assert resp.headers["Content-Encoding"] == "gzip"
                assert int(resp.headers["Content-Length"]) == len(wire_body)
                assert await resp.read() == wire_body
            assert calls == []

    asyncio.run(scenario())


def test_upstream_401_forces_single_refresh_and_replays_once(tmp_path):
    request_body = b'{"stream":true}'

    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, calls):
            upstream.responses_script.append(
                ("status", 401, {}, b'{"detail":"Unauthorized"}')
            )
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=request_body,
                headers=AUTH,
            )
            assert resp.status == 200
            body = await resp.read()
            assert body == b"".join(SSE_CHUNKS)
            assert len(calls) == 1  # exactly one forced refresh
            first, second = upstream.requests
            assert first["headers"]["authorization"] == f"Bearer {UPSTREAM_ACCESS_A}"
            assert (
                second["headers"]["authorization"] == f"Bearer {UPSTREAM_ACCESS_A2}"
            )
            assert second["body"] == request_body  # replayed byte-for-byte

    asyncio.run(scenario())


def test_second_401_returns_unchanged_without_refresh_loop(tmp_path):
    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, calls):
            upstream.responses_script.append(
                ("status", 401, {}, b'{"detail":"Unauthorized"}')
            )
            upstream.responses_script.append(
                ("status", 401, {}, b'{"detail":"Still unauthorized"}')
            )
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 401
            assert await resp.read() == b'{"detail":"Still unauthorized"}'
            assert len(calls) == 1
            assert len(upstream.requests) == 2

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Fail-closed slot and upstream errors
# ---------------------------------------------------------------------------


def test_missing_grant_fails_closed_503(tmp_path):
    async def scenario():
        slot, _ = make_slot(tmp_path, grant_present=False)
        async with broker_env(tmp_path, slots={"A": slot}) as (
            upstream,
            base,
            session,
            _,
        ):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            body = await resp.text()
            assert resp.status == 503
            assert "not_found" in body
            assert UPSTREAM_ACCESS_A not in body
            assert upstream.requests == []

    asyncio.run(scenario())


def test_unreachable_upstream_returns_502(tmp_path):
    async def scenario():
        slot, _ = make_slot(tmp_path)
        app = create_proxy_app(
            slots={"A": slot},
            local_key=LOCAL_KEY,
            upstream_origin="http://127.0.0.1:9",  # nothing listens here
            allow_test_upstream=True,
        )
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        host, port = runner.addresses[0][:2]
        session = aiohttp.ClientSession()
        try:
            resp = await session.post(
                f"http://{host}:{port}/accounts/A/backend-api/codex/responses",
                data=b"{}",
                headers=AUTH,
            )
            assert resp.status == 502
        finally:
            await session.close()
            await runner.cleanup()

    asyncio.run(scenario())


def test_request_body_over_limit_rejected(tmp_path):
    async def scenario():
        async with broker_env(
            tmp_path, request_body_limit=1024
        ) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=b"x" * 4096,
                headers=AUTH,
            )
            assert resp.status == 413
            assert upstream.requests == []

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Log hygiene
# ---------------------------------------------------------------------------


def test_logs_carry_ids_only_never_bodies_or_secrets(tmp_path, caplog):
    request_body = b'{"input":"synthetic-user-prompt-content"}'

    async def scenario():
        async with broker_env(tmp_path) as (upstream, base, session, _):
            resp = await session.post(
                f"{base}/accounts/A/backend-api/codex/responses",
                data=request_body,
                headers=AUTH,
            )
            assert resp.status == 200
            await resp.read()

    with caplog.at_level("DEBUG", logger="agent.oauth_broker.proxy"):
        asyncio.run(scenario())

    text = caplog.text
    assert "alias=A" in text
    assert "status=200" in text
    for secret in (
        LOCAL_KEY,
        UPSTREAM_ACCESS_A,
        UPSTREAM_REFRESH_A,
        "synthetic-user-prompt-content",
        "response.output_text.delta",
    ):
        assert secret not in text
