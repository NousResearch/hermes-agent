"""Tests for the `hermes proxy` subcommand and its upstream adapters."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import socket
import threading
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter
from hermes_cli.proxy.adapters.xai import XAIGrokAdapter


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------








# ---------------------------------------------------------------------------
# NousPortalAdapter
# ---------------------------------------------------------------------------


def _write_auth_store(hermes_home: Path, nous_state: Dict[str, Any]) -> Path:
    """Write an auth.json with the given nous state into a hermetic HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {"nous": nous_state},
    }))
    return auth_path




def test_nous_adapter_concurrent_refresh_serialized(tmp_path, monkeypatch):
    """Two parallel get_credential() calls must serialize through the lock."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth_store(tmp_path, {
        "access_token": "a", "refresh_token": "r",
    })

    call_log: list = []
    in_flight = threading.Event()
    overlap_detected = threading.Event()
    counter = [0]
    counter_lock = threading.Lock()

    def serializing_refresh(**kwargs):
        # If another thread is already inside refresh, the lock is broken.
        if in_flight.is_set():
            overlap_detected.set()
        in_flight.set()
        try:
            call_log.append(threading.current_thread().ident)
            # Simulate refresh latency so any race window is exposed.
            import time
            time.sleep(0.05)
            with counter_lock:
                counter[0] += 1
                idx = counter[0]
            return {
                "api_key": f"key-{idx}",
                "expires_at": "2099-01-01T00:00:00Z",
                "base_url": "https://inference-api.nousresearch.com/v1",
            }
        finally:
            in_flight.clear()

    adapter = NousPortalAdapter()
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(adapter.get_credential().bearer)
        except Exception as exc:  # pragma: no cover - shouldn't happen
            errors.append(exc)

    with patch(
        "hermes_cli.proxy.adapters.nous_portal.resolve_nous_runtime_credentials",
        side_effect=serializing_refresh,
    ):
        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert not errors, f"workers errored: {errors}"
    assert len(results) == 3
    assert len(call_log) == 3
    assert not overlap_detected.is_set(), "refresh calls overlapped — lock is broken"
    assert all(r.startswith("key-") for r in results)


# ---------------------------------------------------------------------------
# XAIGrokAdapter
# ---------------------------------------------------------------------------


def _write_xai_pool_entry(
    hermes_home: Path,
    *,
    access_token: str = "xai-access-token",
    refresh_token: str = "xai-refresh-token",
    base_url: str = "https://api.x.ai/v1",
    source: str = "manual:xai_pkce",
) -> Path:
    """Write an xai-oauth pool entry into a hermetic HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": "xai123",
                    "label": "xai-test",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": source,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "base_url": base_url,
                }
            ]
        },
    }))
    return auth_path


def test_xai_adapter_not_authenticated_when_no_pool_entry(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {},
    }))
    assert not XAIGrokAdapter().is_authenticated()


def test_xai_adapter_retry_rotates_pool_entry_on_429(tmp_path, monkeypatch):
    """429 from xAI must rotate to the next pool entry, not attempt refresh.

    Pre-fix (#28932) ``get_retry_credential`` only fired on 401, so a 429
    rate-limit response flowed back to the client unchanged AND the
    rate-limited bearer stayed active for the next request — defeating
    the whole point of pool rotation.

    Post-fix: 429 lands on ``mark_exhausted_and_rotate`` (no refresh —
    that's irrelevant for rate limits), stamps the 1-hour cooldown
    via ``EXHAUSTED_TTL_429_SECONDS`` on the offending key, and
    returns the next available credential.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    # Two pool entries so rotation has somewhere to go.
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {
            "xai-oauth": [
                {
                    "id": "xai-first",
                    "label": "xai-first",
                    "auth_type": "oauth",
                    "priority": 0,
                    "source": "manual:xai_pkce",
                    "access_token": "first-access-token",
                    "refresh_token": "first-refresh-token",
                    "base_url": "https://api.x.ai/v1",
                },
                {
                    "id": "xai-second",
                    "label": "xai-second",
                    "auth_type": "oauth",
                    "priority": 1,
                    "source": "manual:xai_pkce",
                    "access_token": "second-access-token",
                    "refresh_token": "second-refresh-token",
                    "base_url": "https://api.x.ai/v1",
                },
            ]
        },
    }))

    # Refresh must NOT be called on the 429 path — guard against
    # the fix accidentally trying to refresh-on-rate-limit.
    def _refresh_must_not_run(*args, **kwargs):
        raise AssertionError("refresh_xai_oauth_pure must not run on 429")

    monkeypatch.setattr("hermes_cli.auth.refresh_xai_oauth_pure", _refresh_must_not_run)

    adapter = XAIGrokAdapter()
    failed = adapter.get_credential()
    assert failed.bearer == "first-access-token", "starting bearer should be the first entry"

    retry = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=429,
    )

    assert retry is not None, "429 must rotate to next pool entry"
    assert retry.bearer == "second-access-token", (
        f"expected rotation to second entry, got {retry.bearer!r}"
    )


# ---------------------------------------------------------------------------
# Server: path filtering + forwarding
#
# We run the proxy AND a fake upstream as real aiohttp servers on ephemeral
# ports. Avoids pytest-aiohttp's fixtures (extra dependency for one test file).
# ---------------------------------------------------------------------------

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from hermes_cli.proxy.server import create_app  # noqa: E402


class FakeAdapter(UpstreamAdapter):
    """A test adapter that returns a fixed credential without touching disk."""

    def __init__(self, base_url: str, bearer: str = "test-bearer",
                 allowed=None, raise_on_credential=False,
                 retry_bearer: str | None = None):
        self._base_url = base_url
        self._bearer = bearer
        self._allowed = frozenset(allowed or ["/chat/completions"])
        self._raise = raise_on_credential
        self._retry_bearer = retry_bearer
        self.calls = 0
        self.retry_calls = 0

    @property
    def name(self): return "fake"

    @property
    def display_name(self): return "Fake Provider"

    @property
    def allowed_paths(self): return self._allowed

    def is_authenticated(self): return True

    def get_credential(self):
        self.calls += 1
        if self._raise:
            raise RuntimeError("simulated auth failure")
        return UpstreamCredential(
            bearer=self._bearer, base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )

    def get_retry_credential(self, *, failed_credential, status_code):
        _ = failed_credential
        self.retry_calls += 1
        if status_code != 401 or not self._retry_bearer:
            return None
        return UpstreamCredential(
            bearer=self._retry_bearer,
            base_url=self._base_url,
            expires_at="2099-01-01T00:00:00Z",
        )


async def _start_runner(app: "web.Application"):
    """Spin up an aiohttp app on an ephemeral localhost port. Returns (runner, base_url)."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = list(site._server.sockets)  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _build_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def echo(request):
        body = await request.read()
        captured["requests"].append({
            "method": request.method,
            "path": request.path,
            "auth": request.headers.get("Authorization"),
            "body": body.decode("utf-8") if body else "",
        })
        return web.json_response({"echoed": True, "path": request.path})

    async def sse(request):
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for chunk in [b"data: hello\n\n", b"data: world\n\n", b"data: [DONE]\n\n"]:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", echo)
    app.router.add_route("*", "/v1/embeddings", echo)
    app.router.add_route("*", "/v1/sse", sse)
    return app


def _build_retrying_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def maybe_unauthorized(request):
        body = await request.read()
        auth = request.headers.get("Authorization")
        captured["requests"].append({
            "method": request.method,
            "path": request.path,
            "auth": auth,
            "body": body.decode("utf-8") if body else "",
        })
        if auth == "Bearer jwt-bearer":
            return web.json_response({"error": "bad token"}, status=401)
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_route("*", "/v1/chat/completions", maybe_unauthorized)
    return app






def test_server_strips_client_auth_header():
    """The client's Authorization header MUST NOT reach the upstream."""
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={},
                    headers={"Authorization": "Bearer SHOULD_NOT_LEAK"},
                ) as resp:
                    await resp.read()
            assert captured["requests"][0]["auth"] == "Bearer ours"
            assert "SHOULD_NOT_LEAK" not in captured["requests"][0]["auth"]
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def _build_sse_upstream(
    frames: list[bytes],
    *,
    path: str = "/v1/chat/completions",
) -> "web.Application":
    async def sse(request):
        _ = await request.read()
        resp = web.StreamResponse(
            status=200, headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for chunk in frames:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_route("*", path, sse)
    return app


def test_proxy_appends_done_when_upstream_omits_sentinel():
    """#90848: complete Portal-shaped SSE without [DONE] gets one appended."""
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"LONGCAT_OK"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices":[],"lastOne":true,"usage":{"prompt_tokens":1}}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            text = body.decode("utf-8")
            assert 'data: {"choices":[{"delta":{"content":"LONGCAT_OK"}}]}' in text
            assert '"finish_reason":"stop"' in text
            assert '"lastOne":true' in text
            assert text.count("data: [DONE]") == 1
            assert text.rstrip().endswith("data: [DONE]")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_duplicate_existing_done():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert body.decode("utf-8").count("data: [DONE]") == 1
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_append_done_after_error_event():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            b'data: {"error":{"message":"boom","type":"api_error"}}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert "data: [DONE]" not in body.decode("utf-8")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_proxy_does_not_append_done_after_malformed_trailing_frame():
    async def run():
        frames = [
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
            b'data: {"choices": [MALFORMED]}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_sse_upstream(frames)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="ours")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"stream": True},
                ) as resp:
                    body = await resp.read()
            assert "data: [DONE]" not in body.decode("utf-8")
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Upstream-leg proxy routing
#
# The upstream leg must honour this machine's proxy policy — HTTP(S)_PROXY /
# ALL_PROXY with NO_PROXY, else the macOS system proxy — through the same
# shared resolver the gateway adapters use, instead of a bare ``ClientSession``.
#
# aiohttp's own ``trust_env`` is deliberately NOT used here: it does more than
# proxy discovery, it also enables netrc-derived authentication, which collides
# with the explicit ``Authorization`` header this proxy attaches
# (``ValueError: Cannot combine AUTHORIZATION header with AUTH argument``) and
# it ignores ``ALL_PROXY`` and port-qualified ``NO_PROXY`` entries.
#
# Every test below drives the real ``create_app`` server against real local
# fixtures: an origin and, where relevant, a forwarding proxy.
# ---------------------------------------------------------------------------

_PROXY_ENV_VARS = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
)


@pytest.fixture(autouse=True)
def _hermetic_proxy_environment(monkeypatch):
    """No ambient proxy, netrc or system-proxy state leaks into these tests."""
    for var in _PROXY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("NETRC", raising=False)
    # The live macOS system proxy (SystemConfiguration) would otherwise route loopback
    # fixtures through a local proxy on a developer machine.
    monkeypatch.setattr("gateway.platforms.base._detect_macos_system_proxy", lambda: None)


async def _pump(reader, writer) -> None:
    """Copy one direction of a tunnel; either side closing ends it."""
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except Exception:  # noqa: BLE001 - a closed peer is a normal tunnel end
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


async def _read_head(reader) -> bytes:
    data = b""
    while b"\r\n\r\n" not in data and len(data) < 65536:
        chunk = await reader.read(4096)
        if not chunk:
            break
        data += chunk
    return data


def _build_forwarding_proxy(captured: Dict[str, Any]):
    """A real HTTP proxy fixture speaking both proxy modes.

    The two dependency arms reach a forward proxy differently: request-level ``proxy=``
    sends the absolute request URI, while an ``aiohttp-socks`` connector sends ``CONNECT``
    and expects a tunnel. A real proxy speaks both, so this fixture does too — otherwise the
    proxy matrix would silently become arm-dependent. Records every proxied request.
    """
    async def handle(reader, writer):
        upstream_writer = None
        try:
            head = await asyncio.wait_for(_read_head(reader), timeout=10)
            if not head:
                return
            request_line, _, rest = head.partition(b"\r\n")
            method, target, version = request_line.decode("latin-1").split(" ", 2)
            if method.upper() == "CONNECT":
                captured["proxied"].append({"target": target, "auth": None,
                                            "proxy_authorization": None, "mode": "connect"})
                host, _, port = target.rpartition(":")
                upstream_reader, upstream_writer = await asyncio.open_connection(host, int(port))
                writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
                await writer.drain()
            else:
                parsed = urlsplit(target)
                headers = {k.strip().lower(): v.strip()
                           for k, _, v in (line.partition(":") for line in
                                           rest.decode("latin-1").split("\r\n")) if k}
                captured["proxied"].append({"target": target, "auth": headers.get("authorization"),
                                            "proxy_authorization": headers.get("proxy-authorization"),
                                            "mode": "absolute_uri"})
                path = parsed.path or "/"
                if parsed.query:
                    path = f"{path}?{parsed.query}"
                upstream_reader, upstream_writer = await asyncio.open_connection(
                    parsed.hostname, parsed.port or 80
                )
                upstream_writer.write(f"{method} {path} {version}\r\n".encode("latin-1") + rest)
                await upstream_writer.drain()
            await asyncio.gather(_pump(reader, upstream_writer), _pump(upstream_reader, writer))
        except Exception:  # noqa: BLE001 - fixture: a failed leg closes, the assertion reports it
            pass
        finally:
            for stream_writer in (upstream_writer, writer):
                if stream_writer is not None:
                    try:
                        stream_writer.close()
                    except Exception:  # noqa: BLE001
                        pass
    return handle


class _UpstreamFixture:
    """Origin + (optional) forward proxy + subscription proxy, one lifecycle."""

    def __init__(self, monkeypatch, *, proxy: bool = False, retry_bearer: str | None = None,
                 adapter_base: str | None = None):
        self.monkeypatch = monkeypatch
        self.captured: Dict[str, Any] = {"requests": [], "proxied": [], "socks_targets": []}
        self.origin = _build_fake_upstream(self.captured)
        if retry_bearer is not None:
            self.origin = _build_retrying_fake_upstream(self.captured)
        self._proxy_uses_raw_server = proxy
        self._adapter_base = adapter_base
        self.proxy_base: str | None = None
        self.origin_base: str | None = None
        self.proxy_runner = None
        self.proxy_server = None
        self.origin_runner = None
        self.proxy_runner_public = None
        self.retry_bearer = retry_bearer

    def adapter(self, bearer: str = "ours"):
        base = self._adapter_base or f"{self.origin_base}/v1"
        return FakeAdapter(base, bearer=bearer, retry_bearer=self.retry_bearer)

    async def __aenter__(self):
        self.origin_runner, self.origin_base = await _start_runner(self.origin)
        if self._proxy_uses_raw_server:
            self.proxy_server, port = await _start_raw_server(_build_forwarding_proxy(self.captured))
            self.proxy_base = f"http://127.0.0.1:{port}"
        return self

    async def __aexit__(self, *exc):
        for runner in (self.proxy_runner_public, self.proxy_runner, self.origin_runner):
            if runner is not None:
                await runner.cleanup()
        if self.proxy_server is not None:
            self.proxy_server.close()

    @property
    def origin_host_port(self) -> str:
        assert self.origin_base is not None
        return self.origin_base.replace("http://", "")

    async def request(self, *, path: str = "/v1/chat/completions", bearer: str = "ours"):
        adapter = self.adapter(bearer)
        self.proxy_runner_public, proxy_base = await _start_runner(create_app(adapter))
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{proxy_base}{path}", json={}) as resp:
                return resp.status, await resp.json()


def _run(coro):
    return asyncio.run(coro)


@pytest.mark.parametrize("env_var", ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"])
def test_upstream_leg_follows_environment_proxy(monkeypatch, env_var):
    """A configured proxy env var carries the upstream request, whichever casing wins.

    The origin still receives this proxy's own bearer, and the request the forward proxy
    sees is the one addressed to the origin — either as an absolute request URI
    (request-level ``proxy=``) or as a CONNECT authority (an ``aiohttp-socks`` connector).
    """
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv(env_var, fx.proxy_base)
            status, payload = await fx.request()
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 1, (
                f"upstream leg did not use {env_var}: {fx.captured['proxied']}"
            )
            assert fx.origin_host_port in fx.captured["proxied"][0]["target"], (
                "proxy received a request that was not addressed to the origin"
            )
            assert fx.captured["requests"][0]["auth"] == "Bearer ours"
    _run(scenario())


def test_upstream_leg_bypasses_the_proxy_for_a_no_proxy_target(monkeypatch):
    """NO_PROXY excludes the target: a dead proxy must not be contacted at all.

    The first arm is the control — the identical dead-proxy configuration without NO_PROXY
    fails closed, which is what makes the bypass arm's success attributable to the bypass.
    """
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            status, payload = await fx.request()
            assert status == 502, f"dead proxy was expected to fail bounded, got {status} {payload}"

            monkeypatch.setenv("NO_PROXY", fx.origin_host_port)
            status, payload = await fx.request()
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 0, "bypassed leg must not touch the proxy"
    _run(scenario())


def test_upstream_leg_bypasses_the_proxy_for_a_no_proxy_wildcard(monkeypatch):
    """``NO_PROXY=*`` disables proxy routing for the upstream leg."""
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            monkeypatch.setenv("NO_PROXY", "*")
            status, payload = await fx.request()
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 0
    _run(scenario())


def test_upstream_leg_is_direct_without_proxy_configuration(monkeypatch):
    """No proxy configured: unchanged direct behaviour, no proxy contact."""
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            status, payload = await fx.request()
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 0
            assert fx.captured["requests"][0]["auth"] == "Bearer ours"
    _run(scenario())


def test_upstream_leg_dead_proxy_fails_bounded(monkeypatch):
    """A dead proxy yields the bounded upstream_unreachable error, not a hang or a crash."""
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            status, payload = await fx.request()
            assert status == 502
            assert payload["error"]["code"] == "upstream_unreachable"
            assert fx.captured["requests"] == []
    _run(scenario())


def test_upstream_leg_malformed_proxy_never_falls_back_to_direct(monkeypatch):
    """A proxy value aiohttp cannot use fails bounded — and never silently goes direct.

    ARM A (no ``aiohttp-socks``): the malformed value is a client error at request time.
    """
    if _aiohttp_socks_present():
        pytest.skip("ARM B environment: aiohttp-socks refuses the value at connector construction")

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "127.0.0.1:7890")
            status, payload = await fx.request()
            assert status == 502
            assert payload["error"]["code"] == "upstream_unreachable"
            assert fx.captured["requests"] == [], "a malformed proxy must not fall back to a direct send"
    _run(scenario())


def test_upstream_leg_malformed_proxy_fails_bounded_arm_b(monkeypatch):
    """ARM B: the same malformed value is refused at connector construction — still bounded and
    still never a silent direct send (aiohttp-socks rejects it before any connection)."""
    pytest.importorskip("aiohttp_socks")

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "127.0.0.1:7890")
            status, payload = await fx.request()
            return status, payload, fx.captured

    status, payload, captured = _run(scenario())
    assert status == 500
    assert "proxy session init failed" in payload["error"]["message"], payload
    assert captured["requests"] == [], "a malformed proxy must not fall back to a direct send"
    assert captured["proxied"] == []


def _write_netrc(tmp_path: Path) -> str:
    """A netrc whose ``default`` entry matches every host — the broadest ambient hazard."""
    path = tmp_path / "netrc"
    path.write_text("default\n  login fixture\n  password fixture-secret\n")
    return str(path)


def test_explicit_authorization_survives_matching_netrc(monkeypatch, tmp_path):
    """A matching netrc entry must not collide with our explicit Authorization header.

    Control arm: the very same netrc collapses aiohttp's ``trust_env=True`` with a
    ValueError — that is the defect this call site must not reintroduce.
    """
    netrc_path = _write_netrc(tmp_path)
    monkeypatch.setenv("NETRC", netrc_path)

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            # Control arm: the same netrc collapses aiohttp's own ``trust_env`` path with
            # exactly the ValueError this call site must not reintroduce.
            async with aiohttp.ClientSession(trust_env=True) as session:
                with pytest.raises(ValueError, match="Cannot combine AUTHORIZATION header"):
                    await session.get(f"{fx.origin_base}/v1/models",
                                      headers={"Authorization": "Bearer ours"})

            monkeypatch.setenv("HTTP_PROXY", fx.proxy_base)
            status, payload = await fx.request()
            assert status == 200, payload
            assert fx.captured["requests"][0]["auth"] == "Bearer ours"
            assert fx.captured["proxied"][0]["proxy_authorization"] is None, (
                "netrc must not supply proxy credentials at this call site"
            )
    _run(scenario())


def test_retry_after_401_reuses_the_same_proxy_policy(monkeypatch):
    """The one-shot 401 retry re-resolves the proxy policy instead of bypassing it."""
    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True, retry_bearer="retry-bearer") as fx:
            monkeypatch.setenv("HTTP_PROXY", fx.proxy_base)
            status, payload = await fx.request(bearer="jwt-bearer")
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 2, "retry leg bypassed the proxy policy"
            assert [r["auth"] for r in fx.captured["requests"]] == [
                "Bearer jwt-bearer", "Bearer retry-bearer",
            ]

            monkeypatch.setenv("NO_PROXY", fx.origin_host_port)
            status, payload = await fx.request(bearer="jwt-bearer")
            assert status == 200, payload
            assert len(fx.captured["proxied"]) == 2, "bypassed retry leg must not touch the proxy"
    _run(scenario())


@pytest.mark.parametrize("yaml_body, proxied", [
    ("gateway:\n  trust_env: false\n", False),
    ("gateway:\n  trust_env: true\n", True),
    ("{}\n", True),
])
def test_upstream_proxy_routing_follows_gateway_trust_env(monkeypatch, tmp_path, yaml_body, proxied):
    """The upstream leg inherits the one machine-level proxy-env policy knob.

    No new subscription-proxy setting exists: ``gateway.trust_env`` (default true)
    decides whether the environment/system proxy is honoured at all.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(yaml_body)

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
            status, _payload = await fx.request()
            assert status == (502 if proxied else 200), (
                "gateway.trust_env must control upstream proxy routing"
            )
    _run(scenario())


# ---------------------------------------------------------------------------
# Event-loop ownership of aiohttp transport construction
#
# ``aiohttp-socks`` is a project-declared optional extra (pyproject ``matrix``) and it changes
# this seam: ``proxy_kwargs_for_aiohttp()`` constructs a ``ProxyConnector`` for *every* proxy
# scheme once that package is importable, and connector construction binds the running event
# loop. So the blocking resolution may run off-loop, but transport kwargs must not.
# ARM A = aiohttp-socks absent; ARM B = present.
# ---------------------------------------------------------------------------


def _aiohttp_socks_present() -> bool:
    return importlib.util.find_spec("aiohttp_socks") is not None


async def _start_raw_server(handler) -> tuple:
    """An asyncio server for protocols aiohttp's web layer cannot speak (CONNECT, SOCKS5)."""
    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


def _socks5_handler(captured: Dict[str, Any]):
    """Minimal SOCKS5 CONNECT proxy: no-auth greeting, CONNECT, then tunnel both ways."""
    async def handle(reader, writer):
        try:
            version, nmethods = await reader.readexactly(2)
            assert version == 5, f"unexpected SOCKS version {version}"
            await reader.readexactly(nmethods)
            writer.write(b"\x05\x00")  # no authentication required
            await writer.drain()
            version, command, _reserved, atyp = await reader.readexactly(4)
            assert version == 5 and command == 1, f"unsupported SOCKS request {command}"
            if atyp == 1:
                host = socket.inet_ntoa(await reader.readexactly(4))
            elif atyp == 3:
                (length,) = await reader.readexactly(1)
                host = (await reader.readexactly(length)).decode()
            else:
                host = socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
            port = int.from_bytes(await reader.readexactly(2), "big")
            captured["socks_targets"].append(f"{host}:{port}")
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
            writer.write(b"\x05\x00\x00\x01\x00\x00\x00\x00\x00\x00")
            await writer.drain()
            await asyncio.gather(_pump(reader, upstream_writer), _pump(upstream_reader, writer))
        except Exception:  # noqa: BLE001 - fixture: a failed leg closes, the assertion reports it
            pass
        finally:
            writer.close()
    return handle


def test_proxy_resolution_may_run_off_loop(monkeypatch):
    """The blocking resolution is permitted in a worker thread (it may shell out to scutil)."""
    from hermes_cli.proxy.server import _upstream_proxy_url

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")

    async def scenario():
        return await asyncio.to_thread(_upstream_proxy_url, "https://api.x.ai/v1/models")

    assert _run(scenario()) == "http://127.0.0.1:7890"


def test_transport_kwargs_are_built_on_the_serving_loop(monkeypatch):
    """Transport construction must not happen in the worker thread that resolved the proxy.

    With ``aiohttp-socks`` installed the kwargs carry a connector, and building one off-loop
    fails with ``RuntimeError: no running event loop`` before the request ever leaves.
    """
    from hermes_cli.proxy import server as server_module

    seen: Dict[str, Any] = {}
    original = server_module._proxy_transport_kwargs

    def spy(proxy_url):
        seen["builder_thread"] = threading.get_ident()
        return original(proxy_url)

    monkeypatch.setattr(server_module, "_proxy_transport_kwargs", spy)

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            seen["loop_thread"] = threading.get_ident()
            monkeypatch.setenv("HTTP_PROXY", fx.proxy_base)
            return await fx.request()

    status, payload = _run(scenario())
    assert status == 200, payload
    assert seen["builder_thread"] == seen["loop_thread"], (
        "transport kwargs must be built on the serving event loop"
    )


def test_no_proxy_selected_builds_no_transport():
    """Nothing configured → no connector and no request transport kwargs."""
    from hermes_cli.proxy.server import _proxy_transport_kwargs

    assert _proxy_transport_kwargs(None) == ({}, {})


def test_http_proxy_leg_uses_a_loop_built_connector_arm_b(monkeypatch):
    """ARM B: with aiohttp-socks installed the HTTP-proxy leg is a CONNECT tunnel whose
    connector is built on the serving loop — and it completes end to end.

    A worker-thread build returns HTTP 500 ``proxy session init failed: no running event loop``
    before reaching the proxy, so this test discriminates that regression.
    """
    pytest.importorskip("aiohttp_socks")

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("HTTP_PROXY", fx.proxy_base)
            status, payload = await fx.request()
            return status, payload, fx.captured

    status, payload, captured = _run(scenario())
    assert status == 200, payload
    assert captured["proxied"][0]["mode"] == "connect", captured["proxied"]
    assert captured["requests"][0]["auth"] == "Bearer ours"


def test_socks_proxy_tunnels_end_to_end_arm_b(monkeypatch):
    """ARM B: a SOCKS connector built on the loop really tunnels the request."""
    pytest.importorskip("aiohttp_socks")

    async def scenario():
        captured: Dict[str, Any] = {"requests": [], "proxied": [], "socks_targets": []}
        origin_runner, origin_base = await _start_runner(_build_fake_upstream(captured))
        server, port = await _start_raw_server(_socks5_handler(captured))
        try:
            monkeypatch.setenv("ALL_PROXY", f"socks5://127.0.0.1:{port}")
            runner, proxy_base = await _start_runner(create_app(FakeAdapter(f"{origin_base}/v1", bearer="ours")))
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{proxy_base}/v1/chat/completions", json={}) as resp:
                        return resp.status, await resp.text(), captured
            finally:
                await runner.cleanup()
        finally:
            server.close()
            await origin_runner.cleanup()

    status, text, captured = _run(scenario())
    assert status == 200, text
    assert captured["requests"][0]["auth"] == "Bearer ours"
    assert captured["socks_targets"], "the SOCKS connector did not tunnel to the origin"


def test_socks_proxy_without_aiohttp_socks_is_ignored_arm_a(monkeypatch):
    """ARM A: without aiohttp-socks a SOCKS proxy is ignored (documented) and the leg is direct."""
    if _aiohttp_socks_present():
        pytest.skip("ARM B environment: aiohttp-socks installed")

    async def scenario():
        async with _UpstreamFixture(monkeypatch, proxy=True) as fx:
            monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:1")
            return await fx.request()

    status, payload = _run(scenario())
    assert status == 200, payload
    assert payload is not None


def test_unparseable_upstream_url_fails_bounded(monkeypatch):
    """A base URL the resolver cannot parse is a bounded 500, not a hang or a raw crash."""
    async def scenario():
        async with _UpstreamFixture(monkeypatch, adapter_base="http://[::1/v1") as fx:
            return await fx.request()

    status, payload = _run(scenario())
    assert status == 500
    assert "proxy session init failed" in payload["error"]["message"]


# ---------------------------------------------------------------------------
# NO_PROXY target authority
#
# The authority handed to the shared resolver must be a fully serialized ``host[:port]``:
# an implicit http/https port made explicit, an IPv6 literal bracketed, and no userinfo,
# path, query or fragment — otherwise a port-qualified NO_PROXY entry can never decide.
# ---------------------------------------------------------------------------


def test_upstream_authority_is_fully_serialized():
    """The target handed to the shared resolver carries the effective port and no userinfo."""
    from hermes_cli.proxy.server import _upstream_authority

    assert _upstream_authority("https://api.x.ai/v1/models") == "api.x.ai:443"
    assert _upstream_authority("http://example.test/v1/x") == "example.test:80"
    assert _upstream_authority("https://api.x.ai:8443/v1/models") == "api.x.ai:8443"
    assert _upstream_authority("https://user:pw@api.x.ai/v1/x") == "api.x.ai:443"
    assert _upstream_authority("https://api.x.ai/v1/x?a=1#frag") == "api.x.ai:443"
    assert _upstream_authority("https://[::1]/v1/x") == "[::1]:443"
    assert _upstream_authority("http://[::1]:8080/v1/x") == "[::1]:8080"
    assert _upstream_authority("not-a-url") is None


@pytest.mark.parametrize("url, no_proxy, bypassed", [
    ("https://api.x.ai/v1/models", "api.x.ai:443", True),              # implicit 443
    ("http://example.test/v1/x", "example.test:80", True),             # implicit 80
    ("https://api.x.ai:8443/v1/models", "api.x.ai:8443", True),        # explicit port
    ("https://api.x.ai:8443/v1/models", "api.x.ai:443", False),        # proxy still used
    ("https://[::1]/v1/x", "[::1]:443", True),                         # bracketed IPv6 + port
    ("http://[::1]:8080/v1/x", "[::1]:8080", True),                    # explicit IPv6 port
    ("http://[::1]:8080/v1/x", "::1", True),                           # bare IPv6 host
    ("https://api.x.ai/v1/models", "api.x.ai", True),                  # bare host
    ("https://api.x.ai/v1/models", ".x.ai", True),                     # domain suffix
    ("https://api.x.ai/v1/models", "*.x.ai", True),                    # wildcard suffix
    ("https://api.x.ai/v1/models", "*", True),                         # wildcard all
    ("https://api.x.ai/v1/models", "example.test,api.x.ai:443", True),  # list membership
    ("https://api.x.ai/v1/models", "example.test", False),             # unrelated entry
])
def test_no_proxy_authority_decisions(monkeypatch, url, no_proxy, bypassed):
    """Observable bypass decision for the upstream leg under port-qualified NO_PROXY."""
    from hermes_cli.proxy.server import _upstream_proxy_url

    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    resolved = _upstream_proxy_url(url)
    assert (resolved is None) is bypassed, resolved
