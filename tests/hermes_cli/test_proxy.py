"""Tests for the `hermes proxy` subcommand and its upstream adapters."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

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
from yarl import URL  # noqa: E402

from hermes_cli.proxy.server import (  # noqa: E402
    _open_upstream_request,
    _stream_upstream_response,
    create_app,
)


class FakeAdapter(UpstreamAdapter):
    """A test adapter that returns a fixed credential without touching disk."""

    def __init__(self, base_url: str, bearer: str = "test-bearer",
                 allowed=None, raise_on_credential=False,
                 retry_bearer: str | None = None,
                 upstream_headers: dict[str, str] | None = None,
                 owned_headers: set[str] | None = None):
        self._base_url = base_url
        self._bearer = bearer
        self._allowed = frozenset(allowed or ["/chat/completions"])
        self._raise = raise_on_credential
        self._retry_bearer = retry_bearer
        self._upstream_headers = dict(upstream_headers or {})
        self._owned_headers = frozenset(owned_headers or self._upstream_headers)
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

    def get_owned_upstream_header_names(self):
        return self._owned_headers

    def get_upstream_headers(self, credential):
        _ = credential
        return dict(self._upstream_headers)

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
            "raw_path": request.raw_path,
            "auth": request.headers.get("Authorization"),
            "headers": dict(request.headers),
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
    app.router.add_route("*", "/v1/responses", echo)
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






def _build_rate_limited_fake_upstream(captured: Dict[str, Any]) -> "web.Application":
    async def rate_limited(request):
        captured["requests"].append(
            {
                "auth": request.headers.get("Authorization"),
                "path": request.path,
            }
        )
        return web.json_response({"error": "rate limited"}, status=429)

    app = web.Application()
    app.router.add_post("/v1/chat/completions", rate_limited)
    return app


def _build_codex_sse_upstream(expected_chunks: list[bytes]) -> "web.Application":
    async def responses(request):
        await request.read()
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for chunk in expected_chunks:
            await resp.write(chunk)
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/responses", responses)
    return app


def _build_truncated_sse_upstream() -> "web.Application":
    async def responses(request):
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Content-Length": "512",
            },
        )
        await resp.prepare(request)
        await resp.write(b"event: response.output_text.delta\ndata: partial\n\n")
        if request.transport is not None:
            request.transport.abort()
        return resp

    app = web.Application()
    app.router.add_post("/v1/responses", responses)
    return app


def test_cancelled_upstream_request_closes_new_client_session():
    session = SimpleNamespace(
        request=AsyncMock(side_effect=asyncio.CancelledError()),
        close=AsyncMock(),
    )
    with patch(
        "hermes_cli.proxy.server.aiohttp.ClientSession",
        return_value=session,
    ):
        with pytest.raises(asyncio.CancelledError):
            asyncio.run(
                _open_upstream_request(
                    method="POST",
                    url=URL("http://127.0.0.1:1/v1/responses", encoded=True),
                    body=b"{}",
                    headers={},
                    timeout=None,
                )
            )
    session.close.assert_awaited_once_with()


def test_prepare_failure_releases_upstream_response_and_session():
    upstream = MagicMock()
    upstream.status = 200
    upstream.headers = {}
    session = SimpleNamespace(close=AsyncMock())
    request = SimpleNamespace(transport=None)

    with patch.object(
        web.StreamResponse,
        "prepare",
        new=AsyncMock(side_effect=ConnectionResetError("downstream gone")),
    ):
        with pytest.raises(ConnectionResetError, match="downstream gone"):
            asyncio.run(_stream_upstream_response(request, upstream, session))

    upstream.release.assert_called_once_with()
    session.close.assert_awaited_once_with()


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


def test_server_retries_once_with_adapter_credential_after_401():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(
            _build_retrying_fake_upstream(captured)
        )
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            bearer="jwt-bearer",
            retry_bearer="refreshed-bearer",
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"model": "test"},
                ) as resp:
                    assert resp.status == 200
                    await resp.read()
            assert [row["auth"] for row in captured["requests"]] == [
                "Bearer jwt-bearer",
                "Bearer refreshed-bearer",
            ]
            assert adapter.retry_calls == 1
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_returns_429_when_adapter_has_no_rotation():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(
            _build_rate_limited_fake_upstream(captured)
        )
        adapter = FakeAdapter(f"{upstream_base}/v1", bearer="only-bearer")
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"model": "test"},
                ) as resp:
                    assert resp.status == 429
                    await resp.read()
            assert len(captured["requests"]) == 1
            assert adapter.retry_calls == 1
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_preserves_raw_query_and_does_not_log_it(caplog):
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        raw_query = "x=%252f&a=%2F&secret=do-not-log"
        try:
            async with aiohttp.ClientSession() as session:
                url = URL(f"{proxy_base}/v1/responses?{raw_query}", encoded=True)
                async with session.post(url, json={"model": "gpt-5.6-sol"}) as resp:
                    assert resp.status == 200
                    await resp.read()
            assert captured["requests"][0]["raw_path"] == (
                f"/v1/responses?{raw_query}"
            )
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    caplog.set_level("DEBUG", logger="hermes_cli.proxy.server")
    asyncio.run(run())
    assert "do-not-log" not in caplog.text
    assert "%252f" not in caplog.text


def test_server_preserves_codex_responses_body_bytes():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        payload = (
            b'{"model":"gpt-5.6-sol","input":['
            b'{"type":"function_call_output","call_id":"call_1",'
            b'"output":"exact bytes"}],"stream":false}'
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    assert resp.status == 200
                    await resp.read()
            assert captured["requests"][0]["body"].encode() == payload
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_preserves_codex_responses_sse_bytes():
    async def run():
        chunks = [
            b'event: response.output_item.added\ndata: {"type":"response.output_item.added"}\n\n',
            b'event: response.completed\ndata: {"type":"response.completed"}\n\n',
        ]
        upstream_runner, upstream_base = await _start_runner(
            _build_codex_sse_upstream(chunks)
        )
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    json={"model": "gpt-5.6-sol", "stream": True, "input": []},
                ) as resp:
                    assert resp.status == 200
                    assert await resp.read() == b"".join(chunks)
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_aborts_downstream_on_truncated_upstream_sse():
    async def run():
        upstream_runner, upstream_base = await _start_runner(
            _build_truncated_sse_upstream()
        )
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                with pytest.raises(
                    (
                        aiohttp.ClientPayloadError,
                        aiohttp.ServerDisconnectedError,
                        aiohttp.ClientConnectionError,
                    )
                ):
                    async with session.post(
                        f"{proxy_base}/v1/responses",
                        json={"model": "gpt-5.6-sol", "stream": True, "input": []},
                    ) as resp:
                        await resp.read()
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_codex_proxy_rejects_chat_completions():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses", "/models"],
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/chat/completions",
                    json={"model": "gpt-5.6-sol"},
                ) as resp:
                    body = await resp.json()
                    assert resp.status == 404
                    assert body["error"]["code"] == "path_not_allowed"
            assert captured["requests"] == []
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_strips_owned_account_header_when_adapter_has_no_value():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            allowed=["/responses"],
            upstream_headers={
                "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
                "originator": "codex_cli_rs",
            },
            owned_headers={"User-Agent", "originator", "ChatGPT-Account-ID"},
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    json={"model": "gpt-5.6-sol", "input": []},
                    headers={"chatgpt-account-id": "attacker"},
                ) as resp:
                    assert resp.status == 200
                    await resp.read()
            headers = {k.lower(): v for k, v in captured["requests"][0]["headers"].items()}
            assert "chatgpt-account-id" not in headers
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


def test_server_adapter_headers_override_spoofed_client_values():
    async def run():
        captured: Dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        adapter = FakeAdapter(
            f"{upstream_base}/v1",
            bearer="codex-token",
            allowed=["/responses"],
            upstream_headers={
                "User-Agent": "codex_cli_rs/0.0.0 (Hermes Agent)",
                "originator": "codex_cli_rs",
                "ChatGPT-Account-ID": "acct-real",
            },
        )
        proxy_runner, proxy_base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    json={"model": "gpt-5.6-sol", "input": []},
                    headers={
                        "Authorization": "Bearer CLIENT",
                        "User-Agent": "spoofed",
                        "originator": "spoofed",
                        "ChatGPT-Account-ID": "acct-spoofed",
                    },
                ) as resp:
                    assert resp.status == 200
                    await resp.read()
            headers = {k.lower(): v for k, v in captured["requests"][0]["headers"].items()}
            assert headers["authorization"] == "Bearer codex-token"
            assert headers["user-agent"] == "codex_cli_rs/0.0.0 (Hermes Agent)"
            assert headers["originator"] == "codex_cli_rs"
            assert headers["chatgpt-account-id"] == "acct-real"
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------






