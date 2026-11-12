"""Tests for the `hermes proxy` subcommand and its upstream adapters."""

from __future__ import annotations

import asyncio
import builtins
import json
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import LocalAdapterResponse, UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.adapters.nous_portal import NousPortalAdapter
from hermes_cli.proxy.adapters.openai_codex import OpenAICodexAdapter
from hermes_cli.proxy.adapters.xai import XAIGrokAdapter
from hermes_cli.proxy.cli import cmd_proxy_start


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------


def test_adapter_registry_includes_openai_codex():
    assert "openai-codex" in ADAPTERS


def test_adapter_registry_imports_without_openai_sdk():
    script = """
import importlib.abc
import sys

class _BlockOpenAI(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openai" or fullname.startswith("openai."):
            raise ModuleNotFoundError("blocked for test", name=fullname)
        return None

sys.meta_path.insert(0, _BlockOpenAI())
import hermes_cli.proxy.adapters
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_openai_codex_adapter_translates_nonstreaming_chat_completion(monkeypatch):
    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return {
                "id": "chatcmpl-codex",
                "object": "chat.completion",
                "model": kwargs["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "remembered"},
                             "finish_reason": "stop"}],
            }

    fake_client = MagicMock()
    fake_client.chat.completions = _Completions()
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (fake_client, model),
    )

    result = OpenAICodexAdapter().handle_local_request(
        "/chat/completions",
        json.dumps({
            "model": "gpt-5.6-luna",
            "messages": [{"role": "user", "content": "Extract one fact"}],
            "temperature": 0,
            "max_tokens": 200,
        }).encode(),
    )

    assert result is not None
    assert result.status == 200
    assert json.loads(result.body)["choices"][0]["message"]["content"] == "remembered"
    assert captured == {
        "model": "gpt-5.6-luna",
        "messages": [{"role": "user", "content": "Extract one fact"}],
        "temperature": 0,
        "max_tokens": 200,
    }


def test_openai_codex_adapter_serializes_auxiliary_namespace_response():
    response = SimpleNamespace(
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(role="assistant", content="remembered", tool_calls=None),
            finish_reason="stop",
        )],
        model="gpt-5.6-luna",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2, total_tokens=12),
    )

    payload = json.loads(OpenAICodexAdapter._response_bytes(response))

    assert payload["choices"][0]["message"]["content"] == "remembered"
    assert payload["usage"] == {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}


def test_openai_codex_adapter_lists_luna_for_openai_clients():
    result = OpenAICodexAdapter().handle_local_request("/models", b"")

    assert result is not None
    assert result.status == 200
    payload = json.loads(result.body)
    assert payload["object"] == "list"
    assert "gpt-5.6-luna" in {item["id"] for item in payload["data"]}


def test_openai_codex_adapter_does_not_expose_credential_errors(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (_ for _ in ()).throw(RuntimeError("secret bearer value")),
    )

    result = OpenAICodexAdapter().handle_local_request(
        "/chat/completions",
        json.dumps({"model": "gpt-5.6-luna", "messages": []}).encode(),
    )

    assert result is not None
    assert result.status == 502
    assert "secret bearer value" not in result.body.decode()


def test_openai_codex_adapter_sanitizes_missing_openai_sdk(monkeypatch):
    real_import = builtins.__import__

    def _block_openai(name, *args, **kwargs):
        if name == "openai" or name.startswith("openai."):
            raise ModuleNotFoundError("blocked for test", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_openai)
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (_ for _ in ()).throw(AssertionError("must not resolve without SDK")),
    )

    result = OpenAICodexAdapter().handle_local_request(
        "/chat/completions",
        json.dumps({"model": "gpt-5.6-luna", "messages": []}).encode(),
    )

    assert result is not None
    assert result.status == 502
    assert json.loads(result.body)["error"]["code"] == "codex_inference_failed"


def test_openai_codex_adapter_does_not_sanitize_unrelated_import_errors(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (_ for _ in ()).throw(
            ModuleNotFoundError("unrelated dependency missing", name="unrelated_dependency")
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="unrelated dependency missing"):
        OpenAICodexAdapter().handle_local_request(
            "/chat/completions",
            json.dumps({"model": "gpt-5.6-luna", "messages": []}).encode(),
        )


def test_openai_codex_adapter_does_not_swallow_programming_errors(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (_ for _ in ()).throw(AssertionError("programming defect")),
    )

    with pytest.raises(AssertionError, match="programming defect"):
        OpenAICodexAdapter().handle_local_request(
            "/chat/completions",
            json.dumps({"model": "gpt-5.6-luna", "messages": []}).encode(),
        )


def test_openai_codex_adapter_pool_load_does_not_swallow_programming_errors(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.load_pool",
        lambda provider: (_ for _ in ()).throw(AssertionError("programming defect")),
    )

    with pytest.raises(AssertionError, match="programming defect"):
        OpenAICodexAdapter().is_authenticated()


def test_openai_codex_adapter_rejects_streaming_requests(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.proxy.adapters.openai_codex.resolve_provider_client",
        lambda provider, model: (_ for _ in ()).throw(AssertionError("must reject before inference")),
    )
    result = OpenAICodexAdapter().handle_local_request(
        "/chat/completions",
        json.dumps({"model": "gpt-5.6-luna", "messages": [], "stream": True}).encode(),
    )

    assert result is not None
    assert result.status == 400
    assert json.loads(result.body)["error"]["code"] == "streaming_not_supported"


def test_proxy_cli_requires_named_downstream_key_for_codex(monkeypatch, capsys):
    adapter = MagicMock()
    adapter.is_authenticated.return_value = True
    adapter.requires_downstream_auth = True
    adapter.display_name = "OpenAI Codex OAuth"
    monkeypatch.setattr("hermes_cli.proxy.cli.get_adapter", lambda provider: adapter)
    monkeypatch.delenv("CODEX_PROXY_CLIENT_KEY", raising=False)

    result = cmd_proxy_start(SimpleNamespace(
        provider="openai-codex",
        host="127.0.0.1",
        port=8646,
        auth_key_env="CODEX_PROXY_CLIENT_KEY",
    ))

    assert result == 2
    assert "CODEX_PROXY_CLIENT_KEY" in capsys.readouterr().err








# ---------------------------------------------------------------------------
# NousPortalAdapter
# ---------------------------------------------------------------------------


def _write_auth_store(hermes_home: Path, nous_state: dict[str, Any]) -> Path:
    """Write an auth.json with the given nous state into a hermetic HERMES_HOME."""
    auth_path = hermes_home / "auth.json"
    auth_path.write_text(json.dumps({
        "version": 1,
        "providers": {"nous": nous_state},
    }))
    return auth_path


def test_nous_adapter_never_waits_on_a_free_tier_challenge_under_its_lock(tmp_path, monkeypatch):
    """Every proxied request queues on the adapter lock, so the credential read inside it is a
    background caller: a browser challenge is announced and raised, never waited on."""
    from hermes_cli import anon_challenge
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_auth_store(tmp_path, {"access_token": "a", "refresh_token": "r"})
    seen = []

    def resolve(**_kwargs):
        seen.append(anon_challenge._background.get())
        return {"api_key": "k", "expires_at": "2099-01-01T00:00:00Z",
                "base_url": "https://inference-api.nousresearch.com/v1"}

    with patch("hermes_cli.proxy.adapters.nous_portal.resolve_nous_runtime_credentials", side_effect=resolve):
        NousPortalAdapter().get_credential()
    assert seen == [True]


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
from aiohttp import web

from hermes_cli.proxy.server import create_app, run_server


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


async def _start_runner(app: web.Application):
    """Spin up an aiohttp app on an ephemeral localhost port. Returns (runner, base_url)."""
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = list(site._server.sockets)  # type: ignore[union-attr]
    port = sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


def _build_fake_upstream(captured: dict[str, Any]) -> web.Application:
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


def _build_retrying_fake_upstream(captured: dict[str, Any]) -> web.Application:
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


def test_server_uses_adapter_local_translation_before_upstream_credentials():
    class _LocalAdapter(FakeAdapter):
        def handle_local_request(self, rel_path, body):
            payload = json.loads(body)
            return LocalAdapterResponse(
                status=200,
                body=json.dumps({"translated": payload["model"], "path": rel_path}).encode(),
            )

    async def scenario():
        adapter = _LocalAdapter("https://must-not-be-called.invalid/v1", raise_on_credential=True)
        runner, base = await _start_runner(create_app(adapter))
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base}/v1/chat/completions",
                    json={"model": "gpt-5.6-luna", "messages": []},
                ) as response:
                    assert response.status == 200
                    assert await response.json() == {
                        "translated": "gpt-5.6-luna",
                        "path": "/chat/completions",
                    }
            assert adapter.calls == 0
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_server_does_not_convert_local_adapter_programming_errors():
    class _BuggyLocalAdapter(FakeAdapter):
        def handle_local_request(self, rel_path, body):
            raise AssertionError("programming defect")

    async def scenario():
        adapter = _BuggyLocalAdapter("https://must-not-be-called.invalid/v1")
        runner, base = await _start_runner(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{base}/v1/chat/completions", json={}) as response:
                    assert response.status == 500
                    assert "adapter_translation_failed" not in await response.text()
            assert adapter.calls == 0
        finally:
            await runner.cleanup()

    asyncio.run(scenario())


def test_openai_codex_proxy_refuses_non_loopback_bind():
    async def scenario():
        with pytest.raises(ValueError, match="loopback"):
            await run_server(
                OpenAICodexAdapter(),
                host="0.0.0.0",
                port=0,
                shutdown_event=asyncio.Event(),
            )

    asyncio.run(scenario())


def test_openai_codex_proxy_requires_downstream_bearer_before_inference():
    class _AuthCodexAdapter(OpenAICodexAdapter):
        def handle_local_request(self, rel_path, body):
            return LocalAdapterResponse(status=200, body=b'{"ok":true}')

    async def scenario():
        adapter = _AuthCodexAdapter()
        runner, base = await _start_runner(
            create_app(adapter, inbound_bearer="local-client-secret")
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base}/health") as response:
                    assert response.status == 401
                async with session.get(
                    f"{base}/health",
                    headers={"Authorization": "Bearer local-client-secret"},
                ) as response:
                    assert response.status == 200
                for headers in ({}, {"Authorization": "Bearer wrong"}):
                    async with session.post(
                        f"{base}/v1/chat/completions", json={}, headers=headers
                    ) as response:
                        assert response.status == 401
                async with session.post(
                    f"{base}/v1/chat/completions",
                    json={},
                    headers={"Authorization": "Bearer local-client-secret"},
                ) as response:
                    assert response.status == 200
                    assert await response.json() == {"ok": True}
        finally:
            await runner.cleanup()

    asyncio.run(scenario())






def test_server_strips_client_auth_header():
    """The client's Authorization header MUST NOT reach the upstream."""
    async def run():
        captured: dict[str, Any] = {"requests": []}
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


@pytest.mark.parametrize("bound,headers,allowed", [
    ("127.0.0.1", {}, True),                                              # plain local client
    ("127.0.0.1", {"Origin": "http://{authority}"}, True),                # the proxy's own origin
    ("127.0.0.1", {"Host": "localhost:{port}"}, True),                    # loopback alias
    ("127.0.0.1", {"Host": "localhost.:{port}"}, True),                   # same alias, absolute form
    ("127.0.0.1", {"Host": "rebound.example:{port}"}, False),             # foreign Host
    ("127.0.0.1", {"Origin": "https://site.example"}, False),             # cross-site browser request
    ("127.0.0.1", {"Origin": "null"}, False),                             # opaque browser origin
    ("127.0.0.1", {"Host": "localhost:80", "Origin": "http://localhost"}, True),  # Origin omits :80
    ("127.0.0.1", {"Sec-Fetch-Site": "cross-site"}, False),               # cross-site GET, no Origin
    ("127.0.0.1", {"Host": "user@localhost:{port}"}, False),              # userinfo smuggled into Host
    ("127.0.0.1", {"Origin": "http://{authority}/path"}, False),          # an Origin never has a path
    ("192.0.2.10", {"Host": "192.0.2.10:{port}",                          # specific-IP bind: its own
                    "Origin": "http://192.0.2.10:{port}"}, True),           # address and origin...
    ("192.0.2.10", {"Host": "192.0.2.11:{port}"}, False),                 # ...but not a neighbour's
    ("0.0.0.0", {"Host": "lan-name.example:{port}"}, True),               # LAN API client, no Origin
    ("0.0.0.0", {"Host": "rebound.example:{port}",                        # DNS-rebound page: its
                 "Origin": "http://rebound.example:{port}"}, False),      # Origin equals its Host
    ("0.0.0.0", {"Host": "rebound.example:{port}",                        # ...and its same-origin
                 "Sec-Fetch-Site": "same-origin"}, False),                # GET carries no Origin
    ("::", {"Host": "rebound.example:{port}",
            "Origin": "http://rebound.example:{port}"}, False),
])
def test_loopback_proxy_serves_only_local_non_browser_requests(bound, headers, allowed):
    """The proxy forwards with the operator's credential only for requests whose Host names its
    listener (any name on a wildcard LAN bind) and that carry no browser Origin it cannot vouch for;
    refused ones never go upstream. A wildcard bind has no single origin of its own, and a
    DNS-rebound page's Origin always equals its Host, so no Origin is trusted there."""
    async def run():
        captured: dict[str, Any] = {"requests": []}
        upstream_runner, upstream_base = await _start_runner(_build_fake_upstream(captured))
        proxy_runner, proxy_base = await _start_runner(create_app(FakeAdapter(f"{upstream_base}/v1"), bound_host=bound))
        authority = proxy_base.removeprefix("http://")
        sent = {k: v.format(authority=authority, port=authority.rsplit(":", 1)[1]) for k, v in headers.items()}
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{proxy_base}/v1/chat/completions", json={}, headers=sent) as resp:
                    await resp.read()
                    status = resp.status
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()
        assert status == (200 if allowed else 403)
        assert len(captured["requests"]) == int(allowed)

    asyncio.run(run())


def _build_sse_upstream(
    frames: list[bytes],
    *,
    path: str = "/v1/chat/completions",
) -> web.Application:
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






