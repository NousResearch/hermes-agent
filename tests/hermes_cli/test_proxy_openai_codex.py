"""Behavior tests for the OpenAI Codex subscription proxy upstream."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import UpstreamAdapter


def _jwt(account_id: str, *, exp: int = 4_102_444_800) -> str:
    payload = {
        "exp": exp,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def _write_pool(home: Path, tokens: list[str]) -> None:
    entries = []
    for index, token in enumerate(tokens):
        entries.append({
            "id": f"codex-{index}",
            "label": f"account-{index}",
            "auth_type": "oauth",
            "priority": index,
            "source": f"manual:device-{index}",
            "access_token": token,
            "refresh_token": f"refresh-{index}",
            "request_count": 0,
        })
    (home / "auth.json").write_text(json.dumps({
        "version": 1,
        "providers": {},
        "credential_pool": {"openai-codex": entries},
    }))
    (home / "config.yaml").write_text(
        "credential_pool_strategies:\n  openai-codex: least_used\n"
    )


def test_registry_discovers_openai_codex():
    assert "openai-codex" in ADAPTERS
    adapter = get_adapter("OPENAI-CODEX")
    assert isinstance(adapter, UpstreamAdapter)
    assert adapter.name == "openai-codex"
    assert "/responses" in adapter.allowed_paths


def test_codex_adapter_selects_pool_and_attaches_first_party_headers(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    token = _jwt("acct-company-a")
    _write_pool(tmp_path, [token])

    cred = get_adapter("openai-codex").get_credential()

    assert cred.bearer == token
    assert cred.base_url == "https://chatgpt.com/backend-api/codex"
    assert cred.headers["ChatGPT-Account-ID"] == "acct-company-a"
    assert cred.headers["originator"] == "codex_cli_rs"
    assert cred.headers["User-Agent"].startswith("codex_cli_rs/")


def test_codex_adapter_centralizes_least_used_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first, second = _jwt("acct-a"), _jwt("acct-b")
    _write_pool(tmp_path, [first, second])
    adapter = get_adapter("openai-codex")

    assert adapter.get_credential().bearer == first
    assert adapter.get_credential().bearer == second


def test_codex_adapter_rotates_on_429_and_fails_closed_when_exhausted(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first, second = _jwt("acct-a"), _jwt("acct-b")
    _write_pool(tmp_path, [first, second])
    adapter = get_adapter("openai-codex")

    failed = adapter.get_credential()
    retry = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=429,
        error_context={"message": "usage limit reached", "reset_at": 4_102_444_800},
    )
    assert retry is not None
    assert retry.bearer == second

    final = adapter.get_retry_credential(
        failed_credential=retry,
        status_code=429,
        error_context={"message": "usage limit reached"},
    )
    assert final is None
    assert not adapter.is_authenticated()

    stored = json.loads((tmp_path / "auth.json").read_text())
    entries = stored["credential_pool"]["openai-codex"]
    assert [entry["last_status"] for entry in entries] == ["exhausted", "exhausted"]


def test_codex_adapter_refreshes_then_rotates_on_401(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first, second = _jwt("acct-a"), _jwt("acct-b")
    _write_pool(tmp_path, [first, second])
    adapter = get_adapter("openai-codex")
    failed = adapter.get_credential()

    monkeypatch.setattr(
        "hermes_cli.auth.refresh_codex_oauth_pure",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("invalid grant")),
    )
    retry = adapter.get_retry_credential(
        failed_credential=failed,
        status_code=401,
        error_context={"reason": "invalid_token", "message": "token rejected"},
    )

    assert retry is not None
    assert retry.bearer == second


def test_proxy_codex_rewrites_client_headers_and_fails_closed(tmp_path, monkeypatch):
    aiohttp = pytest.importorskip("aiohttp")
    from aiohttp import web
    from hermes_cli.proxy.server import create_app

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    first, second = _jwt("acct-a"), _jwt("acct-b")
    _write_pool(tmp_path, [first, second])
    captured: Dict[str, Any] = {"requests": []}

    async def upstream(request):
        captured["requests"].append(dict(request.headers))
        return web.json_response({
            "error": {"message": "usage limit reached", "type": "usage_limit_reached"}
        }, status=429)

    async def start(app):
        runner = web.AppRunner(app, access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = list(site._server.sockets)[0].getsockname()[1]
        return runner, f"http://127.0.0.1:{port}"

    async def run():
        upstream_app = web.Application()
        upstream_app.router.add_post("/responses", upstream)
        upstream_runner, upstream_base = await start(upstream_app)
        adapter = get_adapter("openai-codex")
        # Keep the test offline while exercising the real proxy and adapter.
        monkeypatch.setattr(adapter, "upstream_base_url", upstream_base)
        proxy_runner, proxy_base = await start(create_app(adapter))
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    json={"model": "gpt-5.6", "input": "hello", "store": False},
                    headers={
                        "Authorization": "Bearer isolated-client-dummy",
                        "ChatGPT-Account-ID": "must-not-leak",
                        "originator": "must-not-leak",
                    },
                ) as response:
                    body = await response.json()
                    assert response.status == 503
                    assert body["error"]["type"] == "credential_pool_exhausted"
            assert len(captured["requests"]) == 2
            assert captured["requests"][0]["ChatGPT-Account-ID"] == "acct-a"
            assert captured["requests"][1]["ChatGPT-Account-ID"] == "acct-b"
            assert all("isolated-client-dummy" not in h.get("Authorization", "") for h in captured["requests"])
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())
