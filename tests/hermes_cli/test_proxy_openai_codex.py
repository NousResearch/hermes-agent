"""OpenAI Codex subscription-proxy behavior contracts."""

from __future__ import annotations

import asyncio
from typing import Any, FrozenSet

import pytest

from hermes_cli.proxy.adapters import ADAPTERS, get_adapter
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential
from hermes_cli.proxy.adapters.openai_codex import OpenAICodexAdapter


def test_registry_constructs_openai_codex_adapter() -> None:
    assert ADAPTERS["openai-codex"] is OpenAICodexAdapter
    assert isinstance(get_adapter("openai-codex"), OpenAICodexAdapter)


def test_codex_adapter_reports_rate_limited_pool_as_unavailable(monkeypatch) -> None:
    import hermes_cli.proxy.adapters.openai_codex as module

    monkeypatch.setattr(
        module,
        "get_codex_auth_status",
        lambda: {"logged_in": True, "rate_limited": True},
    )
    assert OpenAICodexAdapter().is_authenticated() is False


def test_codex_adapter_adds_provider_headers_and_refreshes_once(monkeypatch) -> None:
    import hermes_cli.proxy.adapters.openai_codex as module

    refresh_calls: list[bool] = []
    monkeypatch.setattr(module, "get_codex_auth_status", lambda: {"logged_in": True})

    def resolve(*, force_refresh: bool):
        refresh_calls.append(force_refresh)
        return {
            "api_key": "header.payload.signature",
            "base_url": "https://chatgpt.com/backend-api/codex",
        }

    monkeypatch.setattr(module, "resolve_codex_runtime_credentials", resolve)
    adapter = OpenAICodexAdapter()

    assert adapter.is_authenticated() is True
    first = adapter.get_credential()
    assert first.base_url == "https://chatgpt.com/backend-api/codex"
    assert dict(first.headers)["originator"] == "codex_cli_rs"
    retry = adapter.get_retry_credential(
        failed_credential=first,
        status_code=401,
    )
    assert retry is not None
    assert refresh_calls == [False, True]
    assert (
        adapter.get_retry_credential(failed_credential=first, status_code=429) is None
    )


aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

from hermes_cli.proxy.server import create_app  # noqa: E402


class _CodexLikeAdapter(UpstreamAdapter):
    def __init__(self, base_url: str) -> None:
        self._base_url = base_url

    @property
    def name(self) -> str:
        return "openai-codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex"

    @property
    def health_contract(self) -> str:
        return "hermes-codex-responses-v1"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return frozenset({"/responses"})

    def is_authenticated(self) -> bool:
        return True

    def get_credential(self) -> UpstreamCredential:
        return UpstreamCredential(
            bearer="upstream-secret",
            base_url=self._base_url,
            headers=(
                ("originator", "codex_cli_rs"),
                ("User-Agent", "codex_cli_rs/0.0.0 (Hermes Agent)"),
            ),
        )


async def _start_runner(app: "web.Application"):
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()
    sockets = list(site._server.sockets)  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{sockets[0].getsockname()[1]}"


def test_authenticated_proxy_forwards_raw_responses_with_zero_agent_health() -> None:
    async def run() -> None:
        captured: dict[str, Any] = {}

        async def responses(request: "web.Request") -> "web.Response":
            captured["body"] = await request.read()
            captured["authorization"] = request.headers.get("Authorization")
            captured["originator"] = request.headers.get("originator")
            return web.Response(
                body=b'data: {"type":"response.completed"}\n\n',
                headers={"Content-Type": "text/event-stream"},
            )

        upstream_app = web.Application()
        upstream_app.router.add_post("/responses", responses)
        upstream_runner, upstream_base = await _start_runner(upstream_app)
        proxy_runner, proxy_base = await _start_runner(
            create_app(
                _CodexLikeAdapter(upstream_base),
                inbound_bearer_key="local-secret",
            )
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{proxy_base}/health") as rejected:
                    assert rejected.status == 401

                headers = {"Authorization": "Bearer local-secret"}
                async with session.get(
                    f"{proxy_base}/health", headers=headers
                ) as health_response:
                    health = await health_response.json()
                assert health == {
                    "status": "ok",
                    "upstream": "OpenAI Codex",
                    "authenticated": True,
                    "contract": "hermes-codex-responses-v1",
                    "provider": "openai-codex",
                    "agent": False,
                    "memory": False,
                    "tools_resolved": 0,
                }

                raw_body = b'{"model":"gpt-test","stream":true}'
                async with session.post(
                    f"{proxy_base}/v1/responses",
                    data=raw_body,
                    headers={
                        **headers,
                        "originator": "untrusted-client-value",
                        "Content-Type": "application/json",
                    },
                ) as response:
                    streamed = await response.read()
                assert response.status == 200
                assert streamed.startswith(b"data:")

            assert captured == {
                "body": raw_body,
                "authorization": "Bearer upstream-secret",
                "originator": "codex_cli_rs",
            }
        finally:
            await proxy_runner.cleanup()
            await upstream_runner.cleanup()

    asyncio.run(run())
