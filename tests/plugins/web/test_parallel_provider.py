"""Regression tests for the keyed Parallel GA/v1 provider path."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from parallel import Parallel

from plugins.web.parallel.provider import (
    ParallelWebSearchProvider,
    _resolve_search_mode,
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "advanced"),
        ("not-a-mode", "advanced"),
        ("agentic", "advanced"),
        ("one-shot", "basic"),
        ("fast", "basic"),
        ("basic", "basic"),
        ("advanced", "advanced"),
        ("turbo", "turbo"),
        ("v1-fast", "fast"),
    ],
)
def test_search_mode_preserves_legacy_semantics_and_explicit_v1_modes(
    monkeypatch: pytest.MonkeyPatch,
    configured: str | None,
    expected: str,
) -> None:
    if configured is None:
        monkeypatch.delenv("PARALLEL_SEARCH_MODE", raising=False)
    else:
        monkeypatch.setenv("PARALLEL_SEARCH_MODE", configured)

    assert _resolve_search_mode() == expected


def test_search_uses_v1_client_and_preserves_normalized_result_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeClient:
        def search(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url="https://docs.parallel.ai",
                        title="Parallel docs",
                        excerpts=["First excerpt", "second excerpt"],
                    )
                ]
            )

    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    monkeypatch.setenv("PARALLEL_SEARCH_MODE", "one-shot")
    with (
        patch(
            "plugins.web.parallel.provider._get_sync_client",
            return_value=FakeClient(),
        ),
        patch("tools.interrupt.is_interrupted", return_value=False),
    ):
        result = ParallelWebSearchProvider().search("Parallel SDK", limit=27)

    assert calls == [
        {
            "search_queries": ["Parallel SDK"],
            "objective": "Parallel SDK",
            "mode": "basic",
            "advanced_settings": {"max_results": 20},
        }
    ]
    assert result == {
        "success": True,
        "data": {
            "web": [
                {
                    "url": "https://docs.parallel.ai",
                    "title": "Parallel docs",
                    "description": "First excerpt second excerpt",
                    "position": 1,
                }
            ]
        },
    }


def test_search_serializes_v1_request_through_real_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [],
                "search_id": "search_test",
                "session_id": "session_test",
            },
        )

    transport = httpx.MockTransport(handle)
    http_client = httpx.Client(transport=transport)
    client = Parallel(api_key="test-key", http_client=http_client)
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    monkeypatch.setenv("PARALLEL_SEARCH_MODE", "agentic")

    try:
        with (
            patch(
                "plugins.web.parallel.provider._get_sync_client",
                return_value=client,
            ),
            patch("tools.interrupt.is_interrupted", return_value=False),
        ):
            result = ParallelWebSearchProvider().search("migration contract", limit=7)
    finally:
        client.close()

    assert result == {"success": True, "data": {"web": []}}
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/search"
    payload = json.loads(requests[0].content)
    assert payload["search_queries"] == ["migration contract"]
    assert payload["mode"] == "advanced"
    assert payload["advanced_settings"]["max_results"] == 7


@pytest.mark.asyncio
async def test_extract_uses_v1_client_and_preserves_per_url_result_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    class FakeAsyncClient:
        async def extract(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        url="https://example.com/ok",
                        title="Example",
                        full_content="Full content",
                        excerpts=["fallback excerpt"],
                    )
                ],
                errors=[
                    SimpleNamespace(
                        url="https://example.com/missing",
                        content="not found",
                        error_type="http_error",
                    )
                ],
            )

    urls = ["https://example.com/ok", "https://example.com/missing"]
    monkeypatch.setenv("PARALLEL_API_KEY", "test-key")
    with (
        patch(
            "plugins.web.parallel.provider._get_async_client",
            return_value=FakeAsyncClient(),
        ),
        patch("tools.interrupt.is_interrupted", return_value=False),
    ):
        result = await ParallelWebSearchProvider().extract(urls)

    assert calls == [
        {
            "urls": urls,
            "advanced_settings": {"full_content": True},
        }
    ]
    assert result == [
        {
            "url": "https://example.com/ok",
            "title": "Example",
            "content": "Full content",
            "raw_content": "Full content",
            "metadata": {
                "sourceURL": "https://example.com/ok",
                "title": "Example",
            },
        },
        {
            "url": "https://example.com/missing",
            "title": "",
            "content": "",
            "error": "not found",
            "metadata": {"sourceURL": "https://example.com/missing"},
        },
    ]
