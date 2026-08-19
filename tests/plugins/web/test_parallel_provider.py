"""Regression tests for the keyed Parallel GA/v1 provider path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from plugins.web.parallel.provider import (
    ParallelWebSearchProvider,
    _resolve_search_mode,
)


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (None, "basic"),
        ("agentic", "basic"),
        ("one-shot", "advanced"),
        ("fast", "fast"),
        ("turbo", "turbo"),
        ("basic", "basic"),
        ("advanced", "advanced"),
        ("  TURBO  ", "turbo"),
        ("not-a-mode", "basic"),
    ],
)
def test_search_mode_resolves_legacy_and_v1_values(
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
            "mode": "advanced",
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


@pytest.mark.asyncio
async def test_extract_uses_v1_client_and_preserves_per_url_result_shapes() -> None:
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
