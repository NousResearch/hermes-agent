"""Tests for the /submit slash-command helpers in the Discord adapter.

The interaction-side flow is integration-shaped (requires a running
discord.py client tree); these tests cover the two pure helpers
that do the actual work: HTTP submission and the polling watcher.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.discord import DiscordAdapter, DISCORD_AVAILABLE


pytestmark = pytest.mark.skipif(
    not DISCORD_AVAILABLE,
    reason="discord.py not installed in this environment",
)


def _adapter() -> DiscordAdapter:
    """Build a minimally-initialized adapter (no client connect)."""
    return DiscordAdapter(PlatformConfig())


# ─── _submit_run_via_local_api ──────────────────────────────────────────────

class _FakeAsyncResp:
    def __init__(self, status_code: int, body):
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        if isinstance(self._body, str):
            return json.loads(self._body)
        return self._body


class _FakeAsyncClient:
    def __init__(self, *, response: _FakeAsyncResp):
        self._response = response
        self.calls: List[Dict[str, Any]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, *, json, headers):
        self.calls.append({"url": url, "json": json, "headers": dict(headers)})
        return self._response

    async def get(self, url, *, headers):
        self.calls.append({"url": url, "headers": dict(headers), "method": "GET"})
        return self._response


@pytest.mark.asyncio
async def test_submit_via_local_api_returns_parsed_body(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "secret-key")
    monkeypatch.setenv("API_SERVER_PORT", "8642")

    fake = _FakeAsyncClient(response=_FakeAsyncResp(200, {"id": "run_xyz"}))
    with patch("httpx.AsyncClient", return_value=fake):
        out = await _adapter()._submit_run_via_local_api("hello world")

    assert out == {"id": "run_xyz"}
    assert fake.calls[0]["url"] == "http://127.0.0.1:8642/v1/runs"
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret-key"
    assert fake.calls[0]["json"] == {"input": "hello world"}


@pytest.mark.asyncio
async def test_submit_via_local_api_returns_none_when_key_missing(monkeypatch):
    monkeypatch.delenv("API_SERVER_KEY", raising=False)
    out = await _adapter()._submit_run_via_local_api("hi")
    assert out is None


@pytest.mark.asyncio
async def test_submit_via_local_api_returns_none_on_5xx(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "k")
    fake = _FakeAsyncClient(response=_FakeAsyncResp(503, "down"))
    with patch("httpx.AsyncClient", return_value=fake):
        out = await _adapter()._submit_run_via_local_api("hi")
    assert out is None


# ─── _watch_run_and_edit_message ────────────────────────────────────────────

class _ReplayingAsyncClient:
    """AsyncClient that returns a fixed list of responses one per .get() call."""

    def __init__(self, responses: List[_FakeAsyncResp]):
        self._responses = list(responses)
        self.call_count = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, *, headers):
        self.call_count += 1
        if self._responses:
            return self._responses.pop(0)
        # Final response repeats forever.
        return _FakeAsyncResp(404, "no more")


def _mock_message():
    msg = MagicMock()
    msg.edit = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_watch_run_edits_with_completed_output(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "k")
    monkeypatch.setenv("API_SERVER_PORT", "8642")

    responses = [
        _FakeAsyncResp(200, {"status": "running"}),
        _FakeAsyncResp(200, {
            "status": "completed",
            "output": "the answer is 42",
            "usage": {"total_tokens": 100},
        }),
    ]
    client = _ReplayingAsyncClient(responses)
    msg = _mock_message()

    with patch("httpx.AsyncClient", return_value=client):
        await _adapter()._watch_run_and_edit_message(
            msg, "run_abc", started_at=time.time(), poll_interval=0.001
        )

    msg.edit.assert_awaited()
    edited = msg.edit.await_args.kwargs["content"]
    assert "the answer is 42" in edited
    assert "✅" in edited
    assert "run_abc" in edited
    assert "100 tokens" in edited


@pytest.mark.asyncio
async def test_watch_run_edits_with_failure_breadcrumb(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "k")
    monkeypatch.setenv("API_SERVER_PORT", "8642")

    responses = [
        _FakeAsyncResp(200, {
            "status": "failed",
            "error": "model returned 503",
        }),
    ]
    client = _ReplayingAsyncClient(responses)
    msg = _mock_message()

    with patch("httpx.AsyncClient", return_value=client):
        await _adapter()._watch_run_and_edit_message(
            msg, "run_xyz", started_at=time.time(), poll_interval=0.001
        )

    edited = msg.edit.await_args.kwargs["content"]
    assert "❌" in edited
    assert "run_xyz" in edited
    assert "failed" in edited
    assert "model returned 503" in edited


@pytest.mark.asyncio
async def test_watch_run_truncates_long_output(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "k")

    huge = "x" * 5000
    responses = [
        _FakeAsyncResp(200, {"status": "completed", "output": huge}),
    ]
    client = _ReplayingAsyncClient(responses)
    msg = _mock_message()

    with patch("httpx.AsyncClient", return_value=client):
        await _adapter()._watch_run_and_edit_message(
            msg, "run_long", started_at=time.time(), poll_interval=0.001
        )

    edited = msg.edit.await_args.kwargs["content"]
    # Discord's per-message limit is 2000.
    assert len(edited) <= 2000
    # Truncation marker present.
    assert "…" in edited


@pytest.mark.asyncio
async def test_watch_run_times_out_and_detaches(monkeypatch):
    monkeypatch.setenv("API_SERVER_KEY", "k")

    # Always running, never terminal.
    forever = _ReplayingAsyncClient([_FakeAsyncResp(200, {"status": "running"})] * 50)
    msg = _mock_message()

    with patch("httpx.AsyncClient", return_value=forever):
        await _adapter()._watch_run_and_edit_message(
            msg, "run_slow",
            started_at=time.time(),
            poll_interval=0.001,
            max_wait_seconds=0.05,  # 50ms total budget
        )

    edited = msg.edit.await_args.kwargs["content"]
    assert "⏱" in edited
    assert "run_slow" in edited
    assert "detaching" in edited
