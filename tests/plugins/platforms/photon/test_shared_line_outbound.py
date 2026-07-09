"""Photon adapter handling of permanent shared-line / free-tier rejections.

Covers issue #51897: Photon's free shared-line pool rejects outbound sends
initiated by Hermes (e.g. cron-delivered messages) with a
`Target not allowed for this project` error. This is a permanent, non-
actionable rejection — the adapter must NOT retry or downgrade to plain
text (which would re-hammer the same rejected target); it must surface the
failure once and return a clear explanation.

No Node sidecar is spawned and no ports are bound.
"""
from __future__ import annotations

from typing import Any, Dict

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(monkeypatch: pytest.MonkeyPatch) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    cfg = PlatformConfig(enabled=True, token="", extra={})
    return PhotonAdapter(cfg)


# -- Fatal (shared-line) classification --------------------------------------

@pytest.mark.parametrize(
    "error",
    [
        "AuthenticationError: [spectrum-imessage] Target not allowed for this project",
        "TARGET NOT ALLOWED FOR THIS PROJECT",  # case-insensitive
        "shared line cannot originate outbound sends",
        "outbound not permitted on free tier",
    ],
)
def test_shared_line_rejection_classified_fatal(error: str) -> None:
    assert PhotonAdapter._is_fatal_error(error) is True


def test_transient_error_not_classified_fatal() -> None:
    # A genuine transient failure must NOT be treated as a permanent block.
    assert PhotonAdapter._is_fatal_error("reset reason: overflow") is False
    assert PhotonAdapter._is_fatal_error(None) is False


# -- Send path short-circuits without retry/downgrade -----------------------

@pytest.mark.asyncio
async def test_send_skips_retry_on_shared_line_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    # First (and only) sidecar call returns the permanent rejection.
    calls: list[str] = []

    async def _fake_sidecar_send(space_id: str, text: str):
        calls.append(text)
        from gateway.platforms.base import SendResult

        return SendResult(success=False, error="Target not allowed for this project")

    monkeypatch.setattr(adapter, "_sidecar_send", _fake_sidecar_send)

    result = await adapter._send_with_retry(chat_id="chat-1", content="hello")

    # Exactly one attempt — no retry loop, no plain-text downgrade.
    assert len(calls) == 1
    assert result.success is False
    assert "target not allowed" in (result.error or "").lower()


@pytest.mark.asyncio
async def test_send_retries_on_transient_error_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _make_adapter(monkeypatch)

    # Overflow first, success on the retry — proves transient errors still
    # exercise the backoff path (regression guard for #50185).
    calls: list[str] = []

    async def _fake_sidecar_send(space_id: str, text: str):
        calls.append(text)
        from gateway.platforms.base import SendResult

        if len(calls) == 1:
            return SendResult(success=False, error="reset reason: overflow", retryable=True)
        return SendResult(success=True, message_id="msg-1")

    monkeypatch.setattr(adapter, "_sidecar_send", _fake_sidecar_send)

    result = await adapter._send_with_retry(chat_id="chat-1", content="hello")

    assert len(calls) == 2
    assert result.success is True
