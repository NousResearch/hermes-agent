"""Opt-in conversational blank-line splitting for PhotonAdapter.send().

Photon shares the ``split_outgoing_*`` extra keys with Telegram/WhatsApp (see
``BasePlatformAdapter._init_conversational_split_config``). With the opt-in
off (default), ``send()`` makes exactly one ``_sidecar_send`` call as before;
opted in, blank-line-separated paragraphs go out as separate iMessage/Photon
bubbles paced by ``split_outgoing_delay_seconds``, and each part is still
individually truncated by ``_sidecar_send``'s MAX_MESSAGE_LENGTH guard.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.photon.adapter import PhotonAdapter


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch, **extra
) -> PhotonAdapter:
    monkeypatch.setenv("PHOTON_PROJECT_ID", "test-project-id")
    monkeypatch.setenv("PHOTON_PROJECT_SECRET", "test-project-secret")
    adapter = PhotonAdapter(PlatformConfig(enabled=True, token="", extra=dict(extra)))
    adapter._sidecar_send = AsyncMock(
        side_effect=[
            SendResult(success=True, message_id=f"msg-{i}") for i in range(10)
        ]
    )
    return adapter


def _sent_texts(adapter: PhotonAdapter) -> list:
    return [c.args[1] for c in adapter._sidecar_send.call_args_list]


@pytest.mark.asyncio
async def test_split_is_opt_out_by_default_single_sidecar_call(monkeypatch):
    adapter = _make_adapter(monkeypatch)

    result = await adapter.send("space-1", "one\n\ntwo")

    assert result.success
    assert result.message_id == "msg-0"
    assert _sent_texts(adapter) == ["one\n\ntwo"]


@pytest.mark.asyncio
async def test_opt_in_splits_on_blank_lines_one_sidecar_call_per_part(monkeypatch):
    adapter = _make_adapter(monkeypatch, split_outgoing_on_blank_lines=True)

    result = await adapter.send("space-1", "one\n\ntwo\nthree")

    assert _sent_texts(adapter) == ["one", "two\nthree"]
    assert result.success
    assert result.message_id == "msg-0"
    assert result.raw_response == {"message_ids": ["msg-0", "msg-1"]}


@pytest.mark.asyncio
async def test_configured_delay_sleeps_between_parts_only(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        split_outgoing_on_blank_lines=True,
        split_outgoing_delay_seconds=1.25,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await adapter.send("space-1", "one\n\ntwo\n\nthree")

    assert len(_sent_texts(adapter)) == 3
    assert [c.args[0] for c in sleep.await_args_list] == [1.25, 1.25]


@pytest.mark.asyncio
async def test_no_delay_when_split_produces_single_part(monkeypatch):
    adapter = _make_adapter(monkeypatch, split_outgoing_on_blank_lines=True)
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await adapter.send("space-1", "one\ntwo")

    assert _sent_texts(adapter) == ["one\ntwo"]
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_blank_lines_inside_fenced_code_stay_in_one_bubble(monkeypatch):
    adapter = _make_adapter(monkeypatch, split_outgoing_on_blank_lines=True)

    await adapter.send("space-1", "intro\n\n```python\nfirst\n\nsecond\n```\n\noutro")

    assert _sent_texts(adapter) == [
        "intro",
        "```python\nfirst\n\nsecond\n```",
        "outro",
    ]


@pytest.mark.asyncio
async def test_max_parts_merges_remainder_into_final_bubble(monkeypatch):
    adapter = _make_adapter(
        monkeypatch,
        split_outgoing_on_blank_lines=True,
        split_outgoing_max_parts=3,
    )

    await adapter.send("space-1", "one\n\ntwo\n\nthree\n\nfour")

    assert _sent_texts(adapter) == ["one", "two", "three\n\nfour"]


@pytest.mark.asyncio
async def test_mid_sequence_failure_returns_failing_result_and_stops(monkeypatch):
    adapter = _make_adapter(monkeypatch, split_outgoing_on_blank_lines=True)
    failure = SendResult(success=False, error="sidecar down", retryable=True)
    adapter._sidecar_send = AsyncMock(
        side_effect=[SendResult(success=True, message_id="msg-0"), failure]
    )

    result = await adapter.send("space-1", "one\n\ntwo\n\nthree")

    assert result is failure
    assert adapter._sidecar_send.await_count == 2
