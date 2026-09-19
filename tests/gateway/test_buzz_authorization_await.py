"""Held attachment await: authority must be fresh for the next profile fetch."""
import asyncio

import pytest

from tests.gateway.test_buzz_authorization_before_side_effects import (
    CHANNEL, OTHER_ALLOWED_USER, SENDER, denied_intake,
)


def attachment_event(event_id="1" * 64):
    return {
        "id": event_id, "kind": 9, "pubkey": SENDER,
        "content": "hello", "created_at": 1,
        "tags": [["h", CHANNEL], ["imeta", "url https://buzz.invalid/media/file.txt",
                 "m text/plain", "x " + "a" * 64, "size 1"]],
    }


async def drain(adapter):
    tasks = tuple(adapter._session_tasks.values())
    if tasks:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=5)


@pytest.mark.asyncio
async def test_grant_revoked_during_attachment_await_blocks_profile_and_seen(denied_intake, monkeypatch):
    probe = denied_intake
    adapter = probe.adapter
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SENDER)
    assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is True
    entered, release = asyncio.Event(), asyncio.Event()
    downloads = []

    async def held_download(metadata):
        downloads.append(metadata)
        entered.set()
        await release.wait()
        return None

    # Keep parsing and _cache_inbound_attachments real; only replace transport.
    monkeypatch.setattr(adapter, "_download_attachment", held_download)
    task = asyncio.create_task(adapter._handle_event(
        CHANNEL, adapter._new_channel_state("group"), attachment_event()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        probe.cli.assert_not_called()
        probe.resolver.assert_not_called()
        monkeypatch.setenv("BUZZ_ALLOWED_USERS", OTHER_ALLOWED_USER)
        assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is False
        release.set()
        await asyncio.wait_for(task, timeout=5)
        await drain(adapter)
        assert len(downloads) == 1  # already-started work is not retroactively cancelled
        assert probe.verdicts == [False]  # preserve real base/central denial dispatch
        probe.reaction.assert_not_called()
        profile_calls = [c for c in probe.cli.call_args_list if list(c.args[0][:2]) == ["users", "get"]]
        assert not profile_calls, "revoked grant triggered a NEW profile fetch after attachment await"
        probe.resolver.assert_not_called()
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await drain(adapter)


@pytest.mark.asyncio
async def test_unchanged_grant_across_attachment_await_keeps_profile_and_seen(denied_intake, monkeypatch):
    probe = denied_intake
    adapter = probe.adapter
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SENDER)
    entered, release = asyncio.Event(), asyncio.Event()
    downloads = []

    async def held_download(metadata):
        downloads.append(metadata)
        entered.set()
        await release.wait()
        return None

    monkeypatch.setattr(adapter, "_download_attachment", held_download)
    task = asyncio.create_task(adapter._handle_event(
        CHANNEL, adapter._new_channel_state("group"), attachment_event()))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert not task.done()
        probe.resolver.assert_not_called()
        assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is True
        release.set()
        await asyncio.wait_for(task, timeout=5)
        await drain(adapter)
        assert len(downloads) == 1
        assert probe.verdicts == [True]
        probe.resolver.assert_awaited_once_with(SENDER)
        assert any(list(c.args[0][:2]) == ["users", "get"] for c in probe.cli.call_args_list)
        probe.reaction.assert_awaited_once_with(CHANNEL, "1" * 64, "👀")
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await drain(adapter)


@pytest.mark.asyncio
async def test_denial_skips_attachment_await_and_later_grant_is_not_cached(denied_intake, monkeypatch):
    # Denial does not reach the attachment await: there is no symmetric held
    # deny-to-grant window here. A subsequent new event observes the new grant.
    from unittest.mock import AsyncMock

    probe = denied_intake
    adapter = probe.adapter
    download = AsyncMock(return_value=None)
    monkeypatch.setattr(adapter, "_download_attachment", download)
    state = adapter._new_channel_state("group")
    assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is False
    await adapter._handle_event(CHANNEL, state, attachment_event())
    await drain(adapter)
    download.assert_not_called()
    probe.resolver.assert_not_called()
    probe.reaction.assert_not_called()
    assert probe.verdicts == [False]
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SENDER)
    assert adapter._is_sender_authorized(SENDER, "group", CHANNEL) is True
    await adapter._handle_event(CHANNEL, state, attachment_event("2" * 64))
    await drain(adapter)
    download.assert_awaited_once()
    probe.resolver.assert_awaited_once_with(SENDER)
    assert probe.verdicts == [False, True]
    probe.reaction.assert_awaited_once_with(CHANNEL, "2" * 64, "👀")
