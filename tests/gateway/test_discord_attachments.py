"""Tests for Discord attachment ingestion into local filesystem cache."""

from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from gateway.platforms.discord import DiscordAdapter


class _FakeDMChannel:
    def __init__(self, channel_id: int = 123):
        self.id = channel_id


class _FakeAttachment:
    def __init__(self, *, filename: str, content_type: str, payload: bytes, url: str = "https://cdn.example/file"):
        self.filename = filename
        self.content_type = content_type
        self.url = url
        self.size = len(payload)
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class _FakeMessage:
    def __init__(self, *, attachments: list[_FakeAttachment], content: str = ""):
        self.content = content
        self.attachments = attachments
        self.channel = _FakeDMChannel()
        self.author = SimpleNamespace(id=7, name="alice", display_name="alice")
        self.mentions = []
        self.reference = None
        self.id = 99
        self.created_at = None


@pytest.fixture(autouse=True)
def _cache_dirs_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr("gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "document_cache")


@pytest.fixture()
def adapter(monkeypatch):
    # Ensure isinstance(message.channel, discord.DMChannel) works with our fake channel.
    monkeypatch.setattr("gateway.platforms.discord.discord.DMChannel", _FakeDMChannel)

    a = DiscordAdapter(PlatformConfig(enabled=True, token="x"))
    a.handle_message = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_markdown_attachment_cached_and_injected(adapter):
    att = _FakeAttachment(
        filename="notes.md",
        content_type="text/markdown",
        payload=b"# Title\nhello markdown",
    )
    msg = _FakeMessage(attachments=[att], content="please summarize")

    await adapter._handle_message(msg)

    event = adapter.handle_message.call_args[0][0]
    assert event.message_type == MessageType.DOCUMENT
    assert len(event.media_urls) == 1
    assert os.path.exists(event.media_urls[0])
    assert event.media_types == ["text/markdown"]
    assert "[Content of notes.md]" in event.text
    assert "hello markdown" in event.text
    assert "please summarize" in event.text


@pytest.mark.asyncio
async def test_archive_attachment_cached_locally(adapter):
    att = _FakeAttachment(
        filename="bundle.zip",
        content_type="application/zip",
        payload=b"PK\x03\x04fakezip",
    )
    msg = _FakeMessage(attachments=[att])

    await adapter._handle_message(msg)

    event = adapter.handle_message.call_args[0][0]
    assert event.message_type == MessageType.DOCUMENT
    assert len(event.media_urls) == 1
    assert os.path.exists(event.media_urls[0])
    assert event.media_types == ["application/zip"]


@pytest.mark.asyncio
async def test_invalid_document_size_env_falls_back(adapter, monkeypatch):
    monkeypatch.setenv("DISCORD_MAX_DOCUMENT_BYTES", "not-an-int")

    att = _FakeAttachment(
        filename="tiny.txt",
        content_type="text/plain",
        payload=b"hello",
    )
    msg = _FakeMessage(attachments=[att], content="process this")

    await adapter._handle_message(msg)

    event = adapter.handle_message.call_args[0][0]
    assert event.message_type == MessageType.DOCUMENT
    assert len(event.media_urls) == 1
    assert os.path.exists(event.media_urls[0])
    assert "[Content of tiny.txt]" in event.text
