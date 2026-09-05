"""Tests for Discord incoming document/file attachment handling.

Covers the document branch in DiscordAdapter._handle_message() —
the `else` clause of the attachment content-type loop that was added
to download, cache, and optionally inject text from non-image/audio files.
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import GatewayConfig, PlatformConfig
from gateway.platforms.base import MessageType, merge_pending_message_event


# ---------------------------------------------------------------------------
# Discord mock setup (copied from test_discord_free_response.py)
# ---------------------------------------------------------------------------

def _ensure_discord_mock():
    """Install a mock discord module when discord.py isn't available."""
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.Client = MagicMock
    discord_mod.File = MagicMock
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.ui = SimpleNamespace(View=object, button=lambda *a, **k: (lambda fn: fn), Button=object)
    discord_mod.ButtonStyle = SimpleNamespace(success=1, primary=2, secondary=2, danger=3, green=1, grey=2, blurple=2, red=3)
    discord_mod.Color = SimpleNamespace(orange=lambda: 1, green=lambda: 2, blue=lambda: 3, red=lambda: 4, purple=lambda: 5)
    discord_mod.Interaction = object
    discord_mod.Embed = MagicMock
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

import plugins.platforms.discord.adapter as discord_platform  # noqa: E402
from plugins.platforms.discord.adapter import (  # noqa: E402
    DiscordAdapter,
    _discord_attachment_name_requires_path_only,
)


# ---------------------------------------------------------------------------
# Fake channel / thread types
# ---------------------------------------------------------------------------

class FakeDMChannel:
    def __init__(self, channel_id: int = 1):
        self.id = channel_id
        self.name = "dm"


class FakeThread:
    def __init__(self, channel_id: int = 10):
        self.id = channel_id
        self.name = "thread"
        self.parent = None
        self.parent_id = None
        self.guild = SimpleNamespace(name="TestServer")
        self.topic = None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    """Point document cache to tmp_path so tests never write to ~/.hermes."""
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(discord_platform.discord, "DMChannel", FakeDMChannel, raising=False)
    monkeypatch.setattr(discord_platform.discord, "Thread", FakeThread, raising=False)
    # These tests mock the actual download. Do not let host DNS/proxy mappings
    # for cdn.discordapp.com decide whether document handling is exercised.
    monkeypatch.setattr(discord_platform, "is_safe_url", lambda _url: True)

    config = PlatformConfig(enabled=True, token="fake-token")
    a = DiscordAdapter(config)
    a._client = SimpleNamespace(user=SimpleNamespace(id=999))
    a.handle_message = AsyncMock()
    return a


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attachment(
    *,
    filename: str,
    content_type: Optional[str],
    size: int = 1024,
    url: str = "https://cdn.discordapp.com/attachments/fake/file",
) -> SimpleNamespace:
    return SimpleNamespace(
        filename=filename,
        content_type=content_type,
        size=size,
        url=url,
    )


def make_message(attachments: list, content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        id=123,
        content=content,
        attachments=attachments,
        mentions=[],
        reference=None,
        created_at=datetime.now(timezone.utc),
        channel=FakeDMChannel(),
        author=SimpleNamespace(id=42, display_name="Tester", name="Tester"),
    )


def _mock_aiohttp_download(raw_bytes: bytes):
    """Return a patch context manager that makes aiohttp return raw_bytes."""
    resp = AsyncMock()
    resp.status = 200
    resp.read = AsyncMock(return_value=raw_bytes)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)

    session = AsyncMock()
    session.get = MagicMock(return_value=resp)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)

    return patch("aiohttp.ClientSession", return_value=session)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIncomingDocumentHandling:

    @pytest.mark.parametrize(
        "filename",
        [
            ".env", ".env.production", ".netrc", ".git-credentials", "id_ed25519",
            "server-private-key.pem", "api-key.txt", "service-auth.txt",
            "client-credentials.json", "oauth-token.txt", "database_password.md",
            "shared.secret.txt",
        ],
    )
    def test_credential_like_filename_requires_path_only(self, filename):
        assert _discord_attachment_name_requires_path_only(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "monkey.txt", "hockey.txt", "keynote.txt", "authentication-guide.md",
            "tokenizer-notes.txt", "secretary-agenda.txt", "ordinary-notes.txt", "readme.md",
        ],
    )
    def test_ordinary_filename_remains_inline_eligible(self, filename):
        assert _discord_attachment_name_requires_path_only(filename) is False


    @pytest.mark.asyncio
    async def test_txt_content_injected(self, adapter):
        """.txt file under 100KB should have its content injected into event.text."""
        file_content = b"Hello from a text file"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="notes.txt", content_type="text/plain")],
                content="summarize this",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of notes.txt]:" in event.text
        assert "Hello from a text file" in event.text
        assert "summarize this" in event.text
        # injection prepended before caption
        assert event.text.index("[Content of") < event.text.index("summarize this")

    @pytest.mark.asyncio
    async def test_credential_attachment_body_absent_from_constructed_prompt(self, adapter):
        from gateway.run import GatewayRunner

        sentinel = "SYNTHETIC_CREDENTIAL_BODY_MUST_NOT_ENTER_PROMPT"
        with _mock_aiohttp_download(sentinel.encode()):
            msg = make_message(
                attachments=[make_attachment(filename="service-token.txt", content_type="text/plain")],
                content="inspect the saved file",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args.args[0]
        saved_path = event.media_urls[0]
        assert Path(saved_path).read_text(encoding="utf-8") == sentinel
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.adapters = {}
        runner._has_setup_skill = lambda: False
        prompt = await runner._prepare_inbound_message_text(event=event, source=event.source, history=[])
        assert sentinel not in prompt
        assert saved_path in prompt
        assert "content was not automatically inlined" in prompt

    @pytest.mark.asyncio
    async def test_queued_credential_followup_keeps_path_only_context(self, adapter):
        from gateway.run import GatewayRunner

        with _mock_aiohttp_download(b"ordinary body"):
            await adapter._handle_message(make_message(
                attachments=[make_attachment(filename="notes.txt", content_type="text/plain")],
                content="first",
            ))
        existing = adapter.handle_message.call_args.args[0]

        sentinel = "QUEUED_CREDENTIAL_BODY_MUST_NOT_ENTER_PROMPT"
        adapter.handle_message.reset_mock()
        with _mock_aiohttp_download(sentinel.encode()):
            await adapter._handle_message(make_message(
                attachments=[make_attachment(filename="service-token.txt", content_type="text/plain")],
                content="second",
            ))
        incoming = adapter.handle_message.call_args.args[0]
        credential_path = incoming.media_urls[0]
        pending = {"session": existing}
        merge_pending_message_event(pending, "session", incoming)
        merged = pending["session"]
        assert credential_path in merged.metadata["discord_path_only_attachment_paths"]

        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig()
        runner.adapters = {}
        runner._has_setup_skill = lambda: False
        prompt = await runner._prepare_inbound_message_text(event=merged, source=merged.source, history=[])
        assert sentinel not in prompt
        assert credential_path in prompt
        assert "content was not automatically inlined" in prompt

    @pytest.mark.asyncio
    async def test_md_content_injected(self, adapter):
        """.md file under 100KB should have its content injected."""
        file_content = b"# Title\nSome markdown content"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="readme.md", content_type="text/markdown")],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of readme.md]:" in event.text
        assert "# Title" in event.text

    @pytest.mark.asyncio
    async def test_log_content_injected(self, adapter):
        """.log file under 100KB should be treated as text/plain and injected."""
        file_content = b"BLE trace line 1\nBLE trace line 2"

        with _mock_aiohttp_download(file_content):
            msg = make_message(
                attachments=[make_attachment(filename="btsnoop_hci.log", content_type="text/plain")],
                content="please inspect this",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of btsnoop_hci.log]:" in event.text
        assert "BLE trace line 1" in event.text
        assert "please inspect this" in event.text


    @pytest.mark.asyncio
    async def test_large_txt_cached_not_injected(self, adapter):
        """.txt over 100KB should be cached but NOT injected into event.text."""
        large_content = b"x" * (200 * 1024)

        with _mock_aiohttp_download(large_content):
            msg = make_message(
                attachments=[make_attachment(filename="big.txt", content_type="text/plain", size=len(large_content))],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        assert "[Content of" not in (event.text or "")

    @pytest.mark.asyncio
    async def test_multiple_text_files_both_injected(self, adapter):
        """Two text file attachments should both be injected into event.text in order."""
        content1 = b"First file content"
        content2 = b"Second file content"

        call_count = 0
        responses = [content1, content2]

        def make_session(_responses):
            idx = 0

            class FakeSession:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, *_):
                    pass

                def get(self, url, **kwargs):
                    nonlocal idx
                    data = _responses[idx % len(_responses)]
                    idx += 1

                    resp = AsyncMock()
                    resp.status = 200
                    resp.read = AsyncMock(return_value=data)
                    resp.__aenter__ = AsyncMock(return_value=resp)
                    resp.__aexit__ = AsyncMock(return_value=False)
                    return resp

            return FakeSession()

        with patch("aiohttp.ClientSession", return_value=make_session([content1, content2])):
            msg = make_message(
                attachments=[
                    make_attachment(filename="file1.txt", content_type="text/plain"),
                    make_attachment(filename="file2.txt", content_type="text/plain"),
                ],
                content="",
            )
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert "[Content of file1.txt]:" in event.text
        assert "First file content" in event.text
        assert "[Content of file2.txt]:" in event.text
        assert "Second file content" in event.text
        assert event.text.index("file1") < event.text.index("file2")


class TestAllowAnyAttachment:
    """Cover accept-any-file-type inbound handling.

    Authorization to message the agent is the gate, not the file extension.
    Unknown file types are cached and surfaced to the agent as DOCUMENT events
    with the source content_type (or application/octet-stream) so gateway/run.py
    emits a path-pointing context note. The legacy ``allow_any_attachment``
    config flag is now a no-op — acceptance is unconditional.
    """

    @pytest.mark.asyncio
    async def test_unknown_type_cached_by_default(self, adapter):
        """Default: unknown extension is cached, not dropped."""
        with _mock_aiohttp_download(b"\x00\x01\x02 binary payload"):
            msg = make_message([
                make_attachment(filename="weird.xyz", content_type="application/x-custom")
            ])
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1
        assert os.path.exists(event.media_urls[0])
        # Falls back to the source content_type when we have one.
        assert event.media_types == ["application/x-custom"]
        assert event.message_type == MessageType.DOCUMENT
        # We deliberately do NOT inline arbitrary (non-UTF-8) bytes — run.py
        # emits the path-pointing note based on DOCUMENT + octet-stream MIME.
        assert "[Content of" not in (event.text or "")


    @pytest.mark.asyncio
    async def test_max_attachment_bytes_caps_uploads(self, adapter):
        """discord.max_attachment_bytes overrides the historical 32 MiB cap."""
        adapter.config.extra["max_attachment_bytes"] = 1024  # 1 KiB

        msg = make_message([
            make_attachment(
                filename="too_big.xyz",
                content_type="application/x-custom",
                size=2048,
            )
        ])
        await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert event.media_urls == []

    @pytest.mark.asyncio
    async def test_max_attachment_bytes_zero_means_unlimited(self, adapter):
        """max_attachment_bytes=0 disables the size cap entirely."""
        adapter.config.extra["max_attachment_bytes"] = 0

        # 64 MiB — would normally exceed the historical 32 MiB hardcoded cap.
        with _mock_aiohttp_download(b"x" * 16):
            msg = make_message([
                make_attachment(
                    filename="huge.xyz",
                    content_type="application/x-custom",
                    size=64 * 1024 * 1024,
                )
            ])
            await adapter._handle_message(msg)

        event = adapter.handle_message.call_args[0][0]
        assert len(event.media_urls) == 1

