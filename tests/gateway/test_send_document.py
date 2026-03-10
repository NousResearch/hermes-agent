"""Tests for the send_document platform adapter methods and FILE: tag extraction.

All tests use mocks -- no real Telegram/Discord connections are needed.
"""

import asyncio
import io
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway.platforms.base import BasePlatformAdapter, SendResult


# =========================================================================
# FILE: Tag Extraction
# =========================================================================

class TestFileTagExtraction:
    """Tests for BasePlatformAdapter.extract_files().

    The tag format is FILE:<path> or FILE:<path|caption>.
    extract_files() returns List[Tuple[str, Optional[str]]].

    All tests mock is_safe_file_path to bypass the cache-dir allowlist
    because we're testing the *parsing* logic, not the path guard.
    The allowlist itself is tested in TestPathAllowlist below.
    """

    @pytest.fixture(autouse=True)
    def _bypass_path_guard(self):
        """Let every path through so we can test parsing in isolation."""
        # Return a SimpleNamespace with is_file()=True so the guard passes,
        # but __str__ returns the original path string unchanged.
        class _FakePath:
            def __init__(self, p):
                self._p = p
            def is_file(self):
                return True
            def __str__(self):
                return self._p
        with patch("tools.file_transfer.is_safe_file_path",
                    side_effect=lambda p: _FakePath(p)):
            yield

    def test_single_file_tag(self):
        content = "Here is your report\nFILE:</tmp/cache/abc_report.csv>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_report.csv", None)
        assert "FILE:<" not in cleaned
        assert "report" in cleaned

    def test_single_file_tag_with_caption(self):
        content = "Done\nFILE:</tmp/cache/abc_report.csv|Your quarterly report>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_report.csv", "Your quarterly report")
        assert "FILE:<" not in cleaned

    def test_multiple_file_tags(self):
        content = "Analysis complete\nFILE:</tmp/a.csv>\nFILE:</tmp/b.png>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 2
        paths = [f[0] for f in files]
        assert "/tmp/a.csv" in paths
        assert "/tmp/b.png" in paths

    def test_no_file_tags(self):
        content = "No files here, just text."
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 0
        assert cleaned == content

    def test_file_tag_cleanup_blank_lines(self):
        content = "Text\n\n\n\nFILE:</tmp/file.txt>\n\n\n\nMore text"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert "\n\n\n" not in cleaned

    def test_file_tag_with_spaces_in_path(self):
        """Paths with spaces should be preserved inside angle brackets."""
        content = "FILE:</tmp/cache/abc_my report (final).csv>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_my report (final).csv", None)

    def test_file_tag_spaces_and_caption(self):
        content = "FILE:</tmp/cache/abc_my report.csv|Here is your report>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_my report.csv", "Here is your report")

    def test_file_tag_caption_with_gt_character(self):
        """Caption containing escaped \\> must be unescaped to >."""
        content = r"FILE:</tmp/a.csv|Results \> 100 items found>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/a.csv", "Results > 100 items found")
        assert "FILE:<" not in cleaned

    def test_multiple_file_tags_same_line(self):
        """Multiple FILE tags on the same line must be parsed separately."""
        content = "FILE:</tmp/a.csv> FILE:</tmp/b.csv>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 2
        assert files[0] == ("/tmp/a.csv", None)
        assert files[1] == ("/tmp/b.csv", None)

    def test_multiple_file_tags_same_line_with_captions(self):
        """Multiple FILE tags with captions on same line."""
        content = "FILE:</tmp/a.csv|Report A> FILE:</tmp/b.csv|Report B>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 2
        assert files[0] == ("/tmp/a.csv", "Report A")
        assert files[1] == ("/tmp/b.csv", "Report B")

    def test_path_with_gt_character(self):
        """Path containing escaped \\> must be unescaped to >."""
        content = r"FILE:</tmp/cache/abc_report \> final.csv|Your report>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_report > final.csv", "Your report")
        assert "FILE:<" not in cleaned

    def test_path_with_gt_no_caption(self):
        """Path containing > without caption must also roundtrip."""
        content = r"FILE:</tmp/cache/abc_data \> 2026.csv>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 1
        assert files[0] == ("/tmp/cache/abc_data > 2026.csv", None)

    def test_mixed_escaped_gt_and_multi_tag(self):
        """Escaped > in caption must not break multi-tag parsing."""
        content = r"FILE:</tmp/a.csv|Count \> 50> FILE:</tmp/b.csv>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 2
        assert files[0] == ("/tmp/a.csv", "Count > 50")
        assert files[1] == ("/tmp/b.csv", None)


# =========================================================================
# Base send_document fallback
# =========================================================================

class TestBaseSendDocument:
    """Tests for the base class send_document fallback."""

    @pytest.mark.asyncio
    async def test_fallback_sends_text(self):
        """Base implementation should fall back to sending file path as text."""
        # Create a minimal concrete adapter
        adapter = _make_mock_adapter()
        adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="1"))

        result = await adapter.send_document(
            chat_id="123",
            file_path="/tmp/cache/report.csv",
            caption="Your report",
        )
        assert result.success
        # The send() call should contain the filename
        call_args = adapter.send.call_args
        assert "report.csv" in call_args.kwargs.get("content", call_args[1].get("content", ""))


# =========================================================================
# Telegram send_document (mocked)
# =========================================================================

class TestTelegramSendDocument:
    """Tests for TelegramAdapter.send_document() with mocked bot."""

    @pytest.mark.asyncio
    async def test_send_document_success(self, tmp_path):
        """Telegram adapter should call bot.send_document."""
        test_file = tmp_path / "report.csv"
        test_file.write_text("col1,col2\n1,2\n")

        adapter = _make_telegram_adapter()
        msg_mock = MagicMock()
        msg_mock.message_id = 42
        adapter._bot.send_document = AsyncMock(return_value=msg_mock)

        result = await adapter.send_document(
            chat_id="12345",
            file_path=str(test_file),
            caption="Your data",
        )
        assert result.success
        assert result.message_id == "42"
        adapter._bot.send_document.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_document_file_not_found(self):
        """Should return error when file doesn't exist."""
        adapter = _make_telegram_adapter()
        result = await adapter.send_document(
            chat_id="12345",
            file_path="/nonexistent/file.csv",
        )
        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, tmp_path):
        """Should reject files over 50MB."""
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (51 * 1024 * 1024))

        adapter = _make_telegram_adapter()
        result = await adapter.send_document(
            chat_id="12345",
            file_path=str(big_file),
        )
        assert not result.success
        assert "50mb" in result.error.lower()

    @pytest.mark.asyncio
    async def test_send_document_not_connected(self):
        """Should return error when bot is None."""
        from gateway.platforms.telegram import TelegramAdapter
        adapter = TelegramAdapter.__new__(TelegramAdapter)
        adapter._bot = None
        result = await adapter.send_document("12345", "/tmp/file.csv")
        assert not result.success


# =========================================================================
# Discord send_document (mocked)
# =========================================================================

class TestDiscordSendDocument:
    """Tests for DiscordAdapter.send_document() with mocked client."""

    @pytest.mark.asyncio
    async def test_send_document_success(self, tmp_path):
        """Discord adapter should send file via discord.File."""
        test_file = tmp_path / "data.json"
        test_file.write_text('{"key": "value"}')

        adapter = _make_discord_adapter()
        msg_mock = MagicMock()
        msg_mock.id = 999
        channel_mock = AsyncMock()
        channel_mock.send = AsyncMock(return_value=msg_mock)
        adapter._client.get_channel = MagicMock(return_value=channel_mock)

        result = await adapter.send_document(
            chat_id="67890",
            file_path=str(test_file),
            caption="JSON data",
        )
        assert result.success
        assert result.message_id == "999"
        channel_mock.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_send_document_file_not_found(self):
        """Should return error when file doesn't exist."""
        adapter = _make_discord_adapter()
        channel_mock = AsyncMock()
        adapter._client.get_channel = MagicMock(return_value=channel_mock)

        result = await adapter.send_document(
            chat_id="67890",
            file_path="/nonexistent/file.json",
        )
        assert not result.success

    @pytest.mark.asyncio
    async def test_send_document_too_large(self, tmp_path):
        """Should reject files over 25MB."""
        big_file = tmp_path / "big.bin"
        big_file.write_bytes(b"x" * (26 * 1024 * 1024))

        adapter = _make_discord_adapter()
        channel_mock = AsyncMock()
        adapter._client.get_channel = MagicMock(return_value=channel_mock)

        result = await adapter.send_document(
            chat_id="67890",
            file_path=str(big_file),
        )
        assert not result.success
        assert "25mb" in result.error.lower()


# =========================================================================
# Helpers
# =========================================================================

def _make_mock_adapter():
    """Create a minimal concrete BasePlatformAdapter for testing."""
    from gateway.config import Platform, PlatformConfig

    class MockAdapter(BasePlatformAdapter):
        async def connect(self): return True
        async def disconnect(self): pass
        async def send(self, chat_id, content, reply_to=None, metadata=None):
            return SendResult(success=True)
        async def get_chat_info(self, chat_id):
            return {"name": "test", "type": "dm"}

    config = PlatformConfig(token="fake")
    return MockAdapter(config, Platform.TELEGRAM)


def _make_telegram_adapter():
    """Create a TelegramAdapter with a mocked bot."""
    from gateway.platforms.telegram import TelegramAdapter
    from gateway.config import PlatformConfig, Platform

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter.config = PlatformConfig(token="fake")
    adapter.platform = Platform.TELEGRAM
    adapter._bot = AsyncMock()
    adapter._app = None
    adapter._message_handler = None
    adapter._running = True
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


def _make_discord_adapter():
    """Create a DiscordAdapter with a mocked client."""
    from gateway.platforms.discord import DiscordAdapter
    from gateway.config import PlatformConfig, Platform

    adapter = DiscordAdapter.__new__(DiscordAdapter)
    adapter.config = PlatformConfig(token="fake")
    adapter.platform = Platform.DISCORD
    adapter._client = MagicMock()
    adapter._ready_event = asyncio.Event()
    adapter._allowed_user_ids = set()
    adapter._message_handler = None
    adapter._running = True
    adapter._active_sessions = {}
    adapter._pending_messages = {}
    return adapter


# =========================================================================
# Path Allowlist (is_safe_file_path guard)
# =========================================================================

class TestPathAllowlist:
    """Tests for the cache-directory allowlist in extract_files().

    extract_files() calls is_safe_file_path() which only allows paths
    inside ~/.hermes/file_cache/ — this prevents host file exfiltration.
    """

    def test_blocks_path_outside_cache(self):
        """FILE tag pointing to /etc/passwd should be stripped, not returned."""
        content = "Here\nFILE:</etc/passwd|secrets>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 0
        assert "FILE:<" not in cleaned  # tag is still cleaned from text

    def test_blocks_dotenv_exfiltration(self):
        """Attempt to exfiltrate .env file should be blocked."""
        content = "FILE:</home/user/.env>"
        files, cleaned = BasePlatformAdapter.extract_files(content)
        assert len(files) == 0

    def test_allows_path_inside_cache(self, tmp_path):
        """FILE tag inside the cache dir should pass through."""
        from tools.file_transfer import get_file_cache_dir
        cache_dir = get_file_cache_dir()
        test_file = cache_dir / "test_allowlist_probe.csv"
        test_file.write_text("col1,col2\n1,2\n")
        try:
            content = f"Report\nFILE:<{test_file}|test>"
            files, cleaned = BasePlatformAdapter.extract_files(content)
            assert len(files) == 1
            assert files[0][1] == "test"
        finally:
            test_file.unlink(missing_ok=True)

    def test_blocks_traversal_via_cache_path(self, tmp_path):
        """Path traversal via cache dir should still be blocked."""
        from tools.file_transfer import FILE_CACHE_DIR
        evil = str(FILE_CACHE_DIR / ".." / ".." / ".env")
        content = f"FILE:<{evil}>"
        files, _ = BasePlatformAdapter.extract_files(content)
        assert len(files) == 0


# =========================================================================
# Discord inbound document → queue_injection flow
# =========================================================================

class TestDiscordInboundDocumentFlow:
    """Verify that Discord document attachments are cached locally
    before being queued for sandbox injection (not left as URLs)."""

    @pytest.mark.asyncio
    async def test_document_attachment_cached_locally(self):
        """Discord document should be downloaded and cached, not kept as URL."""
        from gateway.platforms.base import cache_document_from_bytes

        # Simulate what Discord adapter does with a document attachment
        doc_bytes = b"col1,col2\n1,2\n"
        cached_path = cache_document_from_bytes(doc_bytes, "report.csv")

        assert os.path.exists(cached_path), "Document must be cached locally"
        assert not cached_path.startswith("http"), "Path must be local, not URL"
        # Cleanup
        os.unlink(cached_path)

    @pytest.mark.asyncio
    async def test_cached_document_can_be_queued(self):
        """Cached document path should work with queue_injection."""
        from gateway.platforms.base import cache_document_from_bytes
        from tools.file_transfer import queue_injection, _pending_injections

        doc_bytes = b"hello world"
        cached_path = cache_document_from_bytes(doc_bytes, "test.txt")
        task_id = "discord_e2e_test"

        try:
            queue_injection(task_id, cached_path, "/workspace/uploads/test.txt")
            assert task_id in _pending_injections
            items = _pending_injections[task_id]
            assert len(items) == 1
            assert items[0]["host_path"] == cached_path
            assert os.path.exists(cached_path)
        finally:
            _pending_injections.pop(task_id, None)
            if os.path.exists(cached_path):
                os.unlink(cached_path)
