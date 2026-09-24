"""Tests for delivery edge cases and uncovered code paths.

Tests for gateway/delivery.py lines that were not covered:
- Unknown platform handling (lines 84-86, 92-94)
- to_string() with thread_id (line 103)
- deliver() success/error handling (lines 150-169)
- _deliver_local() file writing (lines 179-212)
- _save_full_output() (lines 219-224)
- _deliver_to_platform() with adapter (lines 233-254)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from gateway.config import Platform
from gateway.delivery import DeliveryTarget, DeliveryRouter, MAX_PLATFORM_OUTPUT, TRUNCATED_VISIBLE
from gateway.session import SessionSource


class TestUnknownPlatformHandling:
    """Test unknown platform handling in parse()."""

    def test_unknown_platform_with_colon_treated_as_local(self):
        """Unknown platform string with colon should fall back to LOCAL."""
        target = DeliveryTarget.parse("unknown:12345")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_unknown_platform_without_colon_treated_as_local(self):
        """Unknown platform string without colon should fall back to LOCAL."""
        target = DeliveryTarget.parse("completely_unknown")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None

    def test_unknown_platform_case_insensitive(self):
        """Unknown platform names should be case-insensitive."""
        target = DeliveryTarget.parse("UNKNOWN_PLATFORM")
        assert target.platform == Platform.LOCAL

    def test_unknown_platform_with_thread_id(self):
        """Unknown platform with thread_id should still fall back to LOCAL."""
        target = DeliveryTarget.parse("unknown_platform:chat:thread123")
        assert target.platform == Platform.LOCAL
        assert target.chat_id is None


class TestToStringWithThreadId:
    """Test to_string() roundtrip with thread_id."""

    def test_platform_chat_id_thread_id_roundtrip(self):
        """Test roundtrip for platform:chat_id:thread_id format."""
        target = DeliveryTarget.parse("telegram:12345:thread678")
        assert target.platform == Platform.TELEGRAM
        assert target.chat_id == "12345"
        assert target.thread_id == "thread678"
        
        result = target.to_string()
        assert result == "telegram:12345:thread678"
        
        # Verify roundtrip
        reparsed = DeliveryTarget.parse(result)
        assert reparsed.platform == Platform.TELEGRAM
        assert reparsed.chat_id == "12345"
        assert reparsed.thread_id == "thread678"

    def test_slack_channel_thread_roundtrip(self):
        """Test Slack channel:thread format preserves case."""
        target = DeliveryTarget.parse("slack:C123ABC:thread456")
        assert target.platform == Platform.SLACK
        assert target.chat_id == "C123ABC"
        assert target.thread_id == "thread456"
        
        result = target.to_string()
        assert result == "slack:C123ABC:thread456"


class TestDeliveryRouterSuccessPath:
    """Test deliver() success path for LOCAL and platform targets."""

    @pytest.mark.asyncio
    async def test_deliver_local_success(self):
        """Test successful local delivery saves file and returns path."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            result = await router.deliver(
                content="Test content",
                targets=[DeliveryTarget.parse("local")],
                job_id="test_job",
                job_name="Test Job"
            )
            
            # Check that local delivery succeeded
            assert "local" in result
            assert result["local"]["success"] is True
            assert "path" in result["local"]["result"]
            assert "timestamp" in result["local"]["result"]

    @pytest.mark.asyncio
    async def test_deliver_platform_success(self):
        """Test successful platform delivery calls adapter.send()."""
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value={"sent": True})
        
        config = MagicMock()
        router = DeliveryRouter(
            config=config,
            adapters={Platform.TELEGRAM: mock_adapter}
        )
        
        result = await router.deliver(
            content="Test message",
            targets=[
                DeliveryTarget(
                    platform=Platform.TELEGRAM,
                    chat_id="12345",
                    thread_id="thread678",
                    is_explicit=True
                )
            ],
            job_id="test_job",
            metadata={"key": "value"}
        )
        
        # Check that delivery succeeded
        assert "telegram:12345:thread678" in result
        assert result["telegram:12345:thread678"]["success"] is True

        # Verify adapter.send was called with correct positional args
        # (metadata may include platform-specific additions like DM topic info)
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert call_args.args[0] == "12345"
        assert call_args.args[1] == "Test message"
        assert call_args.kwargs["metadata"]["key"] == "value"


class TestDeliveryRouterErrorPath:
    """Test deliver() error handling."""

    @pytest.mark.asyncio
    async def test_deliver_platform_no_adapter(self):
        """Test error when no adapter configured for platform."""
        config = MagicMock()
        router = DeliveryRouter(
            config=config,
            adapters={}
        )
        
        result = await router.deliver(
            content="Test message",
            targets=[
                DeliveryTarget(
                    platform=Platform.TELEGRAM,
                    chat_id="12345"
                )
            ]
        )
        
        # Check that delivery failed with error
        assert "telegram:12345" in result
        assert result["telegram:12345"]["success"] is False
        assert "error" in result["telegram:12345"]
        assert "No adapter configured" in result["telegram:12345"]["error"]

    @pytest.mark.asyncio
    async def test_deliver_platform_no_chat_id(self):
        """Test error when no chat_id provided for platform delivery."""
        mock_adapter = AsyncMock()
        config = MagicMock()
        router = DeliveryRouter(
            config=config,
            adapters={Platform.TELEGRAM: mock_adapter}
        )
        
        result = await router.deliver(
            content="Test message",
            targets=[
                DeliveryTarget(
                    platform=Platform.TELEGRAM,
                    chat_id=None  # No chat_id
                )
            ]
        )
        
        assert "telegram" in result
        assert result["telegram"]["success"] is False
        assert "No chat ID" in result["telegram"]["error"]

    @pytest.mark.asyncio
    async def test_deliver_mixed_success_failure(self):
        """Test deliver() with mix of successful and failed targets."""
        mock_telegram_adapter = AsyncMock()
        mock_telegram_adapter.send = AsyncMock(return_value={"sent": True})
        
        mock_discord_adapter = AsyncMock()
        mock_discord_adapter.send = AsyncMock(side_effect=Exception("Discord error"))
        
        config = MagicMock()
        router = DeliveryRouter(
            config=config,
            adapters={
                Platform.TELEGRAM: mock_telegram_adapter,
                Platform.DISCORD: mock_discord_adapter
            }
        )
        
        result = await router.deliver(
            content="Test message",
            targets=[
                DeliveryTarget(
                    platform=Platform.TELEGRAM,
                    chat_id="12345"
                ),
                DeliveryTarget(
                    platform=Platform.DISCORD,
                    chat_id="98765"
                )
            ]
        )
        
        # Telegram should succeed
        assert "telegram:12345" in result
        assert result["telegram:12345"]["success"] is True
        
        # Discord should fail
        assert "discord:98765" in result
        assert result["discord:98765"]["success"] is False
        assert "Discord error" in result["discord:98765"]["error"]


class TestDeliverLocalFileWriting:
    """Test _deliver_local() file writing logic."""

    def test_deliver_local_creates_directory_structure(self):
        """Test that _deliver_local creates necessary directory structure."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            # Should not raise even if directory doesn't exist
            result = router._deliver_local(
                content="Test content",
                job_id="test_job",
                job_name="Test Job",
                metadata={"key": "value"}
            )
            
            assert "path" in result
            # Verify file was created in job directory
            path = Path(result["path"])
            assert path.parent.name == "test_job"
            assert path.suffix == ".md"

    def test_deliver_local_includes_job_name(self):
        """Test that local output includes job name in header."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            result = router._deliver_local(
                content="Test content",
                job_id="test_job",
                job_name="My Job",
                metadata=None
            )
            
            # Check file was created
            assert "path" in result
            path = Path(result["path"])
            assert path.exists()
            
            # Read file content
            content = path.read_text()
            assert "# My Job" in content
            assert "**Job ID:** test_job" in content
            assert "Test content" in content

    def test_deliver_local_without_job_id(self):
        """Test local delivery without job_id goes to misc directory."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            result = router._deliver_local(
                content="Test content",
                job_id=None,
                job_name=None,
                metadata=None
            )
            
            # Check file is in misc directory
            assert "path" in result
            assert "misc" in str(result["path"])


class TestSaveFullOutput:
    """Test _save_full_output() method."""

    def test_save_full_output_creates_file(self):
        """Test that _save_full_output creates the output file."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            path = router._save_full_output("Full output content", "job123")
            
            assert isinstance(path, Path)
            assert path.exists()
            assert path.read_text() == "Full output content"

    def test_save_full_output_file_naming(self):
        """Test that output file follows expected naming pattern."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={}
            )
            
            path = router._save_full_output("Content", "test_job_456")
            
            # File should be named job_id_timestamp.txt
            assert "test_job_456_" in str(path)
            assert str(path).endswith(".txt")


class TestDeliverToPlatform:
    """Test _deliver_to_platform() async method."""

    @pytest.mark.asyncio
    async def test_deliver_to_platform_calls_adapter(self):
        """Test that _deliver_to_platform correctly calls adapter.send()."""
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value={"sent": True})
        
        target = DeliveryTarget(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            is_explicit=True
        )
        
        router = DeliveryRouter(
            config=MagicMock(),
            adapters={Platform.TELEGRAM: mock_adapter}
        )
        
        result = await router._deliver_to_platform(target, "Message content", {"key": "value"})
        
        assert result == {"sent": True}
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        assert call_args.args[0] == "12345"
        assert call_args.args[1] == "Message content"
        # metadata should contain the original metadata
        assert call_args.kwargs["metadata"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_deliver_to_platform_adds_thread_id_to_metadata(self):
        """Test that thread_id is added to metadata when provided."""
        mock_adapter = AsyncMock()
        mock_adapter.send = AsyncMock(return_value={"sent": True})
        
        target = DeliveryTarget(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            thread_id="thread678",
            is_explicit=True
        )
        
        router = DeliveryRouter(
            config=MagicMock(),
            adapters={Platform.TELEGRAM: mock_adapter}
        )
        
        result = await router._deliver_to_platform(target, "Message", {})
        
        mock_adapter.send.assert_called_once()
        call_args = mock_adapter.send.call_args
        # thread_id is present in metadata; platform may override its value
        # (e.g. Telegram may resolve/replace it with a DM topic ID)
        assert "thread_id" in call_args.kwargs["metadata"]

    @pytest.mark.asyncio
    async def test_deliver_to_platform_truncates_oversized_content(self):
        """Test that oversized content is truncated and saved to file."""
        with patch("gateway.delivery.get_hermes_home") as mock_home:
            mock_home.return_value = Path("/tmp/hermes")
            
            mock_adapter = AsyncMock()
            mock_adapter.send = AsyncMock(return_value={"sent": True})
            
            # Create content larger than MAX_PLATFORM_OUTPUT
            oversized_content = "A" * (MAX_PLATFORM_OUTPUT + 100)
            
            router = DeliveryRouter(
                config=MagicMock(),
                adapters={Platform.TELEGRAM: mock_adapter}
            )
            
            result = await router._deliver_to_platform(
                DeliveryTarget(
                    platform=Platform.TELEGRAM,
                    chat_id="12345"
                ),
                oversized_content,
                {"job_id": "test_job"}
            )
            
            # Should still succeed (content was truncated)
            assert result["sent"] is True
            
            # Verify adapter was called with truncated content
            call_args = mock_adapter.send.call_args
            assert len(call_args.args[1]) <= MAX_PLATFORM_OUTPUT
            assert "[truncated" in call_args.args[1]
            
            # Verify metadata includes job_id (for file saving)
            assert call_args.kwargs["metadata"].get("job_id") == "test_job"
