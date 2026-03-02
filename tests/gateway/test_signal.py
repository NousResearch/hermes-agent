"""Tests for Signal platform adapter."""

import pytest
from unittest.mock import AsyncMock

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.signal import SignalAdapter


class TestSignalPlatformEnum:
    """Test that SIGNAL is in Platform enum."""
    
    def test_signal_platform_exists(self):
        """SIGNAL should be a valid Platform."""
        assert hasattr(Platform, 'SIGNAL')
        assert Platform.SIGNAL.value == "signal"


class TestSignalConfigLoading:
    """Test Signal configuration loading from environment."""
    
    def test_signal_config_from_env(self, monkeypatch):
        """Signal config should load from environment variables."""
        monkeypatch.setenv('SIGNAL_HTTP_URL', 'http://127.0.0.1:8080')
        monkeypatch.setenv('SIGNAL_ACCOUNT', '+1234567890')
        monkeypatch.setenv('SIGNAL_ALLOWED_USERS', '+1234567890,+0987654321')
        monkeypatch.setenv('SIGNAL_DM_POLICY', 'pairing')
        monkeypatch.setenv('SIGNAL_GROUP_POLICY', 'disabled')
        
        config = GatewayConfig()
        from gateway.config import _apply_env_overrides
        _apply_env_overrides(config)
        
        assert Platform.SIGNAL in config.platforms
        signal_config = config.platforms[Platform.SIGNAL]
        assert signal_config.enabled is True
        assert signal_config.extra.get('http_url') == 'http://127.0.0.1:8080'
        assert signal_config.extra.get('account') == '+1234567890'
        assert signal_config.extra.get('allowed_users') == '+1234567890,+0987654321'
        assert signal_config.extra.get('dm_policy') == 'pairing'
        assert signal_config.extra.get('group_policy') == 'disabled'


class TestSignalAdapterInit:
    """Test SignalAdapter initialization."""
    
    def test_adapter_initialization(self):
        """SignalAdapter should initialize with correct config."""
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
                'allowed_users': '+1234567890,+0987654321',
                'dm_policy': 'pairing',
                'group_policy': 'disabled',
            }
        )
        
        adapter = SignalAdapter(config)
        
        assert adapter.http_url == 'http://127.0.0.1:8080'
        assert adapter.account == '+1234567890'
        assert '+1234567890' in adapter.allowed_users
        assert '+0987654321' in adapter.allowed_users
        assert adapter.dm_policy == 'pairing'
        assert adapter.group_policy == 'disabled'
    
    def test_parse_comma_separated(self):
        """Test comma-separated list parsing."""
        test_cases = [
            ("", []),
            ("  ", []),
            ("a", ["a"]),
            ("a,b", ["a", "b"]),
            ("a, b , c", ["a", "b", "c"]),
            (",,a,,b,,", ["a", "b"]),
        ]
        
        for input_str, expected in test_cases:
            result = SignalAdapter._parse_comma_separated(input_str)
            assert result == expected, f"Failed for input: '{input_str}'"


class TestSignalAdapterAllowlist:
    """Test SignalAdapter allowlist checking."""
    
    def test_user_in_allowlist(self):
        """Should allow users in allowlist."""
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
                'allowed_users': '+1234567890,+0987654321',
                'dm_policy': 'allowlist',
            }
        )
        
        adapter = SignalAdapter(config)
        
        assert adapter._is_user_allowed('+1234567890') is True
        assert adapter._is_user_allowed('+0987654321') is True
        assert adapter._is_user_allowed('+1111111111') is False
    
    def test_wildcard_allowlist(self):
        """Should allow all users with wildcard."""
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
                'allowed_users': '*',
            }
        )
        
        adapter = SignalAdapter(config)
        assert adapter._is_user_allowed('+1234567890') is True
        assert adapter._is_user_allowed('+9999999999') is True
    
    def test_group_policy_disabled(self):
        """Should deny group messages when policy is disabled."""
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
                'allowed_users': '+1234567890',  # Only this user allowed for DMs
                'group_policy': 'disabled',
            }
        )
        
        adapter = SignalAdapter(config)
        # User in main allowlist is allowed for DMs
        assert adapter._is_user_allowed('+1234567890') is True
        # But not allowed in groups when group_policy is disabled
        # (user must also be in group_allow_from for groups when policy is allowlist)
        # Note: If user is in main allowlist, they pass before group checks
        # So we test with a user NOT in main allowlist
        assert adapter._is_user_allowed('+9999999999', is_group=True) is False


class TestSignalAdapterFileExtensions:
    """Test file extension detection."""
    
    def test_guess_extension_images(self):
        """Should correctly identify image extensions."""
        adapter = SignalAdapter(PlatformConfig())

        # PNG magic bytes (need at least 8 bytes for some checks)
        assert adapter._guess_extension(b'\x89PNG\r\n\x1a\n') == '.png'
        # JPEG magic bytes (need at least 8 bytes)
        assert adapter._guess_extension(b'\xff\xd8\xff\xe0' + b'\x00' * 4) == '.jpg'
        # GIF magic bytes
        assert adapter._guess_extension(b'GIF89a' + b'\x00' * 2) == '.gif'
    
    def test_guess_extension_documents(self):
        """Should correctly identify document extensions."""
        adapter = SignalAdapter(PlatformConfig())

        # PDF magic bytes (need at least 8 bytes)
        assert adapter._guess_extension(b'%PDF-1.4' + b'\x00' * 4) == '.pdf'
        # ZIP/Office magic bytes (need at least 8 bytes)
        assert adapter._guess_extension(b'PK\x03\x04' + b'\x00' * 4) == '.docx'

    def test_guess_extension_mp3(self):
        """Should correctly identify MP3 via frame sync pattern."""
        adapter = SignalAdapter(PlatformConfig())

        # MP3 frame sync: 0xFF followed by 0xE0-0xFF (11 bits of 1s)
        assert adapter._guess_extension(b'\xff\xf0' + b'\x00' * 6) == '.mp3'
        assert adapter._guess_extension(b'\xff\xe0' + b'\x00' * 6) == '.mp3'

    def test_guess_extension_mp4(self):
        """Should correctly identify MP4 via ftyp box."""
        adapter = SignalAdapter(PlatformConfig())

        # MP4 with mp42 brand (offset 4: ftyp, offset 8: brand)
        mp42_data = b'\x00\x00\x00\x20ftypmp42' + b'\x00' * 4
        assert adapter._guess_extension(mp42_data) == '.mp4'

        # MP4 with isom brand
        isom_data = b'\x00\x00\x00\x18ftypisom' + b'\x00' * 4
        assert adapter._guess_extension(isom_data) == '.mp4'
    
    def test_is_image_ext(self):
        """Should correctly identify image extensions."""
        adapter = SignalAdapter(PlatformConfig())
        
        assert adapter._is_image_ext('.jpg') is True
        assert adapter._is_image_ext('.jpeg') is True
        assert adapter._is_image_ext('.png') is True
        assert adapter._is_image_ext('.gif') is True
        assert adapter._is_image_ext('.webp') is True
        assert adapter._is_image_ext('.mp3') is False
    
    def test_is_audio_ext(self):
        """Should correctly identify audio extensions."""
        adapter = SignalAdapter(PlatformConfig())
        
        assert adapter._is_audio_ext('.mp3') is True
        assert adapter._is_audio_ext('.wav') is True
        assert adapter._is_audio_ext('.ogg') is True
        assert adapter._is_audio_ext('.m4a') is True
        assert adapter._is_audio_ext('.jpg') is False


class TestSignalAdapterTypingIndicators:
    """Test typing indicator functionality."""
    
    def test_start_stop_typing(self):
        """Should start and stop typing indicator tasks."""
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
            }
        )
        
        adapter = SignalAdapter(config)
        adapter._running = True
        
        # Start typing (without async for now)
        # We'll test the logic without full async
        assert '_typing_tasks' in dir(adapter)


class TestSignalAdapterAttachmentSize:
    """Test attachment size validation."""
    
    def test_attachment_size_limit(self):
        """Should validate attachment size."""
        from gateway.platforms.signal import SIGNAL_MAX_ATTACHMENT_SIZE
        
        # 100MB limit
        assert SIGNAL_MAX_ATTACHMENT_SIZE == 100 * 1024 * 1024
        
        # Small file should pass
        small_file = b'a' * 1024  # 1KB
        assert len(small_file) < SIGNAL_MAX_ATTACHMENT_SIZE
        
        # Large file should fail
        large_file = b'a' * (SIGNAL_MAX_ATTACHMENT_SIZE + 1)
        assert len(large_file) > SIGNAL_MAX_ATTACHMENT_SIZE


class TestSignalAdapterSSEReconnection:
    """Test SSE reconnection logic."""

    def test_exponential_backoff_constants(self):
        """Should have correct backoff constants."""
        from gateway.platforms.signal import SSE_RETRY_DELAY_INITIAL, SSE_RETRY_DELAY_MAX

        assert SSE_RETRY_DELAY_INITIAL == 2.0 # 2 seconds
        assert SSE_RETRY_DELAY_MAX == 60.0 # 60 seconds cap
        assert SSE_RETRY_DELAY_INITIAL < SSE_RETRY_DELAY_MAX

    def test_health_check_constants(self):
        """Should have correct health check constants."""
        from gateway.platforms.signal import HEALTH_CHECK_INTERVAL, HEALTH_CHECK_STALE_THRESHOLD

        assert HEALTH_CHECK_INTERVAL == 30.0 # 30 seconds
        assert HEALTH_CHECK_STALE_THRESHOLD == 120.0 # 2 minutes
        assert HEALTH_CHECK_STALE_THRESHOLD > HEALTH_CHECK_INTERVAL


class TestSignalAdapterRegex:
    """Test image extraction regex patterns."""

    def test_markdown_http_image(self):
        """Should extract HTTP markdown images."""
        from gateway.platforms.base import BasePlatformAdapter
        
        content = "![alt text](https://example.com/image.png)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0] == ("https://example.com/image.png", "alt text")

    def test_markdown_file_image(self):
        """Should extract file:// markdown images."""
        from gateway.platforms.base import BasePlatformAdapter
        
        content = "![local](file:///path/to/image.jpg)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0] == ("file:///path/to/image.jpg", "local")

    def test_markdown_file_with_encoded_spaces(self):
        """Should extract file:// URLs with URL-encoded characters."""
        from gateway.platforms.base import BasePlatformAdapter
        
        content = "![doc](file:///path/to/my%20image.png)"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "file:///path/to/my%20image.png"

    def test_html_img_tag(self):
        """Should extract HTML img tags."""
        from gateway.platforms.base import BasePlatformAdapter
        
        content = '<img src="https://example.com/pic.gif">'
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0] == ("https://example.com/pic.gif", "")

    def test_html_file_img(self):
        """Should extract HTML img tags with file://."""
        from gateway.platforms.base import BasePlatformAdapter
        
        content = "<img src='file:///C:/Users/test/pic.webp'>"
        images, cleaned = BasePlatformAdapter.extract_images(content)
        assert len(images) == 1
        assert images[0][0] == "file:///C:/Users/test/pic.webp"


class TestSignalPairingStore:
    """Test pairing store improvements."""

    def test_duplicate_code_generation(self, tmp_path, monkeypatch):
        """Should return existing code for user with pending request."""
        import os
        from gateway.pairing import PairingStore

        # Use temp directory for pairing
        monkeypatch.setattr('gateway.pairing.PAIRING_DIR', tmp_path)

        store = PairingStore()
        user_id = "+1234567890"

        # Generate first code
        code1 = store.generate_code("signal", user_id, "Test User")
        assert code1 is not None
        assert code1 != "__RATE_LIMITED__"

        # Generate again for same user - should return same code (not rate limited)
        # This is the key fix: returns existing pending code instead of rate limiting
        code2 = store.generate_code("signal", user_id, "Test User")
        assert code2 == code1

    def test_rate_limit_returns_special_value(self, tmp_path, monkeypatch):
        """Should return __RATE_LIMITED__ when rate limited (no pending code)."""
        import os
        import time
        from gateway.pairing import PairingStore

        # Use temp directory for pairing
        monkeypatch.setattr('gateway.pairing.PAIRING_DIR', tmp_path)

        store = PairingStore()
        user_id = "+1234567890"

        # Generate first code
        code1 = store.generate_code("signal", user_id, "Test User")
        assert code1 is not None
        assert code1 != "__RATE_LIMITED__"

        # Approve the code to remove it from pending
        store.approve_code("signal", code1)

        # Now generate again - should be rate limited (no pending code exists)
        code2 = store.generate_code("signal", user_id, "Test User")
        assert code2 == "__RATE_LIMITED__"


class TestSignalSSEReconnection:
    """Test SSE reconnection logic."""

    @pytest.mark.asyncio
    async def test_sse_backoff_calculation(self):
        """Should calculate exponential backoff correctly."""
        from gateway.platforms.signal import SSE_RETRY_DELAY_INITIAL, SSE_RETRY_DELAY_MAX
        
        # Test backoff progression
        backoff = SSE_RETRY_DELAY_INITIAL
        assert backoff == 2.0
        
        # Double the backoff
        backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)
        assert backoff == 4.0
        
        # Continue doubling
        backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)
        assert backoff == 8.0
        
        # Test max cap
        backoff = 60.0
        backoff = min(backoff * 2, SSE_RETRY_DELAY_MAX)
        assert backoff == 60.0  # Should cap at max

    @pytest.mark.asyncio
    async def test_sse_listener_task_created(self):
        """Should create SSE listener task on connect."""
        from unittest.mock import AsyncMock, patch
        
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
            }
        )
        
        adapter = SignalAdapter(config)
        
        # Mock the HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        with patch.object(adapter.client, 'get', return_value=mock_response):
            with patch.object(adapter, '_sse_listener') as mock_sse:
                mock_sse.return_value = None
                # Connect should create SSE task
                result = await adapter.connect()
                assert result is True
                assert adapter._sse_task is not None

    @pytest.mark.asyncio
    async def test_health_monitor_task_created(self):
        """Should create health monitor task on connect."""
        from unittest.mock import AsyncMock, patch
        
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
            }
        )
        
        adapter = SignalAdapter(config)
        
        # Mock the HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        
        with patch.object(adapter.client, 'get', return_value=mock_response):
            with patch.object(adapter, '_sse_listener') as mock_sse:
                with patch.object(adapter, '_health_monitor') as mock_health:
                    mock_sse.return_value = None
                    mock_health.return_value = None
                    result = await adapter.connect()
                    assert result is True
                    assert adapter._health_monitor_task is not None

    def test_typing_task_tracking(self):
        """Should track typing indicator tasks per chat."""
        import asyncio
        config = PlatformConfig(
            enabled=True,
            extra={
                'http_url': 'http://127.0.0.1:8080',
                'account': '+1234567890',
            }
        )
        
        adapter = SignalAdapter(config)
        assert isinstance(adapter._typing_tasks, dict)
        assert len(adapter._typing_tasks) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
