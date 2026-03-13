"""Tests for Matrix protocol platform adapter.

Follows the same patterns and coverage depth as test_signal.py.
All test data uses generic placeholder identifiers (example.org, matrix.org)
rather than any real user or homeserver names.
"""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Platform Enum
# ---------------------------------------------------------------------------

class TestMatrixPlatformEnum:
    def test_matrix_enum_value(self):
        assert Platform.MATRIX.value == "matrix"

    def test_matrix_in_platform_list(self):
        assert "matrix" in [p.value for p in Platform]


# ---------------------------------------------------------------------------
# Config loading via _apply_env_overrides
# ---------------------------------------------------------------------------

class TestMatrixConfigLoading:
    def test_all_three_vars_required(self, monkeypatch):
        """Matrix is not enabled unless all three required vars are present."""
        from gateway.config import GatewayConfig, _apply_env_overrides

        # Only two of three vars — should NOT enable Matrix
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.platforms

    def test_missing_token(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.platforms

    def test_missing_homeserver(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.platforms

    def test_all_vars_present_enables_matrix(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert Platform.MATRIX in config.platforms
        mc = config.platforms[Platform.MATRIX]
        assert mc.enabled is True
        assert mc.token == "syt_test_token_abc"
        assert mc.extra["homeserver_url"] == "https://matrix.example.org"
        assert mc.extra["user_id"] == "@hermes:matrix.example.org"

    def test_strips_trailing_slash_from_homeserver(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org/")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        config = GatewayConfig()
        _apply_env_overrides(config)

        assert not config.platforms[Platform.MATRIX].extra["homeserver_url"].endswith("/")

    def test_verify_ssl_default_true(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")
        monkeypatch.delenv("MATRIX_VERIFY_SSL", raising=False)

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].extra["verify_ssl"] is True

    def test_verify_ssl_disabled(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")
        monkeypatch.setenv("MATRIX_VERIFY_SSL", "false")

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].extra["verify_ssl"] is False

    def test_home_channel_loaded(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")
        monkeypatch.setenv("MATRIX_HOME_CHANNEL", "!roomabc:matrix.example.org")
        monkeypatch.setenv("MATRIX_HOME_CHANNEL_NAME", "HQ")

        config = GatewayConfig()
        _apply_env_overrides(config)

        hc = config.platforms[Platform.MATRIX].home_channel
        assert hc is not None
        assert hc.chat_id == "!roomabc:matrix.example.org"
        assert hc.name == "HQ"

    def test_home_channel_default_name(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")
        monkeypatch.setenv("MATRIX_HOME_CHANNEL", "!roomabc:matrix.example.org")
        monkeypatch.delenv("MATRIX_HOME_CHANNEL_NAME", raising=False)

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert config.platforms[Platform.MATRIX].home_channel.name == "Home"

    def test_get_connected_platforms_includes_matrix(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX in config.get_connected_platforms()

    def test_get_connected_platforms_excludes_unconfigured_matrix(self, monkeypatch):
        from gateway.config import GatewayConfig, _apply_env_overrides
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.MATRIX not in config.get_connected_platforms()


# ---------------------------------------------------------------------------
# Adapter Init
# ---------------------------------------------------------------------------

class TestMatrixAdapterInit:
    def _make_config(self, **overrides):
        config = PlatformConfig()
        config.enabled = True
        config.token = overrides.pop("token", "syt_test_token_abc")
        config.extra = {
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@hermes:matrix.example.org",
            "verify_ssl": True,
            **overrides,
        }
        return config

    def test_platform_class_attribute(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            assert MatrixAdapter.platform == Platform.MATRIX

    def test_parses_homeserver_url(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())
            assert adapter.homeserver_url == "https://matrix.example.org"

    def test_parses_token(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config(token="syt_mytoken"))
            assert adapter.access_token == "syt_mytoken"

    def test_parses_user_id(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())
            assert adapter.user_id == "@hermes:matrix.example.org"

    def test_verify_ssl_true(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config(verify_ssl=True))
            assert adapter.verify_ssl is True

    def test_verify_ssl_false(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config(verify_ssl=False))
            assert adapter.verify_ssl is False

    def test_initial_state(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())
            assert adapter._client is None
            assert adapter._sync_task is None
            assert adapter._running is False
            assert adapter._next_batch is None
            assert adapter._seen_event_ids == {}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

class TestMatrixHelpers:
    def test_redact_matrix_id_standard(self):
        from gateway.platforms.matrix import _redact_matrix_id
        result = _redact_matrix_id("@alice:example.org")
        assert result.startswith("@al")
        assert "alice" not in result
        assert "example.org" in result
        assert "**" in result

    def test_redact_matrix_id_two_char_localpart(self):
        from gateway.platforms.matrix import _redact_matrix_id
        result = _redact_matrix_id("@ab:matrix.org")
        # Two-char localpart: both chars masked
        assert result.startswith("@")
        assert "matrix.org" in result

    def test_redact_matrix_id_one_char_localpart(self):
        from gateway.platforms.matrix import _redact_matrix_id
        result = _redact_matrix_id("@a:matrix.org")
        assert result.startswith("@")
        assert "matrix.org" in result

    def test_redact_matrix_id_no_at(self):
        from gateway.platforms.matrix import _redact_matrix_id
        assert _redact_matrix_id("notamatrixid") == "notamatrixid"

    def test_redact_matrix_id_empty(self):
        from gateway.platforms.matrix import _redact_matrix_id
        assert _redact_matrix_id("") == "<none>"

    def test_redact_matrix_id_with_port(self):
        from gateway.platforms.matrix import _redact_matrix_id
        result = _redact_matrix_id("@bob:homeserver.example.org:8448")
        assert "bob" not in result
        assert "8448" in result

    def test_parse_comma_list_standard(self):
        from gateway.platforms.matrix import _parse_comma_list
        result = _parse_comma_list("@alice:matrix.org, @bob:example.org , @carol:example.com")
        assert result == [
            "@alice:matrix.org",
            "@bob:example.org",
            "@carol:example.com",
        ]

    def test_parse_comma_list_empty(self):
        from gateway.platforms.matrix import _parse_comma_list
        assert _parse_comma_list("") == []
        assert _parse_comma_list("  ,  ,  ") == []

    def test_is_image_type(self):
        from gateway.platforms.matrix import _is_image_type
        assert _is_image_type("image/png") is True
        assert _is_image_type("image/jpeg") is True
        assert _is_image_type("image/gif") is True
        assert _is_image_type("audio/ogg") is False
        assert _is_image_type("application/pdf") is False

    def test_is_audio_type(self):
        from gateway.platforms.matrix import _is_audio_type
        assert _is_audio_type("audio/ogg") is True
        assert _is_audio_type("audio/mpeg") is True
        assert _is_audio_type("image/png") is False

    def test_is_video_type(self):
        from gateway.platforms.matrix import _is_video_type
        assert _is_video_type("video/mp4") is True
        assert _is_video_type("video/webm") is True
        assert _is_video_type("image/png") is False

    def test_ext_from_mime_common_types(self):
        from gateway.platforms.matrix import _ext_from_mime
        assert _ext_from_mime("image/jpeg") in (".jpg", ".jpeg")  # mimetypes may return either
        assert _ext_from_mime("image/png") == ".png"
        assert _ext_from_mime("audio/ogg") == ".ogg"
        assert _ext_from_mime("video/mp4") == ".mp4"
        assert _ext_from_mime("application/pdf") == ".pdf"

    def test_ext_from_mime_unknown(self):
        from gateway.platforms.matrix import _ext_from_mime
        assert _ext_from_mime("application/unknown-type") == ".bin"

    def test_mime_from_path_images(self):
        from pathlib import Path
        from gateway.platforms.matrix import _mime_from_path
        assert _mime_from_path(Path("photo.jpg")) == "image/jpeg"
        assert _mime_from_path(Path("screenshot.png")) == "image/png"
        assert _mime_from_path(Path("animation.gif")) == "image/gif"

    def test_mime_from_path_audio(self):
        from pathlib import Path
        from gateway.platforms.matrix import _mime_from_path
        assert _mime_from_path(Path("voice.ogg")) == "audio/ogg"
        assert _mime_from_path(Path("music.mp3")) == "audio/mpeg"

    def test_mime_from_path_unknown(self):
        from pathlib import Path
        from gateway.platforms.matrix import _mime_from_path
        assert _mime_from_path(Path("file.xyz")) == "application/octet-stream"

    def test_localpart_extraction(self):
        from gateway.platforms.matrix import _localpart
        assert _localpart("@alice:example.org") == "alice"
        assert _localpart("@hermes:matrix.org") == "hermes"
        assert _localpart("notamatrixid") == "notamatrixid"

    def test_markdown_to_html_with_markdown_installed(self):
        """_markdown_to_html should convert markdown to HTML when library available."""
        from gateway.platforms.matrix import _markdown_to_html
        try:
            import markdown  # noqa: F401
            result = _markdown_to_html("**bold** and `code`")
            assert result is not None
            assert "<strong>" in result or "<b>" in result
            assert "<code>" in result
        except ImportError:
            pytest.skip("markdown library not installed")

    def test_markdown_to_html_without_markdown(self):
        """_markdown_to_html should return None gracefully if library missing."""
        from gateway.platforms.matrix import _markdown_to_html
        with patch.dict("sys.modules", {"markdown": None}):
            result = _markdown_to_html("**bold**")
            assert result is None


# ---------------------------------------------------------------------------
# check_matrix_requirements
# ---------------------------------------------------------------------------

class TestCheckMatrixRequirements:
    def test_all_present_and_nio_available(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is True

    def test_nio_not_installed(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", False):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False

    def test_missing_homeserver(self, monkeypatch):
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False

    def test_missing_token(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False

    def test_missing_user_id(self, monkeypatch):
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False

    def test_all_missing(self, monkeypatch):
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import check_matrix_requirements
            assert check_matrix_requirements() is False


# ---------------------------------------------------------------------------
# Connect failures
# ---------------------------------------------------------------------------

class TestMatrixConnectFailures:
    def _make_config(self, **overrides):
        config = PlatformConfig()
        config.enabled = True
        config.token = overrides.pop("token", "syt_test_token_abc")
        config.extra = {
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@hermes:matrix.example.org",
            "verify_ssl": True,
            **overrides,
        }
        return config

    @pytest.mark.asyncio
    async def test_connect_fails_without_nio(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", False):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_with_empty_homeserver(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config(homeserver_url=""))
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_with_empty_token(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config(token=""))
            result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_when_whoami_errors(self):
        """connect() returns False when the homeserver is unreachable."""
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            mock_client = MagicMock()
            mock_client.whoami = AsyncMock(side_effect=Exception("Connection refused"))
            mock_client.close = AsyncMock()

            with patch("gateway.platforms.matrix.AsyncClient", return_value=mock_client):
                with patch("gateway.platforms.matrix.AsyncClientConfig"):
                    adapter = MatrixAdapter(self._make_config())
                    result = await adapter.connect()
            assert result is False

    @pytest.mark.asyncio
    async def test_connect_fails_when_whoami_returns_error_object(self):
        """connect() returns False when whoami() returns an error response."""
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            mock_client = MagicMock()
            # Error object has no user_id attribute
            mock_client.whoami = AsyncMock(return_value=MagicMock(spec=[]))
            mock_client.close = AsyncMock()

            with patch("gateway.platforms.matrix.AsyncClient", return_value=mock_client):
                with patch("gateway.platforms.matrix.AsyncClientConfig"):
                    adapter = MatrixAdapter(self._make_config())
                    result = await adapter.connect()
            assert result is False


# ---------------------------------------------------------------------------
# Session Source
# ---------------------------------------------------------------------------

class TestMatrixSessionSource:
    def test_dm_source(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!roomabc123:matrix.example.org",
            chat_name="DM",
            chat_type="dm",
            user_id="@alice:example.org",
            user_name="Alice",
        )
        assert source.platform == Platform.MATRIX
        assert source.chat_id == "!roomabc123:matrix.example.org"
        assert source.user_id == "@alice:example.org"
        assert source.chat_type == "dm"

    def test_group_source(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!grouproom:matrix.example.org",
            chat_name="Team",
            chat_type="group",
            user_id="@bob:example.org",
            user_name="Bob",
        )
        assert source.chat_type == "group"

    def test_round_trip_serialization(self):
        from gateway.session import SessionSource
        source = SessionSource(
            platform=Platform.MATRIX,
            chat_id="!roomabc123:matrix.example.org",
            chat_name="General",
            chat_type="group",
            user_id="@carol:example.org",
            user_name="Carol",
        )
        data = source.to_dict()
        restored = SessionSource.from_dict(data)
        assert restored.platform == Platform.MATRIX
        assert restored.chat_id == source.chat_id
        assert restored.user_id == source.user_id
        assert restored.chat_type == source.chat_type


# ---------------------------------------------------------------------------
# Redaction integration (agent/redact.py)
# ---------------------------------------------------------------------------

class TestMatrixIdRedaction:
    def test_matrix_id_localpart_is_masked(self):
        from agent.redact import redact_sensitive_text
        text = "Message from @alice:example.org in room !abc:example.org"
        result = redact_sensitive_text(text)
        # Full localpart "alice" should not appear (only first 2 chars kept)
        assert "lice" not in result
        assert "@al" in result
        assert "example.org" in result

    def test_matrix_id_short_localpart(self):
        from agent.redact import redact_sensitive_text
        text = "User @ab:matrix.org sent a message"
        result = redact_sensitive_text(text)
        assert "matrix.org" in result

    def test_multiple_matrix_ids_redacted(self):
        from agent.redact import redact_sensitive_text
        text = "From @alice:example.org to @bob:matrix.org"
        result = redact_sensitive_text(text)
        assert "lice" not in result
        assert "ob" not in result.split("@")[1]  # "bob" localpart after first @


# ---------------------------------------------------------------------------
# Authorization (gateway/run.py)
# ---------------------------------------------------------------------------

class TestMatrixAuthorization:
    def test_matrix_in_allowlist_maps(self):
        """Matrix must be in both authorization maps in GatewayRunner."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.MATRIX
        source.user_id = "@alice:example.org"

        # No allowlists or allow-all configured — should deny
        with patch.dict("os.environ", {}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False

    def test_matrix_allowed_via_matrix_allowed_users(self):
        """User in MATRIX_ALLOWED_USERS should be authorized."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.MATRIX
        source.user_id = "@alice:example.org"

        env = {"MATRIX_ALLOWED_USERS": "@alice:example.org,@bob:matrix.org"}
        with patch.dict("os.environ", env, clear=True):
            result = gw._is_user_authorized(source)
            assert result is True

    def test_matrix_denied_when_not_in_allowlist(self):
        """User NOT in MATRIX_ALLOWED_USERS should be denied."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.MATRIX
        source.user_id = "@stranger:example.org"

        env = {"MATRIX_ALLOWED_USERS": "@alice:example.org,@bob:matrix.org"}
        with patch.dict("os.environ", env, clear=True):
            result = gw._is_user_authorized(source)
            assert result is False

    def test_matrix_allowed_via_allow_all(self):
        """MATRIX_ALLOW_ALL_USERS=true should authorize any user."""
        from gateway.run import GatewayRunner
        from gateway.config import GatewayConfig

        gw = GatewayRunner.__new__(GatewayRunner)
        gw.config = GatewayConfig()
        gw.pairing_store = MagicMock()
        gw.pairing_store.is_approved.return_value = False

        source = MagicMock()
        source.platform = Platform.MATRIX
        source.user_id = "@anyone:example.org"

        with patch.dict("os.environ", {"MATRIX_ALLOW_ALL_USERS": "true"}, clear=True):
            result = gw._is_user_authorized(source)
            assert result is True


# ---------------------------------------------------------------------------
# Toolset
# ---------------------------------------------------------------------------

class TestMatrixToolset:
    def test_hermes_matrix_toolset_exists(self):
        from toolsets import TOOLSETS
        assert "hermes-matrix" in TOOLSETS

    def test_hermes_matrix_has_core_tools(self):
        from toolsets import TOOLSETS, _HERMES_CORE_TOOLS
        toolset = TOOLSETS["hermes-matrix"]
        assert toolset["tools"] == _HERMES_CORE_TOOLS

    def test_hermes_matrix_has_description(self):
        from toolsets import TOOLSETS
        assert TOOLSETS["hermes-matrix"].get("description")

    def test_hermes_gateway_includes_matrix(self):
        from toolsets import TOOLSETS
        assert "hermes-matrix" in TOOLSETS["hermes-gateway"]["includes"]


# ---------------------------------------------------------------------------
# Send Message Tool
# ---------------------------------------------------------------------------

class TestMatrixSendMessageTool:
    def test_matrix_in_platform_map(self):
        """Matrix must be in the platform_map dict in send_message_tool."""
        # Import the module to verify Platform.MATRIX is handled
        from gateway.config import Platform
        import tools.send_message_tool as smt
        import inspect

        source = inspect.getsource(smt)
        assert '"matrix": Platform.MATRIX' in source or "'matrix': Platform.MATRIX" in source

    def test_send_matrix_function_exists(self):
        from tools.send_message_tool import _send_matrix
        import asyncio
        import inspect
        assert callable(_send_matrix)
        assert asyncio.iscoroutinefunction(_send_matrix)

    def test_send_matrix_returns_error_without_nio(self):
        """When matrix-nio is blocked, _send_matrix must return an error dict (not raise)."""
        import asyncio
        import sys
        from unittest.mock import patch

        async def _run():
            # Block the nio package so the ImportError branch is exercised
            with patch.dict(sys.modules, {"nio": None}):
                # Re-import inside the patch so the function re-evaluates its import
                import importlib
                import tools.send_message_tool as smt_mod
                importlib.reload(smt_mod)
                config = PlatformConfig()
                config.token = "syt_test_token_abc"
                config.extra = {
                    "homeserver_url": "https://matrix.example.org",
                    "user_id": "@hermes:matrix.example.org",
                    "verify_ssl": True,
                }
                result = await smt_mod._send_matrix(
                    config, "!roomabc:matrix.example.org", "hello"
                )
                return result

        result = asyncio.run(_run())
        assert "error" in result, f"Expected error key, got: {result}"
        assert "matrix-nio" in result["error"]

    def test_send_matrix_validates_user_id(self):
        """_send_matrix must return error when user_id is empty."""
        import asyncio

        async def _run():
            try:
                from nio import AsyncClient, AsyncClientConfig, RoomSendResponse
            except ImportError:
                pytest.skip("matrix-nio not installed")

            config = PlatformConfig()
            config.token = "syt_test_token_abc"
            config.extra = {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "",  # Empty user_id
                "verify_ssl": True,
            }
            from tools.send_message_tool import _send_matrix
            result = await _send_matrix(config, "!roomabc:matrix.example.org", "hello")
            return result

        result = asyncio.run(_run())
        assert "error" in result
        assert "MATRIX_USER_ID" in result["error"]


# ---------------------------------------------------------------------------
# Cron Scheduler
# ---------------------------------------------------------------------------

class TestMatrixCronScheduler:
    def test_matrix_in_platform_map(self):
        """Matrix must be in the platform_map in cron/scheduler.py."""
        import cron.scheduler as scheduler_mod
        import inspect
        source = inspect.getsource(scheduler_mod)
        assert '"matrix": Platform.MATRIX' in source or "'matrix': Platform.MATRIX" in source


# ---------------------------------------------------------------------------
# Channel Directory
# ---------------------------------------------------------------------------

class TestMatrixChannelDirectory:
    def test_matrix_in_session_discovery(self):
        """Matrix rooms must be discovered from session history."""
        import gateway.channel_directory as cd_mod
        import inspect
        source = inspect.getsource(cd_mod)
        assert '"matrix"' in source


# ---------------------------------------------------------------------------
# Prompt Hints
# ---------------------------------------------------------------------------

class TestMatrixPromptHints:
    def test_matrix_hint_exists(self):
        from agent.prompt_builder import PLATFORM_HINTS
        assert "matrix" in PLATFORM_HINTS

    def test_matrix_hint_mentions_markdown(self):
        from agent.prompt_builder import PLATFORM_HINTS
        hint = PLATFORM_HINTS["matrix"].lower()
        assert "markdown" in hint

    def test_matrix_hint_mentions_media(self):
        from agent.prompt_builder import PLATFORM_HINTS
        hint = PLATFORM_HINTS["matrix"]
        assert "MEDIA:" in hint or "media" in hint.lower()

    def test_matrix_hint_mentions_matrix(self):
        from agent.prompt_builder import PLATFORM_HINTS
        hint = PLATFORM_HINTS["matrix"]
        assert "Matrix" in hint or "matrix" in hint


# ---------------------------------------------------------------------------
# Platform Status (_platform_status in gateway.py)
# ---------------------------------------------------------------------------

class TestMatrixPlatformStatus:
    def _find_matrix_platform(self):
        from hermes_cli.gateway import _PLATFORMS
        return next(p for p in _PLATFORMS if p["key"] == "matrix")

    def test_matrix_not_configured(self, monkeypatch):
        from hermes_cli.gateway import _platform_status
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        platform = self._find_matrix_platform()
        status = _platform_status(platform)
        assert status == "not configured"

    def test_matrix_partially_configured_token_only(self, monkeypatch):
        from hermes_cli.gateway import _platform_status
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.delenv("MATRIX_HOMESERVER_URL", raising=False)
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        platform = self._find_matrix_platform()
        status = _platform_status(platform)
        assert status == "partially configured"

    def test_matrix_partially_configured_homeserver_only(self, monkeypatch):
        from hermes_cli.gateway import _platform_status
        monkeypatch.delenv("MATRIX_ACCESS_TOKEN", raising=False)
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.delenv("MATRIX_USER_ID", raising=False)

        platform = self._find_matrix_platform()
        status = _platform_status(platform)
        assert status == "partially configured"

    def test_matrix_fully_configured(self, monkeypatch):
        from hermes_cli.gateway import _platform_status
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "syt_test_token_abc")
        monkeypatch.setenv("MATRIX_HOMESERVER_URL", "https://matrix.example.org")
        monkeypatch.setenv("MATRIX_USER_ID", "@hermes:matrix.example.org")

        platform = self._find_matrix_platform()
        status = _platform_status(platform)
        assert status == "configured"

    def test_matrix_wizard_entry_has_required_fields(self):
        platform = self._find_matrix_platform()
        assert platform["key"] == "matrix"
        assert platform["label"] == "Matrix"
        assert platform["token_var"] == "MATRIX_ACCESS_TOKEN"
        assert "vars" in platform
        assert "setup_instructions" in platform

    def test_matrix_wizard_vars_include_all_required(self):
        platform = self._find_matrix_platform()
        var_names = [v["name"] for v in platform["vars"]]
        assert "MATRIX_HOMESERVER_URL" in var_names
        assert "MATRIX_ACCESS_TOKEN" in var_names
        assert "MATRIX_USER_ID" in var_names
        assert "MATRIX_ALLOWED_USERS" in var_names
        assert "MATRIX_VERIFY_SSL" in var_names


# ---------------------------------------------------------------------------
# Room helper functions
# ---------------------------------------------------------------------------

class TestMatrixRoomHelpers:
    def test_get_room_type_dm(self):
        from gateway.platforms.matrix import _get_room_type
        mock_client = MagicMock()
        mock_room = MagicMock()
        mock_room.users = {"@alice:example.org": MagicMock(), "@hermes:example.org": MagicMock()}
        mock_client.rooms = MagicMock()
        mock_client.rooms.get = MagicMock(return_value=mock_room)

        assert _get_room_type(mock_client, "!room1:example.org") == "dm"

    def test_get_room_type_group(self):
        from gateway.platforms.matrix import _get_room_type
        mock_client = MagicMock()
        mock_room = MagicMock()
        mock_room.users = {
            "@alice:example.org": MagicMock(),
            "@bob:example.org": MagicMock(),
            "@hermes:example.org": MagicMock(),
        }
        mock_client.rooms.get = MagicMock(return_value=mock_room)

        assert _get_room_type(mock_client, "!room1:example.org") == "group"

    def test_get_room_type_no_client(self):
        from gateway.platforms.matrix import _get_room_type
        assert _get_room_type(None, "!room1:example.org") == "group"

    def test_get_room_name_from_display_name(self):
        from gateway.platforms.matrix import _get_room_name
        mock_client = MagicMock()
        mock_room = MagicMock()
        mock_room.display_name = "General Chat"
        mock_client.rooms.get = MagicMock(return_value=mock_room)

        assert _get_room_name(mock_client, "!room1:example.org") == "General Chat"

    def test_get_room_name_fallback_to_room_id(self):
        from gateway.platforms.matrix import _get_room_name
        mock_client = MagicMock()
        mock_client.rooms.get = MagicMock(return_value=None)

        assert _get_room_name(mock_client, "!room1:example.org") == "!room1:example.org"

    def test_get_room_name_no_client(self):
        from gateway.platforms.matrix import _get_room_name
        assert _get_room_name(None, "!room1:example.org") == "!room1:example.org"

    def test_get_display_name_from_room_member(self):
        from gateway.platforms.matrix import _get_display_name
        mock_client = MagicMock()
        mock_room = MagicMock()
        mock_member = MagicMock()
        mock_member.display_name = "Alice Smith"
        mock_room.users = {"@alice:example.org": mock_member}
        mock_client.rooms = MagicMock()
        mock_client.rooms.values = MagicMock(return_value=[mock_room])

        assert _get_display_name(mock_client, "@alice:example.org") == "Alice Smith"

    def test_get_display_name_fallback_to_localpart(self):
        from gateway.platforms.matrix import _get_display_name
        mock_client = MagicMock()
        mock_room = MagicMock()
        mock_room.users = {}
        mock_client.rooms.values = MagicMock(return_value=[mock_room])

        assert _get_display_name(mock_client, "@carol:example.org") == "carol"

    def test_get_display_name_no_client(self):
        from gateway.platforms.matrix import _get_display_name
        assert _get_display_name(None, "@dave:matrix.org") == "dave"


# ---------------------------------------------------------------------------
# Invite handling (allowlist integration)
# ---------------------------------------------------------------------------

class TestMatrixInviteHandling:
    def _make_config(self):
        config = PlatformConfig()
        config.enabled = True
        config.token = "syt_test_token_abc"
        config.extra = {
            "homeserver_url": "https://matrix.example.org",
            "user_id": "@hermes:matrix.example.org",
            "verify_ssl": True,
        }
        return config

    @pytest.mark.asyncio
    async def test_invite_accepted_when_allow_all(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOW_ALL_USERS", "true")
        monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())

            mock_client = MagicMock()
            join_resp = MagicMock()
            join_resp.room_id = "!newroom:example.org"
            mock_client.join = AsyncMock(return_value=join_resp)
            adapter._client = mock_client

            await adapter._handle_invite("!newroom:example.org", "@anyone:example.org")
            mock_client.join.assert_called_once()

    @pytest.mark.asyncio
    async def test_invite_rejected_when_inviter_not_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:example.org")
        monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())

            mock_client = MagicMock()
            mock_client.join = AsyncMock()
            adapter._client = mock_client

            await adapter._handle_invite("!newroom:example.org", "@stranger:example.org")
            mock_client.join.assert_not_called()

    @pytest.mark.asyncio
    async def test_invite_accepted_when_inviter_in_allowlist(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@alice:example.org,@bob:matrix.org")
        monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())

            mock_client = MagicMock()
            join_resp = MagicMock()
            join_resp.room_id = "!newroom:example.org"
            mock_client.join = AsyncMock(return_value=join_resp)
            adapter._client = mock_client

            await adapter._handle_invite("!newroom:example.org", "@alice:example.org")
            mock_client.join.assert_called_once()

    @pytest.mark.asyncio
    async def test_invite_accepted_when_no_allowlist_configured(self, monkeypatch):
        """When no allowlist is set, bot joins (gateway auth handles message-level checks)."""
        monkeypatch.delenv("MATRIX_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("MATRIX_ALLOW_ALL_USERS", raising=False)

        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            adapter = MatrixAdapter(self._make_config())

            mock_client = MagicMock()
            join_resp = MagicMock()
            join_resp.room_id = "!newroom:example.org"
            mock_client.join = AsyncMock(return_value=join_resp)
            adapter._client = mock_client

            await adapter._handle_invite("!newroom:example.org", "@anyone:example.org")
            mock_client.join.assert_called_once()


# ---------------------------------------------------------------------------
# Send (mocked client)
# ---------------------------------------------------------------------------

class TestMatrixSend:
    def _make_adapter(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter
            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test_token_abc"
            config.extra = {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "@hermes:matrix.example.org",
                "verify_ssl": True,
            }
            return MatrixAdapter(config)

    @pytest.mark.asyncio
    async def test_send_returns_error_when_not_connected(self):
        from gateway.platforms.matrix import MatrixAdapter
        adapter = self._make_adapter()
        # _client is None (not connected)
        result = await adapter.send("!room:example.org", "hello")
        assert result.success is False
        assert "Not connected" in result.error

    @pytest.mark.asyncio
    async def test_send_success(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter, RoomSendResponse

            adapter = self._make_adapter()
            mock_client = MagicMock()
            send_resp = MagicMock(spec=RoomSendResponse)
            send_resp.event_id = "$event123"
            mock_client.room_send = AsyncMock(return_value=send_resp)
            adapter._client = mock_client

            result = await adapter.send("!room:example.org", "hello world")
            assert result.success is True
            assert result.message_id == "$event123"

    @pytest.mark.asyncio
    async def test_send_failure_propagates_error(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            adapter = self._make_adapter()
            mock_client = MagicMock()
            mock_client.room_send = AsyncMock(side_effect=Exception("network error"))
            adapter._client = mock_client

            result = await adapter.send("!room:example.org", "hello")
            assert result.success is False
            assert "network error" in result.error

    @pytest.mark.asyncio
    async def test_edit_message_success(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter, RoomSendResponse

            adapter = self._make_adapter()
            mock_client = MagicMock()
            send_resp = MagicMock(spec=RoomSendResponse)
            send_resp.event_id = "$edit_event456"
            mock_client.room_send = AsyncMock(return_value=send_resp)
            adapter._client = mock_client

            result = await adapter.edit_message("!room:example.org", "$orig_event", "updated text")
            assert result.success is True
            assert result.message_id == "$edit_event456"

            # Verify the event content has the m.replace relation
            call_args = mock_client.room_send.call_args
            content = call_args.kwargs.get("content") or call_args[1].get("content") or call_args[0][2]
            assert content["m.relates_to"]["rel_type"] == "m.replace"
            assert content["m.relates_to"]["event_id"] == "$orig_event"

    @pytest.mark.asyncio
    async def test_send_image_not_found(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            adapter = self._make_adapter()
            mock_client = MagicMock()
            adapter._client = mock_client

            result = await adapter.send_image(
                "!room:example.org", "/nonexistent/path/image.png"
            )
            assert result.success is False
            assert "/nonexistent/path/image.png" in result.error

    @pytest.mark.asyncio
    async def test_send_document_not_found(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            adapter = self._make_adapter()
            mock_client = MagicMock()
            adapter._client = mock_client

            result = await adapter.send_document(
                "!room:example.org", "/nonexistent/report.pdf"
            )
            assert result.success is False
            assert "/nonexistent/report.pdf" in result.error

    @pytest.mark.asyncio
    async def test_get_chat_info(self):
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter

            adapter = self._make_adapter()
            mock_client = MagicMock()
            mock_room = MagicMock()
            mock_room.display_name = "Team Chat"
            mock_room.users = {
                "@alice:example.org": MagicMock(),
                "@bob:example.org": MagicMock(),
                "@hermes:example.org": MagicMock(),
            }
            mock_client.rooms.get = MagicMock(return_value=mock_room)
            mock_client.rooms.values = MagicMock(return_value=[mock_room])
            adapter._client = mock_client

            info = await adapter.get_chat_info("!room:example.org")
            assert info["chat_id"] == "!room:example.org"
            assert info["name"] == "Team Chat"
            assert info["type"] == "group"


# ---------------------------------------------------------------------------
# Self-message filtering
# ---------------------------------------------------------------------------

class TestMatrixSelfMessageFilter:
    @pytest.mark.asyncio
    async def test_self_messages_are_ignored(self):
        """handle_message must not be called for events from the bot's own user_id."""
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter, RoomMessageText

            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test_token_abc"
            config.extra = {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "@hermes:matrix.example.org",
                "verify_ssl": True,
            }
            adapter = MatrixAdapter(config)
            adapter.handle_message = AsyncMock()
            adapter._client = MagicMock()
            adapter._client.rooms.get = MagicMock(return_value=None)
            adapter._client.rooms.values = MagicMock(return_value=[])

            # Create a mock event that appears to come from the bot itself
            bot_event = MagicMock(spec=RoomMessageText)
            bot_event.sender = "@hermes:matrix.example.org"  # same as user_id
            bot_event.event_id = "$bot_event_001"
            bot_event.body = "I am the bot"
            bot_event.server_timestamp = 0

            await adapter._handle_room_event("!room:example.org", bot_event)
            adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_user_messages_are_dispatched(self):
        """handle_message must be called for events from other users."""
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter, RoomMessageText

            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test_token_abc"
            config.extra = {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "@hermes:matrix.example.org",
                "verify_ssl": True,
            }
            adapter = MatrixAdapter(config)
            adapter.handle_message = AsyncMock()
            mock_client = MagicMock()
            mock_client.rooms.get = MagicMock(return_value=None)
            mock_client.rooms.values = MagicMock(return_value=[])
            adapter._client = mock_client

            user_event = MagicMock(spec=RoomMessageText)
            user_event.sender = "@alice:example.org"  # different from bot user_id
            user_event.event_id = "$user_event_001"
            user_event.body = "Hello bot!"
            user_event.server_timestamp = 0

            await adapter._handle_room_event("!room:example.org", user_event)
            adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_duplicate_events_are_ignored(self):
        """Second delivery of the same event_id must be silently dropped."""
        with patch("gateway.platforms.matrix._NIO_AVAILABLE", True):
            from gateway.platforms.matrix import MatrixAdapter, RoomMessageText

            config = PlatformConfig()
            config.enabled = True
            config.token = "syt_test_token_abc"
            config.extra = {
                "homeserver_url": "https://matrix.example.org",
                "user_id": "@hermes:matrix.example.org",
                "verify_ssl": True,
            }
            adapter = MatrixAdapter(config)
            adapter.handle_message = AsyncMock()
            mock_client = MagicMock()
            mock_client.rooms.get = MagicMock(return_value=None)
            mock_client.rooms.values = MagicMock(return_value=[])
            adapter._client = mock_client

            event = MagicMock(spec=RoomMessageText)
            event.sender = "@alice:example.org"
            event.event_id = "$dup_event_001"
            event.body = "Duplicate"
            event.server_timestamp = 0

            await adapter._handle_room_event("!room:example.org", event)
            await adapter._handle_room_event("!room:example.org", event)  # duplicate

            # handle_message should only be called once
            assert adapter.handle_message.call_count == 1
