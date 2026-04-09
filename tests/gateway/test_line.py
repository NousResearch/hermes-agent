"""Tests for the LINE platform adapter."""
import base64
import hashlib
import hmac as _hmac

import pytest

from gateway.config import Platform, PlatformConfig


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _line_sig(body: bytes, secret: str) -> str:
    """Compute valid LINE HMAC-SHA256 signature (base64-encoded)."""
    h = _hmac.new(secret.encode("utf-8"), body, hashlib.sha256)
    return base64.b64encode(h.digest()).decode("utf-8")


def _make_adapter(platform=None, **extra):
    """Create a LineAdapter with test credentials."""
    from gateway.platforms.line import LineAdapter

    cfg = PlatformConfig(
        enabled=True,
        extra={
            "channel_access_token": "test-token",
            "channel_secret": "test-secret",
            **extra,
        },
    )
    p = platform or Platform.LINE
    return LineAdapter(cfg, platform=p)


# ---------------------------------------------------------------------------
# Chunk 1: Platform enum
# ---------------------------------------------------------------------------

class TestLinePlatformEnum:
    def test_line_enum_exists(self):
        assert Platform.LINE.value == "line"

    def test_line_lynx_enum_exists(self):
        assert Platform.LINE_LYNX.value == "line_lynx"


class TestLineConfigDetection:
    def test_line_connected_when_token_set(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "sec")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE in config.get_connected_platforms()

    def test_line_not_connected_without_token(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE not in config.get_connected_platforms()

    def test_line_lynx_connected_when_token_set(self, monkeypatch):
        monkeypatch.setenv("LINE_LYNX_CHANNEL_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("LINE_LYNX_CHANNEL_SECRET", "sec")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.LINE_LYNX in config.get_connected_platforms()


# ---------------------------------------------------------------------------
# Chunk 2: Helpers and check_requirements
# ---------------------------------------------------------------------------

class TestLineHelpers:
    def test_strip_markdown_bold(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("**hello** world") == "hello world"

    def test_strip_markdown_italic(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("_hello_ world") == "hello world"

    def test_strip_markdown_code_inline(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("`code`") == "code"

    def test_strip_markdown_code_block(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("```python\ncode\n```") == "code"

    def test_strip_markdown_headers(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("## Heading\ntext") == "Heading\ntext"

    def test_strip_markdown_link(self):
        from gateway.platforms.line import _strip_markdown
        assert _strip_markdown("[text](http://example.com)") == "text"

    def test_split_text_within_limit(self):
        from gateway.platforms.line import _split_text
        assert _split_text("hello world", max_len=100) == ["hello world"]

    def test_split_text_at_whitespace(self):
        from gateway.platforms.line import _split_text
        text = "word1 word2 word3"
        chunks = _split_text(text, max_len=12)
        assert all(len(c) <= 12 for c in chunks)

    def test_split_text_long_word_forced_split(self):
        from gateway.platforms.line import _split_text
        # A word longer than max_len must still be split
        chunks = _split_text("a" * 20, max_len=10)
        assert all(len(c) <= 10 for c in chunks)

    def test_check_requirements_missing_sdk(self, monkeypatch):
        import gateway.platforms.line as line_mod
        monkeypatch.setattr(line_mod, "LINE_SDK_AVAILABLE", False)
        from gateway.platforms.line import check_line_requirements
        assert check_line_requirements() is False

    def test_check_requirements_missing_token(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_LYNX_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.platforms.line import check_line_requirements
        assert check_line_requirements() is False

# ---------------------------------------------------------------------------
# Chunk 2: Adapter init + connect/disconnect
# ---------------------------------------------------------------------------

class TestLineAdapterInit:
    def test_reads_token_from_extra(self):
        adapter = _make_adapter(channel_access_token="my-token")
        assert adapter.channel_access_token == "my-token"

    def test_reads_secret_from_extra(self):
        adapter = _make_adapter(channel_secret="my-secret")
        assert adapter.channel_secret == "my-secret"

    def test_default_port_for_line(self):
        adapter = _make_adapter(platform=Platform.LINE)
        assert adapter.webhook_port == 18791

    def test_default_port_for_line_lynx(self):
        adapter = _make_adapter(platform=Platform.LINE_LYNX)
        assert adapter.webhook_port == 18792

    def test_default_path_for_line(self):
        adapter = _make_adapter(platform=Platform.LINE)
        assert adapter.webhook_path == "/webhook/line"

    def test_default_path_for_line_lynx(self):
        adapter = _make_adapter(platform=Platform.LINE_LYNX)
        assert adapter.webhook_path == "/webhook/line/lynx"

    def test_custom_port_overrides_default(self):
        adapter = _make_adapter(webhook_port=19999)
        assert adapter.webhook_port == 19999

    def test_group_personas_loaded(self):
        adapter = _make_adapter(group_personas={"C123": "mochi-line"})
        assert adapter.group_personas == {"C123": "mochi-line"}

    def test_default_persona_loaded(self):
        adapter = _make_adapter(default_persona="cattia-line")
        assert adapter.default_persona == "cattia-line"

    def test_allow_from_loaded(self):
        adapter = _make_adapter(allow_from=["U123", "U456"])
        assert adapter.allow_from == ["U123", "U456"]

    def test_platform_is_line(self):
        adapter = _make_adapter(platform=Platform.LINE)
        assert adapter.platform == Platform.LINE

    def test_platform_is_line_lynx(self):
        adapter = _make_adapter(platform=Platform.LINE_LYNX)
        assert adapter.platform == Platform.LINE_LYNX


class TestLineAdapterConnect:
    def test_connect_fails_without_sdk(self, monkeypatch):
        import gateway.platforms.line as line_mod
        monkeypatch.setattr(line_mod, "LINE_SDK_AVAILABLE", False)
        adapter = _make_adapter()

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert result is False

    def test_connect_fails_without_token(self):
        from gateway.platforms.line import LineAdapter
        cfg = PlatformConfig(enabled=True, extra={"channel_secret": "sec"})
        adapter = LineAdapter(cfg)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert result is False

    def test_connect_fails_without_secret(self):
        from gateway.platforms.line import LineAdapter
        cfg = PlatformConfig(enabled=True, extra={"channel_access_token": "tok"})
        adapter = LineAdapter(cfg)

        import asyncio
        result = asyncio.get_event_loop().run_until_complete(adapter.connect())
        assert result is False


# ---------------------------------------------------------------------------
# Chunk 3: Webhook handler
# ---------------------------------------------------------------------------

class TestLineWebhookSignature:
    """Test HTTP endpoint signature verification via aiohttp TestClient."""

    def _make_app(self, **extra):
        """Build an aiohttp app from the adapter (without starting a real server)."""
        from aiohttp import web
        adapter = _make_adapter(**extra)
        app = web.Application()
        app.router.add_get("/health", lambda _: web.Response(text="ok"))
        app.router.add_post(adapter.webhook_path, adapter._handle_webhook)
        return app, adapter

    @pytest.mark.asyncio
    async def test_missing_signature_returns_401(self):
        from aiohttp.test_utils import TestClient, TestServer
        app, _ = self._make_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post("/webhook/line", data=b"{}")
            assert resp.status == 401

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_403(self):
        from aiohttp.test_utils import TestClient, TestServer
        app, _ = self._make_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/line",
                data=b'{"destination":"U123","events":[]}',
                headers={"X-Line-Signature": "invalidsignature"},
            )
            assert resp.status == 403

    @pytest.mark.asyncio
    async def test_valid_signature_empty_events_returns_200(self):
        from aiohttp.test_utils import TestClient, TestServer
        body = b'{"destination":"U123","events":[]}'
        sig = _line_sig(body, "test-secret")
        app, _ = self._make_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                "/webhook/line",
                data=body,
                headers={"X-Line-Signature": sig},
            )
            assert resp.status == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_ok(self):
        from aiohttp.test_utils import TestClient, TestServer
        app, _ = self._make_app()
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.get("/health")
            assert resp.status == 200
            text = await resp.text()
            assert text == "ok"

class TestLineAuthorization:
    """Test that _process_line_event respects dm_policy and group_policy."""

    def _make_text_event(self, source_type="user", user_id="U123", group_id=None, text="hello"):
        """Build a minimal mock LineMessageEvent."""
        from unittest.mock import MagicMock
        event = MagicMock()
        event.source.type = source_type
        event.source.user_id = user_id
        if group_id:
            event.source.group_id = group_id
        else:
            # When no group, getattr(event.source, "group_id", None) should return None
            del event.source.group_id
        event.message.id = "msg-001"
        event.message.type = "text"
        event.message.text = text
        return event

    @pytest.mark.asyncio
    async def test_dm_allowlist_drops_unknown_user(self):
        from unittest.mock import AsyncMock
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["U999"])
        adapter.handle_message = AsyncMock()
        event = self._make_text_event(source_type="user", user_id="U123")
        await adapter._process_line_event(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dm_allowlist_accepts_allowed_user(self):
        from unittest.mock import AsyncMock
        adapter = _make_adapter(dm_policy="allowlist", allow_from=["U123"])
        adapter.handle_message = AsyncMock()
        event = self._make_text_event(source_type="user", user_id="U123", text="hello")
        await adapter._process_line_event(event)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_dm_open_accepts_all(self):
        from unittest.mock import AsyncMock
        adapter = _make_adapter(dm_policy="open")
        adapter.handle_message = AsyncMock()
        event = self._make_text_event(source_type="user", user_id="U999", text="hello")
        await adapter._process_line_event(event)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_group_allowlist_drops_unknown_group(self):
        from unittest.mock import AsyncMock
        adapter = _make_adapter(
            group_policy="allowlist",
            group_personas={"C-known": "mochi-line"},
        )
        adapter.handle_message = AsyncMock()
        event = self._make_text_event(source_type="group", group_id="C-unknown")
        await adapter._process_line_event(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_open_accepts_all(self):
        from unittest.mock import AsyncMock
        adapter = _make_adapter(group_policy="open", default_persona="cattia-line")
        adapter.handle_message = AsyncMock()
        event = self._make_text_event(source_type="group", group_id="C-any", text="hello")
        await adapter._process_line_event(event)
        adapter.handle_message.assert_called_once()


class TestLinePersonaRouting:
    def _make_event(self, source_type="group", group_id="C123", user_id="U1", text="hi"):
        from unittest.mock import MagicMock
        event = MagicMock()
        event.source.type = source_type
        event.source.user_id = user_id
        if group_id:
            event.source.group_id = group_id
        else:
            del event.source.group_id
        event.message.id = "m1"
        event.message.type = "text"
        event.message.text = text
        return event

    @pytest.mark.asyncio
    async def test_group_in_personas_gets_persona_skill(self):
        adapter = _make_adapter(
            group_personas={"C123": "mochi-line"},
            default_persona="cattia-line",
        )
        captured = []
        async def _capture(ev):
            captured.append(ev)
        adapter.handle_message = _capture
        await adapter._process_line_event(self._make_event(group_id="C123"))
        assert captured[0].auto_skill == "mochi-line"

    @pytest.mark.asyncio
    async def test_group_not_in_personas_gets_default(self):
        adapter = _make_adapter(
            group_personas={"C999": "mochi-line"},
            default_persona="cattia-line",
        )
        captured = []
        async def _capture(ev):
            captured.append(ev)
        adapter.handle_message = _capture
        await adapter._process_line_event(self._make_event(group_id="C-other"))
        assert captured[0].auto_skill == "cattia-line"

    @pytest.mark.asyncio
    async def test_dm_always_gets_default_persona(self):
        adapter = _make_adapter(
            default_persona="cattia-line",
            dm_policy="open",
        )
        captured = []
        async def _capture(ev):
            captured.append(ev)
        adapter.handle_message = _capture
        await adapter._process_line_event(self._make_event(source_type="user", group_id=None))
        assert captured[0].auto_skill == "cattia-line"
