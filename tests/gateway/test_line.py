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
