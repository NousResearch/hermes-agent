"""Tests for gateway.daimon.tool_limiter module."""

from gateway.daimon.tool_limiter import ToolLimiter


class TestToolLimiterUnlisted:
    """Test behavior for tools not in the limits dict (denied by default)."""

    def test_unlisted_tool_is_denied(self):
        limiter = ToolLimiter({"web_search": 5})
        assert limiter.check("some_other_tool") is False

    def test_unlisted_tool_remaining_is_zero(self):
        limiter = ToolLimiter({"web_search": 5})
        assert limiter.remaining("some_other_tool") == 0

    def test_unlisted_tool_denial_message(self):
        limiter = ToolLimiter({"web_search": 5})
        msg = limiter.denial_message("some_other_tool")
        assert "not permitted" in msg

    def test_explicitly_unlimited_tool(self):
        """Tools with limit=-1 are explicitly unlimited."""
        limiter = ToolLimiter({"terminal": -1})
        assert limiter.check("terminal") is True
        limiter.record("terminal")
        limiter.record("terminal")
        limiter.record("terminal")
        assert limiter.check("terminal") is True
        assert limiter.remaining("terminal") is None


class TestToolLimiterLimited:
    """Test behavior for tools with a positive limit."""

    def test_allowed_within_limit(self):
        limiter = ToolLimiter({"web_search": 3})
        assert limiter.check("web_search") is True
        limiter.record("web_search")
        assert limiter.check("web_search") is True
        limiter.record("web_search")
        assert limiter.check("web_search") is True

    def test_denied_at_limit(self):
        limiter = ToolLimiter({"web_search": 2})
        limiter.record("web_search")
        limiter.record("web_search")
        assert limiter.check("web_search") is False

    def test_remaining_decreases(self):
        limiter = ToolLimiter({"web_search": 3})
        assert limiter.remaining("web_search") == 3
        limiter.record("web_search")
        assert limiter.remaining("web_search") == 2
        limiter.record("web_search")
        assert limiter.remaining("web_search") == 1
        limiter.record("web_search")
        assert limiter.remaining("web_search") == 0

    def test_denial_message_at_limit(self):
        limiter = ToolLimiter({"web_search": 2})
        limiter.record("web_search")
        limiter.record("web_search")
        msg = limiter.denial_message("web_search")
        assert "limit reached" in msg
        assert "2/2" in msg


class TestToolLimiterDisabled:
    """Test behavior for tools with limit=0 (disabled)."""

    def test_disabled_tool_not_allowed(self):
        limiter = ToolLimiter({"text_to_speech": 0})
        assert limiter.check("text_to_speech") is False

    def test_disabled_tool_remaining_is_zero(self):
        limiter = ToolLimiter({"text_to_speech": 0})
        assert limiter.remaining("text_to_speech") == 0

    def test_disabled_tool_denial_message(self):
        limiter = ToolLimiter({"text_to_speech": 0})
        msg = limiter.denial_message("text_to_speech")
        assert "disabled" in msg


class TestToolLimiterBrowserNormalization:
    """Test that browser_* tools are normalized to 'browser'."""

    def test_browser_navigate_normalized(self):
        limiter = ToolLimiter({"browser": 5})
        assert limiter.check("browser_navigate") is True
        limiter.record("browser_navigate")
        assert limiter.remaining("browser_navigate") == 4

    def test_browser_click_normalized(self):
        limiter = ToolLimiter({"browser": 2})
        limiter.record("browser_click")
        limiter.record("browser_scroll")
        assert limiter.check("browser_type") is False

    def test_browser_itself_works(self):
        limiter = ToolLimiter({"browser": 1})
        assert limiter.check("browser") is True
        limiter.record("browser")
        assert limiter.check("browser") is False

    def test_normalize_static_method(self):
        assert ToolLimiter._normalize("browser_navigate") == "browser"
        assert ToolLimiter._normalize("browser_click") == "browser"
        assert ToolLimiter._normalize("browser") == "browser"
        assert ToolLimiter._normalize("web_search") == "web_search"

    def test_mixed_browser_tools_share_limit(self):
        limiter = ToolLimiter({"browser": 3})
        limiter.record("browser_navigate")
        limiter.record("browser_click")
        limiter.record("browser_type")
        assert limiter.check("browser_scroll") is False
        assert limiter.remaining("browser") == 0
