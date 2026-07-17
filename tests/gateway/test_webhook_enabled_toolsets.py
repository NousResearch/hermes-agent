"""
Tests for per-route enabled_toolsets on webhook SessionSource.
Covers the fix that allows webhook-triggered sessions to override
the platform default toolset list via route config.

PR: fix(webhook): honor per-route enabled_toolsets in gateway-triggered sessions
"""

import pytest
from gateway.session import SessionSource
from gateway.platforms.base import Platform


class TestSessionSourceEnabledToolsets:
    """SessionSource.enabled_toolsets field — round-trip and backward compat."""

    def test_field_defaults_to_none(self):
        """New field must default to None (backward-compat sentinel)."""
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="test")
        assert src.enabled_toolsets is None

    def test_to_dict_omits_field_when_none(self):
        """None must NOT appear in to_dict output (backward compat)."""
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="test")
        d = src.to_dict()
        assert "enabled_toolsets" not in d

    def test_to_dict_includes_field_when_set(self):
        """Non-None value must be serialized."""
        src = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="test",
            enabled_toolsets=["file", "terminal"],
        )
        d = src.to_dict()
        assert d["enabled_toolsets"] == ["file", "terminal"]

    def test_to_dict_includes_empty_list(self):
        """Empty list is a valid explicit directive — must round-trip."""
        src = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="test",
            enabled_toolsets=[],
        )
        d = src.to_dict()
        assert "enabled_toolsets" in d
        assert d["enabled_toolsets"] == []

    def test_from_dict_restores_toolsets(self):
        """from_dict must restore enabled_toolsets from serialized form."""
        src = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="test",
            enabled_toolsets=["file", "terminal"],
        )
        d = src.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.enabled_toolsets == ["file", "terminal"]

    def test_from_dict_restores_none_when_absent(self):
        """from_dict must yield None when key is absent (backward compat)."""
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="test")
        d = src.to_dict()
        assert "enabled_toolsets" not in d
        restored = SessionSource.from_dict(d)
        assert restored.enabled_toolsets is None

    def test_from_dict_restores_empty_list(self):
        """Empty list must survive the round-trip unchanged."""
        src = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="test",
            enabled_toolsets=[],
        )
        d = src.to_dict()
        restored = SessionSource.from_dict(d)
        assert restored.enabled_toolsets == []

    def test_to_dict_returns_copy_not_reference(self):
        """to_dict must return a copy so mutations don't affect the source."""
        src = SessionSource(
            platform=Platform.WEBHOOK,
            chat_id="test",
            enabled_toolsets=["file"],
        )
        d = src.to_dict()
        d["enabled_toolsets"].append("terminal")
        assert src.enabled_toolsets == ["file"]  # original unchanged


class TestWebhookRouteToolsetWiring:
    """Verify that route_config.enabled_toolsets is stamped onto source."""

    def test_route_toolsets_reach_source(self):
        """
        Simulate the webhook.py wiring logic in isolation:
        route_config.get("enabled_toolsets") -> source.enabled_toolsets.
        """
        route_config = {
            "prompt": "test",
            "enabled_toolsets": ["file", "terminal"],
        }
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="wh:test:1")

        # Mirror the patch in webhook.py _handle_webhook
        _route_toolsets = route_config.get("enabled_toolsets")
        if isinstance(_route_toolsets, list):
            src.enabled_toolsets = _route_toolsets

        assert src.enabled_toolsets == ["file", "terminal"]

    def test_missing_route_toolsets_leaves_none(self):
        """Route without enabled_toolsets must not touch source (None preserved)."""
        route_config = {"prompt": "test"}
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="wh:test:2")

        _route_toolsets = route_config.get("enabled_toolsets")
        if isinstance(_route_toolsets, list):
            src.enabled_toolsets = _route_toolsets

        assert src.enabled_toolsets is None

    def test_non_list_route_toolsets_leaves_none(self):
        """Malformed route config (non-list) must not raise and must leave None."""
        route_config = {"prompt": "test", "enabled_toolsets": "file,terminal"}
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="wh:test:3")

        _route_toolsets = route_config.get("enabled_toolsets")
        if isinstance(_route_toolsets, list):
            src.enabled_toolsets = _route_toolsets

        assert src.enabled_toolsets is None

    def test_empty_list_route_toolsets_is_stamped(self):
        """Empty list is explicit — must be stamped, not treated as falsy."""
        route_config = {"prompt": "test", "enabled_toolsets": []}
        src = SessionSource(platform=Platform.WEBHOOK, chat_id="wh:test:4")

        _route_toolsets = route_config.get("enabled_toolsets")
        if isinstance(_route_toolsets, list):
            src.enabled_toolsets = _route_toolsets

        assert src.enabled_toolsets == []


class TestRunPyGateSemantics:
    """
    Unit-test the gate logic from run.py _run_agent_inner in isolation.
    The real function is too heavy to instantiate here; we test the
    decision logic directly.
    """

    def _resolve_toolsets(self, source_enabled_toolsets, platform_default):
        """Mirror the gate logic added to run.py."""
        if source_enabled_toolsets is not None:
            return list(source_enabled_toolsets)
        else:
            return sorted(platform_default)

    def test_source_override_used_when_set(self):
        result = self._resolve_toolsets(["file", "terminal"], ["web_search", "web_extract"])
        assert result == ["file", "terminal"]

    def test_platform_default_used_when_none(self):
        result = self._resolve_toolsets(None, ["web_search", "web_extract"])
        assert result == ["web_extract", "web_search"]  # sorted

    def test_empty_list_not_collapsed_to_default(self):
        """[] must yield [] not the platform default."""
        result = self._resolve_toolsets([], ["web_search", "web_extract"])
        assert result == []

    def test_single_tool_override(self):
        result = self._resolve_toolsets(["memory"], ["web_search"])
        assert result == ["memory"]
