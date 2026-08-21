"""Tests for ``hermes tools web reorder`` (_provider_display_name, _tools_web_reorder)."""
from __future__ import annotations

from unittest.mock import patch, MagicMock
import pytest


class TestProviderDisplayName:
    """_provider_display_name() uses registry, not hardcoded labels."""

    def test_falls_back_on_unregistered(self):
        """When provider is not in registry, falls back to title case."""
        from hermes_cli.tools_config import _provider_display_name
        with patch("agent.web_search_registry.get_provider") as mock_gp:
            mock_gp.return_value = None
            label = _provider_display_name("my-custom-provider")
            assert label == "My Custom Provider"

    def test_uses_display_name_from_registry(self):
        """When provider exists, uses its display_name property."""
        from hermes_cli.tools_config import _provider_display_name
        with patch("agent.web_search_registry.get_provider") as mock_gp:
            mock_provider = MagicMock()
            mock_provider.display_name = "My Engine"
            mock_gp.return_value = mock_provider
            label = _provider_display_name("test-engine")
            assert label == "My Engine"


class TestToolsWebReorder:
    """_tools_web_reorder() validates input correctly."""

    def test_reorder_validation_rejects_invalid_numbers(self, monkeypatch):
        """Non-numeric input should be rejected."""
        monkeypatch.setattr("builtins.input", lambda _: "a b c")
        config = {}
        from hermes_cli.tools_config import _tools_web_reorder
        # Should print error and return without modifying config
        _tools_web_reorder(config)
        assert "fallback_backends" not in config.get("web", {})

    def test_reorder_validation_rejects_duplicates(self, monkeypatch):
        """Duplicate indices should be rejected."""
        monkeypatch.setattr("builtins.input", lambda _: "1 1 2")
        config = {"web": {"fallback_backends": ["a", "b", "c"]}}
        from hermes_cli.tools_config import _tools_web_reorder
        _tools_web_reorder(config)
        # Config should be unchanged (duplicate rejected)
        assert config["web"]["fallback_backends"] == ["a", "b", "c"]

    def test_reorder_validation_rejects_out_of_range(self, monkeypatch):
        """Index > length should be rejected."""
        monkeypatch.setattr("builtins.input", lambda _: "1 2 5")
        config = {"web": {"fallback_backends": ["a", "b", "c"]}}
        from hermes_cli.tools_config import _tools_web_reorder
        _tools_web_reorder(config)
        assert config["web"]["fallback_backends"] == ["a", "b", "c"]

    def test_reorder_empty_input_keeps_current(self, monkeypatch):
        """Empty input preserves existing order."""
        monkeypatch.setattr("builtins.input", lambda _: "")
        config = {"web": {"fallback_backends": ["a", "b", "c"]}}
        from hermes_cli.tools_config import _tools_web_reorder
        _tools_web_reorder(config)
        assert config["web"]["fallback_backends"] == ["a", "b", "c"]

    def test_empty_config_populates_from_registry(self, monkeypatch):
        """No fallback_backends configured → seed from the registry's
        search-capable providers (no dependency on the companion
        fallback-chain PR's resolver), and a reorder writes back."""
        monkeypatch.setattr("builtins.input", lambda _: "2 1")
        config = {}
        with patch(
            "agent.web_search_registry.list_providers"
        ) as mock_list:
            p1 = MagicMock()
            p1.name = "serper"
            p1.supports_search.return_value = True
            p2 = MagicMock()
            p2.name = "baidu"
            p2.supports_search.return_value = True
            p3 = MagicMock()
            p3.name = "extract-only"
            p3.supports_search.return_value = False
            mock_list.return_value = [p1, p2, p3]
            from hermes_cli.tools_config import _tools_web_reorder
            _tools_web_reorder(config)
        assert config["web"]["fallback_backends"] == ["baidu", "serper"]
