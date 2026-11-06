"""Tests for apply_yaml_config_fn ValueError surfacing.

Verifies that ValueError raised by a platform's apply_yaml_config_fn
is logged at ERROR level (not silently swallowed at debug), so the
user knows why a platform's YAML bridge was skipped.
"""

import logging
from unittest.mock import patch, MagicMock

import pytest


class TestApplyYamlConfigValidationError:
    """Validation errors from apply_yaml_config_fn must be visible."""

    @pytest.fixture
    def temp_hermes_home(self, monkeypatch, tmp_path):
        """Isolate HERMES_HOME for E2E config tests."""
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def _make_entry(self, name, side_effect):
        entry = MagicMock()
        entry.name = name
        entry.source = "plugin"
        entry.apply_yaml_config_fn = MagicMock(side_effect=side_effect)
        return entry

    def test_value_error_logged_at_error_level(
        self, temp_hermes_home, caplog
    ):
        """ValueError from apply_yaml_config_fn is caught by the
        ValueError-specific handler and logged at ERROR level.

        We mock the platform registry to inject a fake
        apply_yaml_config_fn that raises ValueError, simulating a
        platform validation failure (e.g. duplicate app_id, exceeding
        app limits). This is provider-agnostic — the fix applies to
        ALL platforms that use apply_yaml_config_fn.
        """
        config_path = temp_hermes_home / "config.yaml"
        config_path.write_text(
            "broken_platform:\n  some_key: val\n", encoding="utf-8"
        )

        mock_entry = self._make_entry(
            "broken_platform",
            ValueError("duplicate app_id"),
        )
        mock_registry = MagicMock()
        mock_registry.configured_entries.return_value = [mock_entry]
        mock_registry.all_entries.return_value = [mock_entry]
        mock_registry.get.return_value = mock_entry

        with patch("gateway.platform_registry.platform_registry", mock_registry):
            from gateway.config import load_gateway_config

            with caplog.at_level(logging.ERROR, logger="gateway.config"):
                # ValueError must NOT abort config load — the handler logs
                # and continues.
                gw_config = load_gateway_config()

        assert gw_config is not None
        assert any(
            "Configuration error" in record.message
            and "YAML bridge" in record.message
            and "duplicate app_id" in record.message
            and record.levelno == logging.ERROR
            for record in caplog.records
        ), (
            "ValueError from apply_yaml_config_fn should be surfaced at "
            "ERROR, not silently swallowed"
        )

    def test_generic_exception_still_at_debug(
        self, temp_hermes_home, caplog
    ):
        """Non-ValueError exceptions keep the existing debug-level handler.

        A misbehaving plugin (RuntimeError) must not abort config load and
        must not be surfaced at ERROR — only ValueError gets that.
        """
        config_path = temp_hermes_home / "config.yaml"
        config_path.write_text(
            "broken_platform:\n  some_key: val\n", encoding="utf-8"
        )

        mock_entry = self._make_entry(
            "broken_platform",
            RuntimeError("unexpected internal error"),
        )
        mock_registry = MagicMock()
        mock_registry.configured_entries.return_value = [mock_entry]
        mock_registry.all_entries.return_value = [mock_entry]
        mock_registry.get.return_value = mock_entry

        with patch("gateway.platform_registry.platform_registry", mock_registry):
            from gateway.config import load_gateway_config

            with caplog.at_level(logging.DEBUG, logger="gateway.config"):
                # Should NOT raise — generic handler catches it
                gw_config = load_gateway_config()
                assert gw_config is not None

        # RuntimeError was NOT logged at ERROR (only ValueError gets ERROR)
        assert not any(
            record.levelno == logging.ERROR
            and "unexpected internal error" in record.message
            for record in caplog.records
        )