"""Tests for apply_yaml_config_fn ValueError surfacing.

Verifies that ValueError raised by a platform's apply_yaml_config_fn
is logged at ERROR level (not silently swallowed at debug), so the
user knows why a platform was disabled.
"""

import logging
import os
from pathlib import Path
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
        import yaml

        config_path = temp_hermes_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"test_platform": {"some_key": "val"}}, f)

        mock_entry = MagicMock()
        mock_entry.name = "test_platform"
        mock_entry.apply_yaml_config_fn = MagicMock(
            side_effect=ValueError("too many apps configured: limit is 10")
        )
        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False
        mock_registry.all_entries.return_value = [mock_entry]
        mock_registry.get.return_value = mock_entry

        with patch("gateway.platform_registry.platform_registry", mock_registry):
            from gateway.config import load_gateway_config

            with caplog.at_level(logging.ERROR):
                load_gateway_config()

        # The ValueError was logged at ERROR, not swallowed at debug
        assert any(
            "Configuration error" in record.message
            and record.levelno == logging.ERROR
            for record in caplog.records
        ), (
            "Expected ERROR log for ValueError from apply_yaml_config_fn, "
            f"got: {[r.message for r in caplog.records]}"
        )

    def test_generic_exception_still_handled_at_debug(
        self, temp_hermes_home, caplog
    ):
        """Non-ValueError exceptions are still caught by the generic
        except Exception handler — backward compatibility preserved.
        """
        import yaml

        config_path = temp_hermes_home / "config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump({"broken_platform": {"some_key": "val"}}, f)

        mock_entry = MagicMock()
        mock_entry.name = "broken_platform"
        mock_entry.apply_yaml_config_fn = MagicMock(
            side_effect=RuntimeError("unexpected internal error")
        )
        mock_registry = MagicMock()
        mock_registry.is_registered.return_value = False
        mock_registry.all_entries.return_value = [mock_entry]
        mock_registry.get.return_value = mock_entry

        with patch("gateway.platform_registry.platform_registry", mock_registry):
            from gateway.config import load_gateway_config

            # Should NOT raise — generic handler catches it
            gw_config = load_gateway_config()
            assert gw_config is not None

        # RuntimeError was NOT logged at ERROR (only ValueError gets ERROR)
        assert not any(
            record.levelno == logging.ERROR
            and "broken_platform" in record.message
            for record in caplog.records
        )
