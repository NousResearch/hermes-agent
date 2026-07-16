"""Tests for launch-time staleness refusal (phase 3 task 3.3).

In a checkout, launching app surfaces checks the ArtifactStamp and refuses
when stale. In a slot, no check. --build bypasses the check.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from hermes_cli.surface_staleness import check_staleness, is_slot_install


@pytest.fixture
def checkout_root(tmp_path):
    """A checkout layout (pyproject.toml + .git, no manifest.json)."""
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hermes-agent"')
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture
def slot_root(tmp_path):
    """A slot layout (has manifest.json)."""
    (tmp_path / "manifest.json").write_text('{"schema": 1}')
    return tmp_path


class TestIsSlotInstall:
    def test_slot_has_manifest(self, slot_root):
        assert is_slot_install(slot_root) is True

    def test_checkout_no_manifest(self, checkout_root):
        assert is_slot_install(checkout_root) is False


class TestCheckStaleness:
    def test_slot_skips_check(self, slot_root):
        """In a slot, always returns True (no staleness check)."""
        result = check_staleness(
            slot_root, "desktop", ".desktop_stamp",
            ["apps/desktop/src/**"], "apps/desktop/release",
        )
        assert result is True

    def test_build_flag_bypasses(self, checkout_root):
        """--build flag skips the check."""
        result = check_staleness(
            checkout_root, "desktop", ".desktop_stamp",
            ["apps/desktop/src/**"], "apps/desktop/release",
            has_build_flag=True,
        )
        assert result is True

    def test_stale_checkout_refuses(self, checkout_root):
        """In a checkout with a stale stamp, returns False."""
        mock_stamp = MagicMock()
        mock_stamp.needs_build.return_value = True
        with patch("hermes_cli.dev_sync.ArtifactStamp", return_value=mock_stamp):
            result = check_staleness(
                checkout_root, "desktop", ".desktop_stamp",
                ["apps/desktop/src/**"], "apps/desktop/release",
            )
        assert result is False

    def test_fresh_checkout_allows(self, checkout_root):
        """In a checkout with a fresh stamp, returns True."""
        mock_stamp = MagicMock()
        mock_stamp.needs_build.return_value = False
        with patch("hermes_cli.dev_sync.ArtifactStamp", return_value=mock_stamp):
            result = check_staleness(
                checkout_root, "desktop", ".desktop_stamp",
                ["apps/desktop/src/**"], "apps/desktop/release",
            )
        assert result is True

    def test_artifactstamp_failure_doesnt_block(self, checkout_root):
        """If ArtifactStamp fails, don't block the launch."""
        with patch("hermes_cli.dev_sync.ArtifactStamp", side_effect=Exception("boom")):
            result = check_staleness(
                checkout_root, "desktop", ".desktop_stamp",
                ["apps/desktop/src/**"], "apps/desktop/release",
            )
        assert result is True
