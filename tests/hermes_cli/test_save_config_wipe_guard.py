"""Tests for save_config/set_config_value parse-failure guard.

Verifies the inline-parse guard correctly:
- ALLOWS writes when config is a valid empty {} (the Societus blind-spot fix)
- BLOCKS writes when config exists but is genuinely unparseable (garbage/truncated YAML)
- ALLOWS writes when config doesn't exist (new install, no file to parse)
- ALLOWS writes when config has valid non-empty content
"""

from __future__ import annotations

import logging
import os
from unittest.mock import patch

import pytest

from hermes_cli.config import save_config, set_config_value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_path(tmp_path):
    return tmp_path / "config.yaml"


def _write_config(tmp_path, content):
    """Write raw bytes to config.yaml (allows writing invalid YAML)."""
    path = _config_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, str):
        content = content.encode("utf-8")
    path.write_bytes(content)


def _read_config(tmp_path):
    """Read config.yaml content as bytes, or empty if not present."""
    path = _config_path(tmp_path)
    return path.read_bytes() if path.exists() else b""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_hermes_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir so tests never touch real config."""
    # Ensure the config path setup functions resolve to tmp_path
    env_file = tmp_path / ".env"
    env_file.touch()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


# ---------------------------------------------------------------------------
# save_config parse guard tests
# ---------------------------------------------------------------------------

class TestSaveConfigParseGuard:
    """save_config must refuse to write when the existing file is unparseable,
    but allow writes for valid empty {}, valid non-empty, and missing files."""

    def test_valid_empty_dict_is_writable(self, caplog, _isolated_hermes_home):
        """A valid empty {} config must be writable (the Societus blind-spot fix)."""
        _write_config(_isolated_hermes_home, b"{}\n")
        caplog.set_level(logging.WARNING)

        save_config({"model": {"provider": "test"}}, strip_defaults=False)
        result = _read_config(_isolated_hermes_home)

        assert "Refusing to save config" not in caplog.text
        assert b"test" in result or b"provider" in result, (
            "save_config should have written through a valid {} config"
        )

    def test_valid_non_empty_is_writable(self, caplog, _isolated_hermes_home):
        """A valid non-empty config must be writable."""
        _write_config(_isolated_hermes_home, b"model:\n  provider: existing\n")
        caplog.set_level(logging.WARNING)

        save_config({"model": {"provider": "updated"}}, strip_defaults=False)
        result = _read_config(_isolated_hermes_home)

        assert "Refusing to save config" not in caplog.text

    def test_unparseable_garbage_is_blocked(self, caplog, _isolated_hermes_home):
        """An unparseable (garbage) config must be refused — guard fires."""
        _write_config(_isolated_hermes_home, b"unclosed: [\n")
        caplog.set_level(logging.WARNING)

        # save_config will hit the guard and return early, so the file stays
        save_config({"model": {"provider": "test"}}, strip_defaults=False)

        assert "Refusing to save config" in caplog.text, (
            "guard should log warning for unparseable config"
        )
        # File content must remain unchanged (the garbage)
        assert _read_config(_isolated_hermes_home) == b"unclosed: [\n"

    def test_missing_file_is_writable(self, caplog, _isolated_hermes_home):
        """A non-existent config file must be writable (new install)."""
        assert not _config_path(_isolated_hermes_home).exists()
        caplog.set_level(logging.WARNING)

        save_config({"model": {"provider": "test"}}, strip_defaults=False)

        assert "Refusing to save config" not in caplog.text
        assert _config_path(_isolated_hermes_home).exists()


# ---------------------------------------------------------------------------
# set_config_value parse guard tests
# ---------------------------------------------------------------------------

class TestSetConfigValueParseGuard:
    """set_config_value must refuse when the existing file is unparseable,
    but allow writes for valid and missing files."""

    def test_valid_empty_dict_is_writable(self, caplog, _isolated_hermes_home):
        """set_config_value must write through a valid empty {} config."""
        _write_config(_isolated_hermes_home, b"{}\n")
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "test-provider")
        result = _read_config(_isolated_hermes_home)

        assert "Refusing to set config value" not in caplog.text
        assert b"test-provider" in result

    def test_valid_non_empty_is_writable(self, caplog, _isolated_hermes_home):
        """set_config_value must write through a valid non-empty config."""
        _write_config(_isolated_hermes_home, b"model:\n  provider: existing\n")
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "updated")
        result = _read_config(_isolated_hermes_home)

        assert "Refusing to set config value" not in caplog.text
        assert b"updated" in result

    def test_unparseable_garbage_is_blocked(self, caplog, _isolated_hermes_home):
        """set_config_value must refuse when existing config is unparseable."""
        _write_config(_isolated_hermes_home, b"unclosed: [\n")
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "test")

        assert "Refusing to set config value" in caplog.text, (
            "guard should log warning for unparseable config"
        )
        assert _read_config(_isolated_hermes_home) == b"unclosed: [\n"

    def test_missing_file_is_writable(self, caplog, _isolated_hermes_home):
        """set_config_value must write when no config file exists."""
        assert not _config_path(_isolated_hermes_home).exists()
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "test")
        assert "Refusing to set config value" not in caplog.text


# ---------------------------------------------------------------------------
# Non-mapping root tests (Teknium review: list / scalar root must also refuse)
# ---------------------------------------------------------------------------

class TestNonMappingRootGuard:
    """A top-level list (``[]``) or scalar parses WITHOUT raising, so an
    exception-only guard would let save_config / set_config_value wipe the
    file. read_raw_config() coerces both to {} at config.py:6931, so the
    guard must type-check the parsed result, not just catch parse errors.
    """

    def test_list_root_save_config_blocked(self, caplog, _isolated_hermes_home):
        """A list-root config must be refused on save_config (not coerced to {})."""
        _write_config(_isolated_hermes_home, b"- a\n- b\n")
        caplog.set_level(logging.WARNING)

        save_config({"model": {"provider": "test"}}, strip_defaults=False)

        assert "Refusing to save config" in caplog.text, (
            "list-root config should be refused"
        )
        assert _read_config(_isolated_hermes_home) == b"- a\n- b\n"

    def test_list_root_set_config_value_blocked(self, caplog, _isolated_hermes_home):
        """A list-root config must be refused on set_config_value."""
        _write_config(_isolated_hermes_home, b"- a\n- b\n")
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "test")

        assert "Refusing to set config value" in caplog.text, (
            "list-root config should be refused"
        )
        assert _read_config(_isolated_hermes_home) == b"- a\n- b\n"

    def test_scalar_root_save_config_blocked(self, caplog, _isolated_hermes_home):
        """A scalar-root config must be refused on save_config."""
        _write_config(_isolated_hermes_home, b"just a string\n")
        caplog.set_level(logging.WARNING)

        save_config({"model": {"provider": "test"}}, strip_defaults=False)

        assert "Refusing to save config" in caplog.text, (
            "scalar-root config should be refused"
        )
        assert _read_config(_isolated_hermes_home) == b"just a string\n"

    def test_scalar_root_set_config_value_blocked(self, caplog, _isolated_hermes_home):
        """A scalar-root config must be refused on set_config_value."""
        _write_config(_isolated_hermes_home, b"42\n")
        caplog.set_level(logging.WARNING)

        set_config_value("model.provider", "test")

        assert "Refusing to set config value" in caplog.text, (
            "scalar-root config should be refused"
        )
        assert _read_config(_isolated_hermes_home) == b"42\n"


# ---------------------------------------------------------------------------
# Auth provider writer guard (Teknium review: cover auth.py write path too)
# ---------------------------------------------------------------------------

class TestAuthProviderWriterGuard:
    """_update_config_for_provider must route through the same shared guard,
    so a malformed / non-mapping existing config is not clobbered by the
    provider write either.
    """

    def test_unparseable_blocked(self, caplog, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, b"unclosed: [\n")
        caplog.set_level(logging.WARNING)

        from hermes_cli.auth import _update_config_for_provider

        _update_config_for_provider("zai", "", default_model="glm-5.2")

        assert "Refusing to set provider config" in caplog.text, (
            "auth provider writer should refuse unparseable config"
        )
        assert _read_config(_isolated_hermes_home) == b"unclosed: [\n"

    def test_list_root_blocked(self, caplog, _isolated_hermes_home):
        _write_config(_isolated_hermes_home, b"- a\n- b\n")
        caplog.set_level(logging.WARNING)

        from hermes_cli.auth import _update_config_for_provider

        _update_config_for_provider("zai", "", default_model="glm-5.2")

        assert "Refusing to set provider config" in caplog.text, (
            "auth provider writer should refuse list-root config"
        )
        assert _read_config(_isolated_hermes_home) == b"- a\n- b\n"

    def test_valid_empty_writes(self, caplog, _isolated_hermes_home):
        """A valid empty {} config must be writable through the auth writer."""
        _write_config(_isolated_hermes_home, b"{}\n")
        caplog.set_level(logging.WARNING)

        from hermes_cli.auth import _update_config_for_provider

        _update_config_for_provider("zai", "", default_model="glm-5.2")

        assert "Refusing to set provider config" not in caplog.text
        # The written model section should carry the provider + default.
        assert b"glm-5.2" in _read_config(_isolated_hermes_home)
