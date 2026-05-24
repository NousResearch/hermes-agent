"""Tests for hermes_constants module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from hermes_constants import get_hermes_home, get_default_hermes_root


class TestGetHermesHome:
    """Tests for get_hermes_home()."""

    def test_hermes_home_env_var_takes_precedence(self, monkeypatch):
        """HERMES_HOME env var should take precedence over everything."""
        monkeypatch.setenv("HERMES_HOME", "/custom/path")
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            assert get_hermes_home() == Path("/custom/path")

    def test_hermes_profile_env_var_scopes_home(self, monkeypatch, tmp_path):
        """HERMES_PROFILE should scope home to ~/.hermes/<profile> when HERMES_HOME is unset."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HERMES_PROFILE", "alice")
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = get_hermes_home()
                assert result == tmp_path / ".hermes" / "alice"

    def test_hermes_profile_not_set_falls_back_to_default(self, monkeypatch, tmp_path):
        """When neither HERMES_HOME nor HERMES_PROFILE is set, fall back to ~/.hermes."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.delenv("HERMES_PROFILE", raising=False)
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = get_hermes_home()
                assert result == tmp_path / ".hermes"

    def test_hermes_profile_empty_string_ignored(self, monkeypatch, tmp_path):
        """Empty HERMES_PROFILE should be treated as unset."""
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HERMES_PROFILE", "")
        with patch("hermes_constants.get_hermes_home_override", return_value=None):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result = get_hermes_home()
                assert result == tmp_path / ".hermes"

    def test_hermes_home_override_context_var_takes_precedence(self, monkeypatch):
        """Context-local override should take precedence over env vars."""
        from hermes_constants import set_hermes_home_override
        monkeypatch.setenv("HERMES_HOME", "/custom/path")
        monkeypatch.setenv("HERMES_PROFILE", "alice")
        token = set_hermes_home_override("/override/path")
        try:
            result = get_hermes_home()
            assert result == Path("/override/path")
        finally:
            from hermes_constants import reset_hermes_home_override
            reset_hermes_home_override(token)
