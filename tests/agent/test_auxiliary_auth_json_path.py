"""Tests for profile-aware path resolution in auxiliary client (#40677)."""

import pytest
from pathlib import Path


class TestAuthJsonPath:
    def test_resolves_at_call_time(self, monkeypatch, tmp_path):
        """_auth_json_path() should use the live HERMES_HOME."""
        profile_a = tmp_path / "profile_a"
        profile_a.mkdir()
        monkeypatch.setattr(
            "agent.auxiliary_client.get_hermes_home",
            lambda: profile_a,
        )
        from agent.auxiliary_client import _auth_json_path
        path_a = _auth_json_path()
        assert str(profile_a) in str(path_a)
        assert path_a.name == "auth.json"

    def test_legacy_constant_preserved(self):
        """The module-level _AUTH_JSON_PATH constant still exists."""
        from agent.auxiliary_client import _AUTH_JSON_PATH
        assert isinstance(_AUTH_JSON_PATH, Path)
        assert _AUTH_JSON_PATH.name == "auth.json"
