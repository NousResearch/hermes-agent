"""Tests for profile-aware path resolution in auxiliary client (#60241)."""

import pytest
from pathlib import Path


class TestAuthJsonPath:
    def test_resolves_at_call_time(self, monkeypatch, tmp_path):
        """_auth_json_path() should use the live HERMES_HOME, not import-time."""
        profile_a = tmp_path / "profile_a"
        profile_b = tmp_path / "profile_b"
        profile_a.mkdir()
        profile_b.mkdir()

        # Monkeypatch get_hermes_home BEFORE importing the module
        monkeypatch.setattr(
            "agent.auxiliary_client.get_hermes_home",
            lambda: profile_a,
        )
        from agent.auxiliary_client import _auth_json_path

        path_a = _auth_json_path()
        assert str(profile_a) in str(path_a)
        assert path_a.name == "auth.json"

        # Switch profile
        monkeypatch.setattr(
            "agent.auxiliary_client.get_hermes_home",
            lambda: profile_b,
        )
        import importlib
        import agent.auxiliary_client as aux_mod

        importlib.reload(aux_mod)
        path_b = aux_mod._auth_json_path()
        assert str(profile_b) in str(path_b)
        assert path_a != path_b

    def test_legacy_constant_preserved(self):
        """The module-level _AUTH_JSON_PATH constant still exists for compatibility."""
        from agent.auxiliary_client import _AUTH_JSON_PATH

        assert isinstance(_AUTH_JSON_PATH, Path)
        assert _AUTH_JSON_PATH.name == "auth.json"
