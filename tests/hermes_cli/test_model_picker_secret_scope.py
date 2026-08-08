"""The /model picker must read provider keys through the per-profile scope.

854007d1c ("route remaining main-agent fallback key reads through
secret_scope") swept the fallback/auxiliary key reads. ``key_env`` lookups in
``list_authenticated_providers`` — which the gateway's ``/model`` handler calls
directly — were not covered, so under ``multiplex_profiles`` one profile's
picker resolved another profile's key from the process environment.
"""

import os

from agent import secret_scope
from hermes_cli.model_switch import _scoped_key_env


class TestPickerKeyEnvScope:
    def test_unscoped_read_matches_the_process_environment(self, monkeypatch):
        """Single-profile deployments must behave exactly as before."""
        monkeypatch.setenv("ACME_KEY", "from-environment")

        assert _scoped_key_env("ACME_KEY") == "from-environment"

    def test_installed_scope_wins_over_the_process_environment(self, monkeypatch):
        """The multiplexed gateway installs a scope per turn; the picker must
        read that profile's credential, not whatever the process inherited."""
        monkeypatch.setenv("ACME_KEY", "other-profile-key")
        token = secret_scope.set_secret_scope({"ACME_KEY": "this-profile-key"})
        try:
            assert _scoped_key_env("ACME_KEY") == "this-profile-key"
        finally:
            secret_scope.reset_secret_scope(token)

        assert _scoped_key_env("ACME_KEY") == "other-profile-key"

    def test_absent_key_and_empty_name_resolve_empty(self, monkeypatch):
        monkeypatch.delenv("ACME_KEY", raising=False)

        assert _scoped_key_env("ACME_KEY") == ""
        assert _scoped_key_env("") == ""

    def test_value_is_stripped(self, monkeypatch):
        monkeypatch.setenv("ACME_KEY", "  padded  ")

        assert _scoped_key_env("ACME_KEY") == "padded"
