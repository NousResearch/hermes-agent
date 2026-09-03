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


class TestSwitchModelKeyEnvScope:
    """switch_model's user-provider credential reads (the ${VAR} api_key
    expansion and the key_env fallback) must go through the same scope —
    these feed resolve_runtime_provider as explicit_api_key, so a raw
    environ read here leaks another profile's key into the actual switch,
    not just the picker listing."""

    def _run_switch(self, monkeypatch, user_cfg):
        import hermes_cli.model_switch as ms

        captured = {}

        def _fake_runtime(requested, explicit_api_key=None,
                          explicit_base_url=None, target_model=None, **kw):
            captured["key"] = explicit_api_key
            return {"api_key": explicit_api_key or "", "base_url": explicit_base_url, "api_mode": ""}

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_runtime
        )
        monkeypatch.setattr(ms, "resolve_alias", lambda *a, **k: None)
        result = ms.switch_model(
            "some-model",
            current_provider="openrouter",
            current_model="x",
            explicit_provider="acme",
            user_providers={"acme": user_cfg},
        )
        return captured, result

    def test_key_env_read_honors_installed_scope(self, monkeypatch):
        monkeypatch.setenv("ACME_KEY", "other-profile-key")
        token = secret_scope.set_secret_scope({"ACME_KEY": "this-profile-key"})
        try:
            captured, _ = self._run_switch(
                monkeypatch,
                {"base_url": "https://api.acme.test/v1", "key_env": "ACME_KEY"},
            )
        finally:
            secret_scope.reset_secret_scope(token)
        assert captured["key"] == "this-profile-key"

    def test_dollar_var_expansion_honors_installed_scope(self, monkeypatch):
        monkeypatch.setenv("ACME_KEY", "other-profile-key")
        token = secret_scope.set_secret_scope({"ACME_KEY": "this-profile-key"})
        try:
            captured, _ = self._run_switch(
                monkeypatch,
                {"base_url": "https://api.acme.test/v1", "api_key": "${ACME_KEY}"},
            )
        finally:
            secret_scope.reset_secret_scope(token)
        assert captured["key"] == "this-profile-key"
