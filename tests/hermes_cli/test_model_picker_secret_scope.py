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


class TestPickerKeyEnvDotenvFallback:
    """A ``key_env`` that lives only in ``$HERMES_HOME/.env`` must resolve for the
    picker and the switch validation, matching the chat path's credential chain
    (get_env_prefer_dotenv). Without the fallback the verification probe fires
    unauthenticated and every switch prints a spurious "could not reach this
    custom endpoint's model listing" note against a healthy endpoint."""

    def _isolate_home(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ACME_RELAY_KEY", raising=False)
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.config import invalidate_env_cache

        invalidate_env_cache()

    def test_dotenv_only_key_resolves(self, monkeypatch, tmp_path):
        self._isolate_home(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=from-dotenv\n")

        assert _scoped_key_env("ACME_RELAY_KEY") == "from-dotenv"

    def test_installed_scope_wins_over_dotenv(self, monkeypatch, tmp_path):
        self._isolate_home(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=from-dotenv\n")
        token = secret_scope.set_secret_scope({"ACME_RELAY_KEY": "this-profile-key"})
        try:
            assert _scoped_key_env("ACME_RELAY_KEY") == "this-profile-key"
        finally:
            secret_scope.reset_secret_scope(token)

    def test_absent_everywhere_stays_empty(self, monkeypatch, tmp_path):
        self._isolate_home(monkeypatch, tmp_path)

        assert _scoped_key_env("ACME_RELAY_KEY") == ""
        assert _scoped_key_env("") == ""

    def test_entry_configured_key_reads_dotenv_only_key(self, monkeypatch, tmp_path):
        self._isolate_home(monkeypatch, tmp_path)
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=from-dotenv\n")
        from hermes_cli.model_switch import _entry_configured_key

        cfg = {"key_env": "ACME_RELAY_KEY", "base_url": "https://relay.example/v1"}

        assert _entry_configured_key(cfg, _scoped_key_env) == "from-dotenv"

    def test_rotated_dotenv_beats_stale_process_env(self, monkeypatch, tmp_path):
        """A long-lived shell/service can still hold the pre-rotation key in
        ``os.environ`` while ``$HERMES_HOME/.env`` already carries the rotated
        one. Single-profile reads must resolve through the chat path's chain
        (``get_env_prefer_dotenv``), which gives ``.env`` precedence — the old
        ``get_secret()``-first order returned the stale process value."""
        self._isolate_home(monkeypatch, tmp_path)
        monkeypatch.setenv("ACME_RELAY_KEY", "stale-process")
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=fresh-dotenv\n")

        assert _scoped_key_env("ACME_RELAY_KEY") == "fresh-dotenv"

    def test_unscoped_multiplex_stays_empty_even_with_dotenv(self, monkeypatch, tmp_path):
        """Multiplexing with no scope installed must fail closed — the shared
        ``.env``/process env could belong to another profile, so a key present
        there must not leak into this profile's picker read."""
        self._isolate_home(monkeypatch, tmp_path)
        monkeypatch.setenv("ACME_RELAY_KEY", "stale-process")
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=fresh-dotenv\n")
        secret_scope.set_multiplex_active(True)
        try:
            assert _scoped_key_env("ACME_RELAY_KEY") == ""
        finally:
            secret_scope.set_multiplex_active(False)

    def test_switch_model_probe_uses_rotated_dotenv_credential(self, monkeypatch, tmp_path):
        """End-to-end pin of the user-visible behavior: with a stale key in the
        process env and a rotated one in ``.env``, the ``switch_model`` →
        ``validate_requested_model(api_key=...)`` verification probe must fire
        with the fresh ``.env`` credential, exactly like the chat path."""
        self._isolate_home(monkeypatch, tmp_path)
        monkeypatch.setenv("ACME_RELAY_KEY", "stale-process")
        (tmp_path / ".env").write_text("ACME_RELAY_KEY=fresh-dotenv\n")
        import hermes_cli.model_switch as ms
        import hermes_cli.models_validate as mv

        captured = {}

        def _fake_runtime(requested, explicit_api_key=None,
                          explicit_base_url=None, target_model=None, **kw):
            return {"api_key": explicit_api_key or "", "base_url": explicit_base_url, "api_mode": ""}

        def _fake_validate(model, provider, api_key=None, base_url=None,
                           api_mode=None, headers=None, **kw):
            captured["api_key"] = api_key
            return {"accepted": True, "persist": True, "recognized": True, "message": ""}

        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider", _fake_runtime)
        monkeypatch.setattr(ms, "resolve_alias", lambda *a, **k: None)
        monkeypatch.setattr(mv, "validate_requested_model", _fake_validate)

        ms.switch_model(
            "some-model",
            current_provider="openrouter",
            current_model="x",
            explicit_provider="acme",
            user_providers={"acme": {"base_url": "https://api.acme.test/v1", "key_env": "ACME_RELAY_KEY"}},
        )

        assert captured["api_key"] == "fresh-dotenv"
