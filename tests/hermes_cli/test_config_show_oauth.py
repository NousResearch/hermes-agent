"""Tests for OAuth credential_pool display in show_config() (#20675)."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_home(tmp_path, monkeypatch):
    """Set up a minimal HERMES_HOME with config.yaml so show_config doesn't crash."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("model:\n  default: test/model\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Clear caches so show_config picks up our temp config
    import hermes_cli.config as cfg
    from hermes_cli import managed_scope

    cfg._LOAD_CONFIG_CACHE.clear()
    cfg._RAW_CONFIG_CACHE.clear()
    managed_scope.invalidate_managed_cache()
    return home


class TestShowConfigOAuth:
    """`show_config()` must surface credential_pool OAuth status for providers
    that use OAuth instead of API-key env vars (#20675)."""

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _mock_pool(has_creds: bool) -> MagicMock:
        pool = MagicMock()
        pool.has_credentials.return_value = has_creds
        return pool

    # ── happy path ───────────────────────────────────────────────────────

    def test_oauth_token_displayed_when_credentials_exist(
        self, mock_home, capsys
    ):
        """OAuth providers with credentials should show 'OAuth token'."""
        with patch(
            "agent.credential_pool.load_pool",
            return_value=self._mock_pool(has_creds=True),
        ):
            from hermes_cli.config import show_config

            show_config()

        out = capsys.readouterr().out
        assert "OAuth Tokens (credential pool)" in out
        assert "Nous Portal" in out
        assert "OAuth token" in out

    def test_not_set_when_no_credentials(self, mock_home, capsys):
        """Providers without credentials should show 'not set'."""
        with patch(
            "agent.credential_pool.load_pool",
            return_value=self._mock_pool(has_creds=False),
        ):
            from hermes_cli.config import show_config

            show_config()

        out = capsys.readouterr().out
        assert "OAuth Tokens (credential pool)" in out
        assert "not set" in out

    # ── error resilience ─────────────────────────────────────────────────

    def test_not_set_when_load_pool_raises(self, mock_home, capsys):
        """If load_pool() raises, the provider should still appear as 'not set'."""
        with patch(
            "agent.credential_pool.load_pool",
            side_effect=RuntimeError("auth.json missing"),
        ):
            from hermes_cli.config import show_config

            show_config()

        out = capsys.readouterr().out
        assert "OAuth Tokens (credential pool)" in out
        assert "not set" in out
        # All five providers survive the error path
        assert "Nous Portal" in out

    def test_no_crash_when_credential_pool_import_fails(self, mock_home, capsys):
        """If the entire credential_pool module can't be imported,
        show_config() must not crash — the section is simply skipped."""
        from hermes_cli.config import show_config

        with patch.dict("sys.modules", {"agent.credential_pool": None}):
            show_config()

        out = capsys.readouterr().out
        # The OAuth section should not appear at all
        assert "OAuth Tokens (credential pool)" not in out
        # But the rest of the output is intact
        assert "API Keys" in out
        assert "Model" in out

    # ── completeness ─────────────────────────────────────────────────────

    def test_all_five_oauth_providers_listed(self, mock_home, capsys):
        """All five OAuth-backed providers must appear in the section."""
        with patch(
            "agent.credential_pool.load_pool",
            return_value=self._mock_pool(has_creds=True),
        ):
            from hermes_cli.config import show_config

            show_config()

        out = capsys.readouterr().out
        expected_providers = [
            "Nous Portal",
            "OpenAI Codex",
            "GitHub Copilot",
            "Anthropic (OAuth)",
            "xAI (OAuth)",
        ]
        for provider in expected_providers:
            assert provider in out, f"Missing OAuth provider: {provider}"

    def test_mixed_credentials_shown_correctly(self, mock_home, capsys):
        """Some providers have credentials, some don't — both states shown."""
        real_load_pool = None

        def _selective_load_pool(provider: str):
            # Only nous has credentials
            if provider == "nous":
                return TestShowConfigOAuth._mock_pool(has_creds=True)
            return TestShowConfigOAuth._mock_pool(has_creds=False)

        with patch(
            "agent.credential_pool.load_pool",
            side_effect=_selective_load_pool,
        ):
            from hermes_cli.config import show_config

            show_config()

        out = capsys.readouterr().out
        assert "OAuth token" in out  # at least nous has it
        assert "not set" in out      # others don't
