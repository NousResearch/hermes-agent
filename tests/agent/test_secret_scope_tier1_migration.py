"""Regression tests for the Tier-1 core-gateway secret-scope migration.

Class-closure follow-up to the profile secret-scope cluster (#76462 /
#76574): representative call sites from each migrated cluster are exercised
against the three canonical scope semantics:

- scoped value wins (the installed profile's secret is used),
- scoped miss does NOT borrow the process env under multiplex (no-borrow),
- unscoped-under-multiplex behavior per pattern:
  * in-turn sites (get_secret direct) propagate/honor UnscopedSecretError
    semantics via get_secret's verdict,
  * startup sites (Slack pattern) fall back to os.environ on
    UnscopedSecretError.
"""

import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset_multiplex():
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class _Scope:
    """Context manager installing a secret scope."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.token = None

    def __enter__(self):
        self.token = ss.set_secret_scope(self.mapping)
        return self

    def __exit__(self, *exc):
        ss.reset_secret_scope(self.token)


# ── Cluster A: gateway/pairing.py allowlist reads ─────────────────────────

class TestPairingAllowlistRead:
    def test_scoped_value_wins(self, monkeypatch):
        from gateway.pairing import _read_allowlist_env

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "111")
        ss.set_multiplex_active(True)
        with _Scope({"TELEGRAM_ALLOWED_USERS": "222"}):
            assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == "222"

    def test_scoped_miss_no_borrow(self, monkeypatch):
        from gateway.pairing import _read_allowlist_env

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "other-profile")
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == ""

    def test_unscoped_multiplex_falls_back_to_env(self, monkeypatch):
        # Slack pattern: unscoped read under multiplex uses the process env.
        from gateway.pairing import _read_allowlist_env

        monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "own-env")
        ss.set_multiplex_active(True)
        assert _read_allowlist_env("TELEGRAM_ALLOWED_USERS") == "own-env"


# ── Cluster A: gateway/authz_mixin.py gate reads ───────────────────────────

class TestAuthzPlatformGateEnv:
    def test_scoped_value_wins(self, monkeypatch):
        from gateway.authz_mixin import _platform_gate_env

        monkeypatch.setenv("DISCORD_ALLOW_BOTS", "none")
        ss.set_multiplex_active(True)
        with _Scope({"DISCORD_ALLOW_BOTS": "all"}):
            assert _platform_gate_env("DISCORD_ALLOW_BOTS", "none") == "all"

    def test_scoped_miss_returns_default_not_env(self, monkeypatch):
        from gateway.authz_mixin import _platform_gate_env

        monkeypatch.setenv("DISCORD_ALLOW_BOTS", "all")  # another profile's bridge
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert _platform_gate_env("DISCORD_ALLOW_BOTS", "none") == "none"

    def test_single_profile_legacy_env(self, monkeypatch):
        from gateway.authz_mixin import _platform_gate_env

        monkeypatch.setenv("GATEWAY_ALLOWED_USERS", "42")
        assert _platform_gate_env("GATEWAY_ALLOWED_USERS") == "42"


# ── Cluster B: matrix startup reads (Slack pattern) ────────────────────────

class TestMatrixStartupSecret:
    def _helper(self):
        mod = pytest.importorskip("plugins.platforms.matrix.adapter")
        return mod._startup_env_secret

    def test_scoped_value_wins(self, monkeypatch):
        helper = self._helper()
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "env-token")
        ss.set_multiplex_active(True)
        with _Scope({"MATRIX_ACCESS_TOKEN": "scoped-token"}):
            assert helper("MATRIX_ACCESS_TOKEN") == "scoped-token"

    def test_scoped_miss_no_borrow(self, monkeypatch):
        helper = self._helper()
        monkeypatch.setenv("MATRIX_ACCESS_TOKEN", "other-profile")
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert helper("MATRIX_ACCESS_TOKEN") == ""

    def test_unscoped_multiplex_falls_back(self, monkeypatch):
        helper = self._helper()
        monkeypatch.setenv("MATRIX_PASSWORD", "own-env-pass")
        ss.set_multiplex_active(True)
        assert helper("MATRIX_PASSWORD") == "own-env-pass"


# ── Cluster C: managed tool gateway token override ─────────────────────────

class TestToolGatewayUserToken:
    def test_scoped_value_wins(self, monkeypatch):
        from tools.managed_tool_gateway import _read_user_token_override

        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "env-tok")
        ss.set_multiplex_active(True)
        with _Scope({"TOOL_GATEWAY_USER_TOKEN": "scoped-tok"}):
            assert _read_user_token_override() == "scoped-tok"

    def test_scoped_miss_no_borrow(self, monkeypatch):
        from tools.managed_tool_gateway import _read_user_token_override

        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "other-profile-tok")
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert _read_user_token_override() is None

    def test_unscoped_multiplex_falls_back(self, monkeypatch):
        from tools.managed_tool_gateway import _read_user_token_override

        monkeypatch.setenv("TOOL_GATEWAY_USER_TOKEN", "own-env-tok")
        ss.set_multiplex_active(True)
        assert _read_user_token_override() == "own-env-tok"


class TestOpenRouterCheckApiKey:
    def test_scoped_value_wins(self, monkeypatch):
        from tools.openrouter_client import check_api_key

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        ss.set_multiplex_active(True)
        with _Scope({"OPENROUTER_API_KEY": "sk-or-scoped"}):
            assert check_api_key() is True

    def test_scoped_miss_no_borrow(self, monkeypatch):
        from tools.openrouter_client import check_api_key

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-other-profile")
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert check_api_key() is False


# ── Cluster D: auxiliary client key resolution ──────────────────────────────

class TestAuxiliaryScopedKeyEnv:
    def test_scoped_value_wins(self, monkeypatch):
        from agent.auxiliary_client import _scoped_key_env

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-env")
        ss.set_multiplex_active(True)
        with _Scope({"OPENROUTER_API_KEY": "sk-scoped"}):
            assert _scoped_key_env("OPENROUTER_API_KEY") == "sk-scoped"

    def test_scoped_miss_no_borrow(self, monkeypatch):
        from agent.auxiliary_client import _scoped_key_env

        monkeypatch.setenv("OPENAI_API_KEY", "sk-other-profile")
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            assert _scoped_key_env("OPENAI_API_KEY") == ""

    def test_unscoped_multiplex_falls_back(self, monkeypatch):
        from agent.auxiliary_client import _scoped_key_env

        monkeypatch.setenv("OPENAI_API_KEY", "sk-own-env")
        ss.set_multiplex_active(True)
        assert _scoped_key_env("OPENAI_API_KEY") == "sk-own-env"

    def test_empty_name_returns_empty(self):
        from agent.auxiliary_client import _scoped_key_env

        assert _scoped_key_env("") == ""


# ── Cluster E: azure identity presence reads ────────────────────────────────

class TestAzureIdentityPresence:
    def _describe(self):
        azure = pytest.importorskip("agent.azure_identity_adapter")
        if not azure.has_azure_identity_installed():
            pytest.skip("azure-identity not installed")
        return azure.describe_active_credential

    def test_scoped_client_secret_detected(self, monkeypatch):
        describe = self._describe()
        monkeypatch.setenv("AZURE_CLIENT_ID", "cid")
        monkeypatch.setenv("AZURE_TENANT_ID", "tid")
        monkeypatch.delenv("AZURE_CLIENT_SECRET", raising=False)
        monkeypatch.delenv("AZURE_FEDERATED_TOKEN_FILE", raising=False)
        ss.set_multiplex_active(True)
        with _Scope({"AZURE_CLIENT_SECRET": "scoped-secret"}):
            info = describe(timeout_seconds=0.01, allow_install=False)
        assert any("EnvironmentCredential" in s for s in info.get("env_sources", []))

    def test_scoped_miss_hides_env_secret(self, monkeypatch):
        describe = self._describe()
        monkeypatch.setenv("AZURE_CLIENT_ID", "cid")
        monkeypatch.setenv("AZURE_TENANT_ID", "tid")
        monkeypatch.setenv("AZURE_CLIENT_SECRET", "other-profile-secret")
        monkeypatch.delenv("AZURE_FEDERATED_TOKEN_FILE", raising=False)
        ss.set_multiplex_active(True)
        with _Scope({"UNRELATED": "x"}):
            info = describe(timeout_seconds=0.01, allow_install=False)
        assert not any(
            "EnvironmentCredential" in s for s in info.get("env_sources", [])
        )
