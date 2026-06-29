"""End-to-end credential isolation proof for multiplex mode (Workstream A).

These exercise the REAL resolution path (runtime_provider, secret scope, MCP
interpolation) rather than mocking it, proving the property that matters: two
profiles with different keys never see each other's, and an unscoped read in
multiplex mode fails closed instead of leaking.
"""
import pytest

from agent import secret_scope as ss


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    ss.set_multiplex_active(False)
    yield
    ss.set_multiplex_active(False)


class TestRuntimeProviderUsesScope:
    """hermes_cli.runtime_provider._getenv resolves through the secret scope."""

    def test_getenv_reads_scope_under_multiplex(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-global-leak")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"ANTHROPIC_API_KEY": "sk-profileA"})
        try:
            assert _getenv("ANTHROPIC_API_KEY") == "sk-profileA"
        finally:
            ss.reset_secret_scope(tok)

    def test_getenv_two_profiles_isolated(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        ss.set_multiplex_active(True)

        tok_a = ss.set_secret_scope({"OPENAI_API_KEY": "sk-A"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-A"
        finally:
            ss.reset_secret_scope(tok_a)

        tok_b = ss.set_secret_scope({"OPENAI_API_KEY": "sk-B"})
        try:
            assert _getenv("OPENAI_API_KEY") == "sk-B"
        finally:
            ss.reset_secret_scope(tok_b)

    def test_getenv_fails_closed_unscoped(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-leak")
        ss.set_multiplex_active(True)
        with pytest.raises(ss.UnscopedSecretError):
            _getenv("OPENROUTER_API_KEY")

    def test_getenv_global_var_still_reads_environ(self, monkeypatch):
        from hermes_cli.runtime_provider import _getenv
        monkeypatch.setenv("HERMES_MAX_ITERATIONS", "42")
        ss.set_multiplex_active(True)
        # global var: no scope needed, no raise
        assert _getenv("HERMES_MAX_ITERATIONS") == "42"


class TestMcpInterpolationUsesScope:
    """MCP config ${VAR} interpolation resolves through the secret scope."""

    def test_interpolation_reads_scope(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "global-token")
        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({"MY_MCP_TOKEN": "profile-token"})
        try:
            cfg = {"env": {"TOKEN": "${MY_MCP_TOKEN}"}}
            assert _interpolate_env_vars(cfg) == {"env": {"TOKEN": "profile-token"}}
        finally:
            ss.reset_secret_scope(tok)

    def test_interpolation_unset_keeps_placeholder(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.delenv("UNSET_MCP_VAR", raising=False)
        # multiplex off: unset var keeps literal placeholder (legacy behavior)
        assert _interpolate_env_vars("${UNSET_MCP_VAR}") == "${UNSET_MCP_VAR}"

    def test_interpolation_off_reads_environ(self, monkeypatch):
        from tools.mcp_tool import _interpolate_env_vars
        monkeypatch.setenv("MY_MCP_TOKEN", "env-token")
        # multiplex off: legacy os.environ resolution
        assert _interpolate_env_vars("${MY_MCP_TOKEN}") == "env-token"


class TestGatewayEnvEnablementRespectsExplicitDisable:
    """Global gateway env vars must not re-enable disabled secondary profiles."""

    def test_builtin_env_bridges_keep_explicitly_disabled_platforms_off(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, PlatformConfig, _apply_env_overrides

        monkeypatch.setenv("API_SERVER_KEY", "profile-api-key")
        monkeypatch.setenv("FEISHU_APP_ID", "cli_profile")
        monkeypatch.setenv("FEISHU_APP_SECRET", "feishu-secret")
        monkeypatch.setenv("WEIXIN_TOKEN", "weixin-token")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "weixin-account")

        cfg = GatewayConfig(
            platforms={
                Platform.API_SERVER: PlatformConfig(
                    enabled=False,
                    extra={"_enabled_explicit": True},
                ),
                Platform.FEISHU: PlatformConfig(
                    enabled=False,
                    extra={"_enabled_explicit": True},
                ),
                Platform.WEIXIN: PlatformConfig(
                    enabled=False,
                    extra={"_enabled_explicit": True},
                ),
            }
        )

        _apply_env_overrides(cfg)

        assert cfg.platforms[Platform.API_SERVER].enabled is False
        assert cfg.platforms[Platform.API_SERVER].extra["key"] == "profile-api-key"
        assert cfg.platforms[Platform.FEISHU].enabled is False
        assert cfg.platforms[Platform.FEISHU].extra["app_id"] == "cli_profile"
        assert cfg.platforms[Platform.WEIXIN].enabled is False
        assert cfg.platforms[Platform.WEIXIN].token == "weixin-token"
        assert "_enabled_explicit" not in cfg.platforms[Platform.API_SERVER].extra

    def test_scoped_profile_does_not_inherit_global_platform_env(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        monkeypatch.setenv("API_SERVER_KEY", "default-api-key")
        monkeypatch.setenv("FEISHU_APP_ID", "cli_default")
        monkeypatch.setenv("FEISHU_APP_SECRET", "default-feishu-secret")
        monkeypatch.setenv("WEIXIN_TOKEN", "default-weixin-token")
        monkeypatch.setenv("WEIXIN_ACCOUNT_ID", "default-weixin-account")

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope({})
        try:
            cfg = GatewayConfig()
            _apply_env_overrides(cfg)
        finally:
            ss.reset_secret_scope(tok)

        assert Platform.API_SERVER not in cfg.platforms
        assert Platform.FEISHU not in cfg.platforms
        assert Platform.WEIXIN not in cfg.platforms

    def test_scoped_feishu_env_populates_profile_extra(self, monkeypatch):
        from gateway.config import GatewayConfig, Platform, _apply_env_overrides

        monkeypatch.setenv("FEISHU_APP_ID", "cli_default")
        monkeypatch.setenv("FEISHU_APP_SECRET", "default-secret")
        monkeypatch.setenv("FEISHU_GROUP_POLICY", "disabled")

        ss.set_multiplex_active(True)
        tok = ss.set_secret_scope(
            {
                "FEISHU_APP_ID": "cli_profile",
                "FEISHU_APP_SECRET": "profile-secret",
                "FEISHU_GROUP_POLICY": "open",
                "FEISHU_ALLOWED_USERS": "ou_a,ou_b",
                "FEISHU_REQUIRE_MENTION": "false",
            }
        )
        try:
            cfg = GatewayConfig()
            _apply_env_overrides(cfg)
        finally:
            ss.reset_secret_scope(tok)

        feishu = cfg.platforms[Platform.FEISHU]
        assert feishu.enabled is True
        assert feishu.extra["app_id"] == "cli_profile"
        assert feishu.extra["app_secret"] == "profile-secret"
        assert feishu.extra["group_policy"] == "open"
        assert feishu.extra["allowed_users"] == "ou_a,ou_b"
        assert feishu.extra["require_mention"] == "false"
