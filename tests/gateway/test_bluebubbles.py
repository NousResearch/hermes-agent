"""BlueBubbles configuration, formatting and webhook URL behavior."""

import pytest

from gateway.config import Platform, PlatformConfig

from tests.gateway.bluebubbles_test_support import _make_adapter

pytestmark = pytest.mark.usefixtures("_isolate_bluebubbles_environment")


class TestBlueBubblesConfigLoading:
    def test_apply_env_overrides_bluebubbles(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_REQUIRE_MENTION", "true")
        monkeypatch.setenv("BLUEBUBBLES_MENTION_PATTERNS", r'["(?i)^amos\\b"]')
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig()
        _apply_env_overrides(config)
        assert Platform.BLUEBUBBLES in config.platforms
        bc = config.platforms[Platform.BLUEBUBBLES]
        assert bc.enabled is True
        assert bc.extra["server_url"] == "http://localhost:1234"
        assert bc.extra["password"] == "secret"
        assert bc.extra["webhook_port"] == 9999
        assert bc.extra["require_mention"] is True
        assert bc.extra["mention_patterns"] == ["(?i)^amos\\b"]

    def test_yaml_bridges_reply_ux_and_env_does_not_stomp_it(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.setenv("BLUEBUBBLES_SERVER_URL", "http://localhost:1234")
        monkeypatch.setenv("BLUEBUBBLES_PASSWORD", "secret")
        for key in (
            "BLUEBUBBLES_WEBHOOK_HOST",
            "BLUEBUBBLES_WEBHOOK_PORT",
            "BLUEBUBBLES_WEBHOOK_PATH",
            "BLUEBUBBLES_SEND_READ_RECEIPTS",
        ):
            monkeypatch.delenv(key, raising=False)
        (tmp_path / "config.yaml").write_text(
            """
platforms:
  bluebubbles:
    enabled: true
    auto_react: false
    auto_react_type: loved
    send_read_receipts: false
    split_paragraph_replies: true
    typing_indicators: false
    typing_refresh_interval: 7
    webhook_host: 0.0.0.0
    webhook_port: 9876
    webhook_path: /custom-hook
""".strip()
        )
        from gateway.config import load_gateway_config

        config = load_gateway_config()
        platform_config = config.platforms[Platform.BLUEBUBBLES]
        extra = platform_config.extra

        assert platform_config.typing_indicator is False
        assert extra["auto_react"] is False
        assert extra["auto_react_type"] == "loved"
        assert extra["send_read_receipts"] is False
        assert extra["split_paragraph_replies"] is True
        assert extra["typing_indicators"] is False
        assert extra["typing_refresh_interval"] == 7
        assert extra["webhook_host"] == "0.0.0.0"
        assert extra["webhook_port"] == 9876
        assert extra["webhook_path"] == "/custom-hook"

    def test_explicit_env_values_override_yaml_backed_defaults(self, monkeypatch):
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_HOST", "127.0.0.2")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PORT", "9999")
        monkeypatch.setenv("BLUEBUBBLES_WEBHOOK_PATH", "/env-hook")
        monkeypatch.setenv("BLUEBUBBLES_SEND_READ_RECEIPTS", "true")
        from gateway.config import GatewayConfig, _apply_env_overrides

        config = GatewayConfig(
            platforms={
                Platform.BLUEBUBBLES: PlatformConfig(
                    enabled=True,
                    extra={
                        "server_url": "http://configured.example",
                        "password": "configured-secret",
                        "webhook_host": "0.0.0.0",
                        "webhook_port": 9876,
                        "webhook_path": "/yaml-hook",
                        "send_read_receipts": False,
                    },
                )
            }
        )

        _apply_env_overrides(config)
        extra = config.platforms[Platform.BLUEBUBBLES].extra

        assert extra["server_url"] == "http://configured.example"
        assert extra["password"] == "configured-secret"
        assert extra["webhook_host"] == "127.0.0.2"
        assert extra["webhook_port"] == 9999
        assert extra["webhook_path"] == "/env-hook"
        assert extra["send_read_receipts"] is True


class TestBlueBubblesHelpers:
    def test_format_message_preserves_underscores_in_identifiers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        text = "Use /api_v2 with FEATURE_FLAG_NAME and config_file.json"
        assert adapter.format_message(text) == text

    def test_strip_markdown_headers(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.format_message("## Heading\ntext") == "Heading\ntext"

    def test_init_normalizes_webhook_path(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, webhook_path="bluebubbles-webhook")
        assert adapter.webhook_path == "/bluebubbles-webhook"

    def test_server_url_normalized(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, server_url="http://localhost:1234/")
        assert adapter.server_url == "http://localhost:1234"


class TestBlueBubblesWebhookUrl:
    """_webhook_url property normalises local hosts to 'localhost'."""

    def test_default_host(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Default webhook_host is 0.0.0.0 → normalized to localhost
        assert "localhost" in adapter._webhook_url
        assert str(adapter.webhook_port) in adapter._webhook_url
        assert adapter.webhook_path in adapter._webhook_url

    def test_register_url_omits_query_when_no_password(self, monkeypatch):
        """If no password is configured, the register URL should be the bare URL."""
        monkeypatch.delenv("BLUEBUBBLES_PASSWORD", raising=False)
        from gateway.platforms.bluebubbles import BlueBubblesAdapter

        cfg = PlatformConfig(
            enabled=True,
            extra={"server_url": "http://localhost:1234", "password": ""},
        )
        adapter = BlueBubblesAdapter(cfg)
        assert adapter._webhook_register_url == adapter._webhook_url
