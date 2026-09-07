"""Regression tests for the scoped messaging enablement fix (#104619).

The scoped credential fallback must NOT report an empty-`required_env`
platform (whatsapp, api_server, webhook) as enabled when nothing
is configured. `all()` over an empty tuple would otherwise evaluate True.
"""

from hermes_cli.web_routers.messaging import _platform_enablement


def _entry(required_env=("DISCORD_BOT_TOKEN",)):
    return {"required_env": required_env}


class TestScopedEmptyRequiredEnv:
    def test_empty_required_env_with_no_config_is_NOT_enabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config", lambda: {}
        )
        enabled, configured, _ = _platform_enablement(
            "whatsapp", _entry(required_env=()), {}, scoped=True
        )
        assert enabled is False

    def test_empty_required_env_with_explicit_enabled_false(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config",
            lambda: {"platforms": {"whatsapp": {"enabled": False}}},
        )
        enabled, _, _ = _platform_enablement(
            "whatsapp", _entry(required_env=()), {}, scoped=True
        )
        assert enabled is False

    def test_empty_required_env_with_explicit_enabled_true(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config",
            lambda: {"platforms": {"whatsapp": {"enabled": True}}},
        )
        enabled, _, _ = _platform_enablement(
            "whatsapp", _entry(required_env=()), {}, scoped=True
        )
        assert enabled is True


class TestScopedWithCredentials:
    def test_env_credentials_alone_enable(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config", lambda: {}
        )
        enabled, configured, _ = _platform_enablement(
            "discord", _entry(), {"DISCORD_BOT_TOKEN": "tok"}, scoped=True
        )
        assert enabled is True
        assert configured is True

    def test_missing_credentials_stay_disabled(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config", lambda: {}
        )
        enabled, configured, _ = _platform_enablement(
            "discord", _entry(), {}, scoped=True
        )
        assert enabled is False
        assert configured is False


class TestScopedConfiguredField:
    # `configured` came from the same `all()`-over-empty expression, so an
    # empty-`required_env` platform with nothing set up also reported
    # configured=True; it must stay False for that shape too.
    def test_empty_required_env_configured_stays_false(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config", lambda: {}
        )
        enabled, configured, _ = _platform_enablement(
            "whatsapp", _entry(required_env=()), {}, scoped=True
        )
        assert enabled is False
        assert configured is False

    def test_credentials_report_configured_true(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.web_routers.messaging.load_config", lambda: {}
        )
        _, configured, _ = _platform_enablement(
            "discord", _entry(), {"DISCORD_BOT_TOKEN": "tok"}, scoped=True
        )
        assert configured is True
