"""Update-time missing-deps warning must cover platforms with no platform_registry entry (#126028).

weixin (and every other platform still on the legacy ``gateway.run._BUILTIN_ADAPTERS`` table) has
no ``PlatformEntry``, so ``platform_registry.get()`` returns ``None`` for it and its missing
dependencies were silently skipped instead of surfaced in the update warning.
"""

from gateway.config import Platform
from hermes_cli import main_install_repair


class _FakeGatewayConfig:
    def __init__(self, platforms):
        self._platforms = platforms

    def get_connected_platforms(self):
        return self._platforms


def test_legacy_builtin_platform_missing_deps_is_reported(monkeypatch):
    """weixin is not in platform_registry; its deps must still be checked via the legacy table."""
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: _FakeGatewayConfig([Platform.WEIXIN]))
    monkeypatch.setattr("gateway.platforms.weixin.check_weixin_requirements", lambda: False)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {})

    missing = main_install_repair._configured_features_missing_deps()

    assert any(feature == "weixin" for feature, _hint in missing)


def test_legacy_builtin_platform_with_deps_present_is_not_reported(monkeypatch):
    monkeypatch.setattr(
        "gateway.config.load_gateway_config", lambda: _FakeGatewayConfig([Platform.WEIXIN]))
    monkeypatch.setattr("gateway.platforms.weixin.check_weixin_requirements", lambda: True)
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly", lambda: {})

    missing = main_install_repair._configured_features_missing_deps()

    assert not any(feature == "weixin" for feature, _hint in missing)
