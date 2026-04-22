from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.homeassistant import HomeAssistantAdapter, check_ha_requirements
from gateway.platforms.mattermost import MattermostAdapter, check_mattermost_requirements
from gateway.platforms.signal import SignalAdapter, check_signal_requirements
from gateway.run import GatewayRunner


def _clear_gateway_env(monkeypatch) -> None:
    for key in (
        "HASS_TOKEN",
        "HASS_URL",
        "MATTERMOST_TOKEN",
        "MATTERMOST_URL",
        "SIGNAL_HTTP_URL",
        "SIGNAL_ACCOUNT",
    ):
        monkeypatch.delenv(key, raising=False)


def test_homeassistant_requirements_accept_platform_config(monkeypatch):
    _clear_gateway_env(monkeypatch)

    config = PlatformConfig(enabled=True, token="tok", extra={"url": "http://ha.local"})

    assert check_ha_requirements(config) is True


def test_mattermost_requirements_accept_platform_config(monkeypatch):
    _clear_gateway_env(monkeypatch)

    config = PlatformConfig(enabled=True, token="tok", extra={"url": "https://mm.example.com"})

    assert check_mattermost_requirements(config) is True


def test_signal_requirements_accept_platform_config(monkeypatch):
    _clear_gateway_env(monkeypatch)

    config = PlatformConfig(
        enabled=True,
        extra={"http_url": "http://127.0.0.1:8080", "account": "+15551234567"},
    )

    assert check_signal_requirements(config) is True


def test_runner_creates_config_backed_homeassistant_adapter_without_env(monkeypatch):
    _clear_gateway_env(monkeypatch)

    runner = GatewayRunner(GatewayConfig())
    config = PlatformConfig(enabled=True, token="tok", extra={"url": "http://ha.local"})

    adapter = runner._create_adapter(Platform.HOMEASSISTANT, config)

    assert isinstance(adapter, HomeAssistantAdapter)
    assert adapter._hass_token == "tok"
    assert adapter._hass_url == "http://ha.local"


def test_runner_creates_config_backed_mattermost_adapter_without_env(monkeypatch):
    _clear_gateway_env(monkeypatch)

    runner = GatewayRunner(GatewayConfig())
    config = PlatformConfig(enabled=True, token="tok", extra={"url": "https://mm.example.com"})

    adapter = runner._create_adapter(Platform.MATTERMOST, config)

    assert isinstance(adapter, MattermostAdapter)
    assert adapter._token == "tok"
    assert adapter._base_url == "https://mm.example.com"


def test_runner_creates_config_backed_signal_adapter_without_env(monkeypatch):
    _clear_gateway_env(monkeypatch)

    runner = GatewayRunner(GatewayConfig())
    config = PlatformConfig(
        enabled=True,
        extra={"http_url": "http://127.0.0.1:8080", "account": "+15551234567"},
    )

    adapter = runner._create_adapter(Platform.SIGNAL, config)

    assert isinstance(adapter, SignalAdapter)
    assert adapter.http_url == "http://127.0.0.1:8080"
    assert adapter.account == "+15551234567"
