from unittest.mock import MagicMock, patch

from cron.scheduler_preflight import _preflight_check_delivery
from cron.scheduler_delivery import cron_delivery_targets
from gateway.config import HomeChannel, Platform


def _proxy_host_config():
    config = MagicMock()
    config.get_connected_platforms.return_value = [Platform.API_SERVER]
    config.get_home_channel.return_value = HomeChannel(
        platform=Platform.MATRIX,
        chat_id="!room:example.org",
        name="Matrix",
    )
    return config


def test_proxy_fronted_platform_passes_cron_preflight():
    with (
        patch("gateway.config.load_gateway_config", return_value=_proxy_host_config()),
        patch("gateway.proxy_outbox.enabled_platforms", return_value={"matrix"}),
    ):
        assert _preflight_check_delivery({"deliver": "matrix:!room:example.org"}) is None


def test_proxy_fronted_platform_is_listed_as_cron_target(monkeypatch):
    monkeypatch.setenv("MATRIX_HOME_CHANNEL", "!room:example.org")
    with (
        patch("gateway.config.load_gateway_config", return_value=_proxy_host_config()),
        patch("gateway.proxy_outbox.enabled_platforms", return_value={"matrix"}),
    ):
        targets = cron_delivery_targets()

    matrix = next(target for target in targets if target["id"] == "matrix")
    assert matrix["home_target_set"] is True
