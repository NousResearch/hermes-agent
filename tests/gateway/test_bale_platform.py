from unittest.mock import patch

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner, _telegramize_command_mentions


def test_telegramize_command_mentions_treats_bale_like_telegram():
    rewritten = _telegramize_command_mentions("Try /my-command now", Platform.BALE)
    assert rewritten == "Try /my_command now"


def test_gateway_runner_creates_bale_adapter():
    runner = GatewayRunner(GatewayConfig())
    config = PlatformConfig(enabled=True, token="bale-token")

    with patch("gateway.platforms.bale.check_bale_requirements", return_value=True):
        adapter = runner._create_adapter(Platform.BALE, config)

    assert adapter is not None
    assert adapter.platform == Platform.BALE
    assert adapter.PLATFORM == Platform.BALE

