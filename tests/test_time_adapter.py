from gateway.config import PlatformConfig, Platform
from gateway.platforms.slack import SlackAdapter


def _cfg():
    return PlatformConfig(enabled=True, token="xoxb-test")


def test_slack_seam_defaults():
    a = SlackAdapter(_cfg())
    assert a._app_token_env() == "SLACK_APP_TOKEN"
    assert a._api_base_url() is None
