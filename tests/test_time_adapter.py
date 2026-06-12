from gateway.config import PlatformConfig, Platform
from gateway.platforms.slack import SlackAdapter


def _cfg():
    return PlatformConfig(enabled=True, token="xoxb-test")


def test_slack_seam_defaults():
    a = SlackAdapter(_cfg())
    assert a._app_token_env() == "SLACK_APP_TOKEN"
    assert a._api_base_url() is None


def test_seam_override_flows_through_factories():
    class _Sub(SlackAdapter):
        def _app_token_env(self):
            return "CUSTOM_APP_TOKEN"
        def _api_base_url(self):
            return "https://example.test/api/"
    a = _Sub(_cfg())
    assert a._app_token_env() == "CUSTOM_APP_TOKEN"
    client = a._make_web_client("tok")
    assert str(client.base_url).rstrip("/") == "https://example.test/api"
    app = a._make_async_app("tok")
    assert app is not None
