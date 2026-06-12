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


def test_time_adapter_overrides(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru/api/")
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())
    assert a._app_token_env() == "TIME_APP_TOKEN"
    assert a._api_base_url() == "https://time.tbank.ru/api/"


def test_time_make_web_client_uses_base_url(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru/api/")
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())
    client = a._make_web_client("t-bot")
    assert str(client.base_url).rstrip("/") == "https://time.tbank.ru/api"


def test_time_api_base_url_none_when_env_unset(monkeypatch):
    monkeypatch.delenv("TIME_API_BASE_URL", raising=False)
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())
    assert a._api_base_url() is None


def test_time_adapter_has_time_platform_identity():
    from gateway.config import Platform
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())
    assert a.platform == Platform("time")
    assert a.platform.value == "time"


def test_time_build_source_carries_time_platform():
    from gateway.config import Platform
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())
    src = a.build_source(chat_id="C1", chat_type="group", user_id="U1")
    assert src.platform == Platform("time")


def test_slack_adapter_keeps_slack_identity():
    from gateway.config import Platform
    from gateway.platforms.slack import SlackAdapter
    a = SlackAdapter(_cfg())
    assert a.platform == Platform.SLACK
