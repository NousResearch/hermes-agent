from gateway.config import PlatformConfig, Platform, is_slack_compatible
from gateway.platforms.slack import SlackAdapter


def _cfg():
    return PlatformConfig(enabled=True, token="xoxb-test")


def test_time_plugin_package_exports_register():
    # Regression guard: the gateway plugin loader imports the PACKAGE
    # (__init__.py) and calls getattr(module, "register"). An empty
    # __init__.py would silently disable the whole platform.
    import importlib

    pkg = importlib.import_module("plugins.platforms.time")
    assert callable(getattr(pkg, "register", None)), (
        "plugins/platforms/time/__init__.py must re-export register()"
    )


def test_time_is_slack_protocol_compatible():
    # Time shares Slack's wire protocol, so Slack-protocol behaviors
    # (slash dispatch, thread_ts progress fallback) must apply to it.
    assert is_slack_compatible(Platform("time")) is True
    assert is_slack_compatible(Platform.SLACK) is True
    assert is_slack_compatible(Platform.TELEGRAM) is False


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


def test_env_enablement_returns_flat_dict(monkeypatch):
    monkeypatch.setenv("TIME_BOT_TOKEN", "t-bot")
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru/api/")
    monkeypatch.setenv("TIME_HOME_CHANNEL", "C123")
    from plugins.platforms.time.adapter import _env_enablement
    seed = _env_enablement()
    # Flat: api_base_url is a top-level key (so it merges into extra), not nested under "extra"
    assert seed["api_base_url"] == "https://time.tbank.ru/api/"
    assert "extra" not in seed
    assert seed["home_channel"] == {"chat_id": "C123"}


def test_env_enablement_none_without_token(monkeypatch):
    monkeypatch.delenv("TIME_BOT_TOKEN", raising=False)
    from plugins.platforms.time.adapter import _env_enablement
    assert _env_enablement() is None


def test_validate_config_uses_extra_when_env_absent(monkeypatch):
    monkeypatch.delenv("TIME_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TIME_API_BASE_URL", raising=False)
    from plugins.platforms.time.adapter import _validate_config
    class _Cfg:
        extra = {"token": "t-bot", "api_base_url": "https://time.tbank.ru/api/"}
    assert _validate_config(_Cfg()) is True
    class _Empty:
        extra = {}
    assert _validate_config(_Empty()) is False
