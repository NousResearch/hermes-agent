"""Tests for the Time (Т‑Банк) platform adapter.

Time is a Mattermost-based messenger, so ``TimeAdapter`` subclasses the
bundled ``MattermostAdapter`` and carries its own ``Platform("time")``
identity (load-bearing for the corp_safe security profile). Its inbound
transport is REST long-polling because Time's proxy blocks the Mattermost
WebSocket.

The Slack-seam tests at the bottom exercise the generic overridable seams on
``SlackAdapter`` itself (kept for any future Slack-compatible platform); they
are independent of Time.
"""

from gateway.config import PlatformConfig, Platform, is_slack_compatible
from gateway.platforms.slack import SlackAdapter
from plugins.platforms.mattermost.adapter import MattermostAdapter


def _cfg(token: str = ""):
    return PlatformConfig(enabled=True, token=token)


# ── Plugin wiring ───────────────────────────────────────────────────────

def test_time_plugin_package_exports_register():
    # Regression guard: the gateway plugin loader imports the PACKAGE
    # (__init__.py) and calls getattr(module, "register"). An empty
    # __init__.py would silently disable the whole platform.
    import importlib

    pkg = importlib.import_module("plugins.platforms.time")
    assert callable(getattr(pkg, "register", None)), (
        "plugins/platforms/time/__init__.py must re-export register()"
    )


# ── Identity & transport ────────────────────────────────────────────────

def test_time_subclasses_mattermost():
    from plugins.platforms.time.adapter import TimeAdapter
    assert issubclass(TimeAdapter, MattermostAdapter)


def test_time_adapter_has_time_platform_identity():
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg(token="t-bot"))
    assert a.platform == Platform("time")
    assert a.platform.value == "time"


def test_time_build_source_carries_time_platform():
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg(token="t-bot"))
    src = a.build_source(chat_id="C1", chat_type="group", user_id="U1")
    assert src.platform == Platform("time")


def test_time_maps_env_to_mattermost_config(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru")
    monkeypatch.setenv("TIME_BOT_TOKEN", "bot-token-xyz")
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg())  # no token on config → read from TIME_BOT_TOKEN
    assert a._base_url == "https://time.tbank.ru"
    assert a._token == "bot-token-xyz"


def test_time_is_not_slack_protocol_compatible():
    # Time is Mattermost-based; Slack-protocol behaviors must NOT apply to it.
    assert is_slack_compatible(Platform("time")) is False
    assert is_slack_compatible(Platform.SLACK) is True


def test_time_ws_handshake_headers_send_bearer(monkeypatch):
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru")
    from plugins.platforms.time.adapter import TimeAdapter
    a = TimeAdapter(_cfg(token="bot-token-xyz"))
    headers = a._ws_handshake_headers()
    assert headers.get("Authorization") == "Bearer bot-token-xyz"


def test_time_uses_polling_receive_loop():
    from plugins.platforms.time.adapter import TimeAdapter
    # TimeAdapter overrides the receive loop and provides a poll loop.
    assert TimeAdapter._start_receive_loop is not MattermostAdapter._start_receive_loop
    assert hasattr(TimeAdapter, "_poll_loop")
    assert hasattr(TimeAdapter, "_discover_channels")


# ── Plugin config helpers ───────────────────────────────────────────────

def test_env_enablement_returns_flat_dict(monkeypatch):
    monkeypatch.setenv("TIME_BOT_TOKEN", "t-bot")
    monkeypatch.setenv("TIME_API_BASE_URL", "https://time.tbank.ru")
    monkeypatch.setenv("TIME_HOME_CHANNEL", "C123")
    from plugins.platforms.time.adapter import _env_enablement
    seed = _env_enablement()
    # Flat: "url" is a top-level key (merged into extra by the registry),
    # NOT nested under "extra". Mattermost transport reads config.extra["url"].
    assert seed["url"] == "https://time.tbank.ru"
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
        token = "t-bot"
        extra = {"url": "https://time.tbank.ru"}
    assert _validate_config(_Cfg()) is True

    class _Empty:
        token = None
        extra = {}
    assert _validate_config(_Empty()) is False


# ── Generic SlackAdapter seams (kept for future Slack-compatible platforms) ──

def test_slack_seam_defaults():
    a = SlackAdapter(_cfg(token="xoxb-test"))
    assert a._app_token_env() == "SLACK_APP_TOKEN"
    assert a._api_base_url() is None


def test_slack_seam_override_flows_through_factories():
    class _Sub(SlackAdapter):
        def _app_token_env(self):
            return "CUSTOM_APP_TOKEN"

        def _api_base_url(self):
            return "https://example.test/api/"

    a = _Sub(_cfg(token="xoxb-test"))
    assert a._app_token_env() == "CUSTOM_APP_TOKEN"
    client = a._make_web_client("tok")
    assert str(client.base_url).rstrip("/") == "https://example.test/api"
    assert a._make_async_app("tok") is not None


def test_slack_adapter_keeps_slack_identity():
    a = SlackAdapter(_cfg(token="xoxb-test"))
    assert a.platform == Platform.SLACK
