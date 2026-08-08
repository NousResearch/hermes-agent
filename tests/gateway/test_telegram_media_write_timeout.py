"""Behavior tests for Telegram's configurable media upload timeout."""

import asyncio
import sys
from unittest.mock import MagicMock

from gateway.config import PlatformConfig


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    telegram_mod = MagicMock()
    telegram_mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    telegram_mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    telegram_mod.constants.ChatType.GROUP = "group"
    telegram_mod.constants.ChatType.SUPERGROUP = "supergroup"
    telegram_mod.constants.ChatType.CHANNEL = "channel"
    telegram_mod.constants.ChatType.PRIVATE = "private"
    telegram_modules = (
        "telegram",
        "telegram.ext",
        "telegram.constants",
        "telegram.request",
    )
    for name in telegram_modules:
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class _StopConnect(Exception):
    """Stop connect after request construction and before network access."""


class _RecordingHTTPXRequest:
    """Record each HTTPXRequest constructor call."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.__class__.instances.append(self)


def _build_requests(monkeypatch, *, telegram_config=None):
    _RecordingHTTPXRequest.instances = []

    async def _no_fallback_ips():
        return []

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback_ips)
    monkeypatch.setattr(tg_adapter, "resolve_proxy_url", lambda *args, **kwargs: None)
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _RecordingHTTPXRequest)

    extra = tg_adapter._apply_yaml_config({}, telegram_config or {}) or {}
    adapter = TelegramAdapter(
        PlatformConfig(enabled=True, token="test-token", extra=extra)
    )
    monkeypatch.setattr(
        adapter,
        "_acquire_platform_lock",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])

    builder = MagicMock()
    builder.token.return_value = builder
    builder.request.return_value = builder
    builder.get_updates_request.return_value = builder
    builder.build.side_effect = _StopConnect
    application = MagicMock()
    application.builder.return_value = builder
    monkeypatch.setattr(tg_adapter, "Application", application)

    asyncio.run(adapter.connect())
    return list(_RecordingHTTPXRequest.instances)


def test_configured_media_write_timeout_is_passed_to_httpx_request(monkeypatch):
    requests = _build_requests(
        monkeypatch,
        telegram_config={"extra": {"media_write_timeout": 180.0}},
    )

    assert len(requests) == 2
    assert all(
        request.kwargs.get("media_write_timeout") == 180.0
        for request in requests
    )


def test_unset_media_write_timeout_uses_ptb_default(monkeypatch):
    requests = _build_requests(monkeypatch)

    assert len(requests) == 2
    assert all("media_write_timeout" not in request.kwargs for request in requests)


# --------------------------------------------------------------------------
# End-to-end config resolution.
#
# The constructor assertions above build ``extra`` by calling
# ``_apply_yaml_config`` directly, which skips the real YAML resolution. This
# drives ``load_gateway_config()`` so a regression in the loader itself cannot
# pass unnoticed.
# --------------------------------------------------------------------------


def _load_with_yaml_dict(yaml_dict: dict):
    """Drive the real ``load_gateway_config()`` with *yaml_dict* as config.yaml."""
    from pathlib import Path
    from unittest.mock import patch

    from gateway.config import load_gateway_config

    fake_home = Path("/tmp/fake_hermes_home_67655")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with patch("gateway.config.get_hermes_home", return_value=fake_home), \
         patch.object(Path, "exists", fake_exists), \
         patch("builtins.open", create=True) as mock_file:
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


def _telegram_extra(yaml_dict: dict) -> dict:
    from gateway.config import Platform

    cfg = _load_with_yaml_dict(yaml_dict)
    return cfg.platforms[Platform.TELEGRAM].extra or {}


def test_media_write_timeout_survives_the_real_config_loader():
    extra = _telegram_extra({
        "platforms": {
            "telegram": {
                "enabled": True,
                "token": "test-token",
                "extra": {"media_write_timeout": 180.0},
            }
        }
    })

    assert extra.get("media_write_timeout") == 180.0


def test_config_loader_leaves_media_write_timeout_unset_when_absent():
    extra = _telegram_extra({
        "platforms": {
            "telegram": {"enabled": True, "token": "test-token"},
        }
    })

    assert "media_write_timeout" not in extra


# --------------------------------------------------------------------------
# Standalone sender.
#
# tools/send_message_tool.py::_send_telegram builds its own Bot rather than
# reusing the gateway adapter's, so the configured timeout has to be threaded
# through separately or media uploads there keep PTB's 20 second default.
# --------------------------------------------------------------------------


class _StopAfterBot(Exception):
    """Abort _send_telegram once the Bot has been constructed."""


_built_bots: list = []


class _StubBot:
    def __init__(self, *args, **kwargs):
        object.__setattr__(self, "init_kwargs", kwargs)
        _built_bots.append(self)

    def __getattr__(self, name):
        raise _StopAfterBot(name)


def _capture_standalone_requests(monkeypatch, *, media_write_timeout, proxy=None):
    """Call ``_send_telegram`` far enough to build its Bot, returning the
    HTTPXRequest kwargs it used and the Bot constructor kwargs.

    ``_send_telegram`` resolves ``telegram`` and ``telegram.request`` lazily
    through ``sys.modules``, and the module-level mock above may already own
    those entries, so patch the objects sys.modules actually holds rather than
    whatever ``import telegram.request`` binds.
    """
    import tools.send_message_tool as smt
    from gateway.platforms import base as gateway_base

    recorded: list = []
    _built_bots.clear()

    class _RecordingRequest:
        def __init__(self, *args, **kwargs):
            recorded.append(kwargs)

    tg_mod = sys.modules["telegram"]
    tg_request_mod = sys.modules.get("telegram.request", tg_mod)

    monkeypatch.setattr(tg_mod, "Bot", _StubBot, raising=False)
    monkeypatch.setattr(tg_request_mod, "HTTPXRequest", _RecordingRequest, raising=False)
    monkeypatch.setattr(
        gateway_base, "resolve_proxy_url", lambda *a, **k: proxy, raising=False
    )

    asyncio.run(
        smt._send_telegram(
            "test-token",
            "12345",
            "hello",
            media_write_timeout=media_write_timeout,
        )
    )
    return recorded, list(_built_bots)


def test_standalone_direct_path_applies_configured_media_write_timeout(monkeypatch):
    recorded, bots = _capture_standalone_requests(monkeypatch, media_write_timeout=180.0)

    assert recorded, "the direct path should build a configured HTTPXRequest"
    assert all(kw.get("media_write_timeout") == 180.0 for kw in recorded)
    assert bots and "request" in bots[0].init_kwargs


def test_standalone_proxy_path_applies_configured_media_write_timeout(monkeypatch):
    recorded, bots = _capture_standalone_requests(
        monkeypatch, media_write_timeout=180.0, proxy="http://proxy.example:8080"
    )

    assert len(recorded) == 2, "proxy path builds request and get_updates_request"
    assert all(kw.get("media_write_timeout") == 180.0 for kw in recorded)
    assert all(kw.get("proxy") == "http://proxy.example:8080" for kw in recorded)
    assert bots and "get_updates_request" in bots[0].init_kwargs


def test_standalone_unset_media_write_timeout_keeps_ptb_defaults(monkeypatch):
    recorded, bots = _capture_standalone_requests(monkeypatch, media_write_timeout=None)

    # With nothing configured the direct path builds a bare Bot, so PTB's own
    # request defaults apply untouched.
    assert all("media_write_timeout" not in kw for kw in recorded)
    assert bots, "a Bot should still be constructed"
    assert "request" not in bots[0].init_kwargs


def test_standalone_non_numeric_media_write_timeout_is_ignored(monkeypatch):
    recorded, bots = _capture_standalone_requests(
        monkeypatch, media_write_timeout="not-a-number"
    )

    assert all("media_write_timeout" not in kw for kw in recorded)
    assert bots and "request" not in bots[0].init_kwargs
