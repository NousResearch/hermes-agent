"""A Telegram error raised through a proxy whose URL holds no secret keeps its full text (and the
polling-error traceback) at every site that redacts for a proxy: only a URL carrying something the
log rendering leaves out (userinfo, path, query) reduces the text to ``<Type> via <host:port>``.

Corporate proxies without credentials are where the real reason ("407 Proxy Authentication
Required") matters most; a regression printed ``Telegram startup failed: None`` at all 8 sites."""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_agent.gateway.config import PlatformConfig

CLEAN_PROXY = "http://proxy.example.com:3128"
DETAIL = "proxy refused CONNECT: 407 Proxy Authentication Required"


def _adapter(monkeypatch, proxy):
    from plugins.platforms.telegram import adapter as tg

    monkeypatch.setenv("TELEGRAM_PROXY", proxy)
    monkeypatch.setenv("HERMES_TELEGRAM_DISABLE_FALLBACK_IPS", "true")
    adapter = tg.TelegramAdapter(PlatformConfig(enabled=True, token="123456:TOKEN"))
    adapter._ptb_proxy_url = proxy  # what _build_ptb_requests() binds at connect
    return tg, adapter


async def _until_fatal(adapter):
    for _ in range(2000):
        if adapter.has_fatal_error:
            return
        await asyncio.sleep(0)
    raise AssertionError("recovery ladder never went fatal")


def _ladder_patches(adapter):
    return (patch.object(adapter, "_stop_updater_or_go_fatal", AsyncMock(return_value=True)),
            patch.object(adapter, "_drain_polling_connections", AsyncMock()),
            patch.object(adapter, "_start_polling_once", AsyncMock(side_effect=ConnectionError(DETAIL))),
            patch.object(adapter, "_handoff_polling_fatal_error", AsyncMock()))


async def _run_ladders(tg, adapter):
    real_sleep = asyncio.sleep
    adapter._app = MagicMock()
    with patch.object(tg.asyncio, "sleep", lambda *_a, **_k: real_sleep(0)):
        p1, p2, p3, p4 = _ladder_patches(adapter)
        with p1, p2, p3, p4:
            # network-retry, reconnect-failed, network-final
            await adapter._handle_polling_network_error(ConnectionError(DETAIL))
            await _until_fatal(adapter)
            adapter._fatal_error_code = adapter._fatal_error_message = None
            # conflict-retry, conflict-final
            await adapter._handle_polling_conflict(ConnectionError(DETAIL))
            await _until_fatal(adapter)


async def _run_polling_callback(adapter):
    """The polling error callback: network (warning) and unknown (error + traceback) branches."""
    with patch.object(adapter, "_delete_webhook_best_effort", AsyncMock()), \
            patch.object(adapter, "_start_polling_resilient", AsyncMock(return_value=True)), \
            patch.object(adapter, "_handle_polling_network_error", AsyncMock()):
        await adapter._start_polling_mode(is_reconnect=True)
        callback = adapter._polling_error_callback_ref
        callback(ConnectionError(DETAIL))
        await asyncio.sleep(0)
        adapter._polling_error_task = None
        try:
            raise RuntimeError(DETAIL)
        except RuntimeError as exc:
            callback(exc)


@pytest.mark.parametrize("proxy", [CLEAN_PROXY])
def test_secret_free_proxy_keeps_error_text_at_every_site(monkeypatch, caplog, proxy):
    tg, adapter = _adapter(monkeypatch, proxy)

    def _raise(**_kw):
        raise ConnectionError(DETAIL)

    with caplog.at_level(logging.DEBUG), patch.object(tg, "HTTPXRequest", _raise), \
            patch.object(adapter, "_acquire_platform_lock", return_value=True), \
            patch.object(adapter, "_write_runtime_status_safe"):
        assert asyncio.run(adapter.connect()) is False
        fatal = adapter.fatal_error_message
        _, adapter = _adapter(monkeypatch, proxy)
        asyncio.run(_run_ladders(tg, adapter))
        asyncio.run(_run_polling_callback(adapter))

    assert fatal == f"Telegram startup failed: {DETAIL}"
    sites = {
        "connect": "Failed to connect to Telegram:",
        "network-retry": "reconnecting in",
        "reconnect-failed": "Telegram polling reconnect failed:",
        "network-final": "Last error:",
        "conflict-retry": "Scheduling next attempt",
        "conflict-final": "Original error:",
        "callback-network": "Telegram network error, scheduling reconnect:",
        "callback-error": "Telegram polling error:",
    }
    records = [r for r in caplog.records if r.name.startswith("plugins.platforms.telegram")]
    for site, marker in sites.items():
        hits = [r for r in records if marker in r.getMessage()]
        assert hits, f"{site}: no log line"
        assert all(DETAIL in r.getMessage() for r in hits), (site, [r.getMessage() for r in hits])
    polling_error = next(r for r in records if sites["callback-error"] in r.getMessage())
    assert polling_error.exc_info, "the polling-error traceback was dropped for a secret-free proxy"


def test_secret_bearing_proxy_still_drops_text_and_traceback(monkeypatch, caplog):
    _, adapter = _adapter(monkeypatch, "http://bob:Sekr3tPW@proxy.example.com:3128")
    with caplog.at_level(logging.DEBUG):
        asyncio.run(_run_polling_callback(adapter))
    hit = next(r for r in caplog.records if "Telegram polling error:" in r.getMessage())
    assert hit.getMessage().endswith("RuntimeError via http://proxy.example.com:3128")
    assert not hit.exc_info
