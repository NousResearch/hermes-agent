"""Regression test for #31599 — Telegram general-pool CLOSE_WAIT fd leak.

Background
----------
PTB's ``telegram.request.HTTPXRequest`` builds the underlying
``httpx.AsyncClient`` with ``limits = httpx.Limits(max_connections=...)``
and *no* keepalive tuning, so httpx's default ``keepalive_expiry=5.0``
applies.  Behind an HTTP proxy (Cloudflare Warp etc.) a peer-initiated
FIN can sit in ``CLOSE_WAIT`` longer than that, leaking fds in the
general request pool (``_request[1]`` — the pool that routes
``bot.send_message`` / ``set_my_commands``), which
``_drain_polling_connections`` never resets.

The fix wires the shared ``gateway.platforms._http_client_limits``
``platform_httpx_limits()`` helper into *every* HTTPXRequest the adapter
builds — the fallback-transport branch, the proxy branch, and the plain
branch — so idle keepalive sockets drain aggressively.

Contract asserted here (mutation-survivable)
---------------------------------------------
Every ``HTTPXRequest`` constructed by ``TelegramAdapter.connect()`` must
receive ``httpx_kwargs["limits"]`` that is an ``httpx.Limits`` with a
``keepalive_expiry`` strictly below httpx's 5.0 default and a positive,
bounded ``max_keepalive_connections``.  Reverting the limits wiring (so
HTTPXRequest falls back to PTB's default 5.0s keepalive) fails this test.
"""

import asyncio
import sys
from unittest.mock import MagicMock, patch

import httpx
import pytest

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
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, telegram_mod)


_ensure_telegram_mock()

from plugins.platforms.telegram import adapter as tg_adapter  # noqa: E402
from plugins.platforms.telegram.adapter import TelegramAdapter  # noqa: E402


class _StopConnect(Exception):
    """Sentinel raised to abort connect() once requests are built."""


class _RecordingHTTPXRequest:
    """Stand-in for PTB's HTTPXRequest that records constructor kwargs."""

    instances: list = []

    def __init__(self, *args, httpx_kwargs=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        if httpx_kwargs is not None:
            self.kwargs["httpx_kwargs"] = httpx_kwargs
        _RecordingHTTPXRequest.instances.append(self)


def _make_adapter() -> TelegramAdapter:
    return TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))


def _drive_connect(monkeypatch, *, proxy_url):
    """Run connect() far enough to build the HTTPXRequests, then abort.

    Returns the list of recorded _RecordingHTTPXRequest instances.
    """
    _RecordingHTTPXRequest.instances = []

    # No DoH auto-discovery → exercise the proxy / plain branches, not fallback.
    async def _no_fallback():
        return []

    monkeypatch.setattr(tg_adapter, "discover_fallback_ips", _no_fallback)
    monkeypatch.setattr(
        tg_adapter, "resolve_proxy_url", lambda *a, **k: proxy_url
    )
    # Replace the real HTTPXRequest with our recorder.
    monkeypatch.setattr(tg_adapter, "HTTPXRequest", _RecordingHTTPXRequest)

    adapter = _make_adapter()
    # Skip the cross-process token lock.
    monkeypatch.setattr(adapter, "_acquire_platform_lock", lambda *a, **k: True)
    # Ensure the adapter reports no statically-configured fallback IPs.
    monkeypatch.setattr(adapter, "_fallback_ips", lambda: [])

    # builder.request(...).get_updates_request(...).build() must be harmless;
    # make build() raise our sentinel so connect() stops right after the
    # HTTPXRequests are constructed (before any real network/init).
    fake_built_app = MagicMock()
    fake_built_app.initialize = MagicMock(side_effect=_StopConnect)

    chainable = MagicMock()
    chainable.token.return_value = chainable
    chainable.base_url.return_value = chainable
    chainable.base_file_url.return_value = chainable
    chainable.local_mode.return_value = chainable
    chainable.request.return_value = chainable
    chainable.get_updates_request.return_value = chainable
    chainable.build.side_effect = _StopConnect

    builder_root = MagicMock()
    builder_root.builder.return_value = chainable
    monkeypatch.setattr(tg_adapter, "Application", builder_root)

    try:
        asyncio.run(adapter.connect())
    except _StopConnect:
        pass
    except Exception:
        # connect() wraps work in a try; if it swallows the sentinel and
        # continues to real init, the recorded instances are still valid.
        pass

    return list(_RecordingHTTPXRequest.instances)


def _assert_keepalive_tight(instances):
    assert instances, "connect() built no HTTPXRequest — test setup is wrong"

    # Skip keepalive assertion if PTB doesn't support httpx_kwargs (#61158)
    # Older PTB versions (< 22.6) don't accept httpx_kwargs parameter.
    # In that case, keepalive tuning is unavailable and we accept the fallback.
    import inspect
    from telegram.request import HTTPXRequest
    ptb_supports_httpx_kwargs = "httpx_kwargs" in inspect.signature(HTTPXRequest.__init__).parameters
    if not ptb_supports_httpx_kwargs:
        pytest.skip("PTB < 22.6: httpx_kwargs not supported, keepalive tuning unavailable")

    for inst in instances:
        limits = inst.kwargs.get("httpx_kwargs", {}).get("limits")
        assert isinstance(limits, httpx.Limits), (
            "HTTPXRequest must receive httpx_kwargs['limits'] = httpx.Limits "
            "wired from platform_httpx_limits() (#31599). Missing → PTB falls "
            "back to default keepalive_expiry=5.0 and leaks CLOSE_WAIT fds."
        )
        # The whole point: keepalive must be tighter than httpx's 5.0 default.
        assert limits.keepalive_expiry is not None
        assert limits.keepalive_expiry < 5.0, (
            "keepalive_expiry must be < httpx default 5.0 so idle/CLOSE_WAIT "
            "sockets drain promptly behind a proxy (#31599)."
        )
        assert limits.max_keepalive_connections is not None
        assert 1 <= limits.max_keepalive_connections <= 50
        # PTB's connection_pool_size (max_connections) must be preserved.
        assert limits.max_connections is not None and limits.max_connections > 0


def test_proxy_branch_general_pool_has_tight_keepalive(monkeypatch):
    """The proxy path the #31599 reporter hit must wire tuned limits."""
    instances = _drive_connect(monkeypatch, proxy_url="http://127.0.0.1:9/")
    # Both the general request pool and the get_updates pool are built here.
    assert len(instances) >= 2
    _assert_keepalive_tight(instances)
    # Sanity: the proxy was actually threaded through (we're on the proxy branch).
    assert any(inst.kwargs.get("proxy") == "http://127.0.0.1:9/" for inst in instances)


def test_plain_branch_general_pool_has_tight_keepalive(monkeypatch):
    """No proxy / no fallback IPs → plain branch must also wire tuned limits."""
    instances = _drive_connect(monkeypatch, proxy_url=None)
    assert len(instances) >= 2
    _assert_keepalive_tight(instances)


def test_limits_keepalive_below_ptb_default_is_the_contract():
    """Document the invariant independent of adapter wiring: the shared
    helper itself must tighten keepalive below httpx's 5.0 default."""
    from gateway.platforms._http_client_limits import platform_httpx_limits

    limits = platform_httpx_limits()
    assert isinstance(limits, httpx.Limits)
    assert limits.keepalive_expiry is not None and limits.keepalive_expiry < 5.0
