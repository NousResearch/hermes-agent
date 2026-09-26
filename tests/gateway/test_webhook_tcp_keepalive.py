"""Webhook listener must survive kernels that reject SO_KEEPALIVE (#123327).

On one macOS host every accepted webhook connection died: the pinned
aiohttp's ``tcp_keepalive`` calls ``setsockopt(SO_KEEPALIVE)`` with no
``OSError`` guard (unlike ``tcp_nodelay`` in the same file), so a kernel
that answers EINVAL tears the connection down and the client sees EOF.
The in-repo mitigation lives in ``gateway.platforms.tcp_site``: an
idempotent guard that makes the per-accept keepalive tuning best-effort.
"""

import errno
import socket
from contextlib import contextmanager
from unittest.mock import patch

import aiohttp
import pytest

from gateway.config import PlatformConfig
from gateway.platforms.webhook import WebhookAdapter, _INSECURE_NO_AUTH


@contextmanager
def _reject_so_keepalive():
    """Simulate the affected macOS kernel: SO_KEEPALIVE always fails EINVAL."""
    orig = socket.socket.setsockopt

    def flaky(self, level, opt, value):
        if level == socket.SOL_SOCKET and opt == socket.SO_KEEPALIVE:
            raise OSError(errno.EINVAL, "Invalid argument")
        return orig(self, level, opt, value)

    socket.socket.setsockopt = flaky  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.setsockopt = orig


class _EinvalSocket:
    def setsockopt(self, level, opt, value):
        raise OSError(errno.EINVAL, "Invalid argument")


class _EinvalTransport:
    def get_extra_info(self, name, default=None):
        if name == "socket":
            return _EinvalSocket()
        return default


def test_keepalive_guard_tolerates_einval():
    """The shared guard makes both aiohttp keepalive entry points best-effort."""
    from gateway.platforms import tcp_site

    tcp_site.ensure_tcp_keepalive_guard()

    from aiohttp import tcp_helpers, web_protocol

    # Must not raise even though every setsockopt(SO_KEEPALIVE) fails EINVAL.
    tcp_helpers.tcp_keepalive(_EinvalTransport())  # type: ignore[arg-type]
    web_protocol.tcp_keepalive(_EinvalTransport())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_webhook_serves_post_when_keepalive_rejected():
    """Real repro: a full webhook listener + POST while the kernel rejects SO_KEEPALIVE.

    The route filters on events so the POST is answered 200 without spawning
    an agent run — the assertion is purely that the accepted connection
    survives the keepalive tuning. Unfixed, the client never gets a response.
    """
    config = PlatformConfig(
        enabled=True,
        extra={
            "host": "127.0.0.1",
            "port": 0,
            "routes": {"r1": {"secret": _INSECURE_NO_AUTH, "prompt": "x", "events": ["push"]}},
        },
    )
    adapter = WebhookAdapter(config)
    with _reject_so_keepalive():
        try:
            with patch.object(adapter, "_reload_dynamic_routes"):
                assert await adapter.connect() is True
            port = list(adapter._runner.addresses)[0][1]  # type: ignore[union-attr]
            timeout = aiohttp.ClientTimeout(total=8)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"http://127.0.0.1:{port}/webhooks/r1", data=b"{}"
                ) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["status"] == "ignored"
        finally:
            await adapter.disconnect()
