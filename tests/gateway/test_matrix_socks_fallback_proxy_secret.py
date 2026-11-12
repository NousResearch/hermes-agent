"""With ``aiohttp_socks`` missing, ``_create_matrix_session`` drops the SOCKS ``MATRIX_PROXY`` and returns a
``trust_env`` session that dials HTTPS_PROXY instead: the whoami failure must be redacted against that
environment proxy, traceback included. Real aiohttp request, real adapter catch; mautrix's ``whoami`` is
a one-method shim issuing the request through the session the adapter built."""

import asyncio
import logging
import sys
from types import SimpleNamespace

import pytest

from hermes_agent.gateway.config import PlatformConfig

PASSWORD = "MatrixQzxPassword"


def test_whoami_failure_through_the_env_proxy_after_socks_fallback_hides_its_secret(monkeypatch, caplog, tmp_path):
    pytest.importorskip("aiohttp")
    from plugins.platforms.matrix import adapter as matrix_adapter

    for name in list(__import__("os").environ):
        if name.lower().endswith("_proxy"):
            monkeypatch.delenv(name)
    monkeypatch.setitem(sys.modules, "aiohttp_socks", None)
    monkeypatch.setattr(matrix_adapter, "gateway_trust_env", lambda: True)
    monkeypatch.setattr("plugins.platforms.matrix.session_proxy.gateway_trust_env", lambda: True)
    monkeypatch.setenv("MATRIX_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("HTTPS_PROXY", f"http://bob:{PASSWORD}#x@127.0.0.1:1")
    adapter = matrix_adapter.MatrixAdapter(PlatformConfig(
        enabled=True, token="dummy", extra={"homeserver": "https://127.0.0.1:1", "user_id": "@probe:localhost"}))

    async def run():
        session = matrix_adapter._create_matrix_session(adapter._proxy_url)

        class Client:
            async def whoami(self):
                async with session.get("https://127.0.0.1:1/_matrix/client/v3/account/whoami") as response:
                    return await response.json()

        async def _abort(api):
            return False
        monkeypatch.setattr(adapter, "_abort_connect", _abort)
        try:
            return session.trust_env, await adapter._connect_authenticate(Client(), SimpleNamespace(session=session))
        finally:
            await session.close()

    with caplog.at_level(logging.DEBUG):
        trust_env, authenticated = asyncio.run(run())
    assert trust_env is True and authenticated is False
    logs = "\n".join(logging.Formatter().format(r) for r in caplog.records)
    assert "whoami failed" in logs
    assert "Qzx" not in logs, logs[-3000:]
