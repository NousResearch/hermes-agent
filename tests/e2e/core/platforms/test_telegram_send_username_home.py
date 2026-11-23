"""``hermes send --to telegram`` with the home channel set to a private user's ``@username``.

The Bot API resolves ``@username`` only for public channels and supergroups, so every send to such a
home answers a bare "Chat not found" while ``telegram:<numeric id>`` works (#95121). The REAL
``hermes send`` CLI against the Bot API stand-in, which answers an ``@`` chat id the way Telegram
does: the failure must name the cause and the numeric-id fix; a numeric home still delivers.
"""

from __future__ import annotations

import pytest

from tests.e2e.core.platforms._drv_telegram import TelegramDriver
from tests.e2e.core.platforms._helpers import GatewayUnderTest
from tests.fakes.fake_llm_provider import FakeLLMServer

pytestmark = [pytest.mark.platforms("posix")]

_CHAT_NOT_FOUND = {"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}


@pytest.mark.parametrize("home, delivered", [("@shelmecha", False), ("111", True)])
def test_bare_telegram_send_to_a_username_home_names_the_numeric_id_fix(tmp_path, home, delivered) -> None:
    drv = TelegramDriver()
    drv.start()
    llm = FakeLLMServer()
    llm.start()
    for method in ("sendMessage", "sendRichMessage"):
        drv.standin.fail(method, _CHAT_NOT_FOUND, status=400, times=10,
                         match=lambda p: str(p.get("chat_id", "")).startswith("@"))
    gw = GatewayUnderTest(tmp_path / "rig", llm_base_url=llm.base_url, config=drv.gateway_config(),
                          env={**drv.gateway_env(), "TELEGRAM_HOME_CHANNEL": home}, ready=drv.connected)
    try:
        result = gw.run_cli("send", "--to", "telegram", "probe message")
    finally:
        llm.stop()
        drv.stop()
    out = result.stdout + result.stderr
    assert (result.returncode == 0) is delivered, out
    if not delivered:
        assert "Chat not found" in out and "numeric chat id" in out and "/sethome" in out, out
