"""Pairing DM and home-channel prompt come from the i18n catalog and carry the skin's name.

Both notices are the first thing a person sees from a fresh gateway, so they must
follow ``display.language`` like every other messenger reply, and a custom skin
must not leak the stock product name into chat.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.i18n import SUPPORTED_LANGUAGES, t
from gateway.branding import agent_display_name
from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource

PAIRING_KEYS = (
    "gateway.pairing.greeting",
    "gateway.pairing.code",
    "gateway.pairing.approve_hint",
    "gateway.pairing.rate_limited",
)
HOME_KEY = "gateway.home_channel.notice"


def test_keys_exist_in_every_catalog():
    for lang in SUPPORTED_LANGUAGES:
        for key in (*PAIRING_KEYS, HOME_KEY):
            assert t(key, lang=lang) != key, f"{key} missing in {lang}"


def test_english_notice_text_is_unchanged_for_the_stock_skin():
    rendered = t(HOME_KEY, lang="en", platform="Discord", brand="Hermes", cmd="/sethome")
    assert rendered == (
        "📬 No home channel is set for Discord. A home channel is where Hermes delivers "
        "cron job results and cross-platform messages.\n\nType /sethome to make this chat "
        "your home channel, or ignore to skip."
    )


def test_portuguese_notice_is_localized():
    rendered = t(HOME_KEY, lang="pt", platform="Telegram", brand="Atlas", cmd="/sethome")
    assert "canal principal" in rendered
    assert "Atlas" in rendered
    assert "/sethome" in rendered
    assert "Hermes" not in rendered


def test_stock_skin_keeps_the_short_name(monkeypatch):
    from hermes_cli import config as cli_config

    monkeypatch.setattr(cli_config, "load_config_readonly", lambda: {"display": {}})
    assert agent_display_name() == "Hermes"


def test_custom_skin_name_is_used(tmp_path, monkeypatch):
    from hermes_cli import config as cli_config
    from hermes_cli import skin_engine

    (tmp_path / "atlas.yaml").write_text(
        "name: atlas\nbranding:\n  agent_name: Atlas\n", encoding="utf-8"
    )
    monkeypatch.setattr(skin_engine, "_skins_dir", lambda: tmp_path)
    monkeypatch.setattr(cli_config, "load_config_readonly", lambda: {"display": {"skin": "atlas"}})
    assert agent_display_name() == "Atlas"


def _bare_runner(code):
    runner = object.__new__(GatewayRunner)
    store = MagicMock()
    store._is_rate_limited.return_value = False
    store.generate_code.return_value = code
    store.profile = "default"
    runner._pairing_store_for = lambda source: store
    adapter = SimpleNamespace(send=AsyncMock())
    runner._adapter_for_source = lambda source: adapter
    return runner, adapter, store


@pytest.mark.asyncio
async def test_pairing_dm_is_rendered_from_the_catalog(monkeypatch):
    monkeypatch.setattr("agent.i18n.get_language", lambda: "pt")
    runner, adapter, _ = _bare_runner("XDQAAUY6")
    source = SessionSource(platform=Platform.TELEGRAM, chat_id="42", chat_type="dm", user_id="7")

    await runner._hm_offer_pairing_code(source)

    adapter.send.assert_awaited_once()
    chat_id, text = adapter.send.await_args.args
    assert chat_id == "42"
    assert "pareamento" in text
    assert "`XDQAAUY6`" in text
    assert "`hermes pairing approve telegram XDQAAUY6`" in text


@pytest.mark.asyncio
async def test_rate_limited_reply_is_rendered_from_the_catalog(monkeypatch):
    monkeypatch.setattr("agent.i18n.get_language", lambda: "en")
    runner, adapter, store = _bare_runner(None)
    source = SessionSource(platform=Platform.DISCORD, chat_id="9", chat_type="dm", user_id="1")

    await runner._hm_offer_pairing_code(source)

    chat_id, text = adapter.send.await_args.args
    assert text == t("gateway.pairing.rate_limited", lang="en")
    store._record_rate_limit.assert_called_once_with("discord", "1")
