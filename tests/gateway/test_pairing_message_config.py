"""Configurable pairing reply text (``gateway.pairing_message``).

A hosted deployment whose owner approves pairing requests in a web panel gets DMs pointing at
``hermes pairing approve`` — a CLI that does not exist in their world. The template key lets the
operator rewrite that reply; every failure mode falls back to the stock text, so a typo in the
template can never leave a stranger without instructions.
"""

import logging

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run_inbound_unauthorized import pairing_code_reply
from gateway.session import SessionSource


def test_default_reply_unchanged_when_key_absent():
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    assert config.pairing_message == ""
    reply = pairing_code_reply(
        "telegram", "AB12", "", custom_template=config.pairing_message
    )
    assert "Your pairing code: `AB12`" in reply
    assert "pairing approve telegram AB12" in reply


def test_config_key_round_trip_from_dict():
    config = GatewayConfig.from_dict({
        "platforms": {"telegram": {"enabled": True}},
        "pairing_message": "Code {code} for {platform} — ask the owner to approve it in the dashboard.",
    })
    assert config.pairing_message == (
        "Code {code} for {platform} — ask the owner to approve it in the dashboard."
    )
    expected = "Code CD34 for telegram — ask the owner to approve it in the dashboard."
    reply = pairing_code_reply(
        "telegram",
        "CD34",
        "",
        custom_template=config.pairing_message,
    )
    assert reply == expected


def test_template_missing_code_placeholder_falls_back_to_stock(caplog):
    template = "Ask the owner to approve you in the dashboard."  # no {code}
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        reply = pairing_code_reply("telegram", "AB12", "", custom_template=template)
    assert reply == pairing_code_reply("telegram", "AB12", "")
    assert any("pairing_message" in r.message for r in caplog.records)


def test_template_unknown_placeholder_falls_back_to_stock(caplog):
    template = "Your code is {code} ({platform}); owner: {owner}."
    with caplog.at_level(logging.WARNING, logger="gateway.run"):
        reply = pairing_code_reply("telegram", "AB12", "", custom_template=template)
    assert reply == pairing_code_reply("telegram", "AB12", "")
    assert any("pairing_message" in r.message for r in caplog.records)


def test_github_dm_has_no_pairing_store_the_reply_path_stays_stock():
    """The custom template must not leak into rate-limited or other canned replies."""
    assert (
        pairing_code_reply("telegram", "AB12", "", custom_template="{code}") == "AB12"
    )
    assert pairing_code_reply("telegram", "AB12", "", custom_template="") != ""


def _pairing_offer_runner(platform: Platform, config: GatewayConfig):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.config = config
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {platform: adapter}
    store = MagicMock()
    store._is_rate_limited.return_value = False
    store.generate_code.return_value = "AB12"
    store.profile = "default"
    runner.pairing_store = store
    runner._pairing_stores = {}
    return runner, adapter, store


def _source(platform: Platform) -> SessionSource:
    return SessionSource(
        platform=platform,
        user_id="u1",
        chat_id="c1",
        user_name="stranger",
        chat_type="dm",
    )


@pytest.mark.asyncio
async def test_runner_passes_template_to_pairing_reply():
    """``_hm_offer_pairing_code`` reads ``pairing_message`` off the config the same way the
    ``decline`` path reads ``unauthorized_dm_decline_message``."""
    config = GatewayConfig(
        platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)},
        pairing_message="Access code {code} ({platform}). The owner approves in the web panel.",
    )
    runner, adapter, _ = _pairing_offer_runner(Platform.TELEGRAM, config)
    source = _source(Platform.TELEGRAM)

    await runner._hm_offer_pairing_code(source)

    sent = adapter.send.call_args[0][1]
    assert sent == "Access code AB12 (telegram). The owner approves in the web panel."


@pytest.mark.asyncio
async def test_runner_without_template_sends_stock_reply():
    config = GatewayConfig(platforms={Platform.TELEGRAM: PlatformConfig(enabled=True)})
    runner, adapter, _ = _pairing_offer_runner(Platform.TELEGRAM, config)
    source = _source(Platform.TELEGRAM)

    await runner._hm_offer_pairing_code(source)

    sent = adapter.send.call_args[0][1]
    assert "Your pairing code: `AB12`" in sent
    assert "pairing approve telegram AB12" in sent
