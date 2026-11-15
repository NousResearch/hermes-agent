"""A setup notice is the operator's business: ``setup_notices`` decides who receives it.

The measured defect (#137027's sibling): on a served instance a member who wrote into a WhatsApp DM
received the runtime's own "no home channel" setup notice, pasted it back, and the model spent two
more calls explaining it. The audience of both setup notices is declared on the platform block —
``author`` (today's behaviour), ``operator`` (the ``operator_ids`` a profile names) or ``off``.
"""

from unittest.mock import AsyncMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource, SessionStore

OPERATOR = "40746215021@s.whatsapp.net"
MEMBER = "40770273376@s.whatsapp.net"


def _runner(tmp_path, monkeypatch, extra=None):
    monkeypatch.setattr("gateway.run._hermes_home", tmp_path)
    monkeypatch.delenv("WHATSAPP_HOME_CHANNEL", raising=False)
    (tmp_path / "config.yaml").write_text("model: test\n", encoding="utf-8")
    runner = GatewayRunner.__new__(GatewayRunner)
    config = GatewayConfig()
    if extra is not None:
        config.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True, extra=dict(extra))
    runner.config = config
    runner.session_store = SessionStore(sessions_dir=tmp_path / "sessions", config=config)
    runner._deliver_platform_notice = AsyncMock()
    return runner


def _source(user_id):
    return SessionSource(platform=Platform.WHATSAPP, chat_id=user_id, user_id=user_id, chat_type="dm")


async def _first_contact(runner, user_id):
    notes: list[str] = []
    await runner._hmwa_first_contact_notes(_source(user_id), [], notes)
    return notes


@pytest.mark.asyncio
async def test_a_member_gets_neither_setup_notice_when_the_profile_names_its_operator(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, extra={"setup_notices": "operator", "operator_ids": [OPERATOR]})

    notes = await _first_contact(runner, MEMBER)

    assert notes == []
    runner._deliver_platform_notice.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_declared_operator_still_receives_the_no_home_channel_prompt(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, extra={"setup_notices": "operator", "operator_ids": [OPERATOR]})

    await _first_contact(runner, OPERATOR)

    runner._deliver_platform_notice.assert_awaited_once()
    text = runner._deliver_platform_notice.await_args.args[1]
    assert "/sethome" in text and "No home channel" in text


@pytest.mark.asyncio
async def test_a_profile_declaring_no_operator_keeps_todays_behaviour(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch)

    notes = await _first_contact(runner, MEMBER)

    assert len(notes) == 1
    runner._deliver_platform_notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_operator_mode_with_no_ids_declared_is_not_a_mute(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, extra={"setup_notices": "operator", "operator_ids": []})

    await _first_contact(runner, MEMBER)

    runner._deliver_platform_notice.assert_awaited_once()


@pytest.mark.asyncio
async def test_off_sends_no_setup_notice_to_anyone(tmp_path, monkeypatch):
    runner = _runner(tmp_path, monkeypatch, extra={"setup_notices": "off", "operator_ids": [OPERATOR]})

    notes = await _first_contact(runner, OPERATOR)

    assert notes == []
    runner._deliver_platform_notice.assert_not_awaited()


def test_the_audience_reads_author_by_default_and_an_unknown_word_degrades_to_it():
    config = GatewayConfig()
    config.platforms[Platform.WHATSAPP] = PlatformConfig(enabled=True, extra={"setup_notices": "members"})

    assert config.get_setup_notices(Platform.WHATSAPP) == "author"
    assert config.get_setup_notices(Platform.TELEGRAM) == "author"


def test_the_operator_list_accepts_the_list_and_the_separated_string_alike():
    config = GatewayConfig()
    config.platforms[Platform.WHATSAPP] = PlatformConfig(
        enabled=True, extra={"operator_ids": [OPERATOR, " ", ""]},
    )
    config.platforms[Platform.TELEGRAM] = PlatformConfig(
        enabled=True, extra={"operator_ids": "123, 456 789"},
    )

    assert config.get_operator_ids(Platform.WHATSAPP) == (OPERATOR,)
    assert config.get_operator_ids(Platform.TELEGRAM) == ("123", "456", "789")
    assert config.get_operator_ids(Platform.DISCORD) == ()
