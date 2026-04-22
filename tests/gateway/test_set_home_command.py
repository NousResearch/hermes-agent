from types import SimpleNamespace

import pytest
import yaml

from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource


class ForumChannel:
    pass


@pytest.mark.asyncio
async def test_sethome_uses_parent_channel_for_discord_thread(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    guild = SimpleNamespace(name="GSV")
    parent = SimpleNamespace(id=987654321, name="agent-ops", guild=guild)
    channel = SimpleNamespace(parent=parent, parent_id=parent.id, guild=guild)
    raw_message = SimpleNamespace(channel=channel)

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="1496043680090558464",
            chat_name="hermes-apl",
            chat_type="thread",
            thread_id="1496043680090558464",
            user_id="u1",
            user_name="tester",
        ),
        raw_message=raw_message,
        message_id="m1",
    )

    result = await runner._handle_set_home_command(event)

    assert "GSV / #agent-ops" in result
    assert "987654321" in result
    assert "stable Discord home delivery" in result
    assert (tmp_path / "config.yaml").exists()
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["DISCORD_HOME_CHANNEL"] == "987654321"
    assert gateway_run.os.environ["DISCORD_HOME_CHANNEL"] == "987654321"


@pytest.mark.asyncio
async def test_sethome_keeps_forum_thread_as_home_target(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    guild = SimpleNamespace(name="GSV")
    parent = ForumChannel()
    parent.id = 222
    parent.name = "ideas"
    parent.guild = guild
    parent.type = 15
    channel = SimpleNamespace(parent=parent, parent_id=parent.id, guild=guild)
    raw_message = SimpleNamespace(channel=channel)

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="333",
            chat_name="GSV / ideas / post-1",
            chat_type="thread",
            thread_id="333",
            user_id="u1",
            user_name="tester",
        ),
        raw_message=raw_message,
        message_id="m-forum",
    )

    result = await runner._handle_set_home_command(event)

    assert "GSV / ideas / post-1" in result
    assert "(ID: 333)" in result
    assert "stable Discord home delivery" not in result
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["DISCORD_HOME_CHANNEL"] == "333"


@pytest.mark.asyncio
async def test_sethome_keeps_thread_when_parent_object_is_missing(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    channel = SimpleNamespace(parent=None, parent_id=987654321, guild=SimpleNamespace(name="GSV"))
    raw_message = SimpleNamespace(channel=channel)

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="1496043680090558464",
            chat_name="GSV / #agent-ops / hermes-apl",
            chat_type="thread",
            thread_id="1496043680090558464",
            user_id="u1",
            user_name="tester",
        ),
        raw_message=raw_message,
        message_id="m-missing-parent",
    )

    result = await runner._handle_set_home_command(event)

    assert "(ID: 1496043680090558464)" in result
    assert "stable Discord home delivery" not in result
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["DISCORD_HOME_CHANNEL"] == "1496043680090558464"


@pytest.mark.asyncio
async def test_sethome_keeps_current_chat_for_non_thread_discord(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("DISCORD_HOME_CHANNEL", raising=False)

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="444",
            chat_name="GSV / #agent-ops",
            chat_type="group",
            user_id="u1",
            user_name="tester",
        ),
        message_id="m3",
    )

    result = await runner._handle_set_home_command(event)

    assert "GSV / #agent-ops" in result
    assert "(ID: 444)" in result
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["DISCORD_HOME_CHANNEL"] == "444"


@pytest.mark.asyncio
async def test_sethome_keeps_current_chat_for_non_thread_sources(monkeypatch, tmp_path):
    import gateway.run as gateway_run

    runner = object.__new__(gateway_run.GatewayRunner)
    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.delenv("TELEGRAM_HOME_CHANNEL", raising=False)

    event = MessageEvent(
        text="/sethome",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="12345",
            chat_name="Primary DM",
            chat_type="dm",
            user_id="u1",
            user_name="tester",
        ),
        message_id="m2",
    )

    result = await runner._handle_set_home_command(event)

    assert "Primary DM" in result
    assert "12345" in result
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert config["TELEGRAM_HOME_CHANNEL"] == "12345"
    assert gateway_run.os.environ["TELEGRAM_HOME_CHANNEL"] == "12345"
