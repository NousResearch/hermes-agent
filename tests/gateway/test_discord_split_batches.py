"""Opt-in bounded delivery through real config loading and Discord adapter code."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import plugins.platforms.discord.adapter as discord_platform
from gateway.config import Platform, PlatformConfig, load_gateway_config
from plugins.platforms.discord.adapter import DiscordAdapter


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def make_adapter(limit=40):
    return DiscordAdapter(PlatformConfig(enabled=True, extra={"max_split_messages": limit}))


def report(words=6000):
    return " ".join(f"record-{i:05d}" for i in range(words))


def test_yaml_limit_reaches_adapter_and_does_not_leak(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("discord:\n  max_split_messages: 40\n", encoding="utf-8")
    config = load_gateway_config()
    adapter = DiscordAdapter(config.platforms[Platform.DISCORD])
    assert len(adapter._prepare_split_chunks(report())) > adapter.MAX_SPLIT_MESSAGES
    config_path.write_text("discord:\n  require_mention: true\n", encoding="utf-8")
    default = DiscordAdapter(load_gateway_config().platforms[Platform.DISCORD])
    assert len(default._prepare_split_chunks(report())) == default.MAX_SPLIT_MESSAGES
    assert len(adapter._prepare_split_chunks(report())) > default.MAX_SPLIT_MESSAGES


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 41, 1000000, 8.5, float("inf"), "40", [], {}])
def test_invalid_limit_retains_default_flood_guard(invalid):
    adapter = make_adapter(invalid)
    chunks = adapter._prepare_split_chunks(report())
    assert len(chunks) == adapter.MAX_SPLIT_MESSAGES
    assert "Response truncated" in chunks[-1]
    assert not chunks[0].startswith("**Part")


@pytest.mark.parametrize("limit", [1, 8, 20, 21, 40])
def test_configured_cap_includes_notice(limit):
    chunks = make_adapter(limit)._prepare_split_chunks(report(10000))
    assert len(chunks) == limit
    assert "Response truncated" in chunks[-1]
    assert all(len(chunk) <= DiscordAdapter.MAX_MESSAGE_LENGTH for chunk in chunks)


def test_short_replies_keep_existing_format():
    adapter = make_adapter()
    for content in ["hello", report(1000)]:
        assert adapter._prepare_split_chunks(content) == adapter.truncate_message(
            content, adapter.MAX_MESSAGE_LENGTH
        )


def test_labels_preserve_fences_order_and_partial_batch():
    adapter = make_adapter()
    content = "```text\n" + "\n".join(f"line-{i:05d}" for i in range(4500)) + "\n```"
    chunks = adapter._prepare_split_chunks(content)
    assert 20 < len(chunks) < 40
    assert chunks[0].startswith("**Part 1 of 2 — 1/20**\n")
    remaining = len(chunks) - 20
    assert chunks[20].startswith(f"**Part 2 of 2 — 1/{remaining}**\n")
    assert chunks[-1].startswith(f"**Part 2 of 2 — {remaining}/{remaining}**\n")
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in chunks)
    assert all(chunk.count("```") == 2 for chunk in chunks)
    actual = [line for chunk in chunks for line in chunk.splitlines() if line.startswith("line-")]
    assert actual == [f"line-{i:05d}" for i in range(4500)]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["send", "forum", "edit"])
@pytest.mark.parametrize("limit", [8, 40])
async def test_real_delivery_paths_cap_and_pause_between_batches(monkeypatch, path, limit):
    adapter = make_adapter(limit)
    delivered = []
    pauses = []

    async def fake_send(*, content, reference=None):
        delivered.append(content)
        return SimpleNamespace(id=9000 + len(delivered))

    async def pause(seconds):
        pauses.append((len(delivered), seconds))

    monkeypatch.setattr(discord_platform.asyncio, "sleep", pause)
    channel = SimpleNamespace(id=555, send=AsyncMock(side_effect=fake_send))
    adapter._client = SimpleNamespace(get_channel=lambda _: channel, fetch_channel=AsyncMock())
    if path == "send":
        result = await adapter.send("555", report(10000))
    elif path == "forum":
        async def create_thread(*, name, content):
            first = await fake_send(content=content)
            return SimpleNamespace(thread=channel, message=first)
        forum = SimpleNamespace(id=666, create_thread=AsyncMock(side_effect=create_thread))
        result = await adapter._send_to_forum(forum, report(10000))
    else:
        msg = SimpleNamespace(id=42, edit=AsyncMock(side_effect=fake_send))
        result = await adapter._edit_overflow_split(channel, msg, "42", report(10000))
    assert result.success
    assert len(delivered) == limit
    assert "Response truncated" in delivered[-1]
    assert all(len(chunk) <= adapter.MAX_MESSAGE_LENGTH for chunk in delivered)
    if limit > adapter.SPLIT_BATCH_SIZE:
        assert pauses == [(adapter.SPLIT_BATCH_SIZE, adapter.SPLIT_BATCH_DELAY_SECONDS)]
        assert delivered[20].startswith("**Part 2 of 2 — 1/20**\n")
    else:
        assert pauses == []
