import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from plugins.platforms.discord import background_activity as bg
from plugins.platforms.discord.background_activity import (
    DiscordActivityPublisher,
    PublishGate,
    dashboard_channel_id_from_adapter,
    parse_channel_id,
    render_dashboard,
    render_presence,
)


def _item(key="one", elapsed=1, **overrides):
    item = {
        "key": key, "title": "Build safe thing", "profile": "coder", "model": "gpt-safe",
        "provider": "openai-codex", "worker": "kanban", "started_at": 100.0,
        "elapsed_seconds": elapsed, "elapsed": f"{elapsed}s", "state": "running",
    }
    item.update(overrides)
    return item


def test_presence_is_compact_and_aggregates_workers():
    text = render_presence([_item("one"), _item("two")])
    assert text.startswith("2 workers · Build safe thing")
    assert len(text) <= 96


def test_dashboard_has_one_idle_or_active_document():
    assert "Idle" in render_dashboard([], now=120)
    active = render_dashboard([_item()], now=120)
    assert "1 active worker" in active
    assert "Build safe thing" in active
    assert "coder" in active
    assert "gpt-safe" in active


def test_dashboard_lists_each_worker_separately_with_identity_and_state():
    dashboard = render_dashboard(
        [
            _item("one", title="First card", worker="kanban", profile="coder"),
            _item(
                "two", title="Second card", worker="delegate", profile="default",
                model="claude-x", provider="anthropic",
            ),
        ],
        now=120,
    )
    assert "2 active workers" in dashboard
    assert "• **First card**" in dashboard
    assert "• **Second card**" in dashboard
    # Every row carries its own identity line — no aggregate-only rendering.
    assert "worker=kanban · profile=coder · model=gpt-safe · provider=openai-codex · state=running" in dashboard
    assert "worker=delegate · profile=default · model=claude-x · provider=anthropic · state=running" in dashboard
    assert dashboard.count("state=running") == 2


def test_dashboard_omits_absent_model_and_provider_but_keeps_state():
    row = render_dashboard([_item(model="", provider="")], now=120)
    assert "model=" not in row
    assert "provider=" not in row
    assert "worker=kanban · profile=coder · state=running" in row


def test_dashboard_order_is_deterministic_and_independent_of_input_order():
    early = _item("early", title="Early card", started_at=10.0)
    late = _item("late", title="Late card", started_at=900.0)
    forward = render_dashboard([early, late], now=1000)
    reversed_input = render_dashboard([late, early], now=1000)
    assert forward == reversed_input
    assert forward.index("Early card") < forward.index("Late card")


def test_publish_gate_flushes_transitions_but_throttles_elapsed_ticks():
    gate = PublishGate(periodic_seconds=15, minimum_interval_seconds=1)
    active = [_item(elapsed=1)]
    assert gate.should_publish(active, now=100)
    gate.mark_published(active, now=100)
    assert not gate.should_publish([_item(elapsed=2)], now=101)
    assert gate.should_publish([_item(elapsed=16)], now=116)
    gate.mark_published([_item(elapsed=16)], now=116)
    assert not gate.should_publish([], now=116.1)
    assert gate.should_publish([], now=117)
    gate.mark_published([], now=117)
    assert not gate.should_publish([], now=118)


def test_publish_gate_retries_failed_idle_transition():
    gate = PublishGate(minimum_interval_seconds=0)
    active = [_item()]
    gate.mark_published(active, now=100)
    assert gate.should_publish([], now=101)
    # No mark_published: the failed terminal transition remains pending.
    assert gate.should_publish([], now=101.5)


# --- Configuration: opt-in dashboard channel ---------------------------------


class _FakeConfig:
    def __init__(self, extra=None):
        self.extra = dict(extra or {})


class _FakeAdapter:
    """Minimal adapter stand-in: only ``config.extra`` matters for resolution."""

    def __init__(self, extra=None, client=None):
        self.config = _FakeConfig(extra)
        self._client = client


def _fake_message(message_id=7):
    return SimpleNamespace(id=message_id, edit=AsyncMock(), pin=AsyncMock())


def _channel_with_no_history():
    async def _history(*_args, **_kwargs):
        return
        yield  # pragma: no cover - makes this an async generator

    channel = SimpleNamespace(id=4242, send=AsyncMock(return_value=_fake_message()), history=_history)
    channel.fetch_message = AsyncMock(side_effect=Exception("unknown message"))
    return channel


def test_parse_channel_id_accepts_ids_and_mentions_rejects_junk():
    assert parse_channel_id(4242) == 4242
    assert parse_channel_id("4242") == 4242
    assert parse_channel_id("  <#4242>  ") == 4242
    for malformed in (None, "", "   ", "usage", "general", "<#general>", 0, -1, True, False, "12.5"):
        assert parse_channel_id(malformed) is None


def test_configured_channel_id_reads_extra_and_ignores_malformed_value(monkeypatch):
    monkeypatch.delenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", raising=False)
    adapter = _FakeAdapter({"background_activity_channel_id": "4242"})
    assert dashboard_channel_id_from_adapter(adapter) == 4242
    malformed = _FakeAdapter({"background_activity_channel_id": "usage"})
    assert dashboard_channel_id_from_adapter(malformed) is None
    assert dashboard_channel_id_from_adapter(_FakeAdapter({})) is None


def test_configured_channel_id_falls_back_to_scoped_env(monkeypatch):
    monkeypatch.setenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", "77")
    assert dashboard_channel_id_from_adapter(_FakeAdapter({})) == 77


def test_apply_yaml_config_seeds_background_activity_channel(monkeypatch):
    monkeypatch.delenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", raising=False)
    from plugins.platforms.discord import adapter as discord_adapter

    seeded = discord_adapter._apply_yaml_config(
        {"platforms": {"discord": {"extra": {"background_activity_channel_id": "4242"}}}},
        {},
    )
    assert seeded is not None
    assert seeded["background_activity_channel_id"] == "4242"
    assert os.environ["DISCORD_BACKGROUND_ACTIVITY_CHANNEL"] == "4242"

    monkeypatch.delenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", raising=False)
    assert discord_adapter._apply_yaml_config({}, {}) is None
    monkeypatch.delenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", raising=False)


@pytest.mark.asyncio
async def test_presence_always_published_without_configured_channel():
    client = SimpleNamespace(change_presence=AsyncMock(), get_channel=Mock(), fetch_channel=AsyncMock())
    publisher = DiscordActivityPublisher(_FakeAdapter({}, client), channel_id=None)
    await publisher._publish([_item()])
    assert client.change_presence.await_count == 1
    activity = client.change_presence.await_args.kwargs["activity"]
    assert activity is not None
    assert "Build safe thing" in render_presence([_item()])
    # No channel configured: nothing is fetched, nothing is sent.
    client.get_channel.assert_not_called()
    client.fetch_channel.assert_not_called()


@pytest.mark.asyncio
async def test_configured_channel_publishes_single_dashboard_message(monkeypatch, tmp_path):
    monkeypatch.setattr(bg, "get_hermes_home", lambda: tmp_path)
    channel = _channel_with_no_history()
    client = SimpleNamespace(
        change_presence=AsyncMock(),
        get_channel=Mock(return_value=channel),
        fetch_channel=AsyncMock(),
        user=object(),
    )
    publisher = DiscordActivityPublisher(_FakeAdapter({"background_activity_channel_id": "4242"}, client))
    assert publisher.channel_id == 4242
    await publisher._publish([_item()])
    assert client.change_presence.await_count == 1
    assert channel.send.await_count == 1
    message = channel.send.return_value
    assert message.pin.await_count == 1
    # The placeholder is filled in with the first payload, all on the SAME message.
    assert message.edit.await_count == 1
    await publisher._publish([_item("two")])
    assert channel.send.await_count == 1
    assert message.edit.await_count == 2


@pytest.mark.asyncio
async def test_inaccessible_channel_keeps_presence_and_does_not_raise(caplog):
    client = SimpleNamespace(
        change_presence=AsyncMock(),
        get_channel=Mock(return_value=None),
        fetch_channel=AsyncMock(side_effect=Exception("403 Forbidden")),
        user=object(),
    )
    publisher = DiscordActivityPublisher(_FakeAdapter({"background_activity_channel_id": "4242"}, client))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(RuntimeError):
            await publisher._publish([_item()])
    # Presence was still updated before the dashboard attempt failed.
    assert client.change_presence.await_count == 1
    assert any("unavailable" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_publisher_loop_survives_inaccessible_channel(monkeypatch):
    """The polling loop swallows the dashboard failure and keeps presence flowing."""
    monkeypatch.setattr(bg, "list_all_active_work", lambda _home: [_item()])
    client = SimpleNamespace(
        change_presence=AsyncMock(),
        get_channel=Mock(return_value=None),
        fetch_channel=AsyncMock(side_effect=Exception("403 Forbidden")),
        user=object(),
    )
    publisher = DiscordActivityPublisher(
        _FakeAdapter({"background_activity_channel_id": "4242"}, client), poll_seconds=0.01,
    )
    publisher.start()
    await asyncio.sleep(0.05)
    await publisher.stop()
    assert client.change_presence.await_count >= 1
    assert publisher.task is None


@pytest.mark.asyncio
async def test_explicit_none_channel_ignores_configured_value(monkeypatch):
    monkeypatch.setenv("DISCORD_BACKGROUND_ACTIVITY_CHANNEL", "4242")
    client = SimpleNamespace(change_presence=AsyncMock(), get_channel=Mock(), fetch_channel=AsyncMock())
    publisher = DiscordActivityPublisher(_FakeAdapter({"background_activity_channel_id": "4242"}, client), channel_id=None)
    assert publisher.channel_id is None
    await publisher._publish([_item()])
    client.get_channel.assert_not_called()
    client.fetch_channel.assert_not_called()


@pytest.mark.asyncio
async def test_existing_dashboard_message_is_reused_not_duplicated(monkeypatch, tmp_path):
    monkeypatch.setattr(bg, "get_hermes_home", lambda: tmp_path)

    client = SimpleNamespace(change_presence=AsyncMock(), user=object())
    existing = SimpleNamespace(
        id=55, author=client.user, content="`Hermes background work`\n🟢 Idle",
        edit=AsyncMock(), pin=AsyncMock(),
    )

    async def _history_with_marker(*_args, **_kwargs):
        yield existing

    channel = SimpleNamespace(
        id=4242, send=AsyncMock(return_value=_fake_message(99)),
        history=_history_with_marker, fetch_message=AsyncMock(),
    )
    client.get_channel = Mock(return_value=channel)
    client.fetch_channel = AsyncMock()
    publisher = DiscordActivityPublisher(_FakeAdapter({"background_activity_channel_id": "4242"}, client))
    await publisher._publish([_item()])
    channel.send.assert_not_called()
    # The marker row found in history is adopted and edited in place.
    assert publisher._message is existing
    assert existing.edit.await_count == 1
    assert publisher._stored_message_id() == 55


def test_message_state_is_profile_safe_and_channel_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(bg, "get_hermes_home", lambda: tmp_path)
    publisher = DiscordActivityPublisher(_FakeAdapter({}, None), channel_id=4242)
    assert publisher._stored_message_id() is None
    publisher._store_message_id(99)
    assert publisher._stored_message_id() == 99
    # The atomic write leaves no temp file behind.
    assert not (tmp_path / "gateway" / "discord_background_activity.tmp").exists()
    # A stored id from another channel is never reused.
    other = DiscordActivityPublisher(_FakeAdapter({}, None), channel_id=777)
    assert other._stored_message_id() is None
