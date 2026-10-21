"""Telegram ``/topic`` mode: opt-in root-DM auto transfer (``dm_topic_auto_transfer``, #139136).

Driven through the real ``TelegramAdapter.handle_message`` ingress (topic recovery, session keying,
background processing, final send) into the real ``GatewayRunner._handle_message``; only the Bot API
(``adapter._bot``) and the model call (``runner._run_agent``) are faked.

* default (key unset): a root-DM message keeps the v0.22 lobby recovery — it is answered in the
  user's most recently bound topic (#136567);
* opted in: the message opens a fresh topic via ``createForumTopic`` and the reply lands there;
* opted in but ``createForumTopic`` fails: it falls back to lobby recovery.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.conftest import _ensure_telegram_mock
from tests.waiting import async_wait_until

_ensure_telegram_mock()

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from hermes_state import SessionDB
from plugins.platforms.telegram.adapter import TelegramAdapter
from tests.gateway.test_telegram_topic_mode import (
    _make_runner,
    _make_source,
    _seed_two_topic_bindings,
)

NEW_TOPIC_ID = 777


def _fake_bot(*, create_fails: bool = False) -> MagicMock:
    bot = MagicMock()
    if create_fails:
        bot.create_forum_topic = AsyncMock(side_effect=RuntimeError("Bad Request: chat is not a forum"))
    else:
        bot.create_forum_topic = AsyncMock(return_value=SimpleNamespace(message_thread_id=NEW_TOPIC_ID))
    bot.send_message = AsyncMock(return_value=SimpleNamespace(message_id=9001))
    bot.send_chat_action = AsyncMock(return_value=True)
    return bot


def _rig(tmp_path, monkeypatch, *, extra: dict, create_fails: bool = False):
    import gateway.run as gateway_run

    db = SessionDB(db_path=tmp_path / "state.db")
    _seed_two_topic_bindings(db)  # topic mode on; topic "222" is the most recently bound
    # The stub session store names every topic-lane session "sess-topic"; the binding FK needs its row.
    db.create_session(session_id="sess-topic", source="telegram", user_id="208214988")
    runner = _make_runner(session_db=db)
    runner.config.platforms[Platform.TELEGRAM].extra.update(extra)
    runs: list[dict] = []

    async def fake_run_agent(*args, **kwargs):
        runs.append(kwargs)
        return {"success": True, "final_response": "agent reply", "session_id": kwargs.get("session_id"),
                "messages": []}

    runner._run_agent = AsyncMock(side_effect=fake_run_agent)
    monkeypatch.setattr(gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "***"})

    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra=dict(extra)))
    adapter._bot = _fake_bot(create_fails=create_fails)
    runner.adapters[Platform.TELEGRAM] = adapter
    # The production wiring: every callback GatewayRunner installs on an adapter (topic recovery and,
    # where it exists, the topic-transfer hook), with the message handler pointed at the real runner.
    runner._busy_text_mode = "interrupt"
    runner._wire_adapter_handlers(
        adapter, message_handler=runner._handle_message,
        authorization_check=lambda *_a, **_k: True, platform_event_handler=AsyncMock())
    return runner, adapter, db, runs


async def _deliver_root_dm(adapter: TelegramAdapter, text: str = "plan the migration") -> MessageEvent:
    """Plain text takes the adapter's real text-batching ingress (``_enqueue_text_event``, whose batch
    key already runs topic recovery) before ``handle_message``; commands dispatch directly."""
    event = MessageEvent(text=text, source=_make_source(thread_id=None), message_id="4242")
    if event.get_command():
        await adapter.handle_message(event)
    else:
        adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.0
        adapter._enqueue_text_event(event)
        await async_wait_until(lambda: not adapter._pending_text_batches and adapter._session_tasks,
                               what="text batch flushed into a session task")
    await async_wait_until(
        lambda: adapter._bot.send_message.await_count and all(t.done() for t in adapter._session_tasks.values()),
        what="session tasks done and a reply sent")
    await asyncio.gather(*adapter._session_tasks.values(), return_exceptions=True)
    return event


def _reply_calls(adapter: TelegramAdapter) -> list:
    """The agent's final reply sends (not the one-off "no home channel" onboarding notice)."""
    return [c for c in adapter._bot.send_message.await_args_list if "agent reply" in str(c.kwargs.get("text"))]


@pytest.mark.asyncio
async def test_default_root_dm_keeps_lobby_recovery_to_last_bound_topic(tmp_path, monkeypatch):
    """Key unset: no topic is created; the turn and its reply go to the last bound topic (#136567)."""
    _runner, adapter, db, runs = _rig(tmp_path, monkeypatch, extra={})

    event = await _deliver_root_dm(adapter)

    adapter._bot.create_forum_topic.assert_not_awaited()
    assert event.source.thread_id == "222"
    assert len(runs) == 1
    (reply,) = _reply_calls(adapter)
    assert reply.kwargs.get("message_thread_id") == 222
    db.close()


@pytest.mark.asyncio
async def test_opt_in_root_dm_opens_new_topic_and_reply_lands_there(tmp_path, monkeypatch):
    _runner, adapter, db, runs = _rig(tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": True})

    event = await _deliver_root_dm(adapter)

    adapter._bot.create_forum_topic.assert_awaited_once()
    assert adapter._bot.create_forum_topic.await_args.kwargs["chat_id"] == 208214988
    assert event.source.thread_id == str(NEW_TOPIC_ID)
    assert len(runs) == 1, "the root-DM message must run as a turn, not get the lobby reminder"
    (reply,) = _reply_calls(adapter)
    assert reply.kwargs.get("message_thread_id") == NEW_TOPIC_ID
    # The root-DM message lives outside the new topic: it can't be the reply anchor there.
    assert reply.kwargs.get("reply_to_message_id") is None
    # The new topic is a bound session lane, so the next message inside it continues this session.
    binding = db.get_telegram_topic_binding(chat_id="208214988", thread_id=str(NEW_TOPIC_ID))
    assert binding is not None and binding["user_id"] == "208214988"
    db.close()


@pytest.mark.asyncio
async def test_opt_in_topic_create_failure_falls_back_to_lobby_recovery(tmp_path, monkeypatch):
    _runner, adapter, db, runs = _rig(
        tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": True}, create_fails=True)

    event = await _deliver_root_dm(adapter)

    adapter._bot.create_forum_topic.assert_awaited_once()
    assert event.source.thread_id == "222"
    assert len(runs) == 1
    (reply,) = _reply_calls(adapter)
    assert reply.kwargs.get("message_thread_id") == 222
    db.close()


@pytest.mark.asyncio
async def test_opt_in_root_dm_command_stays_in_the_lobby(tmp_path, monkeypatch):
    """``/topic`` (and every other command) typed in the root DM is a lobby command: no topic."""
    _runner, adapter, db, runs = _rig(tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": True})

    await _deliver_root_dm(adapter, text="/topic help")

    adapter._bot.create_forum_topic.assert_not_awaited()
    assert runs == []
    db.close()


def test_top_level_telegram_yaml_key_reaches_extra(monkeypatch):
    """``telegram: dm_topic_auto_transfer: true`` (top-level YAML) lands in PlatformConfig.extra,
    like its neighbour ``disable_topic_auto_rename``."""
    from plugins.platforms.telegram.adapter import _apply_yaml_config

    extras = _apply_yaml_config({}, {"dm_topic_auto_transfer": True}) or {}
    assert extras.get("dm_topic_auto_transfer") is True


@pytest.mark.asyncio
@pytest.mark.parametrize("launch,secondary", [(False, True), (True, False), (False, False), (True, True)])
async def test_receiving_secondary_bot_controls_opt_in(tmp_path, monkeypatch, launch, secondary):
    """Multiplexed: the bot that RECEIVED the root DM owns ``dm_topic_auto_transfer``, not the launch bot."""
    runner, primary, db, _runs = _rig(tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": launch})
    runner.config.multiplex_profiles = True
    runner._primary_profile_name = "default"
    second = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={"dm_topic_auto_transfer": secondary}))
    second._bot = _fake_bot()
    second.gateway_runner = runner
    second.set_owner_profile("probe-secondary")
    runner._profile_adapters = {"probe-secondary": {Platform.TELEGRAM: second}}
    db.enable_telegram_topic_mode(chat_id="208214988", user_id="208214988", profile_name="probe-secondary")
    db.create_session(session_id="secondary-old", source="telegram", user_id="208214988",
                      profile_name="probe-secondary")
    db.bind_telegram_topic(chat_id="208214988", thread_id="333", user_id="208214988",
                           profile_name="probe-secondary", session_key="secondary-old-route",
                           session_id="secondary-old")
    done = asyncio.Event()

    async def capture(_event):
        done.set()

    runner._wire_adapter_handlers(second, message_handler=capture, authorization_check=lambda *_a, **_k: True,
                                  platform_event_handler=AsyncMock())
    source = second.build_source(chat_id="208214988", user_id="208214988", chat_type="dm", message_id="4242")
    event = MessageEvent(text="root prompt", source=source, message_id="4242")
    second._text_batch_delay_seconds = second._text_batch_split_delay_seconds = 0
    second._enqueue_text_event(event)
    await asyncio.wait_for(done.wait(), 10)
    await async_wait_until(lambda: not second._session_tasks and not second._pending_text_batch_tasks,
                           what="secondary batch settles")
    db.close()
    assert second._bot.create_forum_topic.await_count == int(secondary)
    assert event.source.thread_id == (str(NEW_TOPIC_ID) if secondary else "333")
    primary._bot.create_forum_topic.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_topic_reply_anchor_survives_native_codec(tmp_path, monkeypatch):
    """A durably admitted (snapshot -> restore) transferred event still sends without quoting the root DM."""
    from gateway.session_envelope import restore_native, snapshot_native
    from gateway.session_ingress import deliver_response
    from plugins.platforms.telegram.telegram_topic_transfer import apply_root_dm_topic_transfer

    runner, adapter, db, _runs = _rig(tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": True})
    runner.config.multiplex_profiles = False
    runner._is_user_authorized = lambda _source, **_kwargs: True
    event = MessageEvent(text="root prompt", source=_make_source(), message_id="4242")
    await apply_root_dm_topic_transfer(adapter, event)
    assert event.source.thread_id == str(NEW_TOPIC_ID)
    payload = snapshot_native(runner, event)
    restored = restore_native(payload, runner)
    assert restored.message_id == "4242"  # inbound identity is kept
    await deliver_response(adapter, restored, payload["native_text_v1"]["route"], "restored reply")
    send = adapter._bot.send_message.await_args.kwargs
    db.close()
    assert send.get("message_thread_id") == NEW_TOPIC_ID
    assert send.get("reply_to_message_id") is None
    # The default (None) override stays out of the envelope, so older retries keep their fingerprint.
    plain = MessageEvent(text="x", source=_make_source(), message_id="4243")
    assert "reply_anchor_override" not in snapshot_native(runner, plain)["native_text_v1"]["event"]


@pytest.mark.asyncio
async def test_second_root_dm_does_not_cancel_an_in_flight_topic_transfer(tmp_path, monkeypatch):
    """A second root-DM chunk arriving while ``createForumTopic`` for the first is still in flight must
    not cancel the committed flush: the first message keeps its topic (no hold + replay into a second
    topic) and every created topic receives its message."""
    _runner, adapter, db, runs = _rig(tmp_path, monkeypatch, extra={"dm_topic_auto_transfer": True})
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.0
    release = asyncio.Event()
    ids = iter(range(NEW_TOPIC_ID, NEW_TOPIC_ID + 10))

    async def paused_create(**_kwargs):
        await release.wait()
        return SimpleNamespace(message_thread_id=next(ids))

    adapter._bot.create_forum_topic = AsyncMock(side_effect=paused_create)
    first = MessageEvent(text="plan the migration", source=_make_source(thread_id=None), message_id="4242")
    second = MessageEvent(text="and the rollback", source=_make_source(thread_id=None), message_id="4243")
    adapter._enqueue_text_event(first)
    await async_wait_until(lambda: adapter._bot.create_forum_topic.await_count == 1, what="first create in flight")
    adapter._enqueue_text_event(second)
    await async_wait_until(lambda: not adapter._pending_text_batches, what="second chunk's flush popped it")
    release.set()
    await async_wait_until(lambda: len(runs) >= 2 and not adapter._pending_text_batches
                           and all(t.done() for t in adapter._session_tasks.values()), what="both turns ran")
    await asyncio.gather(*adapter._session_tasks.values(), return_exceptions=True)
    db.close()
    assert not getattr(adapter, "_held_inbound_events", None), "the first message was held for replay"
    created = adapter._bot.create_forum_topic.await_count
    assert first.source.thread_id == str(NEW_TOPIC_ID)
    landed = {first.source.thread_id, second.source.thread_id}
    assert landed == {str(NEW_TOPIC_ID + i) for i in range(created)}, "a created topic was left empty"
    assert len(runs) == 2
