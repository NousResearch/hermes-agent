"""Thread-history backfill for the Google Chat adapter (user-OAuth reads)."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from tests.gateway.test_google_chat import (  # ``adapter`` is a fixture
    _FakeHttpError,
    _make_chat_envelope,
    adapter,
)

SPACE, THREAD = "spaces/S", "spaces/S/threads/T"


def _payload(text="what now?", msg_name="spaces/S/messages/T.CUR", create_time="2026-09-29T10:00:00Z",
             space_type="SPACE"):
    env = _make_chat_envelope(text=text, msg_name=msg_name, thread_name=THREAD)
    payload = env["chat"]["messagePayload"]
    payload["message"].update(createTime=create_time, threadReply=True)
    payload["message"]["space"]["spaceType"] = space_type
    payload["space"]["spaceType"] = space_type
    return payload["message"], payload


def _history_api(messages_newest_first):
    api = MagicMock()
    api.spaces.return_value.messages.return_value.list.return_value.execute.return_value = {
        "messages": messages_newest_first}
    return api


def _list_kwargs(api):
    return api.spaces.return_value.messages.return_value.list.call_args.kwargs


@pytest.fixture(autouse=True)
def _no_session(adapter):
    adapter._bot_user_id = "users/bot"
    adapter._has_active_session = MagicMock(return_value=False)


@pytest.mark.asyncio
async def test_full_backfill_on_first_turn(adapter):
    api = _history_api([
        {"name": "spaces/S/messages/T.CUR", "sender": {"name": "users/12345", "type": "HUMAN"}, "text": "what now?"},
        {"name": "spaces/S/messages/T.B", "sender": {"name": "users/bot", "type": "BOT"}, "text": "deploy finished"},
        {"name": "spaces/S/messages/T.T", "sender": {"name": "users/bot", "type": "BOT"}, "text": "starting deploy"},
    ])
    adapter._history_chat_api = AsyncMock(return_value=api)
    event = await adapter._build_message_event(*_payload())
    ctx = event.channel_context
    assert ctx.startswith("[Thread context — earlier messages")
    assert ctx.index("Hermes (you): starting deploy") < ctx.index("deploy finished")
    assert "what now?" not in ctx and event.text == "what now?"
    assert _list_kwargs(api)["filter"] == f"thread.name = {THREAD}"
    assert _list_kwargs(api)["orderBy"] == "createTime desc"
    assert adapter._thread_count_store.get_last_seen(SPACE, THREAD) == "2026-09-29T10:00:00Z"


@pytest.mark.asyncio
async def test_dm_main_flow_gets_no_backfill(adapter):
    adapter._history_chat_api = AsyncMock()
    event = await adapter._build_message_event(*_payload(space_type="DIRECT_MESSAGE"))
    assert event.channel_context is None
    adapter._history_chat_api.assert_not_called()


@pytest.mark.asyncio
async def test_dm_side_thread_gets_backfill(adapter):
    adapter._thread_count_store.incr(SPACE, THREAD)  # thread seen before → side thread
    adapter._history_chat_api = AsyncMock(return_value=_history_api(
        [{"name": "spaces/S/messages/T.X", "sender": {"name": "users/bot", "type": "BOT"}, "text": "cron: build red"}]))
    event = await adapter._build_message_event(*_payload(space_type="DIRECT_MESSAGE"))
    assert "cron: build red" in event.channel_context


@pytest.mark.asyncio
async def test_delta_uses_watermark_and_skips_own_replies(adapter):
    store = adapter._thread_count_store
    store.set_last_seen(SPACE, THREAD, "2026-09-29T09:00:00Z")
    store.add_skip(SPACE, THREAD, "spaces/S/messages/T.OWN")
    adapter._has_active_session = MagicMock(return_value=True)
    adapter._display_names["users/777"] = "Colleague"
    api = _history_api([
        {"name": "spaces/S/messages/T.OWN", "sender": {"name": "users/bot", "type": "BOT"}, "text": "my answer"},
        {"name": "spaces/S/messages/T.C", "sender": {"name": "users/777", "type": "HUMAN"}, "text": "DB is down"},
    ])
    adapter._history_chat_api = AsyncMock(return_value=api)
    event = await adapter._build_message_event(*_payload())
    assert "Colleague: DB is down" in event.channel_context
    assert "my answer" not in event.channel_context and "since your last reply" in event.channel_context
    assert _list_kwargs(api)["filter"] == f'thread.name = {THREAD} AND create_time > "2026-09-29T09:00:00Z"'


@pytest.mark.asyncio
async def test_existing_session_without_watermark_starts_one(adapter):
    adapter._has_active_session = MagicMock(return_value=True)
    adapter._history_chat_api = AsyncMock()
    event = await adapter._build_message_event(*_payload())
    assert event.channel_context is None
    adapter._history_chat_api.assert_not_called()
    assert adapter._thread_count_store.get_last_seen(SPACE, THREAD) == "2026-09-29T10:00:00Z"


@pytest.mark.asyncio
async def test_root_message_of_new_thread_makes_no_call(adapter):
    adapter._history_chat_api = AsyncMock()
    msg, env = _payload(msg_name="spaces/S/messages/T.T")
    msg["threadReply"] = False
    assert (await adapter._build_message_event(msg, env)).channel_context is None
    adapter._history_chat_api.assert_not_called()


@pytest.mark.asyncio
async def test_commands_do_not_consume_delta(adapter):
    adapter._history_chat_api = AsyncMock()
    event = await adapter._build_message_event(*_payload(text="/approve"))
    assert event.text == "/approve" and event.channel_context is None
    assert "spaces/S/messages/T.CUR" in adapter._thread_count_store.get_skip(SPACE, THREAD)


@pytest.mark.asyncio
async def test_read_failure_keeps_watermark(adapter):
    api = MagicMock()
    api.spaces.return_value.messages.return_value.list.return_value.execute.side_effect = _FakeHttpError(400)
    adapter._history_chat_api = AsyncMock(return_value=api)
    event = await adapter._build_message_event(*_payload())
    assert event is not None and event.channel_context is None
    assert adapter._thread_count_store.get_last_seen(SPACE, THREAD) is None


@pytest.mark.asyncio
async def test_char_budget_keeps_newest(adapter):
    adapter._thread_context_max_chars = 600
    msgs = [{"name": f"spaces/S/messages/T.{i}", "sender": {"name": "users/9", "type": "HUMAN"},
             "text": f"msg{i} " + "x" * 200} for i in range(10)]
    adapter._history_chat_api = AsyncMock(return_value=_history_api(list(reversed(msgs))))
    ctx = (await adapter._build_message_event(*_payload())).channel_context
    assert "msg9" in ctx and "msg0" not in ctx and "older message(s) omitted" in ctx


@pytest.mark.asyncio
async def test_disabled_by_config(adapter):
    adapter._thread_context_enabled = False
    adapter._history_chat_api = AsyncMock()
    assert (await adapter._build_message_event(*_payload())).channel_context is None
    adapter._history_chat_api.assert_not_called()


@pytest.mark.asyncio
async def test_no_token_degrades_and_hints_once(adapter, tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    adapter._load_per_user_chat_api = AsyncMock(return_value=None)
    adapter._acquire_user_chat_api = AsyncMock(return_value=(None, None))
    caplog.set_level("INFO")
    assert await adapter._history_chat_api(["u@example.com"]) is None
    assert await adapter._history_chat_api(["u@example.com"]) is None
    hints = [r.getMessage() for r in caplog.records if "thread context unavailable" in r.getMessage()]
    assert len(hints) == 1 and "/setup-files" in hints[0]


@pytest.mark.asyncio
async def test_token_without_readonly_scope_is_skipped(adapter, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.platforms.google_chat import oauth
    path = oauth._token_path("u@example.com")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scopes": ["https://www.googleapis.com/auth/chat.messages.create"]}))
    adapter._load_per_user_chat_api = AsyncMock(return_value=MagicMock())
    adapter._acquire_user_chat_api = AsyncMock(return_value=(None, None))
    assert await adapter._history_chat_api(["u@example.com"]) is None
    adapter._load_per_user_chat_api.assert_not_called()


@pytest.mark.asyncio
async def test_outbound_messages_join_skip_set(adapter):
    adapter._chat_api.spaces.return_value.messages.return_value.create.return_value.execute.return_value = {
        "name": "spaces/S/messages/T.R", "thread": {"name": THREAD}}
    await adapter._create_message(SPACE, {"text": "hi", "thread": {"name": THREAD}})
    assert "spaces/S/messages/T.R" in adapter._thread_count_store.get_skip(SPACE, THREAD)


def test_store_meta_roundtrip_keeps_int_counts_readable(tmp_path):
    from plugins.platforms.google_chat.adapter import _ThreadCountStore
    path = tmp_path / "c.json"
    s1 = _ThreadCountStore(path)
    s1.incr(SPACE, THREAD)
    s1.set_last_seen(SPACE, THREAD, "2026-09-29T10:00:00.5Z")
    s1.set_last_seen(SPACE, THREAD, "2026-09-29T09:00:00Z")  # never backwards
    s1.add_skip(SPACE, THREAD, "spaces/S/messages/T.A")
    s2 = _ThreadCountStore(path)
    s2.load()
    assert s2.get(SPACE, THREAD) == 1
    assert s2.get_last_seen(SPACE, THREAD) == "2026-09-29T10:00:00.5Z"
    assert s2.get_skip(SPACE, THREAD) == {"spaces/S/messages/T.A"}
    assert json.loads(path.read_text())[SPACE] == {THREAD: 1}


def test_scopes_add_only_readonly_history():
    from plugins.platforms.google_chat.oauth import SCOPES
    assert SCOPES == ["https://www.googleapis.com/auth/chat.messages.create",
                      "https://www.googleapis.com/auth/chat.messages.readonly"]
