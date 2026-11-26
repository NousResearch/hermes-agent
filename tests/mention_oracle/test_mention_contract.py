"""Owner contract (2026-11-23): a valid ``@mention`` reaches the bot and gets a reply, and a
group-scoped turn never receives DM-scoped context.

False premise this suite exists to kill: "if the intake gate recognises a group @mention, the
pipeline downstream stays group-scoped". ``SessionSource.chat_type`` already reaches
``agent._chat_type`` at construction, but no context consumer read it — ``_init_memory`` loaded
MEMORY.md/USER.md and the external memory provider in every scope, ``session_search`` could read
DM sessions mid-turn, and the ``memory`` tool could write DM files from a group.

Boundary rules (owner contract): only external edges are substituted — the loopback model
server, the recording adapter, and the plugin's reader file. Expected values are independent
literals (``CANARY_DM_SECRET``), never computed by the code under test.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

# Imported at module top on purpose: run_agent's import-time early-recovery probe stats a
# marker inside the repo .git dir, which must fire during collection — inside a test the
# per-test home-io guard would flag that legitimate probe of the checkout's own .git.
import gateway.run  # collection-time import is the point; the module is exercised via _make_runner
import run_agent
from gateway.config import Platform, PlatformConfig, StreamingConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.session import SessionSource
from hermes_cli import plugins

from tests.fakes.fake_llm_provider import FakeLLMServer, Text, write_hermes_home
from tests.fixtures.storage_seed import seed_messages, seed_session
from tests.gateway.test_run_progress_topics import _make_runner
from tests.gateway.test_telegram_group_gating import (
    _group_message, _make_adapter, _mention_entities,
)

CANARY_DM_SECRET = "CANARY_DM_SECRET_9481"
GROUP_CHAT_ID = "-100"
BOT_HANDLES = ("hqecea_bot", "sqecea_bot", "yqecea_bot")


def _event_messages_payload(server: "FakeLLMServer") -> str:
    """Every byte of message content the model was asked to look at."""
    parts = []
    for body in server.main_requests():
        for msg in body.get("messages", []):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                parts.extend(str(item.get("text", "")) for item in content if isinstance(item, dict))
            parts.append(json.dumps(msg.get("tool_calls", ""), ensure_ascii=False))
    return "\n".join(parts)


class _RecordingChat(BasePlatformAdapter):
    """External transport boundary: records what the gateway tries to deliver."""

    def __init__(self):
        super().__init__(PlatformConfig(enabled=True, token="***"), Platform.TELEGRAM)
        self.log: list[str] = []
        self._ids = 0

    async def connect(self, *, is_reconnect=False):
        return True

    async def disconnect(self):
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self._ids += 1
        self.log.append(content)
        return SendResult(success=True, message_id=f"m{self._ids}")

    async def edit_message(self, chat_id, message_id, content, **_):
        return SendResult(success=True, message_id=message_id)

    async def send_typing(self, chat_id, metadata=None):
        return None

    async def get_chat_info(self, chat_id):
        return {"id": chat_id}


# ── A. Allowed matrix: Telegram group @mention → enqueued, typed event ─────────

@pytest.mark.asyncio
@pytest.mark.parametrize("handle", BOT_HANDLES)
async def test_telegram_group_mention_is_admitted_and_typed_group(handle):
    """A real-shaped group update mentioning the bot passes the gate once, with a typed
    group scope on the source and a trigger mark the watchdog can key on."""
    adapter = _make_adapter(
        bot_username=handle, require_mention=True, exclusive_bot_mentions=True,
        allowed_chats=[GROUP_CHAT_ID], group_allowed_chats=[GROUP_CHAT_ID],
    )
    events = []
    adapter._enqueue_text_event = events.append
    adapter.handle_message = AsyncMock(side_effect=events.append)
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_inbound_av = AsyncMock(return_value=False)

    text = f"@{handle} status?"
    msg = _group_message(text, entities=_mention_entities(text, [f"@{handle}"]))
    update = SimpleNamespace(update_id=1001, message=msg, effective_message=None)
    await adapter._handle_text_message(update, SimpleNamespace())

    assert len(events) == 1, "a valid @mention in an allowed group must dispatch exactly one event"
    event = events[0]
    assert event.source.chat_type == "group", "group intake must arrive typed as group"
    assert event.source.user_id, "a dispatched group mention keeps the sender id"
    assert event.mention_trigger is True, "an admitted @mention must carry the watchdog mark"


@pytest.mark.asyncio
async def test_telegram_group_message_without_mention_is_not_dispatched():
    """The deny side of the matrix: group chatter without a mention must not reach the agent."""
    adapter = _make_adapter(
        bot_username="hqecea_bot", require_mention=True, exclusive_bot_mentions=True,
        allowed_chats=[GROUP_CHAT_ID], group_allowed_chats=[GROUP_CHAT_ID],
    )
    events = []
    adapter._enqueue_text_event = events.append
    adapter.handle_message = AsyncMock(side_effect=events.append)
    adapter._ensure_forum_commands = AsyncMock()
    adapter._cache_inbound_av = AsyncMock(return_value=False)

    msg = _group_message("anyone seen the deploy?")
    update = SimpleNamespace(update_id=1002, message=msg, effective_message=None)
    await adapter._handle_text_message(update, SimpleNamespace())
    assert not events


# ── B. Canary isolation: a group turn must never see DM-scoped memory ───────────

@pytest.mark.asyncio
async def test_group_turn_excludes_dm_memory_and_marks_scope(monkeypatch, tmp_path):
    """A DM canary in MEMORY.md/USER.md must appear 0 times in the model request of a
    group turn; the assembled context must carry an explicit chat_scope=group marker."""

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = Path(os.environ["HERMES_HOME"])  # per-test sandbox from conftest
    with FakeLLMServer([Text("pong from group")]) as srv:
        # Real deployment inputs, not code substitutions: config.yaml keys and the
        # production home-override are the user's own mechanisms — the real resolver
        # (config.yaml -> resolve_runtime_with_fallback) must reach the fake server by itself.
        write_hermes_home(home, srv.base_url, api_mode="chat_completions",
                          model_api_key="sk-fake-e2e")
        mem_dir = home / "memories"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "MEMORY.md").write_text(f"- dm note {CANARY_DM_SECRET}\n", encoding="utf-8")
        (mem_dir / "USER.md").write_text(f"- dm profile {CANARY_DM_SECRET}\n", encoding="utf-8")

        plugins._reset_plugin_managers_for_tests()
        # gateway.run froze its module-level _hermes_home at import (the session sandbox);
        # conftest gives each test its own HERMES_HOME — bind it the same way production
        # profile routing binds a home, so _load_gateway_config reads this test's config.yaml.
        home_token = set_hermes_home_override(str(home))
        chat = _RecordingChat()
        runner = _make_runner(chat)
        runner.adapters[Platform.TELEGRAM] = chat
        runner.config.streaming = StreamingConfig(enabled=False)

        source = SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_CHAT_ID,
                               chat_type="group", user_id="u-owner")
        session_key = "agent:main:telegram:group:g-100:u-owner"
        try:
            result = await runner._run_agent(
                message="hqecea_bot status?", context_prompt="", history=[],
                source=source, session_id="sess-group-1", session_key=session_key)
        finally:
            reset_hermes_home_override(home_token)
            plugins._reset_plugin_managers_for_tests()

    assert result["final_response"], "a valid mention must produce a non-empty reply"
    payload = _event_messages_payload(srv)
    assert srv.main_requests(), "the turn must actually call the model"
    assert CANARY_DM_SECRET not in payload, (
        "DM-scoped memory leaked into a group turn's model request")
    assert "chat_scope" in payload and "group" in payload, (
        "the assembled group context must carry an explicit chat_scope=group marker")


@pytest.mark.asyncio
async def test_dm_turn_still_sees_dm_memory(monkeypatch, tmp_path):
    """Control arm: the same seeded store in a DM turn keeps memory — the fix must narrow
    group scope only, never break DM recall."""

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    home = Path(os.environ["HERMES_HOME"])  # per-test sandbox from conftest
    with FakeLLMServer([Text("pong from dm")]) as srv:
        write_hermes_home(home, srv.base_url, api_mode="chat_completions",
                          model_api_key="sk-fake-e2e")
        mem_dir = home / "memories"
        mem_dir.mkdir(parents=True, exist_ok=True)
        (mem_dir / "MEMORY.md").write_text(f"- dm note {CANARY_DM_SECRET}\n", encoding="utf-8")

        plugins._reset_plugin_managers_for_tests()
        home_token = set_hermes_home_override(str(home))
        chat = _RecordingChat()
        runner = _make_runner(chat)
        runner.adapters[Platform.TELEGRAM] = chat
        runner.config.streaming = StreamingConfig(enabled=False)

        source = SessionSource(platform=Platform.TELEGRAM, chat_id="dm-1",
                               chat_type="dm", user_id="u-owner")
        try:
            result = await runner._run_agent(
                message="hi", context_prompt="", history=[],
                source=source, session_id="sess-dm-1",
                session_key="agent:main:telegram:dm:dm-1")
        finally:
            reset_hermes_home_override(home_token)
            plugins._reset_plugin_managers_for_tests()

    assert result["final_response"]
    assert CANARY_DM_SECRET in _event_messages_payload(srv), (
        "DM turns must keep built-in memory — scope isolation is group-only")


# ── C. Recall tools under group scope ─────────────────────────────────────────

def _seeded_store(tmp_path):
    """Real persistence store: one DM session holding the canary, one group session clean."""
    from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

    writer = open_persistence(profile_scope(home=tmp_path), eager=True)
    seed_session(writer, "sess-dm-leak", "telegram", chat_type="dm", chat_id="dm-1")
    seed_messages(writer, "sess-dm-leak", {"role": "user", "content": f"secret: {CANARY_DM_SECRET}"})
    seed_session(writer, "sess-group-ok", "telegram", chat_type="group", chat_id=GROUP_CHAT_ID)
    seed_messages(writer, "sess-group-ok", {"role": "user", "content": "group chatter"})
    return writer


def _search_agent(chat_type, persistence, db):
    """The agent-shaped caller the inline executor dispatches under."""
    return SimpleNamespace(
        _chat_type=chat_type, session_id="sess-group-ok",
        persistence=persistence, _get_session_db_for_recall=lambda: db)


def test_session_search_in_group_scope_never_returns_dm_sessions(tmp_path):
    """Mid-turn recall must not surface the DM canary session in a group-scoped turn."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from persistence_assembly import open_persistence_reader

    writer = _seeded_store(tmp_path)
    try:
        with open_persistence_reader(tmp_path) as reader:
            group_agent = _search_agent("group", reader, writer)
            result = INLINE_TOOL_EXECUTORS["session_search"](
                group_agent, {"query": CANARY_DM_SECRET, "detail": "full"}, None)
            payload = result if isinstance(result, dict) else json.loads(str(result))
            assert payload.get("count") == 0 and not payload.get("results"), (
                "session_search in group scope returned DM session content")

            dm_agent = _search_agent("dm", reader, writer)
            control = INLINE_TOOL_EXECUTORS["session_search"](
                dm_agent, {"query": CANARY_DM_SECRET, "detail": "full"}, None)
            assert CANARY_DM_SECRET in str(control), (
                "control: the same search in DM scope must find the session")
    finally:
        writer.close()


def test_memory_tool_in_group_scope_cannot_write_dm_files(tmp_path):
    """A group turn must not write the owner's DM memory files (same store, other direction)."""
    from agent.inline_tool_executors import INLINE_TOOL_EXECUTORS
    from tools.memory_tool import MemoryStore

    store = MemoryStore(memory_enabled=True, user_profile_enabled=True)
    agent = SimpleNamespace(_chat_type="group", _memory_store=store, _memory_manager=None)
    agent._build_memory_write_metadata = lambda **kw: {}

    result = INLINE_TOOL_EXECUTORS["memory"](
        agent, {"action": "add", "target": "memory", "content": CANARY_DM_SECRET}, None)
    text = str(result)
    assert "success\": false" in text.replace("'", '"') or "unavailable" in text.lower() \
        or "group" in text.lower(), (
        f"memory writes from a group-scoped turn must be refused, got: {text[:200]}")
    assert CANARY_DM_SECRET not in str(store.memory_text() if hasattr(store, "memory_text") else ""), (
        "a refused write must not reach the DM store")


# ── D. Watchdog: mention_received without reply_sent ───────────────────────────

@pytest.mark.asyncio
async def test_mention_trigger_without_delivery_logs_mention_unanswered(caplog):
    """An admitted mention whose turn ends with nothing delivered must raise a structured
    ``mention_unanswered`` error — zero silent failures."""
    import logging

    chat = _RecordingChat()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_CHAT_ID,
                           chat_type="group", user_id="u-owner")
    event = MessageEvent(text="@hqecea_bot status?", source=source,
                         message_type=MessageType.TEXT)
    event.mention_trigger = True

    async def silent_handler(event):
        return ""  # model produced nothing: no reply reaches the chat

    chat.set_message_handler(silent_handler)
    session_key = "agent:main:telegram:group:g-100:u-owner"
    with caplog.at_level(logging.ERROR):
        await chat._process_message_background(event, session_key)
    assert not chat.log
    assert any("mention_unanswered" in rec.message for rec in caplog.records), (
        "a mention turn with no delivery must log mention_unanswered")


@pytest.mark.asyncio
async def test_mention_trigger_with_delivery_stays_quiet(caplog):
    """Control arm: a delivered reply must not trip the watchdog."""
    import logging

    chat = _RecordingChat()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_CHAT_ID,
                           chat_type="group", user_id="u-owner")
    event = MessageEvent(text="@hqecea_bot status?", source=source,
                         message_type=MessageType.TEXT)
    event.mention_trigger = True

    async def handler(event):
        return "all good"

    chat.set_message_handler(handler)
    session_key = "agent:main:telegram:group:g-100:u-owner"
    with caplog.at_level(logging.ERROR):
        await chat._process_message_background(event, session_key)
    assert chat.log == ["all good"]
    assert not any("mention_unanswered" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_streamed_group_mention_delivery_does_not_trip_watchdog(caplog):
    """Replay of the live hit (gateway.log 2026-11-23 09:19:24): a group mention whose final
    reached the chat through the stream path — the draft send was edited in place by
    ``edit_streamed_message`` (``already_sent``), so ``_hmwa_deliver_turn_response`` stashed the
    delivered body on ``event._streamed_final_response`` and returned None. The turn's send paths
    never ran (``delivery_attempted=False``) while the user already had the reply on screen:
    outcome was SUCCESS and the watchdog still logged mention_unanswered delivered=False — a
    false alarm on every streamed mention until the seam counted the stash."""
    import logging

    chat = _RecordingChat()
    source = SessionSource(platform=Platform.TELEGRAM, chat_id=GROUP_CHAT_ID,
                           chat_type="group", user_id="u-owner")
    event = MessageEvent(text="@hqecea_bot status?", source=source,
                         message_type=MessageType.TEXT, message_id="10")
    event.mention_trigger = True

    async def streamed_handler(ev):
        # The production contract after a confirmed streamed/edit delivery
        # (run_turn_persistence._hmwa_deliver_turn_response): stash the delivered text, return
        # None so the adapter does not send it a second time.
        ev._streamed_final_response = "Неизвестно."

    chat.set_message_handler(streamed_handler)
    session_key = "agent:main:telegram:group:g-100:u-owner"
    with caplog.at_level(logging.ERROR):
        await chat._process_message_background(event, session_key)
    assert not any("mention_unanswered" in rec.message for rec in caplog.records), (
        "stream-mediated delivery must count as delivered at the outcome seam")
