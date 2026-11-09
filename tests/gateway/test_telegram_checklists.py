"""Inbound Telegram checklists and their task-update service messages (#78718, duplicate #78749).

A checklist (Bot API 9.1 ``Message.checklist``) has no ``text``, so no handler matched it: it reached
neither the agent nor the log. It is now a text turn through the same intake as a text message.
``checklist_tasks_done`` / ``checklist_tasks_added`` service messages are observed, never a turn.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from tests.gateway.conftest import _ensure_telegram_mock
from tests.waiting import async_wait_until

_ensure_telegram_mock()
sys.modules.pop("plugins.platforms.telegram.adapter", None)
from plugins.platforms.telegram.adapter import TelegramAdapter

USER, OTHER, BOT, GROUP = 42, 77, 999, -100
# Verbatim Bot API payload of a checklist sent to a bot in a DM (it never reached the agent).
LIVE = json.loads(
    '{"title": "can you read checklists?", "tasks": [{"id":1,"text":"what time is now"},'
    '{"id":2,"text":"test task"},{"id":3,"text":"weather today"},{"id":4,"text":"who am I"}],'
    ' "others_can_add_tasks": true, "others_can_mark_tasks_as_done": true}',
    object_hook=lambda fields: SimpleNamespace(**fields))
LIVE_TEXT = (
    "[Checklist] can you read checklists?\n- [ ] what time is now\n- [ ] test task\n"
    "- [ ] weather today\n- [ ] who am I")


class _Store:
    def __init__(self):
        self.entries = []

    def get_or_create_session(self, source):
        return SimpleNamespace(session_id="observed-session")

    def append_to_transcript(self, session_id, entry, skip_db=False):
        self.entries.append(entry)


def _adapter(**extra):
    base = {"allow_from": [str(USER)], "group_allow_from": [str(USER)], "allowed_chats": [],
            "group_allowed_chats": [], "allowed_topics": []}
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="***", extra={**base, **extra}))
    adapter._bot = SimpleNamespace(id=BOT, username="hermes_bot")
    adapter._session_store = _Store()
    adapter._text_batch_delay_seconds = adapter._text_batch_split_delay_seconds = 0.01
    adapter.handle_message = AsyncMock()
    return adapter


def _message(*, text=None, checklist=None, group=False, user_id=USER, reply_to=None, **fields):
    chat = SimpleNamespace(id=GROUP if group else USER, type="group" if group else "private",
                           title="Team" if group else None, full_name=None if group else "Alice", is_forum=False)
    return SimpleNamespace(
        message_id=7, text=text, caption=None, entities=[], caption_entities=[], message_thread_id=None,
        is_topic_message=False, chat=chat, date=None, reply_to_message=reply_to, checklist=checklist,
        from_user=SimpleNamespace(id=user_id, full_name="Alice", first_name="Alice", is_bot=False), **fields)


def _update(msg, *, edited=False):
    return SimpleNamespace(update_id=11, message=None if edited else msg, edited_message=msg if edited else None,
                           effective_message=msg)


def _checklist(title, *tasks):
    return SimpleNamespace(title=title, tasks=list(tasks))


def _task(task_id, text, **completion):
    return SimpleNamespace(id=task_id, text=text, **completion)


def _event_text(msg):
    return _adapter()._build_message_event(msg, MessageType.TEXT).text


# --- rendering --------------------------------------------------------------------------------


def test_live_checklist_renders_as_readable_text():
    assert _event_text(_message(checklist=LIVE)) == LIVE_TEXT


def test_done_tasks_are_checked_with_who_completed_them():
    done_at = datetime(2026, 11, 2, 9, 30, tzinfo=UTC)
    checklist = _checklist(
        "Groceries",
        _task(1, "milk", completion_date=done_at, completed_by_user=SimpleNamespace(full_name="Anna Lee")),
        _task(2, "bread", completion_date=done_at, completed_by_chat=SimpleNamespace(full_name=None, title="Ops")),
        _task(3, "eggs", completion_date=datetime(1970, 1, 1, tzinfo=UTC)),  # PTB's value for "0: not done"
        _task(4, "tea", completion_date=done_at),
        _task(5, "jam"))
    assert _event_text(_message(checklist=checklist)) == (
        "[Checklist] Groceries\n- [x] milk (done by Anna Lee)\n- [x] bread (done by Ops)\n- [ ] eggs\n"
        "- [x] tea\n- [ ] jam")


def test_empty_title_keeps_the_header():
    assert _event_text(_message(checklist=_checklist("", _task(1, "only task")))) == "[Checklist]\n- [ ] only task"


def test_long_checklist_keeps_one_line_per_task():
    tasks = [_task(i, f"task {i}") for i in range(1, 31)]  # 30 is the Bot API maximum
    tasks[4] = _task(5, "spans\nseveral\tlines ")
    lines = _event_text(_message(checklist=_checklist("Sprint", *tasks))).split("\n")
    assert len(lines) == 31
    assert lines[5] == "- [ ] spans several lines"
    assert lines[-1] == "- [ ] task 30"


def test_message_without_a_checklist_renders_nothing():
    assert _event_text(_message(text=None)) == ""
    assert _event_text(_message(text="hello")) == "hello"


# --- routing: PTB handler selection ---------------------------------------------------------------


class _Filter:
    """Composable stand-in for a PTB filter, evaluated against an update."""

    def __init__(self, check):
        self.check = check

    def __or__(self, other):
        return _Filter(lambda update: self.check(update) or other.check(update))

    def __and__(self, other):
        return _Filter(lambda update: self.check(update) and other.check(update))

    def __invert__(self):
        return _Filter(lambda update: not self.check(update))


def _has(attr, test=bool):
    return _Filter(lambda update: test(getattr(update.effective_message, attr, None)))


class _MessageFilter(_Filter):
    """PTB's MessageFilter: a subclass overrides ``filter(message)``; the update check applies it to the message."""

    def __init__(self, name=None):
        super().__init__(lambda update: bool(self.filter(update.effective_message)))
        self.name = name

    def filter(self, message):  # the subclass provides it, as with PTB
        raise NotImplementedError


_FILTERS = SimpleNamespace(
    MessageFilter=_MessageFilter,
    TEXT=_has("text"), COMMAND=_has("text", lambda text: bool(text) and text.startswith("/")),
    LOCATION=_has("location"), VENUE=_has("venue"), PHOTO=_has("photo"), VIDEO=_has("video"),
    VIDEO_NOTE=_has("video_note"), AUDIO=_has("audio"), VOICE=_has("voice"),
    Document=SimpleNamespace(ALL=_has("document")), Sticker=SimpleNamespace(ALL=_has("sticker")),
    CHECKLIST=_has("checklist"),
    UpdateType=SimpleNamespace(EDITED=_Filter(lambda update: update.edited_message is not None)),
    StatusUpdate=SimpleNamespace(FORUM_TOPIC_CREATED=_has("forum_topic_created"),
                                 CHECKLIST_TASKS_DONE=_has("checklist_tasks_done"),
                                 CHECKLIST_TASKS_ADDED=_has("checklist_tasks_added")))


def _message_handlers(adapter, filters):
    registered = []
    # Patch the module object the adapter's code runs in: a later-collected test file may have re-imported
    # ``plugins.platforms.telegram.adapter`` since this one bound ``TelegramAdapter``.
    with patch.dict(type(adapter)._register_handlers.__globals__,
                    {"filters": filters, "TelegramMessageHandler": lambda flt, callback: registered.append((flt, callback))}):
        adapter._register_handlers(MagicMock())
    return registered


def _routed_to(handlers, update):
    return next((callback for flt, callback in handlers if flt.check(update)), None)  # PTB: first match wins


def test_checklists_route_to_the_text_handler_and_task_updates_to_the_observer():
    adapter = _adapter()
    handlers = _message_handlers(adapter, _FILTERS)
    done = SimpleNamespace(checklist_message=None, marked_as_done_task_ids=[3], marked_as_not_done_task_ids=[])
    added = SimpleNamespace(checklist_message=None, tasks=[_task(5, "new")])

    assert _routed_to(handlers, _update(_message(checklist=LIVE))) == adapter._handle_text_message
    assert _routed_to(handlers, _update(_message(text="hi"))) == adapter._handle_text_message
    assert _routed_to(handlers, _update(_message(checklist_tasks_done=done))) == adapter._handle_checklist_update
    assert _routed_to(handlers, _update(_message(checklist_tasks_added=added))) == adapter._handle_checklist_update
    # An edited checklist is not a new request (task changes arrive as their own service messages).
    assert _routed_to(handlers, _update(_message(checklist=LIVE), edited=True)) is None


def test_ptb_without_checklist_filters_still_registers_the_core_handlers():
    old_ptb = SimpleNamespace(**{k: v for k, v in vars(_FILTERS).items() if k not in {"CHECKLIST", "StatusUpdate"}},
                              StatusUpdate=None)
    adapter = _adapter()
    handlers = _message_handlers(adapter, old_ptb)
    assert _routed_to(handlers, _update(_message(text="hi"))) == adapter._handle_text_message
    assert adapter._handle_checklist_update not in [callback for _flt, callback in handlers]


# --- routing: the text intake ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dm_checklist_reaches_the_agent_as_a_text_turn():
    adapter = _adapter()
    await adapter._handle_text_message(_update(_message(checklist=LIVE)), None)
    await async_wait_until(lambda: adapter.handle_message.await_count, what="checklist dispatched")
    event = adapter.handle_message.await_args.args[0]
    assert (event.text, event.message_type) == (LIVE_TEXT, MessageType.TEXT)
    assert (event.source.chat_type, event.source.user_id) == ("dm", str(USER))


def _replied(message_id, *, user_id, text=None, checklist=None):
    """A replied-to message: media fields present (the replied-media cache reads them) but empty."""
    return SimpleNamespace(
        message_id=message_id, text=text, caption=None, checklist=checklist, photo=None, video=None,
        video_note=None, voice=None, audio=None, document=None,
        from_user=SimpleNamespace(id=user_id, is_bot=user_id == BOT, full_name="Hermes" if user_id == BOT else "Alice"))


def _bot_message():
    return _replied(5, user_id=BOT, text="earlier answer")


GATING = [
    ("dm", {}, {}, "dispatched"),
    ("unauthorized group member", {}, {"group": True, "user_id": OTHER}, "dropped"),
    ("unmentioned under require_mention", {"require_mention": True}, {"group": True}, "dropped"),
    ("observed group", {"require_mention": True, "observe_unmentioned_group_messages": True,
                        "group_allowed_chats": [str(GROUP)]}, {"group": True}, "observed"),
    ("reply to the bot", {"require_mention": True}, {"group": True, "reply_to": "bot"}, "dispatched"),
    ("free-response chat", {"require_mention": True, "free_response_chats": [str(GROUP)]}, {"group": True},
     "dispatched"),
    ("chat outside allowed_chats", {"allowed_chats": ["-200"]}, {"group": True}, "dropped"),
]


async def _intake_outcome(extra, fields, **payload):
    adapter = _adapter(**extra)
    dispatched = []
    adapter._enqueue_text_event = dispatched.append
    if fields.get("reply_to") == "bot":
        fields = {**fields, "reply_to": _bot_message()}
    await adapter._handle_text_message(_update(_message(**fields, **payload)), None)
    if dispatched:
        return "dispatched", dispatched[0].text
    if adapter._session_store.entries:
        return "observed", adapter._session_store.entries[0]["content"]
    return "dropped", None


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,fields,expected", [case[1:] for case in GATING], ids=[case[0] for case in GATING])
async def test_checklist_gating_matches_text(extra, fields, expected):
    text_outcome, _ = await _intake_outcome(extra, fields, text="please look")
    checklist_outcome, content = await _intake_outcome(extra, fields, checklist=LIVE)
    assert text_outcome == checklist_outcome == expected
    if expected != "dropped":
        assert LIVE_TEXT.split("\n", 1)[1] in content


_MENTION = SimpleNamespace(type="mention", offset=0, length=len("@hermes_bot"))


@pytest.mark.asyncio
@pytest.mark.parametrize("extra,text,checklist", [
    ({}, "@hermes_bot please look",
     SimpleNamespace(title="@hermes_bot please look", title_entities=[_MENTION], tasks=[_task(1, "first")])),
    ({}, "please look @hermes_bot", _checklist("Todo", _task(1, "please look @hermes_bot"))),
    ({"mention_patterns": [r"^\s*hermes\b"]}, "hermes, check this", _checklist("Todo", _task(1, "hermes, check this"))),
], ids=["mention entity in the title", "entity-less mention in a task", "wake word in a task"])
async def test_mention_in_a_checklist_addresses_the_bot_like_one_in_text(extra, text, checklist):
    extra = {"require_mention": True, **extra}
    assert (await _intake_outcome(extra, {"group": True}, text=text))[0] == "dispatched"
    assert (await _intake_outcome(extra, {"group": True}, checklist=checklist))[0] == "dispatched"


# --- task-update service messages -------------------------------------------------------------------


def _done(ids=(), not_done=(), checklist=LIVE):
    return SimpleNamespace(checklist_message=SimpleNamespace(checklist=checklist) if checklist else None,
                           marked_as_done_task_ids=list(ids), marked_as_not_done_task_ids=list(not_done))


_OBSERVING = {"require_mention": True, "observe_unmentioned_group_messages": True,
              "group_allowed_chats": [str(GROUP)]}


async def _observed_update(**service):
    adapter = _adapter(**_OBSERVING)
    await adapter._handle_checklist_update(_update(_message(group=True, **service)), None)
    adapter.handle_message.assert_not_awaited()
    assert not adapter._pending_text_batches
    [entry] = adapter._session_store.entries
    assert entry["observed"] is True
    attribution, _, text = entry["content"].partition("\n")
    assert attribution == f"[Alice|{USER}]"
    return text


@pytest.mark.asyncio
async def test_tasks_done_update_names_the_tasks():
    assert await _observed_update(checklist_tasks_done=_done([3, 4], [2])) == (
        '[Checklist update] can you read checklists?\nMarked done: "weather today", "who am I"\n'
        'Marked not done: "test task"')


@pytest.mark.asyncio
async def test_tasks_done_update_without_the_checklist_falls_back_to_ids():
    assert await _observed_update(checklist_tasks_done=_done([3], checklist=None)) == (
        "[Checklist update]\nMarked done: task #3")


@pytest.mark.asyncio
async def test_tasks_added_update_lists_the_new_tasks():
    added = SimpleNamespace(checklist_message=SimpleNamespace(checklist=LIVE),
                            tasks=[_task(5, "call mom"), _task(6, "buy\nmilk")])
    assert await _observed_update(checklist_tasks_added=added) == (
        '[Checklist update] can you read checklists?\nAdded: "call mom", "buy milk"')


@pytest.mark.asyncio
async def test_dm_task_update_is_logged_but_never_a_turn(caplog):
    adapter = _adapter()
    with caplog.at_level(logging.INFO):
        await adapter._handle_checklist_update(_update(_message(checklist_tasks_done=_done([3]))), None)
    adapter.handle_message.assert_not_awaited()
    assert not adapter._pending_text_batches and not adapter._session_store.entries
    logged = [r.getMessage() for r in caplog.records if "Checklist update" in r.getMessage()]
    assert len(logged) == 1 and f"chat={USER}" in logged[0]
    assert "weather today" not in logged[0]  # task text is user content: never in INFO logs


@pytest.mark.asyncio
async def test_task_update_from_unauthorized_member_is_not_observed(caplog):
    adapter = _adapter(**_OBSERVING)
    with caplog.at_level(logging.INFO):
        await adapter._handle_checklist_update(
            _update(_message(group=True, user_id=OTHER, checklist_tasks_done=_done([3]))), None)
    assert not adapter._session_store.entries
    assert not [r for r in caplog.records if "Checklist update" in r.getMessage()]


# --- reply context -------------------------------------------------------------------------------------


def _checklist_message():
    return _replied(8, user_id=USER, checklist=LIVE)


def test_reply_to_a_checklist_carries_it_as_reply_context():
    event = _adapter()._build_message_event(_message(text="here", reply_to=_checklist_message()), MessageType.TEXT)
    assert (event.text, event.reply_to_message_id, event.reply_to_text) == ("here", "8", LIVE_TEXT)


def test_reply_to_one_task_carries_only_that_task():
    msg = _message(text="do this one", reply_to=_checklist_message(), reply_to_checklist_task_id=3)
    event = _adapter()._build_message_event(msg, MessageType.TEXT)
    assert event.reply_to_text == "[Checklist] can you read checklists?\n- [ ] weather today"
