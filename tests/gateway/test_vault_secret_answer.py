"""A secret typed into a chat (vault password, master password, one-time code) is deleted on
receipt, only its owner can answer, and it never reaches the agent as an ordinary turn.

Built on the messaging vault prompts (``gateway/vault_prompts.py``). These pin the promise a
secret prompt makes: the gateway posts one only where the bot can delete the answer, deletes it
BEFORE handing the value over, checks the deletion and says so in the chat when it fails, and
keeps the value out of the agent in every edge it can see — another member of a shared topic
answering, a voice note, a password starting with "/", a message merged from several, an answer
that lands just after the prompt ended.
"""

import asyncio
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.i18n import t
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult, merge_pending_message_event
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.vault_prompts import VaultPromptRefused, bound_owner
from tools import clarify_gateway as cm


SESSION_KEY = "agent:main:telegram:group:-100777:42"
OWNER = "1001"
OTHER = "2002"


class _ChatAdapter(BasePlatformAdapter):
    """Records deletions and sends; deletion succeeds unless told otherwise."""

    def __init__(self, delete_ok=True):
        super().__init__(PlatformConfig(enabled=True, token="test"), Platform.TELEGRAM)
        self.delete_ok = delete_ok
        self.deleted: list[str] = []
        self.sent: list[tuple[str, dict]] = []
        self.retired: list[tuple[str, str]] = []
        self.on_delete = None

    async def connect(self, *, is_reconnect: bool = False):
        return True

    async def disconnect(self):
        pass

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((content, dict(metadata or {}), reply_to))
        return SendResult(success=True, message_id="bot-1")

    async def get_chat_info(self, chat_id):
        return {"id": chat_id, "type": "supergroup"}

    async def delete_message(self, chat_id, message_id):
        if self.on_delete is not None:
            self.on_delete(message_id)
        self.deleted.append(message_id)
        return self.delete_ok

    async def retire_clarify_card(self, clarify_id, notice):
        self.retired.append((clarify_id, notice))


class _FellThroughIntercept(Exception):
    """Sentinel: _handle_message got PAST the clarify intercept (the message would become a turn)."""


def _event(text="", *, user=OWNER, message_id="m1", **kwargs):
    return MessageEvent(
        text=text,
        message_type=kwargs.pop("message_type", MessageType.TEXT),
        source=SessionSource(platform=Platform.TELEGRAM, chat_id="-100777", chat_type="group",
                             user_id=user, thread_id="42"),
        message_id=message_id,
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _clean_clarify_state():
    def clear():
        with cm._lock:
            cm._entries.clear()
            cm._session_index.clear()
            cm._notify_cbs.clear()
            cm._secret_graces.clear()
    clear()
    yield
    clear()


@pytest.fixture(autouse=True)
def _no_retry_wait():
    with patch("gateway.run_inbound.asyncio.sleep", AsyncMock()):
        yield


def _make_runner(adapter):
    from gateway.run import GatewayRunner

    runner = GatewayRunner.__new__(GatewayRunner)
    runner._startup_restore_in_progress = False
    runner._scale_to_zero_note_real_inbound = lambda: None
    runner._is_user_authorized = lambda source: True
    runner._session_key_for_source = lambda source: SESSION_KEY
    runner._delivery_adapter_for = lambda source: adapter
    runner._update_prompt_pending = {}
    runner._thread_metadata_for_source = lambda source, reply_to=None: {
        "thread_id": "42", "telegram_reply_to_message_id": "m1"}
    return runner


async def _dispatch(runner, event):
    import tools.slash_confirm as slash_confirm_mod

    def _tripwire(_key):
        raise _FellThroughIntercept()

    with patch("hermes_cli.plugins.invoke_hook", return_value=[]), \
            patch.object(slash_confirm_mod, "get_pending", _tripwire):
        return await runner._handle_message(event)


def _secret_prompt(cid="cl-secret"):
    return cm.register(cid, SESSION_KEY, "password?", None, secret=True, owner_user_id=OWNER)


# -- receipt ------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_secret_is_deleted_before_the_value_is_handed_over():
    adapter = _ChatAdapter()
    entry = _secret_prompt()
    seen_at_deletion = []
    adapter.on_delete = lambda _mid: seen_at_deletion.append(entry.event.is_set())

    assert await _dispatch(_make_runner(adapter), _event("  hunter2  ")) == ""

    assert adapter.deleted == ["m1"]
    assert seen_at_deletion == [False], "the tool must not hold the value before the message is gone"
    assert entry.response == "hunter2"
    assert adapter.retired == [("cl-secret", t("gateway.clarify.answered_secret"))]
    assert all("hunter2" not in notice for _, notice in adapter.retired)
    assert adapter.sent == [], "nothing to report when the deletion worked"


@pytest.mark.asyncio
async def test_failed_deletion_is_retried_then_reported_in_the_topic_without_quoting():
    adapter = _ChatAdapter(delete_ok=False)
    entry = _secret_prompt()

    assert await _dispatch(_make_runner(adapter), _event("hunter2")) == ""

    assert adapter.deleted == ["m1", "m1"], "one retry"
    assert entry.response == "hunter2", "the answer still serves; the user is told to delete it"
    [(content, metadata, reply_to)] = adapter.sent
    assert content == t("gateway.vault_prompt.not_deleted")
    assert "hunter2" not in content
    assert reply_to is None and not any("reply_to" in key for key in metadata), \
        "a reply would quote the secret"
    assert metadata.get("thread_id") == "42", "the notice lands in the user's topic"


@pytest.mark.asyncio
async def test_every_message_merged_into_the_answer_is_deleted():
    adapter = _ChatAdapter()
    _secret_prompt()
    event = _event("hunter2", message_id="m1")
    event.absorb_merged(_event("\nvoilà", message_id="m2"))
    event.absorb_merged(_event("!", message_id="m3"))

    await _dispatch(_make_runner(adapter), event)

    assert adapter.deleted == ["m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_skip_is_a_password_not_a_decline():
    adapter = _ChatAdapter()
    entry = _secret_prompt()

    await _dispatch(_make_runner(adapter), _event("skip"))

    assert entry.response == "skip"


@pytest.mark.asyncio
async def test_a_password_starting_with_a_slash_is_taken_as_the_secret():
    adapter = _ChatAdapter()
    entry = _secret_prompt()

    assert await _dispatch(_make_runner(adapter), _event("/Abc123xyz")) == ""

    assert entry.response == "/Abc123xyz"
    assert adapter.deleted == ["m1"]


@pytest.mark.asyncio
async def test_a_real_command_still_works_and_leaves_the_prompt_armed():
    adapter = _ChatAdapter()
    entry = _secret_prompt()

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(_make_runner(adapter), _event("/stop"))

    assert not entry.event.is_set()
    assert adapter.deleted == []


# -- who may answer -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_another_member_of_the_topic_cannot_answer():
    adapter = _ChatAdapter()
    entry = _secret_prompt()

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(_make_runner(adapter), _event("on en parle demain ?", user=OTHER))

    assert not entry.event.is_set(), "the owner's real password must still be awaited"
    assert adapter.deleted == []


# -- voice --------------------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_a_voice_note_is_deleted_with_its_audio_and_never_transcribed(tmp_path):
    adapter = _ChatAdapter()
    entry = _secret_prompt()
    audio = tmp_path / "voice.ogg"
    audio.write_bytes(b"OggS")
    runner = _make_runner(adapter)
    runner._transcribe_pending_audio_event_once = AsyncMock(side_effect=AssertionError("transcribed"))

    event = _event("", message_type=MessageType.VOICE, media_urls=[str(audio)], media_types=["audio/ogg"])
    assert await _dispatch(runner, event) == ""

    assert adapter.deleted == ["m1"]
    assert not audio.exists(), "the cached recording is the secret too"
    assert not entry.event.is_set(), "the prompt waits for the typed answer"
    assert [content for content, _, _ in adapter.sent] == [t("gateway.vault_prompt.voice_refused")]


# -- just after the prompt ended ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_an_answer_right_after_the_prompt_ended_is_deleted_and_dropped():
    adapter = _ChatAdapter()
    _secret_prompt()
    assert cm.wait_for_response("cl-secret", 0.01) is None  # timed out unanswered

    assert await _dispatch(_make_runner(adapter), _event("hunter2")) == ""

    assert adapter.deleted == ["m1"]
    assert [content for content, _, _ in adapter.sent] == [t("gateway.vault_prompt.expired")]
    # One message only: the next one is an ordinary turn again.
    with pytest.raises(_FellThroughIntercept):
        await _dispatch(_make_runner(adapter), _event("bon, on reprend", message_id="m2"))


@pytest.mark.asyncio
async def test_the_window_opens_on_cancellation_too_and_only_for_the_owner():
    adapter = _ChatAdapter()
    _secret_prompt()
    assert cm.clear_session(SESSION_KEY) == 1  # /new, eviction, shutdown

    with pytest.raises(_FellThroughIntercept):
        await _dispatch(_make_runner(adapter), _event("salut", user=OTHER))
    assert await _dispatch(_make_runner(adapter), _event("hunter2", message_id="m2")) == ""
    assert adapter.deleted == ["m2"]


# -- busy session routing -----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_busy_session_routes_a_slash_secret_to_the_intercept():
    """While the agent waits on the prompt the session is busy; "/Abc..." must reach the intercept,
    not the follow-up queue (which would hand it to the agent verbatim)."""
    adapter = _ChatAdapter()
    adapter._dispatch_inline_reply = AsyncMock()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    _secret_prompt()

    await adapter._handle_message_while_active(_event("/Abc123xyz"), SESSION_KEY)

    adapter._dispatch_inline_reply.assert_awaited_once()
    adapter._busy_session_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_busy_session_still_queues_slash_text_for_an_ordinary_prompt():
    adapter = _ChatAdapter()
    adapter._dispatch_inline_reply = AsyncMock()
    adapter._busy_session_handler = AsyncMock(return_value=True)
    cm.register("cl-open", SESSION_KEY, "which file?", None)

    await adapter._handle_message_while_active(_event("/Abc123xyz"), SESSION_KEY)

    adapter._dispatch_inline_reply.assert_not_awaited()


# -- merged ids ---------------------------------------------------------------------------------

def test_a_merged_event_keeps_every_chat_message_id():
    head = _event("hunter", message_id="m1")
    pending = {SESSION_KEY: head}
    merge_pending_message_event(pending, SESSION_KEY, _event("2", message_id="m2"), merge_text=True)
    tail = _event("!", message_id="m3")
    tail.absorb_merged(_event("?", message_id="m4"))
    head.absorb_merged(tail)
    head.message_id = "m4"  # the queue-mode debounce re-anchors on the latest message
    assert head.chat_message_ids() == ["m4", "m1", "m2", "m3"]


# -- the guard refuses where a secret cannot be taken safely --------------------------------------

def _ctx(*, user_id=OWNER, can_delete=True, chat_type="group", probe=True):
    adapter = SimpleNamespace()
    if probe:
        async def can_delete_user_messages(chat_id, kind):
            return can_delete
        adapter.can_delete_user_messages = can_delete_user_messages
    return SimpleNamespace(source=SessionSource(platform=Platform.TELEGRAM, chat_id="-100777",
                                                chat_type=chat_type, user_id=user_id),
                           _status_adapter=adapter, _status_chat_id="-100777")


def _schedule(coro, _log_message):
    fut = Future()
    fut.set_result(asyncio.run(coro))
    return fut


def test_owner_is_the_turns_sender_when_the_bot_can_delete():
    assert bound_owner(_ctx(), _schedule) == OWNER


def test_refused_when_the_bot_cannot_delete_the_answer():
    with pytest.raises(VaultPromptRefused, match="Delete messages"):
        bound_owner(_ctx(can_delete=False), _schedule)


def test_refused_when_the_adapter_cannot_tell():
    """A platform has to prove it can delete; an adapter without the probe cannot."""
    with pytest.raises(VaultPromptRefused, match="Delete messages"):
        bound_owner(_ctx(probe=False), _schedule)


def test_refused_when_nobody_can_be_bound():
    with pytest.raises(VaultPromptRefused, match="unknown"):
        bound_owner(_ctx(user_id=None), _schedule)


def test_refused_without_a_chat_to_ask_in():
    ctx = _ctx()
    ctx._status_adapter = None
    with pytest.raises(VaultPromptRefused, match="no chat"):
        bound_owner(ctx, _schedule)


def test_a_refusal_posts_no_card():
    runner = object.__new__(TurnRunner)
    runner._ctx, runner._schedule = _ctx(can_delete=False), _schedule
    runner._ask_clarify_question = MagicMock()
    with pytest.raises(VaultPromptRefused):
        runner._vault_ask_sync("email?", secret=False, last=False)
    runner._ask_clarify_question.assert_not_called()


def test_a_skipped_prompt_is_a_decline():
    runner = object.__new__(TurnRunner)
    runner._ctx, runner._schedule = _ctx(), _schedule
    runner._ask_clarify_question = lambda *a, **k: (cm.SKIPPED, True)
    assert runner._vault_ask_sync("q", secret=True) == ""
