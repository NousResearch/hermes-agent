"""Processing badges must ride the bot's own messages (#119033)."""

import asyncio
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from gateway.platforms.event import ProcessingOutcome


class _FakeSdkBuilder:
    """Stands in for a lark-oapi ``*.builder()``: every setter records its value."""

    def __init__(self):
        self._fields = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)

        def _set(value):
            self._fields[name] = value
            return self

        return _set

    def build(self):
        return SimpleNamespace(**self._fields)


class _FakeSdkRequestType:
    @staticmethod
    def builder():
        return _FakeSdkBuilder()


@pytest.fixture
def fake_lark_requests(monkeypatch):
    """Fake the lazily imported lark-oapi request modules and the raw tenant GET builder."""
    import sys
    import types

    import plugins.platforms.feishu.adapter as feishu_mod

    for parent in ("lark_oapi", "lark_oapi.api", "lark_oapi.api.im", "lark_oapi.api.contact"):
        if parent not in sys.modules:
            monkeypatch.setitem(sys.modules, parent, types.ModuleType(parent))
    im_v1 = types.ModuleType("lark_oapi.api.im.v1")
    for name in ("CreateMessageReactionRequest", "CreateMessageReactionRequestBody", "DeleteMessageReactionRequest"):
        setattr(im_v1, name, _FakeSdkRequestType)
    contact_v3 = types.ModuleType("lark_oapi.api.contact.v3")
    for name in ("GetUserRequest",):
        setattr(contact_v3, name, _FakeSdkRequestType)
    monkeypatch.setitem(sys.modules, "lark_oapi.api.im.v1", im_v1)
    monkeypatch.setitem(sys.modules, "lark_oapi.api.contact.v3", contact_v3)
    monkeypatch.setattr(
        feishu_mod, "_tenant_get_request",
        lambda uri, *, queries=None: SimpleNamespace(uri=uri, queries=queries),
    )


def _plain_feishu_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.feishu.adapter import FeishuAdapter

    return FeishuAdapter(PlatformConfig())


def _reaction_adapter(*, delete_success=True):
    adapter = _plain_feishu_adapter()
    tracker = SimpleNamespace(created=[], deleted=[], created_on=[], deleted_on=[], sent=[])
    tracker.response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_bot", chat_id="oc_chat"))

    def _send(request):
        tracker.sent.append(request)
        return tracker.response

    def _create(request):
        tracker.created.append(request.request_body.reaction_type["emoji_type"])
        tracker.created_on.append(request.message_id)
        return SimpleNamespace(success=lambda: True, data=SimpleNamespace(reaction_id="r_typing"))

    def _delete(request):
        tracker.deleted.append(request.reaction_id)
        tracker.deleted_on.append(request.message_id)
        return SimpleNamespace(success=lambda: delete_success, code=0 if delete_success else 99, msg="")

    adapter._client = SimpleNamespace(
        im=SimpleNamespace(v1=SimpleNamespace(
            message=SimpleNamespace(create=_send, reply=_send),
            message_reaction=SimpleNamespace(create=_create, delete=_delete),
        ))
    )
    return adapter, tracker


def _run_processing(adapter, outcome):
    from gateway.platforms.event import MessageEvent

    event = MessageEvent(text="next turn", message_id="om_msg", source=adapter.build_source(chat_id="oc_chat"))

    async def run():
        assert (await adapter.send("oc_chat", "previous answer")).success
        await adapter.on_processing_start(event)
        await adapter.on_processing_complete(event, outcome)

    asyncio.run(run())


def test_processing_success_removes_typing_and_adds_nothing(fake_lark_requests, monkeypatch):
    monkeypatch.delenv("FEISHU_REACTIONS", raising=False)
    adapter, tracker = _reaction_adapter()
    _run_processing(adapter, ProcessingOutcome.SUCCESS)
    assert tracker.created == ["Typing"]
    assert tracker.deleted == ["r_typing"]
    assert tracker.created_on == tracker.deleted_on == ["om_bot"]
    assert "om_msg" not in adapter._pending_processing_reactions


def test_processing_failure_swaps_typing_for_cross_mark(fake_lark_requests, monkeypatch):
    monkeypatch.delenv("FEISHU_REACTIONS", raising=False)
    adapter, tracker = _reaction_adapter()
    _run_processing(adapter, ProcessingOutcome.FAILURE)
    assert tracker.created == ["Typing", "CrossMark"]
    assert tracker.deleted == ["r_typing"]
    assert tracker.created_on == ["om_bot", "om_bot"]
    assert tracker.deleted_on == ["om_bot"]


def test_processing_failure_skips_cross_mark_when_typing_removal_fails(fake_lark_requests, monkeypatch):
    """A Typing badge we couldn't remove must not get a CrossMark stacked next to it (the UI would
    read as both working and failed); the handle is kept for LRU eviction."""
    monkeypatch.delenv("FEISHU_REACTIONS", raising=False)
    adapter, tracker = _reaction_adapter(delete_success=False)
    _run_processing(adapter, ProcessingOutcome.FAILURE)
    assert tracker.created == ["Typing"]
    assert tracker.deleted == ["r_typing"]
    assert adapter._pending_processing_reactions["om_msg"] == ("om_bot", "r_typing")


@pytest.mark.parametrize("thread_id", [None, "omt_current"])
@pytest.mark.parametrize("newer_reply", [False, True])
@pytest.mark.parametrize("delete_success", [False, True])
def test_failed_delivery_does_not_reuse_badge_target_next_turn(
    fake_lark_requests, monkeypatch, thread_id, newer_reply, delete_success,
):
    """A rejected send retires its badge target without discarding a newer bot reply."""
    from gateway.platforms.event import MessageEvent

    monkeypatch.setenv("FEISHU_REACTIONS", "true")
    adapter, tracker = _reaction_adapter(delete_success=delete_success)
    source = adapter.build_source(chat_id="oc_chat", thread_id=thread_id)
    event = MessageEvent(text="current turn", message_id="om_user1", source=source)
    next_event = MessageEvent(text="next turn", message_id="om_user2", source=source)
    metadata = {"thread_id": thread_id}

    async def run():
        assert (await adapter.send("oc_chat", "previous answer", metadata=metadata)).success
        await adapter.on_processing_start(event)
        tracker.response = SimpleNamespace(success=lambda: False, code=400, msg="delivery rejected", data=None)
        assert not (await adapter.send("oc_chat", "failed answer", metadata=metadata)).success
        if newer_reply:
            tracker.response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(
                message_id="om_new", chat_id="oc_chat"))
            assert (await adapter.send("oc_chat", "newer answer", metadata=metadata)).success
        await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
        assert adapter._last_bot_messages.get(("oc_chat", thread_id)) == ("om_new" if newer_reply else None)
        await adapter.on_processing_start(next_event)

    asyncio.run(run())
    terminal = ["CrossMark"] if delete_success else []
    assert tracker.created == ["Typing"] + terminal + (["Typing"] if newer_reply else [])
    assert tracker.created_on == ["om_bot"] * (1 + len(terminal)) + (["om_new"] if newer_reply else [])
    assert tracker.deleted_on == ["om_bot"]
    assert adapter._pending_processing_reactions.get("om_user1") == (
        None if delete_success else ("om_bot", "r_typing"))


@pytest.mark.parametrize("case", ["create", "reply", "thread", "thread-reply", "user-id", "first", "other-chat", "other-thread", "rejected", "missing-id", "disabled"])
def test_processing_badge_targets_only_our_sent_conversation_message(fake_lark_requests, monkeypatch, case):
    """Exercise real sends and lifecycle requests, including a new reply between start and completion."""
    from gateway.platforms.event import MessageEvent

    monkeypatch.setenv("FEISHU_REACTIONS", "false" if case == "disabled" else "true")
    adapter, tracker = _reaction_adapter()
    chat_id = "oc_other" if case == "other-chat" else "oc_chat"
    thread_id = "omt_other" if case == "other-thread" else "omt_current" if case.startswith("thread") else None
    reply_to = "om_user" if case in {"reply", "thread-reply"} else None
    if case == "user-id":
        chat_id = "feishu_user_id:u_human"
    tracker.response.data.chat_id = "oc_other" if case == "other-chat" else "oc_chat"
    if case == "rejected":
        tracker.response.success = lambda: False
    if case == "missing-id":
        tracker.response.data.message_id = None
    event = MessageEvent(
        text="next turn", message_id="om_user",
        source=adapter.build_source(chat_id="oc_chat", thread_id="omt_current" if thread_id else None),
    )
    expected = case in {"create", "reply", "thread", "thread-reply", "user-id"}

    async def run():
        if case != "first":
            response = await adapter._send_raw_message(
                chat_id=chat_id, msg_type="text", payload='{"text":"previous answer"}',
                reply_to=reply_to, metadata={"thread_id": thread_id} if thread_id else None,
            )
            assert response is tracker.response
        await adapter.on_processing_start(event)
        await adapter.on_processing_start(event)  # Never stack duplicate Typing badges.
        tracker.response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id="om_new", chat_id="oc_chat"))
        await adapter.send("oc_chat", "new answer", metadata={"thread_id": event.source.thread_id})
        await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)

    asyncio.run(run())
    assert tracker.created == (["Typing", "CrossMark"] if expected else [])
    assert tracker.created_on == (["om_bot", "om_bot"] if expected else [])
    assert tracker.deleted_on == (["om_bot"] if expected else [])
    assert "om_user" not in adapter._pending_processing_reactions


@pytest.mark.parametrize("reply_to", [None, "om_user"])
def test_bot_message_tracking_is_bounded_and_cannot_retry_delivered_messages(fake_lark_requests, monkeypatch, reply_to):
    import plugins.platforms.feishu.adapter as feishu_mod

    monkeypatch.setattr(feishu_mod, "_FEISHU_PROCESSING_REACTION_CACHE_SIZE", 2)
    adapter, tracker = _reaction_adapter()

    async def send(chat_id):
        tracker.response = SimpleNamespace(success=lambda: True, data=SimpleNamespace(message_id=f"bot_{chat_id}", chat_id=chat_id))
        response = await adapter._send_raw_message(
            chat_id=chat_id, msg_type="text", payload='{"text":"answer"}', reply_to=reply_to, metadata=None,
        )
        assert response is tracker.response

    async def run():
        for chat_id in ("oc_old", "oc_recent", "oc_old", "oc_new"):
            await send(chat_id)
        assert list(adapter._last_bot_messages) == [("oc_old", None), ("oc_new", None)]
        assert adapter._last_bot_messages[("oc_new", None)] == "bot_oc_new"

        class BrokenCache(OrderedDict):
            def __setitem__(self, key, value):
                raise RuntimeError("bookkeeping failed after delivery")

        adapter._last_bot_messages = BrokenCache()
        await send("oc_delivered")

    asyncio.run(run())
    assert len(tracker.sent) == 5  # A bookkeeping failure must not cause a duplicate send.
