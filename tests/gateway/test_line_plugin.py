"""Tests for the LINE platform adapter plugin.

Covers LINE adapter behavior from the PR review:

1. webhook signature verification (HMAC-SHA256, base64) + tampering rejection
2. inbound chat-id resolution for user / group / room sources
3. three-allowlist gating (users / groups / rooms / allow_all)
4. inbound dedup via webhookEventId
5. RequestCache state machine (PENDING → READY → DELIVERED, ERROR)
6. Markdown stripping with URL preservation + LINE-sized chunking
7. inbound media normalization to gateway message types and MIME metadata
8. send routing: reply token preferred → push fallback → batched at 5/call
9. register() metadata + standalone_send shape
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import base64
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.gateway._plugin_adapter_loader import load_plugin_adapter
from tests.waiting import async_wait_until

# Load plugins/platforms/line/adapter.py under plugin_adapter_line so it
# cannot collide with sibling platform-plugin tests in the same process.
_line = load_plugin_adapter("line")

verify_line_signature = _line.verify_line_signature
strip_markdown_preserving_urls = _line.strip_markdown_preserving_urls
split_for_line = _line.split_for_line
build_postback_button_message = _line.build_postback_button_message
_resolve_chat = _line._resolve_chat
_allowed_for_source = _line._allowed_for_source
RequestCache = _line.RequestCache
State = _line.State
LineAdapter = _line.LineAdapter
register = _line.register
check_requirements = _line.check_requirements
validate_config = _line.validate_config
_standalone_send = _line._standalone_send
_env_enablement = _line._env_enablement


# ---------------------------------------------------------------------------
# 1. Signature verification
# ---------------------------------------------------------------------------

class TestSignature:

    def _sign(self, body: bytes, secret: str) -> str:
        digest = hmac.new(secret.encode(), body, hashlib.sha256).digest()
        return base64.b64encode(digest).decode()


    def test_wrong_secret_rejected(self):
        body = b'{"events": []}'
        sig = self._sign(body, "secret")
        assert not verify_line_signature(body, sig, "different")


    def test_empty_secret_rejected(self):
        assert not verify_line_signature(b"x", "AAAA", "")


# ---------------------------------------------------------------------------
# 2. Chat-id / source resolution
# ---------------------------------------------------------------------------

class TestSourceResolution:

    def test_user_source(self):
        chat_id, ctype = _resolve_chat({"type": "user", "userId": "U123"})
        assert chat_id == "U123"
        assert ctype == "dm"

    def test_group_source(self):
        chat_id, ctype = _resolve_chat({"type": "group", "groupId": "C456", "userId": "U123"})
        assert chat_id == "C456"
        assert ctype == "group"


# ---------------------------------------------------------------------------
# 3. Three-allowlist gating
# ---------------------------------------------------------------------------

class TestAllowlist:

    def test_allow_all_short_circuits(self):
        for src in [
            {"type": "user", "userId": "Ufoo"},
            {"type": "group", "groupId": "Cfoo"},
            {"type": "room", "roomId": "Rfoo"},
        ]:
            assert _allowed_for_source(src, allow_all=True, user_ids=set(), group_ids=set(), room_ids=set())

    def test_user_in_allowlist_passes(self):
        src = {"type": "user", "userId": "Uok"}
        assert _allowed_for_source(src, allow_all=False, user_ids={"Uok"}, group_ids=set(), room_ids=set())


# ---------------------------------------------------------------------------
# 4. Inbound dedup
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 5. RequestCache state machine
# ---------------------------------------------------------------------------

class TestRequestCache:


    def test_mark_delivered_from_ready(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "x")
        c.mark_delivered(rid)
        assert c.get(rid).state is State.DELIVERED


    def test_set_ready_on_delivered_is_noop(self):
        c = RequestCache()
        rid = c.register_pending("Uchat")
        c.set_ready(rid, "first")
        c.mark_delivered(rid)
        c.set_ready(rid, "second")
        # DELIVERED is terminal — no further mutation
        assert c.get(rid).payload == "first"
        assert c.get(rid).state is State.DELIVERED

    def test_retire_moves_only_pending(self):
        c = RequestCache()
        pending, ready = c.register_pending("Uchat"), c.register_pending("Uchat")
        c.set_ready(ready, "earlier answer")
        c.retire(pending)
        c.retire(ready)
        assert c.get(pending).state is State.DELIVERED
        assert (c.get(ready).state, c.get(ready).payload) == (State.READY, "earlier answer")


# ---------------------------------------------------------------------------
# 6. Markdown stripping + chunking
# ---------------------------------------------------------------------------

class TestMarkdownAndChunking:

    def test_bold_stripped(self):
        assert strip_markdown_preserving_urls("**hello**") == "hello"

    def test_italic_stripped(self):
        assert strip_markdown_preserving_urls("*hello*") == "hello"


    def test_split_long_chunks_at_paragraph_boundary(self):
        text = "para1\n\npara2\n\npara3"
        chunks = split_for_line(text, max_chars=8)
        assert all(len(c) <= 8 for c in chunks), chunks
        assert len(chunks) >= 2

    def test_split_caps_at_five_chunks(self):
        # 1000 paragraphs of 100 chars each — must cap at 5 LINE bubbles.
        text = "\n\n".join(["x" * 100 for _ in range(1000)])
        chunks = split_for_line(text)
        assert len(chunks) <= 5


# ---------------------------------------------------------------------------
# 7. Inbound media normalization
# ---------------------------------------------------------------------------

class TestInboundMedia:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig

        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.fetch_content = AsyncMock(return_value=b"line-bytes")
        ad.handle_message = AsyncMock()
        return ad

    def _event(self, msg_type, **message):
        payload = {"type": msg_type, "id": f"{msg_type}-1"}
        payload.update(message)
        return {
            "type": "message",
            "replyToken": "reply-token",
            "source": {"type": "group", "groupId": "Cline", "userId": "Uline"},
            "message": payload,
        }

    def _captured_event(self, adapter):
        adapter.handle_message.assert_awaited_once()
        return adapter.handle_message.await_args.args[0]

    def test_image_message_uses_photo_type_and_image_mime(self, adapter):
        with patch.object(
            _line,
            "cache_image_from_bytes_async",
            new=AsyncMock(return_value="/cache/image.jpg"),
        ) as cache:
            asyncio.run(adapter._handle_message_event(self._event("image")))

        cache.assert_awaited_once_with(b"line-bytes", ext=".jpg")
        event = self._captured_event(adapter)
        assert event.message_type is _line.MessageType.PHOTO
        assert event.media_urls == ["/cache/image.jpg"]
        assert event.media_types == ["image/jpeg"]


# ---------------------------------------------------------------------------
# 8. Send routing (reply -> push fallback, batching, system-bypass)
# ---------------------------------------------------------------------------

class TestSendRouting:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={
            "channel_access_token": "tok",
            "channel_secret": "sec",
        })
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad



    def test_send_caps_messages_per_call_at_five(self, adapter):
        # Build a payload that would naturally split into more than 5 LINE
        # bubbles; the chunker should cap at 5 + truncate.
        big = "\n\n".join(["x" * 4500 for _ in range(20)])
        result = asyncio.run(adapter.send("Uchat", big))
        assert result.success
        call_kwargs = adapter._client.push.call_args
        # call_args is (args, kwargs); for our send the messages are the 2nd positional
        sent_messages = call_kwargs.args[1] if call_kwargs.args else call_kwargs.kwargs.get("messages")
        # Without args, fall back to inspecting the call shape
        if sent_messages is None:
            # We invoked client.push(chat_id, messages) — check first batch
            sent_messages = adapter._client.push.call_args.args[1]
        assert len(sent_messages) <= 5

    def test_format_message_strips_markdown(self, adapter):
        out = adapter.format_message("**bold** [link](https://x.com)")
        assert "**" not in out
        assert "https://x.com" in out


# ---------------------------------------------------------------------------
# 9. Register() metadata + plugin entry points
# ---------------------------------------------------------------------------

class TestRegister:

    class _FakeCtx:
        def __init__(self):
            self.kwargs = None

        def register_platform(self, **kw):
            self.kwargs = kw





    def test_max_message_length_below_line_per_bubble_limit(self):
        ctx = self._FakeCtx()
        register(ctx)
        # LINE per-bubble limit is 5000; we register 4500 to leave headroom.
        assert ctx.kwargs["max_message_length"] <= 5000


class TestSlowLLMPostbackRegression:
    """#106446 — the two reporter reproductions, driven through the real ``send()`` /
    ``_handle_postback_event()`` paths with only the HTTP client mocked."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"})
        ad = LineAdapter(cfg)
        ad._client = MagicMock()
        ad._client.reply = AsyncMock()
        ad._client.push = AsyncMock()
        return ad

    def test_repro_a_heartbeat_does_not_become_the_cached_answer(self, adapter):
        # The gateway's periodic heartbeat (stamped ``_interim_send`` by ``_interim_metadata``)
        # arrives while the postback button is PENDING; the real answer follows.
        rid = adapter._cache.register_pending("Uchat")
        adapter._pending_buttons["Uchat"] = rid
        asyncio.run(adapter.send("Uchat", "⏳ Working — 3 min — iteration 1/60, research_market",
                                 metadata={"_interim_send": True}))
        assert adapter._cache.get(rid).state is State.PENDING, "heartbeat must not fill the cache"
        assert adapter._client.push.await_count == 1, "heartbeat lands as a visible bubble"
        asyncio.run(adapter.send("Uchat", "Final answer: task completed"))
        assert adapter._cache.get(rid).payload == "Final answer: task completed"

    def test_repro_b_stale_mapping_does_not_swallow_the_next_answer(self, adapter):
        # A READY entry still mapped to the chat, then a tap on a button whose entry is gone:
        # neither may absorb a later send or leave the chat stuck.
        rid = adapter._cache.register_pending("Uchat")
        adapter._cache.set_ready(rid, "Previous answer")
        adapter._pending_buttons["Uchat"] = rid
        result = asyncio.run(adapter.send("Uchat", "Answer to the NEXT user message"))
        assert result.success and adapter._client.push.await_count == 1, "next answer reaches the wire"
        assert adapter._cache.get(rid).payload == "Previous answer"
        assert "Uchat" not in adapter._pending_buttons
        # Expired button tap: user gets a notice and the dead mapping is dropped.
        adapter._pending_buttons["Uchat"] = "ghost-rid"
        asyncio.run(adapter._handle_postback_event({
            "replyToken": "reply-token", "source": {"type": "user", "userId": "Uchat"},
            "postback": {"data": json.dumps({"action": "show_response", "request_id": "ghost-rid"})}}))
        adapter._client.reply.assert_awaited_once()
        assert adapter._client.reply.await_args.args[1][0]["text"] == adapter.expired_text
        assert "Uchat" not in adapter._pending_buttons


SK = "agent:main:line:dm:Uchat"


@pytest.fixture
def line_adapter(monkeypatch):
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
    from gateway.config import PlatformConfig
    ad = LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"}))
    ad._client = MagicMock()
    ad._client.reply = AsyncMock()
    ad._client.push = AsyncMock()
    ad._client.loading = AsyncMock()
    return ad


@pytest.fixture
def clarify():
    """Register a real pending clarify (``SK`` by default); returns its id."""
    from tools import clarify_gateway
    ids = iter(range(1, 100))
    keys = set()

    def _register(choices, multi_select=False, session_key=SK):
        cid = f"cl-{next(ids)}"
        clarify_gateway.register(cid, session_key, "Which region?", choices, multi_select)
        keys.add(session_key)
        return cid

    yield _register
    for key in keys:
        clarify_gateway.clear_session(key)


def _sent(client_call):
    """Messages of the single reply/push call the LINE client received."""
    client_call.assert_awaited_once()
    return client_call.await_args.args[1]


def _show_response(adapter, rid):
    asyncio.run(adapter._handle_postback_event({
        "replyToken": "tap-token", "source": {"type": "user", "userId": "Uchat"},
        "postback": {"data": json.dumps({"action": "show_response", "request_id": rid})}}))
    return adapter._client.reply.await_args.args[1][0]["text"]


class TestPromptPastSlowButton:
    """A clarify or approval prompt asks the user something, so it never hides behind the
    slow-LLM "Get answer" button; once it is on the wire the button holds nothing."""

    def _button(self, adapter, ready=None):
        rid = adapter._cache.register_pending("Uchat")
        if ready is not None:
            adapter._cache.set_ready(rid, ready)
        adapter._pending_buttons["Uchat"] = rid
        return rid

    def _ask(self, adapter, clarify):
        choices = ["Tokyo", "Osaka"]
        return asyncio.run(adapter.send_clarify("Uchat", "Which region?", choices, clarify(choices), SK))

    def test_clarify_reaches_the_wire_and_retires_the_button(self, line_adapter, clarify):
        rid = self._button(line_adapter)
        assert self._ask(line_adapter, clarify).success
        assert "Which region?" in _sent(line_adapter._client.push)[0]["text"]
        assert line_adapter._cache.get(rid).state is State.DELIVERED
        assert "Uchat" not in line_adapter._pending_buttons

    def test_answer_after_the_prompt_is_sent_not_cached(self, line_adapter, clarify):
        rid = self._button(line_adapter)
        self._ask(line_adapter, clarify)
        assert asyncio.run(line_adapter.send("Uchat", "Osaka it is")).success
        assert line_adapter._client.push.await_args.args[1][0]["text"] == "Osaka it is"
        assert line_adapter._cache.get(rid).payload is None

    def test_old_button_tap_after_the_prompt_says_already_replied(self, line_adapter, clarify):
        rid = self._button(line_adapter)
        self._ask(line_adapter, clarify)
        assert _show_response(line_adapter, rid) == line_adapter.delivered_text

    def test_failed_prompt_keeps_the_button_for_the_answer(self, line_adapter, clarify):
        rid = self._button(line_adapter)
        line_adapter._client.push.side_effect = RuntimeError("LINE push 429: monthly limit")
        assert not self._ask(line_adapter, clarify).success
        assert line_adapter._cache.get(rid).state is State.PENDING
        line_adapter._client.push.side_effect = None
        asyncio.run(line_adapter.send("Uchat", "final"))
        assert line_adapter._cache.get(rid).state is State.READY
        assert line_adapter._cache.get(rid).payload == "final"

    def test_ready_answer_survives_a_later_prompt(self, line_adapter, clarify):
        rid = self._button(line_adapter, ready="Previous answer")
        assert self._ask(line_adapter, clarify).success
        assert "Which region?" in _sent(line_adapter._client.push)[0]["text"]
        assert _show_response(line_adapter, rid) == "Previous answer"

    def test_exec_approval_text_retires_the_button(self, line_adapter):
        rid = self._button(line_adapter)
        asyncio.run(line_adapter.send("Uchat", "Approve `rm -rf build`? Reply /approve or /deny",
                                      metadata={"_interim_send": True, "is_approval_prompt": True}))
        assert "Approve" in _sent(line_adapter._client.push)[0]["text"]
        assert line_adapter._cache.get(rid).state is State.DELIVERED

    @pytest.mark.parametrize("paused", [True, False])
    def test_slow_button_does_not_fire_while_typing_is_paused(self, line_adapter, paused):
        line_adapter.slow_response_threshold = 0.01
        line_adapter._reply_tokens["Uchat"] = ("user-token", time.time() + 50)
        if paused:
            line_adapter.pause_typing_for_chat("Uchat")

        async def run():
            stop = asyncio.Event()
            typing = asyncio.create_task(line_adapter._keep_typing("Uchat", interval=0.01, stop_event=stop))
            if paused:
                await asyncio.sleep(0.2)  # health: allow HX013 -- the threshold elapsing with no button is the behaviour under test
            else:
                await async_wait_until(lambda: line_adapter._client.reply.await_count, timeout=5)
            stop.set()
            await typing

        asyncio.run(run())
        if paused:
            line_adapter._client.reply.assert_not_awaited()
            assert "Uchat" in line_adapter._reply_tokens, "the user's reply token is kept for the answer"
        else:
            token, messages = line_adapter._client.reply.await_args.args
            assert token == "user-token" and messages[0]["type"] == "template"


class TestClarifyQuickReply:
    """A single-select clarify shows one Quick Reply chip per choice. The chip posts the choice's
    number, so a tap and a typed number answer through the same gateway clarify text intercept."""

    CHOICES = ["Tokyo (Recommended)", "Osaka", "Use the Singapore region"]

    def _ask(self, adapter, cid, choices):
        return asyncio.run(adapter.send_clarify("Uchat", "Which region?", choices, cid, SK))

    def _answer(self, text):
        from tools import clarify_gateway
        return clarify_gateway.attempt_text_response_for_session(SK, text)

    def test_chip_per_choice_posts_its_number_under_a_clipped_label(self, line_adapter, clarify):
        self._ask(line_adapter, clarify(self.CHOICES), self.CHOICES)
        actions = [item["action"] for item in _sent(line_adapter._client.push)[-1]["quickReply"]["items"]]
        assert [a["type"] for a in actions] == ["message"] * 3
        assert [a["text"] for a in actions] == ["1", "2", "3"]
        assert [a["label"] for a in actions] == ["1. Tokyo (Recommend…", "2. Osaka", "3. Use the Singapor…"]

    def test_multiline_choice_gets_a_single_line_label(self, line_adapter, clarify):
        choices = ["Keep it\nas is", "Drop it"]
        self._ask(line_adapter, clarify(choices), choices)
        labels = [item["action"]["label"] for item in _sent(line_adapter._client.push)[-1]["quickReply"]["items"]]
        assert labels == ["1. Keep it as is", "2. Drop it"]

    @pytest.mark.parametrize("index", [0, 1, 2])
    def test_each_chip_answers_with_the_full_choice(self, line_adapter, clarify, index):
        from tools import clarify_gateway
        cid = clarify(self.CHOICES)
        entry = clarify_gateway.get_pending(cid)
        self._ask(line_adapter, cid, self.CHOICES)
        chip = _sent(line_adapter._client.push)[-1]["quickReply"]["items"][index]["action"]["text"]
        assert self._answer(chip) == clarify_gateway.TEXT_RESOLVED
        assert entry.response == self.CHOICES[index]

    def test_plugin_labels_get_a_chip_each(self, line_adapter, clarify):
        from tools import clarify_gateway
        labels = [f"label {i}" for i in range(1, 13)]
        cid = clarify(labels)
        entry = clarify_gateway.get_pending(cid)
        self._ask(line_adapter, cid, labels)
        items = _sent(line_adapter._client.push)[-1]["quickReply"]["items"]
        assert len(items) == 12
        assert self._answer(items[-1]["action"]["text"]) == clarify_gateway.TEXT_RESOLVED
        assert entry.response == "label 12"

    @pytest.mark.parametrize("count, chips", [(13, 13), (14, 0)])
    def test_past_thirteen_choices_only_the_numbered_text_goes_out(self, line_adapter, clarify, count, chips):
        choices = [f"choice {i}" for i in range(1, count + 1)]
        self._ask(line_adapter, clarify(choices), choices)
        messages = _sent(line_adapter._client.push)
        assert len(messages[-1].get("quickReply", {}).get("items", [])) == chips
        assert f"{count}. choice {count}" in "".join(m["text"] for m in messages)

    def test_multi_select_gets_no_chips_and_a_typed_list_answers(self, line_adapter, clarify):
        from tools import clarify_gateway
        cid = clarify(self.CHOICES, multi_select=True)
        entry = clarify_gateway.get_pending(cid)
        self._ask(line_adapter, cid, self.CHOICES)
        assert "quickReply" not in _sent(line_adapter._client.push)[-1]
        assert self._answer("1,3") == clarify_gateway.TEXT_RESOLVED
        assert json.loads(entry.response) == [self.CHOICES[0], self.CHOICES[2]]

    def test_open_ended_question_gets_no_chips(self, line_adapter, clarify):
        self._ask(line_adapter, clarify(None), None)
        message = _sent(line_adapter._client.push)[-1]
        assert "quickReply" not in message and "Which region?" in message["text"]

    @pytest.mark.parametrize("typed, expected", [("Somewhere in Europe", "Somewhere in Europe"), ("skip", None)])
    def test_typed_own_answer_or_skip_still_answers(self, line_adapter, clarify, typed, expected):
        from tools import clarify_gateway
        cid = clarify(self.CHOICES)
        entry = clarify_gateway.get_pending(cid)
        self._ask(line_adapter, cid, self.CHOICES)
        assert self._answer(typed) == clarify_gateway.TEXT_RESOLVED
        assert entry.response == (expected or clarify_gateway.SKIPPED)

    def test_clarify_rides_the_stashed_reply_token(self, line_adapter, clarify):
        line_adapter._reply_tokens["Uchat"] = ("user-token", time.time() + 50)
        self._ask(line_adapter, clarify(self.CHOICES), self.CHOICES)
        line_adapter._client.push.assert_not_awaited()
        assert line_adapter._client.reply.await_args.args[0] == "user-token"
        assert "quickReply" in _sent(line_adapter._client.reply)[-1]

    def test_chip_tap_arrives_as_text_and_its_token_carries_the_reply(self, line_adapter):
        line_adapter.handle_message = AsyncMock()
        asyncio.run(line_adapter._handle_message_event({
            "type": "message", "replyToken": "tap-token", "source": {"type": "user", "userId": "Uchat"},
            "message": {"type": "text", "id": "m1", "text": "2"}}))
        assert line_adapter.handle_message.await_args.args[0].text == "2"
        asyncio.run(line_adapter.send("Uchat", "Osaka it is"))
        assert line_adapter._client.reply.await_args.args[0] == "tap-token"

    def test_clarify_past_a_slow_button_carries_chips_and_retires_it(self, line_adapter, clarify):
        rid = line_adapter._cache.register_pending("Uchat")
        line_adapter._pending_buttons["Uchat"] = rid
        self._ask(line_adapter, clarify(self.CHOICES), self.CHOICES)
        assert len(_sent(line_adapter._client.push)[-1]["quickReply"]["items"]) == 3
        assert line_adapter._cache.get(rid).state is State.DELIVERED

    def test_another_group_member_tapping_does_not_answer(self, line_adapter, clarify):
        from tools import clarify_gateway

        def key(user_id):
            return line_adapter._source_session_key(
                line_adapter.build_source(chat_id="Cgrp", chat_type="group", user_id=user_id))

        cid = clarify(self.CHOICES, session_key=key("Uasker"))
        assert clarify_gateway.attempt_text_response_for_session(key("Ubystander"), "1") == clarify_gateway.TEXT_NO_PENDING
        assert clarify_gateway.get_pending(cid) is not None


class TestEnvEnablement:

    def test_returns_none_without_credentials(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert _env_enablement() is None


class TestStandaloneSend:

    def test_missing_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN", raising=False)
        from gateway.config import PlatformConfig
        for extra in ({}, {"channel_access_token": "   "}):  # a blank token never reaches the push API
            result = asyncio.run(_standalone_send(PlatformConfig(enabled=True, extra=extra), "Uchat", "hi"))
            assert "missing token" in result["error"]


class TestPostbackButtonShape:

    def test_template_buttons_structure(self):
        msg = build_postback_button_message("hi", "Tap me", "rid-1")
        assert msg["type"] == "template"
        assert msg["template"]["type"] == "buttons"
        assert msg["template"]["text"] == "hi"
        actions = msg["template"]["actions"]
        assert len(actions) == 1
        assert actions[0]["type"] == "postback"
        data = json.loads(actions[0]["data"])
        assert data == {"action": "show_response", "request_id": "rid-1"}


class TestCheckRequirements:


    def test_rejects_without_secret(self, monkeypatch):
        monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "t")
        monkeypatch.delenv("LINE_CHANNEL_SECRET", raising=False)
        assert not check_requirements()
        monkeypatch.setenv("LINE_CHANNEL_SECRET", "   ")
        assert not check_requirements()


class TestValidateConfig:

    def test_validates_from_extra(self):
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={"channel_access_token": "t", "channel_secret": "s"},
        )
        assert validate_config(cfg)


class TestAdapterInit:

    @pytest.mark.asyncio
    async def test_connect_fails_when_channel_lock_held(self, monkeypatch):
        """``acquire_scoped_lock`` returns ``(acquired, existing)``; a live foreign holder must stop
        connect() before the LINE client is built (the tuple is truthy, so a bare ``if not`` never fired)."""
        import gateway.status as gateway_status
        from gateway.config import PlatformConfig

        monkeypatch.setattr(
            gateway_status, "acquire_scoped_lock",
            lambda scope, identity, metadata=None: (False, {"pid": 4242, "profile": "other"}))
        ad = LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "tok", "channel_secret": "sec"}))
        assert await ad.connect() is False
        assert ad._fatal_error_code == "line_lock"
        assert "other" in ad._fatal_error_message
        assert ad._client is None

    def test_init_from_config_extra(self, monkeypatch):
        for k in ("LINE_CHANNEL_ACCESS_TOKEN", "LINE_CHANNEL_SECRET", "LINE_PORT"):
            monkeypatch.delenv(k, raising=False)
        from gateway.config import PlatformConfig
        cfg = PlatformConfig(
            enabled=True,
            extra={
                "channel_access_token": "tok",
                "channel_secret": "sec",
                "port": 7777,
                "public_url": "https://x.example.com",
                "allowed_users": ["U1", "U2"],
            },
        )
        ad = LineAdapter(cfg)
        assert ad.channel_access_token == "tok"
        assert ad.channel_secret == "sec"
        # A whitespace-only secret reads as unset, so connect fails closed instead of keying HMAC with blanks.
        # So do null and other non-strings, which must not become guessable keys like "None" or "True".
        for blank in ("   ", None, True, 0):
            assert LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": "tok",
                                                                   "channel_secret": blank})).channel_secret == ""
            assert LineAdapter(PlatformConfig(enabled=True, extra={"channel_access_token": blank,
                                                                   "channel_secret": "sec"})).channel_access_token == ""
        assert ad.webhook_port == 7777
        assert ad.public_base_url == "https://x.example.com"
        assert ad.allowed_users == {"U1", "U2"}


# ---------------------------------------------------------------------------
# 9. Inbound message-type classification
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# 10. Dual-stack bind default (NS-603)
# ---------------------------------------------------------------------------

class TestDualStackBind:
    """The LINE webhook server's default bind must serve BOTH IPv4 and IPv6.

    Regression guard for the hosted LINE 502 (NS-603): Fly.io 6PN — the
    private network the edge router reverse-proxies LINE ingest over — is
    IPv6-only (``<app>.internal`` resolves to an ``fdaa:…`` address). The
    adapter used to default to ``host="0.0.0.0"`` (IPv4 only), so the
    router's dial to ``<app>.internal:8646`` hit an address nothing was
    listening on → connection refused → 502 on webhook verification.

    Mirrors gateway/platforms/webhook.py's fix (commit d542894ad):
    ``DEFAULT_HOST = None`` → asyncio binds one socket per address family.
    ``"::"`` is NOT a valid substitute (bindv6only=1 on Fly machines makes
    it IPv6-only, breaking IPv4 loopback health probes).
    """

    def _cfg(self, **extra):
        from gateway.config import PlatformConfig
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return PlatformConfig(enabled=True, extra=base)


    def test_empty_host_normalises_to_none(self, monkeypatch):
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(host=""))
        assert ad.webhook_host is None


    @pytest.mark.asyncio
    async def test_default_bind_serves_both_families(self, monkeypatch):
        """Behavioural proof: host=None opens v4 AND v6 listening sockets."""
        # Guard: on IPv4-only hosts (CI runners with IPv6 disabled) the
        # dual-stack bind legitimately yields no v6 socket — that's the
        # environment, not a regression. Probe an actual ::1 bind rather
        # than trusting socket.has_ipv6 (compile-time constant).
        import socket as _socket
        if not _socket.has_ipv6:
            pytest.skip("IPv6 not supported by this Python build")
        try:
            _probe = _socket.socket(_socket.AF_INET6, _socket.SOCK_STREAM)
            try:
                _probe.bind(("::1", 0))
            finally:
                _probe.close()
        except OSError:
            pytest.skip("IPv6 stack unavailable on this host (cannot bind ::1)")
        monkeypatch.delenv("LINE_HOST", raising=False)
        ad = LineAdapter(self._cfg(port=0))
        ad._client = MagicMock()
        ad._client.get_bot_user_id = AsyncMock(return_value="Ubot")

        # Skip credential/network preamble — drive the aiohttp bind directly
        # the same way connect() does.
        from aiohttp import web
        ad._app = web.Application()
        ad._runner = web.AppRunner(ad._app)
        await ad._runner.setup()
        site = web.TCPSite(ad._runner, ad.webhook_host, 0)
        try:
            await site.start()
            addrs = list(ad._runner.addresses)
            has_v6 = any(len(a) == 4 for a in addrs)
            has_v4 = any(len(a) == 2 for a in addrs)
            assert has_v4, f"IPv4 bind missing — got {addrs}"
            assert has_v6, (
                f"IPv6 bind missing (the 6PN reachability bug, NS-603) — got {addrs}"
            )
        finally:
            await ad._runner.cleanup()


class TestMediaPublicUrlGuard:
    """Outbound media requires LINE_PUBLIC_URL whenever the bind host is not
    a publicly fetchable address — including the new dual-stack ``None``
    default and the legacy wildcard strings."""

    def _adapter(self, monkeypatch, **extra):
        from gateway.config import PlatformConfig
        monkeypatch.delenv("LINE_HOST", raising=False)
        monkeypatch.delenv("LINE_PUBLIC_URL", raising=False)
        base = {"channel_access_token": "tok", "channel_secret": "sec"}
        base.update(extra)
        return LineAdapter(PlatformConfig(enabled=True, extra=base))




    def test_send_image_blocked_without_public_url(self, monkeypatch, tmp_path):
        ad = self._adapter(monkeypatch)
        ad._client = MagicMock()
        img = tmp_path / "x.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n123")
        result = asyncio.run(ad.send_image_file("Uchat", str(img)))
        assert not result.success
        assert "LINE_PUBLIC_URL" in (result.error or "")

