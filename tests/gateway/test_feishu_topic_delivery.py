"""Topic routing contracts through the real Feishu SDK builders and public send APIs.

No live credentials: the SDK's transport methods are replaced, not routing or lookup.
"""
import asyncio
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter


def ok(**fields):
    return NS(success=lambda: True, data=NS(**fields))


def failed(code):
    return NS(success=lambda: False, code=code, msg="raw error SECRET must not reach diagnostic")


@pytest.fixture
def adapter():
    pytest.importorskip("lark_oapi")
    from plugins.platforms.feishu.adapter import _load_lark_oapi
    assert _load_lark_oapi()  # bind real reply/create/upload builders, not the SDK-absent fallback
    a = FeishuAdapter(PlatformConfig())
    a._client = Mock()
    a._client.im.v1.message.list.return_value = ok(items=[])
    a._client.im.v1.message.reply.return_value = ok(message_id="om_reply")
    a._client.im.v1.message.create.return_value = ok(message_id="om_created")
    a._client.im.v1.file.create.return_value = ok(file_key="file_uploaded")
    a._client.im.v1.image.create.return_value = ok(image_key="img_uploaded")
    yield a
    executor = getattr(a, "_sdk_executor", None)
    if executor:
        executor.shutdown(wait=True)


async def send_kind(adapter, kind, tmp_path, metadata, reply_to=None):
    common = dict(chat_id="oc_chat", reply_to=reply_to, metadata=metadata)
    if kind in {"text", "post", "status", "stream"}:
        text = "**ORIGINAL_SECRET**" if kind == "post" else "ORIGINAL_SECRET"
        return await adapter.send(content=text, **common)
    if kind == "image":
        path = tmp_path / "private_image.png"
        path.write_bytes(b"PNG")
        return await adapter.send_image_file(image_path=str(path), caption="ORIGINAL_SECRET", **common)
    path = tmp_path / ("private_voice.ogg" if "audio" in kind else "private_document.pdf")
    path.write_bytes(b"file")
    if "audio" in kind:
        return await adapter.send_voice(audio_path=str(path), caption="ORIGINAL_SECRET" if kind == "captioned_audio" else None,
                                        **common)
    return await adapter.send_document(file_path=str(path), caption="ORIGINAL_SECRET", **common)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["parent_chat", "error_notice", "silent"])
@pytest.mark.parametrize("kind", ["text", "post", "status", "stream", "image", "file", "audio", "captioned_audio"])
@pytest.mark.parametrize("failure", ["no_anchor", 230011, 231003, 99992402])
async def test_topic_policy_is_shared_by_every_payload_and_never_recurses(adapter, tmp_path, policy, kind, failure):
    adapter._topic_delivery_fallback = policy
    if isinstance(failure, int):
        adapter._client.im.v1.message.reply.return_value = failed(failure)
    metadata = {"thread_id": "omt_topic"}
    if failure != "no_anchor":
        metadata["reply_to_message_id"] = " om_old "
    result = await send_kind(adapter, kind, tmp_path, metadata)
    assert result.success is (policy == "parent_chat")
    assert result.retry_suppressed is (policy != "parent_chat")
    creates = adapter._client.im.v1.message.create.call_args_list
    assert len(creates) == (0 if policy == "silent" else 1)
    request = adapter._client.im.v1.message.list.call_args.args[0]
    from lark_oapi.api.im.v1 import CreateMessageRequest, ListMessageRequest
    assert isinstance(request, ListMessageRequest)
    assert (request.container_id_type, request.container_id, request.sort_type, request.page_size) == (
        "thread", "omt_topic", "ByCreateTimeDesc", 20)
    if creates:
        request = creates[0].args[0]
        assert isinstance(request, CreateMessageRequest)
        assert request.receive_id_type == "chat_id"
        assert request.request_body.receive_id == "oc_chat"
        if policy == "error_notice":
            diagnostic = json.loads(request.request_body.content)["text"]
            assert all(word in diagnostic for word in ("ref=", "code=", "stage=", "oc_chat", "omt_topic", "app_id="))
            assert all(word not in diagnostic for word in ("ORIGINAL_SECRET", "raw error", "private_", "file_uploaded"))
            assert result.message_id is None  # never let streaming edit the diagnostic into original content
    if policy != "parent_chat":
        # Same turn, copied metadata: no repeated lookup, notice, media upload or outer plain fallback.
        adapter._client.im.v1.message.create.side_effect = RuntimeError("notice retry forbidden")
        again = await adapter._send_with_retry("oc_chat", "OTHER_SECRET", metadata=dict(metadata))
        assert again is result
        assert adapter._client.im.v1.message.list.call_count == 1
        assert adapter._client.im.v1.message.create.call_count == len(creates)


@pytest.mark.asyncio
@pytest.mark.parametrize("anchor_source", ["explicit", "metadata", "missing"])
async def test_reanchor_uses_newest_candidates_excludes_failed_and_deleted_and_is_bounded(adapter, anchor_source):
    metadata = {"thread_id": "omt_topic"}
    reply_to = " om_old " if anchor_source == "explicit" else None
    if anchor_source == "metadata":
        metadata["reply_to_message_id"] = " om_old "
    adapter._client.im.v1.message.list.return_value = ok(items=[
        NS(message_id="om_deleted", deleted=True), NS(message_id="om_old"),
        NS(message_id="om_elsewhere", thread_id="omt_wrong"), NS(message_id="om_newest"),
        NS(message_id="om_next"), NS(message_id="om_last"), NS(message_id="om_beyond_bound"),
    ])
    replies = []
    def reply(request):
        replies.append(request.message_id)
        assert request.request_body.reply_in_thread is True
        return ok(message_id="om_sent") if request.message_id == "om_next" else failed(230011)
    adapter._client.im.v1.message.reply.side_effect = reply
    result = await adapter.send("oc_chat", "hello", reply_to=reply_to, metadata=metadata)
    assert result.success
    # The list contains the failed original anchor; it must never be selected twice.
    assert replies == ["om_old", "om_newest", "om_next"]
    adapter._client.im.v1.message.create.assert_not_called()
    assert adapter._client.im.v1.message.list.call_count == 1
    # Reuse the recovered anchor for following chunks/status/media, without searching again.
    await adapter.send("oc_chat", "next", reply_to=reply_to, metadata=metadata)
    assert replies[-1] == "om_next"
    assert adapter._client.im.v1.message.list.call_count == 1

    adapter._client.im.v1.message.reply.reset_mock()
    adapter._client.im.v1.message.reply.side_effect = lambda request: failed(231003)
    result = await adapter.send("oc_chat", "bounded", metadata={"thread_id": "omt_topic"})
    assert result.success  # parent_chat default, after exactly three candidates
    assert adapter._client.im.v1.message.reply.call_count == 3
    assert adapter._client.im.v1.message.create.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["reply", "reanchor"])
@pytest.mark.parametrize("error", [99991400, 99991663, 230001, "network"])
async def test_unrelated_failures_never_broaden_recipient(adapter, monkeypatch, stage, error):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    metadata = {"thread_id": "omt_topic"}
    if stage != "lookup":
        metadata["reply_to_message_id"] = "om_old"
    def outcome(request):
        if error == "network":
            raise TimeoutError("SECRET timeout")
        return failed(error)
    if stage == "lookup":
        adapter._client.im.v1.message.list.side_effect = outcome
    elif stage == "reply":
        adapter._client.im.v1.message.reply.side_effect = outcome
    else:
        adapter._client.im.v1.message.list.return_value = ok(items=[NS(message_id="om_new")])
        adapter._client.im.v1.message.reply.side_effect = lambda request: (
            failed(230011) if request.message_id == "om_old" else outcome(request))
    result = await adapter.send("oc_chat", "ORIGINAL_SECRET", metadata=metadata)
    assert not result.success
    assert not result.retry_suppressed
    adapter._client.im.v1.message.create.assert_not_called()
    if stage == "reply":
        adapter._client.im.v1.message.list.assert_not_called()
    if error == "network" and stage != "lookup":
        ids = [c.args[0].request_body.uuid for c in adapter._client.im.v1.message.reply.call_args_list
               if c.args[0].message_id == ("om_old" if stage == "reply" else "om_new")]
        assert len(ids) == 3 and len(set(ids)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("notice_failure", ["response", "exception"])
async def test_failed_notice_and_long_replies_stay_terminal_without_losing_parent_chat_chunks(adapter, notice_failure):
    adapter._topic_delivery_fallback = "error_notice"
    if notice_failure == "response":
        adapter._client.im.v1.message.create.return_value = failed(99991663)
    else:
        adapter._client.im.v1.message.create.side_effect = TimeoutError("SECRET")
    adapter.MAX_MESSAGE_LENGTH = 8
    result = await adapter.send("oc_chat", "first second third fourth", metadata={"thread_id": "omt_topic"})
    assert not result.success and result.retry_suppressed
    assert adapter._client.im.v1.message.create.call_count == 1
    assert adapter._client.im.v1.message.list.call_count == 1

    adapter._topic_delivery_fallback = "parent_chat"
    adapter._client.im.v1.message.create.reset_mock()
    adapter._client.im.v1.message.create.side_effect = None
    adapter._client.im.v1.message.create.return_value = ok(message_id="om_parent")
    content = "first second third fourth"
    result = await adapter.send("oc_chat", content, metadata={"thread_id": "omt_topic"})
    assert result.success
    payloads = [json.loads(c.args[0].request_body.content)["text"] for c in adapter._client.im.v1.message.create.call_args_list]
    assert payloads == adapter.truncate_message(content, adapter.MAX_MESSAGE_LENGTH)
    assert len(payloads) > 1


@pytest.mark.asyncio
async def test_concurrent_outputs_share_one_notice_but_other_topics_and_redirects_recover(adapter):
    adapter._topic_delivery_fallback = "error_notice"
    state = {}
    results = await asyncio.gather(*[
        adapter.send("oc_chat", f"secret {i}", metadata={"thread_id": "omt_topic", "_feishu_topic_delivery": state})
        for i in range(6)])
    assert all(r is results[0] and r.retry_suppressed for r in results)
    assert adapter._client.im.v1.message.create.call_count == 1
    assert adapter._client.im.v1.message.list.call_count == 1
    other = await adapter.send("oc_chat", "unrelated", reply_to="om_foreground", metadata={"thread_id": "omt_foreground"})
    assert other.success
    assert adapter._client.im.v1.message.reply.call_args.args[0].message_id == "om_foreground"

    adapter._topic_delivery_fallback = "parent_chat"
    md = {"thread_id": "omt_redirect"}
    await adapter.send("oc_chat", "before", reply_to="om_initial", metadata=md)
    await adapter.send("oc_chat", "after redirect", reply_to="om_redirected", metadata=md)
    assert adapter._client.im.v1.message.reply.call_args.args[0].message_id == "om_redirected"
    # A new explicit anchor after an earlier policy outcome can re-enter the topic.
    redirected = await adapter.send("oc_chat", "now recoverable", reply_to="om_new_anchor",
                                    metadata={"thread_id": "omt_topic", "_feishu_topic_delivery": state})
    assert redirected.success
    assert adapter._client.im.v1.message.reply.call_args.args[0].message_id == "om_new_anchor"


@pytest.mark.asyncio
async def test_independent_identical_sends_have_distinct_uuids_and_partial_failure_does_not_replay(adapter):
    metadata = {"thread_id": "omt_topic", "reply_to_message_id": "om_anchor"}
    await adapter.send("oc_chat", "identical", metadata=metadata)
    await adapter.send("oc_chat", "identical", metadata=metadata)
    requests = [c.args[0] for c in adapter._client.im.v1.message.reply.call_args_list]
    assert requests[0].request_body.uuid != requests[1].request_body.uuid

    adapter._client.im.v1.message.reply.reset_mock()
    adapter.MAX_MESSAGE_LENGTH = 20
    adapter._client.im.v1.message.reply.side_effect = [ok(message_id="om_head"), failed(230001)]
    result = await adapter._send_with_retry("oc_chat", "long message " * 20, metadata=metadata)
    assert not result.success
    assert result.raw_response["partial_overflow"] is True
    assert adapter._client.im.v1.message.reply.call_count == 2
    adapter._client.im.v1.message.create.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("seed_state", [False, True])
@pytest.mark.parametrize("initial_anchor", [None, "om_old"])
async def test_stream_overflow_after_parent_chat_fallback_never_creates_a_new_topic(adapter, seed_state, initial_anchor):
    from gateway.stream_consumer import GatewayStreamConsumer

    metadata = {"thread_id": "omt_original"}
    if seed_state:
        metadata["_feishu_topic_delivery"] = {}
    if initial_anchor:
        metadata["reply_to_message_id"] = initial_anchor
        adapter._client.im.v1.message.reply.return_value = failed(230011)
    adapter._client.im.v1.message.create.side_effect = [ok(message_id="om_parent_1"), ok(message_id="om_parent_2")]
    consumer = GatewayStreamConsumer(adapter, "oc_chat", metadata=metadata, initial_reply_to_id=initial_anchor)
    first = await consumer._send_new_chunk("first chunk", initial_anchor)
    second = await consumer._send_new_chunk("second chunk", first)
    assert (first, second) == ("om_parent_1", "om_parent_2")
    assert adapter._client.im.v1.message.create.call_count == 2
    assert adapter._client.im.v1.message.reply.call_count == int(initial_anchor is not None)
    assert adapter._client.im.v1.message.list.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["parent_chat", "error_notice", "silent"])
@pytest.mark.parametrize("lookup_failure", [99991400, 99991663, "network"])
async def test_history_lookup_failure_uses_policy_without_treating_it_as_a_send_failure(adapter, monkeypatch, policy, lookup_failure):
    adapter._topic_delivery_fallback = policy
    if lookup_failure == "network":
        adapter._client.im.v1.message.list.side_effect = TimeoutError()
    else:
        adapter._client.im.v1.message.list.return_value = failed(lookup_failure)
    result = await adapter.send("oc_chat", "ORIGINAL_SECRET", metadata={"thread_id": "omt_topic"})
    assert result.success is (policy == "parent_chat")
    assert result.retry_suppressed is (policy != "parent_chat")
    assert adapter._client.im.v1.message.list.call_count == 1
    assert adapter._client.im.v1.message.create.call_count == int(policy != "silent")
    adapter._client.im.v1.message.reply.assert_not_called()
    if policy == "error_notice":
        notice = adapter._client.im.v1.message.create.call_args.args[0].request_body.content
        assert "lookup_failed" in notice and "ORIGINAL_SECRET" not in notice and "raw error" not in notice


@pytest.mark.asyncio
async def test_bare_send_timeout_is_not_replayed_as_plaintext_and_disconnected_adapter_is_honest(adapter, monkeypatch):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    adapter._client.im.v1.message.reply.side_effect = TimeoutError()
    result = await adapter._send_with_retry(
        "oc_chat", "ORIGINAL_SECRET", metadata={"thread_id": "omt_topic", "reply_to_message_id": "om_anchor"})
    assert not result.success and not result.retry_suppressed
    assert "TimeoutError" in result.error
    calls = adapter._client.im.v1.message.reply.call_args_list
    assert len(calls) == 3
    assert len({c.args[0].request_body.uuid for c in calls}) == 1
    adapter._client.im.v1.message.create.assert_not_called()
    adapter._client.im.v1.message.list.assert_not_called()
    adapter._client = None
    result = await adapter.send("oc_chat", "hello", metadata={"thread_id": "omt_topic"})
    assert not result.success and result.error == "Not connected"


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["parent_chat", "error_notice", "silent"])
@pytest.mark.parametrize("code", [230011, 99991663])
async def test_silent_policy_cleans_processing_badge_without_exposing_failure_reaction(adapter, policy, code):
    from unittest.mock import AsyncMock
    from gateway.config import Platform
    from gateway.platforms.base_thread_metadata import _thread_metadata_for_event
    from gateway.platforms.event import MessageEvent, ProcessingOutcome
    from gateway.session import SessionSource

    adapter._topic_delivery_fallback = policy
    adapter._client.im.v1.message.reply.return_value = failed(code)
    adapter._reactions_enabled = lambda: True
    adapter._add_reaction = AsyncMock(return_value="reaction_typing")
    adapter._remove_reaction = AsyncMock(return_value=True)
    event = MessageEvent(text="user request", message_id="om_user", source=SessionSource(
        platform=Platform.FEISHU, chat_id="oc_chat", chat_type="group", thread_id="omt_topic", message_id="om_user"))
    await adapter.on_processing_start(event)
    await adapter.send("oc_chat", "ORIGINAL_SECRET", metadata=_thread_metadata_for_event(event))
    await adapter.on_processing_complete(event, ProcessingOutcome.FAILURE)
    adapter._remove_reaction.assert_awaited_once_with("om_user", "reaction_typing")
    assert "om_user" not in adapter._pending_processing_reactions
    reactions = [c.args[1] for c in adapter._add_reaction.await_args_list]
    assert reactions == (["Typing"] if policy == "silent" and code == 230011 else ["Typing", "CrossMark"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [99991663, "timeout"])
async def test_failed_parent_chat_fallback_is_honest_and_never_reenters_topic_recovery(adapter, monkeypatch, failure):
    async def no_sleep(_):
        pass
    monkeypatch.setattr("plugins.platforms.feishu.adapter_delivery.asyncio.sleep", no_sleep)
    if failure == "timeout":
        adapter._client.im.v1.message.create.side_effect = TimeoutError()
    else:
        adapter._client.im.v1.message.create.return_value = failed(failure)
    result = await adapter._send_with_retry("oc_chat", "ORIGINAL_SECRET", metadata={"thread_id": "omt_topic"})
    assert not result.success and not result.retry_suppressed
    assert adapter._client.im.v1.message.list.call_count == 1
    adapter._client.im.v1.message.reply.assert_not_called()
    creates = adapter._client.im.v1.message.create.call_args_list
    assert len(creates) == (3 if failure == "timeout" else 2)
    for call in creates:
        request = call.args[0]
        assert request.receive_id_type == "chat_id" and request.request_body.receive_id == "oc_chat"
        assert "ORIGINAL_SECRET" in request.request_body.content
        assert "Feishu topic delivery failed" not in request.request_body.content
    if failure == "timeout":
        assert len({c.args[0].request_body.uuid for c in creates}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["home", "lookup", "notice"])
async def test_topic_exception_logs_keep_traceback_but_redact_sdk_text(adapter, monkeypatch, caplog, boundary):
    secret = "PRIVATE_PAYLOAD credential=not-for-logs"

    def fail(*args, **kwargs):
        try:
            raise ValueError(secret)
        except ValueError as cause:
            raise RuntimeError(secret) from cause

    state = {}
    if boundary == "home":
        adapter._topic_delivery_fallback = "parent_then_home"
        adapter._client.im.v1.message.create.return_value = failed(232009)
        monkeypatch.setattr(adapter, "_resolve_topic_home", fail)
        result = await adapter._send_to_topic_fallback(
            chat_id="oc_chat", msg_type="text", payload="body", state=state, metadata={})
    elif boundary == "lookup":
        adapter._client.im.v1.message.list.side_effect = fail
        result = await adapter._list_topic_reply_anchors("omt_topic", set())
    else:
        adapter._topic_delivery_fallback = "error_notice"
        adapter._client.im.v1.message.create.side_effect = fail
        result = await adapter._apply_topic_delivery_fallback(
            chat_id="oc_chat", thread_id="omt_topic", anchor=None, msg_type="text", payload="body",
            state=state, code="missing_anchor", stage="resolve_anchor", metadata={})
    assert not result.success
    assert secret not in str(result.error)
    assert secret not in caplog.text
    records = [record for record in caplog.records if record.exc_info]
    assert len(records) == 1
    _, sanitized, traceback = records[0].exc_info
    assert str(sanitized) == "RuntimeError"
    assert sanitized.__context__ is None and sanitized.__cause__ is None
    assert traceback is not None
