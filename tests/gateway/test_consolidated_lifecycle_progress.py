"""Acknowledged lifecycle events exercise the native queue and turn owners offline."""

import asyncio
import queue
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest

import gateway.run as gateway_run
from agent.conversation_compression import ROUTINE_COMPRESSION_STATUS_SAMPLES, COMPACTION_DONE_STATUS
from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.run_turn_runner import TurnRunner
from gateway.session import SessionSource
from gateway.turn_context import TurnContext
from tests.gateway import test_run_progress_topics as progress


class Capture(progress.MetadataEditProgressCaptureAdapter):
    def __init__(self, platform=Platform.TELEGRAM):
        super().__init__(platform)
        self.send_ok = True
        self.edit_ok = True
        self.visible = {}
        self.entered = asyncio.Event()
        self.release = None

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append(dict(chat_id=chat_id, content=content, reply_to=reply_to, metadata=metadata))
        self.entered.set()
        if self.release is not None:
            await self.release.wait()
        mid = f"p{len(self.sent)}"
        if self.send_ok:
            self.visible[mid] = content
        return SendResult(success=self.send_ok, message_id=mid if self.send_ok else None)

    async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        self.edits.append(dict(chat_id=chat_id, message_id=message_id, content=content, metadata=metadata))
        if self.edit_ok:
            self.visible[message_id] = content
        return SendResult(success=self.edit_ok, message_id=message_id if self.edit_ok else None)


def rail(adapter=None, mode="full"):
    adapter = adapter or Capture()
    owner = progress._make_runner(adapter)
    source = SessionSource(platform=adapter.platform, chat_id="chat", chat_type="group", thread_id="topic")
    ctx = TurnContext(
        source=source, progress_mode=mode, progress_grouping="accumulate",
        progress_queue=queue.Queue(), _run_still_current=lambda: True,
        _progress_metadata={"thread_id": "topic"}, _progress_reply_to="anchor",
        _cleanup_progress=True, _status_adapter=adapter, _status_chat_id="chat",
        _status_thread_metadata={"thread_id": "topic"}, _loop_for_step=asyncio.get_running_loop(),
    )
    return owner, TurnRunner(owner, ctx), ctx, adapter


async def stop(task):
    task.cancel()
    with suppress(asyncio.CancelledError):
        await asyncio.wait_for(task, 4)


async def terminal(ctx, text):
    ack = asyncio.get_running_loop().create_future()
    ctx.progress_queue.put(("__terminal__", text, ack))
    return await asyncio.wait_for(asyncio.shield(ack), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "all"])
@pytest.mark.parametrize("edit_ok,send_ok", [(True, True), (False, False), (False, True)])
async def test_terminal_ack_covers_exact_adapter_payload(mode, edit_ok, send_ok):
    _, tr, ctx, adapter = rail(mode=mode)
    task = asyncio.create_task(tr.send_progress_messages())
    try:
        assert await terminal(ctx, "acknowledged prefix") is True
        adapter.edit_ok, adapter.send_ok = edit_ok, send_ok
        assert await terminal(ctx, "terminal outcome") is (edit_ok or send_ok)
        if edit_ok:
            assert adapter.visible["p1"] == "acknowledged prefix\nterminal outcome"
        else:
            assert adapter.sent[-1]["content"] == "terminal outcome"
        assert adapter.sent[0]["reply_to"] == "anchor"
        assert adapter.sent[0]["metadata"]["thread_id"] == "topic"
        assert adapter.edits[-1]["metadata"]["thread_id"] == "topic"
        assert ctx._cleanup_msg_ids == (["p1", "p2"] if not edit_ok and send_ok else ["p1"])
    finally:
        await stop(task)


@pytest.mark.asyncio
async def test_keyed_heartbeat_moves_to_tail_and_stays_mutable_after_rollover():
    _, tr, ctx, adapter = rail()
    adapter.MAX_MESSAGE_LENGTH = 48
    task = asyncio.create_task(tr.send_progress_messages())
    try:
        ctx.progress_queue.put(("__upsert__", "heartbeat", "working old"))
        await asyncio.wait_for(adapter.entered.wait(), 1)
        ctx.progress_queue.put(("__full__", "A" * 24))
        ctx.progress_queue.put(("__upsert__", "heartbeat", "working new"))
        ctx.progress_queue.put(("__full__", "B" * 24))
        assert await terminal(ctx, "done") is True
        payloads = list(adapter.visible.values())
        assert all(len(text) <= 48 for text in payloads)
        assert "working old" not in "\n".join(payloads)
        assert payloads[-1].endswith("working new\ndone")
        assert "\n".join(payloads).count("working new") == 1
        assert "\n".join(payloads).count("A" * 24) == 1
        assert "\n".join(payloads).count("B" * 24) == 1
    finally:
        await stop(task)


@pytest.mark.asyncio
@pytest.mark.parametrize("state", ["current", "stale", "interrupted"])
async def test_idle_flush_and_dropped_ack(state):
    _, tr, ctx, adapter = rail(mode="all")
    task = asyncio.create_task(tr.send_progress_messages())
    await asyncio.sleep(0)
    if state == "stale":
        ctx._run_still_current = lambda: False
    elif state == "interrupted":
        ctx.agent_holder[0] = SimpleNamespace(is_interrupted=True)
    ack = asyncio.get_running_loop().create_future()
    flush = asyncio.get_running_loop().create_future()
    ctx.progress_queue.put(("__terminal__", "last entry", ack))
    ctx.progress_queue.put(("__flush__", flush))
    await stop(task)
    assert ack.done() and ack.result() is (state == "current")
    assert flush.done() and flush.result() is (state == "current")
    assert bool(adapter.sent) is (state == "current")


@pytest.mark.asyncio
async def test_repeated_cancellation_keeps_exact_inflight_send_and_ordered_receipts():
    _, tr, ctx, adapter = rail()
    adapter.release = asyncio.Event()
    first, last = [asyncio.get_running_loop().create_future() for _ in range(2)]
    ctx.progress_queue.put(("__terminal__", "first", first))
    ctx.progress_queue.put(("__terminal__", "last", last))
    task = asyncio.create_task(tr.send_progress_messages())
    await asyncio.wait_for(adapter.entered.wait(), 1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    adapter.release.set()
    await asyncio.wait_for(task, 3)
    assert first.done() and first.result() is True
    assert last.done() and last.result() is True
    assert [x["content"] for x in adapter.sent] == ["first"]
    assert list(adapter.visible.values()) == ["first\nlast"]
    assert ctx._cleanup_msg_ids == ["p1"]


@pytest.mark.asyncio
async def test_cleanup_fences_callback_already_inside_ingress(monkeypatch):
    owner, tr, ctx, adapter = rail(mode="all")
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"compression": {"progress_notices": True}})
    entered, release = threading.Event(), threading.Event()
    native_prepare = gateway_run._prepare_gateway_status_message

    def prepare(*args):
        entered.set()
        assert release.wait(2)
        return native_prepare(*args)

    monkeypatch.setattr(gateway_run, "_prepare_gateway_status_message", prepare)
    callback = asyncio.create_task(asyncio.to_thread(tr._status_callback_sync, "info", ROUTINE_COMPRESSION_STATUS_SAMPLES[0]))
    assert await asyncio.to_thread(entered.wait, 1)
    close = asyncio.create_task(asyncio.to_thread(tr.close_progress_ingress))
    await asyncio.sleep(0.02)
    assert not close.done()
    release.set()
    await callback
    await close
    tr._status_callback_sync("compacted", COMPACTION_DONE_STATUS)
    task = asyncio.create_task(tr.send_progress_messages())
    await asyncio.sleep(0)
    await stop(task)
    assert len(adapter.sent) == 1
    assert COMPACTION_DONE_STATUS not in "\n".join(adapter.visible.values())


@pytest.mark.asyncio
async def test_native_lifecycle_only_statuses_create_and_share_queue(monkeypatch, tmp_path):
    class Agent(progress.FakeAgent):
        def run_conversation(self, *args, **kwargs):
            self.status_callback("info", ROUTINE_COMPRESSION_STATUS_SAMPLES[0])
            self.status_callback("compacted", COMPACTION_DONE_STATUS)
            self.status_callback("warn", "Retrying in 4.2s (attempt 1/3)...")
            return {"final_response": "done", "messages": [], "api_calls": 1}

    adapter, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="lifecycle-only", adapter_cls=Capture,
        config_data={"compression": {"progress_notices": True}, "display": {
            "tool_progress": "off", "thinking_progress": False, "long_running_notifications": False,
        }},
    )
    assert result["final_response"] == "done"
    assert len(adapter.sent) == 1
    assert COMPACTION_DONE_STATUS in adapter.visible["p1"]
    assert "Retrying" not in adapter.visible["p1"]


@pytest.mark.asyncio
async def test_native_heartbeat_only_is_visible_before_worker_completion(monkeypatch, tmp_path):
    visible = threading.Event()

    class Adapter(Capture):
        async def send(self, *args, **kwargs):
            result = await super().send(*args, **kwargs)
            visible.set()
            return result

    class Agent(progress.FakeAgent):
        def run_conversation(self, *args, **kwargs):
            assert visible.wait(2), "heartbeat was buffered until turn completion"
            return {"final_response": "done", "messages": [], "api_calls": 1}

    original = gateway_run.GatewayRunner._run_agent_notify_long_running
    observed = {}

    async def notify(self, disp, ctx, holder):
        observed["ctx"] = ctx
        await original(self, disp, ctx, holder)

    monkeypatch.setenv("HERMES_AGENT_NOTIFY_INTERVAL", "0.03")
    monkeypatch.setattr(gateway_run.GatewayRunner, "_should_emit_long_running_notification", lambda *a: True)
    monkeypatch.setattr(gateway_run.GatewayRunner, "_run_agent_notify_long_running", notify)
    adapter, _ = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="heartbeat-only", adapter_cls=Adapter,
        config_data={"display": {"tool_progress": "off", "thinking_progress": False}},
    )
    assert observed["ctx"].needs_progress_queue
    assert observed["ctx"].progress_queue is not None
    assert adapter.sent[0]["metadata"]["thread_id"] == "17585"


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["separate", "noneditable", "api", "raw", "taskcard"])
async def test_native_status_fallback_surfaces_keep_standalone_status(monkeypatch, surface):
    adapter = progress.FailingNativeTaskCardAdapter() if surface == "taskcard" else Capture()
    owner, tr, ctx, adapter = rail(adapter, mode="all")
    if surface == "separate":
        ctx.progress_grouping = "separate"
    elif surface == "noneditable":
        adapter.SUPPORTS_MESSAGE_EDITING = False
    elif surface in {"api", "raw"}:
        ctx.source.platform = Platform.API_SERVER if surface == "api" else Platform.LOCAL
    elif surface == "taskcard":
        ctx._native_slack_task_cards = True
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"compression": {"progress_notices": True}})
    tr._status_callback_sync("compacted", COMPACTION_DONE_STATUS)
    await asyncio.sleep(0.05)
    assert ctx.progress_queue.empty()
    assert adapter.sent[-1]["content"] == COMPACTION_DONE_STATUS
    assert adapter.sent[-1]["metadata"]["thread_id"] == "topic"
    if surface == "taskcard":
        tr.native_tool_start_callback("call-1", "web_search", {"query": "safe"})
        task = asyncio.create_task(tr.send_progress_messages())
        await asyncio.sleep(0.05)
        await stop(task)
        assert adapter.native_updates
        assert "web_search" in adapter.sent[-1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["send", "edit", "multipart_send", "multipart_edit", "chunks", "failed_ack", "ordinary"])
async def test_native_terminal_survives_real_post_delivery_cleanup(monkeypatch, tmp_path, delivery):
    captured = {}
    native_make = progress._make_runner

    def make(adapter):
        owner = native_make(adapter)
        native_run = owner._run_agent
        native_wait = owner._run_agent_await_turn_worker

        async def wait(worker, ctx, *args):
            response = await native_wait(worker, ctx, *args)
            # Real queue entries and flush acknowledgements establish temporary
            # bubbles before the native exhaustion/ordinary completion branches.
            async def append(text):
                ctx._enqueue_lifecycle_progress(("__append__", text))
                ack = asyncio.get_running_loop().create_future()
                ctx._enqueue_lifecycle_progress(("__flush__", ack))
                assert await asyncio.wait_for(ack, 2) is True

            await append("temporary progress")
            captured["temporary"] = set(adapter.visible)
            ctx._enqueue_lifecycle_progress(("__reset__",))
            if delivery in {"edit", "multipart_edit"}:
                await append("editable prefix")
                captured["edited_id"] = next(iter(set(adapter.visible) - captured["temporary"]))
            captured["ctx"] = ctx
            return response

        async def run(*args, **kwargs):
            generation = owner._begin_session_run_generation(kwargs["session_key"])
            result = await native_run(*args, **kwargs, run_generation=generation, defer_terminal_lifecycle_progress=True)
            captured.update(owner=owner, source=kwargs["source"], key=kwargs["session_key"], generation=generation)
            return result

        owner._run_agent_await_turn_worker = wait
        owner._run_agent = run
        return owner

    class Adapter(Capture):
        def __init__(self, platform=Platform.TELEGRAM):
            super().__init__(platform)
            self.deleted = []
            self.deletion_done = asyncio.Event()
            if delivery == "chunks":
                self.MAX_MESSAGE_LENGTH = 96

        def multipart(self, result, content, *, edit=False):
            primary = result.message_id
            ids = [primary, primary + "-middle", primary + "-last"]
            first, second = len(content) // 3, 2 * len(content) // 3
            self.visible.update(zip(ids, [content[:first], content[first:second], content[second:]]))
            # The receipt explicitly acknowledges every terminal-bearing ID,
            # including the edited target rather than only its continuations.
            return SendResult(
                success=True, message_id=ids[-1] if edit else primary,
                raw_response={"message_ids": ids},
                continuation_message_ids=tuple(ids[1:]),
            )

        async def send(self, chat_id, content, reply_to=None, metadata=None):
            result = await super().send(chat_id, content, reply_to, metadata)
            if delivery == "multipart_send" and "Session auto-reset" in content:
                return self.multipart(result, content)
            return result

        async def edit_message(self, chat_id, message_id, content, *, finalize=False, metadata=None):
            result = await super().edit_message(chat_id, message_id, content, finalize=finalize, metadata=metadata)
            if delivery == "multipart_edit" and "Session auto-reset" in content:
                return self.multipart(result, content, edit=True)
            return result

        async def delete_message(self, chat_id, message_id):
            self.deleted.append(message_id)
            self.visible.pop(message_id, None)
            self.deletion_done.set()
            return True

    class Agent(progress.FakeAgent):
        def run_conversation(self, *args, **kwargs):
            result = {"final_response": "ordinary final" if delivery == "ordinary" else "context exhausted", "messages": [], "api_calls": 1}
            if delivery != "ordinary":
                result["compression_exhausted"] = True  # Deliberately without failed=True.
            return result

    monkeypatch.setattr(progress, "_make_runner", make)
    adapter, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="terminal-retention", adapter_cls=Adapter,
        config_data={"display": {"tool_progress": "off", "long_running_notifications": False, "cleanup_progress": True}},
    )
    complete = result.get("_complete_lifecycle_progress")
    assert callable(complete) is (delivery != "ordinary")
    owner = captured["owner"]
    owner.session_store.reset_session = lambda key: None
    if delivery == "failed_ack":
        adapter.send_ok = False
    response, _ = await owner._hmwa_compression_exhaustion_reset(
        result, result["final_response"], SimpleNamespace(session_id="old"), captured["key"], captured["source"],
    )
    acknowledged = delivery not in {"failed_ack", "ordinary"}
    assert (response == "") is acknowledged
    if complete:
        before = (list(adapter.sent), list(adapter.edits))
        assert await complete() is acknowledged
        assert (adapter.sent, adapter.edits) == before
    if response:
        assert ("Session auto-reset" in response) is (delivery == "failed_ack")
        adapter.send_ok = True
        assert (await adapter.send(captured["source"].chat_id, response)).success
    final_ids = set(adapter.visible) - captured["temporary"]
    assert final_ids
    if "edited_id" in captured:
        assert captured["edited_id"] in final_ids
    if delivery in {"multipart_send", "multipart_edit", "chunks"}:
        assert len(final_ids) > 1
    surviving = {mid: text for mid, text in adapter.visible.items() if mid in final_ids}
    assert "Session auto-reset" in "".join(surviving.values()) or delivery == "ordinary"
    # Exercise the base adapter's real registration, generation-aware pop and
    # callback firing, including the acknowledged empty-response path.
    event = asyncio.Event()
    event._hermes_run_generation = captured["generation"]
    await adapter._fire_post_delivery_callback(captured["key"], event)
    await asyncio.wait_for(adapter.deletion_done.wait(), 2)
    assert set(adapter.deleted) == captured["temporary"]
    assert adapter.visible == surviving
    assert adapter.pop_post_delivery_callback(captured["key"]) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["declined_code", "declined_text", "failed_edit", "replacement", "partial_edit"])
async def test_relay_terminal_receipt_controls_fallback_and_cleanup(monkeypatch, tmp_path, outcome):
    from gateway.config import PlatformConfig
    from gateway.relay.adapter import RelayAdapter
    from gateway.relay.egress import declined_send
    from tests.gateway.relay.stub_connector import StubConnector
    from tests.gateway.relay.test_relay_live_cards import make_desc

    desc = make_desc(platform="telegram", supports_draft_streaming=False,
                     supported_ops=("send", "edit", "delete", "typing"))
    captured = {}

    class Connector(StubConnector):
        def __init__(self):
            super().__init__(desc)
            self.visible = {}
            self.deleted = asyncio.Event()

        async def send_outbound(self, action, *, platform=None):
            self.sent.append(action)
            op = action["op"]
            if op == "send":
                mid = f"m{len(self.visible) + 1}"
                self.visible[mid] = action["content"]
                return {"success": True, "message_id": mid}
            if op == "edit":
                if "Session auto-reset" not in action["content"]:
                    self.visible[action["message_id"]] = action["content"]
                    return {"success": True, "message_id": action["message_id"]}
                if outcome == "declined_code":
                    return {"success": False, "code": "egress_declined"}
                if outcome == "declined_text":
                    return {"success": False, "error": "telegram egress declined: target is not an approved destination for this connection"}
                if outcome == "failed_edit":
                    return {"success": False, "error": "editing unavailable"}
                self.visible["m2"] = action["content"]
                # A successful replacement receipt names m2 even while the
                # consumer's cached edit target still names temporary m1.
                if outcome == "replacement":
                    return {"success": True, "message_id": "m2"}
                self.visible["m2"] = action["content"][:80]
                return {"success": False, "message_ids": ["m2"], "error": "continuation failed"}
            if op == "delete":
                self.visible.pop(action["message_id"], None)
                self.deleted.set()
            return {"success": True}

    wire = Connector()
    adapter = RelayAdapter(PlatformConfig(), desc, transport=wire)
    native_make = progress._make_runner

    def make(actual_adapter):
        owner = native_make(actual_adapter)
        native_run, native_wait = owner._run_agent, owner._run_agent_await_turn_worker

        async def wait(worker, ctx, *args):
            result = await native_wait(worker, ctx, *args)
            ctx._enqueue_lifecycle_progress(("__append__", "temporary progress"))
            ack = asyncio.get_running_loop().create_future()
            ctx._enqueue_lifecycle_progress(("__flush__", ack))
            assert await asyncio.wait_for(ack, 2) is True
            captured["ctx"] = ctx
            captured["terminal_start"] = len(wire.sent)
            return result

        async def run(**kwargs):
            kwargs["source"].delivered_via_upstream_relay = True
            generation = owner._begin_session_run_generation(kwargs["session_key"])
            captured.update(owner=owner, source=kwargs["source"], key=kwargs["session_key"], generation=generation)
            return await native_run(**kwargs, run_generation=generation, defer_terminal_lifecycle_progress=True)

        owner._run_agent, owner._run_agent_await_turn_worker = run, wait
        return owner

    class Agent(progress.FakeAgent):
        def run_conversation(self, *args, **kwargs):
            return {"final_response": "context exhausted", "compression_exhausted": True, "messages": [], "api_calls": 1}

    monkeypatch.setattr(progress, "_make_runner", make)
    _, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="relay-terminal-receipts",
        adapter_cls=lambda platform: adapter,
        config_data={"display": {"tool_progress": "off", "long_running_notifications": False, "cleanup_progress": True}},
    )
    complete = result["_complete_lifecycle_progress"]
    owner = captured["owner"]
    owner.session_store.reset_session = lambda key: None
    response, _ = await owner._hmwa_compression_exhaustion_reset(
        result, result["final_response"], SimpleNamespace(session_id="old"), captured["key"], captured["source"],
    )
    refused = outcome.startswith("declined")
    assert (response == "") is (outcome != "partial_edit")
    if refused:
        assert [a["op"] for a in wire.sent[captured["terminal_start"]:] if a["op"] in {"send", "edit"}] == ["edit"]
        assert declined_send(await complete()), "a refusal must not become success or a retryable missing acknowledgement"
    else:
        assert await complete() is (outcome != "partial_edit")
        assert "Session auto-reset" in wire.visible["m2"]
        if outcome == "replacement":
            assert all(a["message_id"] == "m1" for a in wire.sent if a["op"] == "edit")
    assert captured["ctx"]._terminal_progress_msg_ids == (set() if refused else {"m2"})
    # The real base callback fires for an empty final as well as for a delivered
    # ordinary fallback. Neither case may preserve the unchanged edit target.
    if response:
        assert (await adapter.send(captured["source"].chat_id, response)).success
    event = asyncio.Event()
    event._hermes_run_generation = captured["generation"]
    await adapter._fire_post_delivery_callback(captured["key"], event)
    await asyncio.wait_for(wire.deleted.wait(), 2)
    assert [a["message_id"] for a in wire.sent if a["op"] == "delete"] == ["m1"]
    expected = set() if refused else {"m2", "m3"} if response else {"m2"}
    assert set(wire.visible) == expected


@pytest.mark.asyncio
async def test_native_cleanup_joins_heartbeat_before_closing_consumer():
    owner, tr, ctx, adapter = rail()
    owner._draining = False
    ctx._close_progress_ingress = tr.close_progress_ingress
    ctx._complete_lifecycle_progress = tr.complete_lifecycle_progress
    tr._progress_task = asyncio.create_task(tr.send_progress_messages())
    started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def notify():
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await release.wait()
            tr.enqueue_lifecycle_progress(("__upsert__", "heartbeat", "too late"))

    heartbeat = asyncio.create_task(notify())
    tracking = asyncio.create_task(asyncio.Event().wait())
    interrupt = asyncio.create_task(asyncio.Event().wait())
    await started.wait()
    cleanup = asyncio.create_task(owner._run_agent_cleanup_turn_tasks(
        ctx, progress_task=tr._progress_task, log_task=None, interrupt_monitor=interrupt,
        _notify_task=heartbeat, tracking_task=tracking, stream_task=None,
    ))
    await asyncio.wait_for(stopping.wait(), 1)
    assert not tr._progress_task.done()
    assert not cleanup.done()
    release.set()
    await asyncio.wait_for(cleanup, 2)
    assert tr._progress_task.done()
    assert ctx.progress_queue.empty()
    assert adapter.sent == []


@pytest.mark.asyncio
@pytest.mark.parametrize("defer_completion", [False, True])
async def test_cleanup_cancellation_at_ingress_fence_settles_owned_tasks(monkeypatch, defer_completion):
    owner, tr, ctx, adapter = rail(mode="all")
    owner._draining = False
    ctx._defer_progress_completion = defer_completion
    ctx._complete_lifecycle_progress = tr.complete_lifecycle_progress
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: {"compression": {"progress_notices": True}})
    entered, release, fence_entered = threading.Event(), threading.Event(), threading.Event()
    native_prepare = gateway_run._prepare_gateway_status_message

    def prepare(*args):
        entered.set()
        assert release.wait(5), "status callback was not released"
        return native_prepare(*args)

    def close_ingress():
        fence_entered.set()
        tr.close_progress_ingress()

    monkeypatch.setattr(gateway_run, "_prepare_gateway_status_message", prepare)
    ctx._close_progress_ingress = close_ingress
    stopping, heartbeat_release = asyncio.Event(), asyncio.Event()

    async def notify():
        try:
            await asyncio.Event().wait()
        finally:
            stopping.set()
            await heartbeat_release.wait()
            tr.enqueue_lifecycle_progress(("__upsert__", "heartbeat", "too late"))

    tr._progress_task = asyncio.create_task(tr.send_progress_messages())
    heartbeat = asyncio.create_task(notify())
    tracking = asyncio.create_task(asyncio.Event().wait())
    interrupt = asyncio.create_task(asyncio.Event().wait())
    log = asyncio.create_task(asyncio.Event().wait())
    callback = asyncio.create_task(asyncio.to_thread(
        tr._status_callback_sync, "info", ROUTINE_COMPRESSION_STATUS_SAMPLES[0],
    ))
    cleanup = None
    tasks = [tr._progress_task, heartbeat, tracking, interrupt, log, callback]
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        cleanup = asyncio.create_task(owner._run_agent_cleanup_turn_tasks(
            ctx, progress_task=tr._progress_task, log_task=log, interrupt_monitor=interrupt,
            _notify_task=heartbeat, tracking_task=tracking, stream_task=None,
        ))
        tasks.append(cleanup)
        assert await asyncio.to_thread(fence_entered.wait, 2)
        # Cancel the actual cleanup owner while its executor fence waits for the callback's lock.
        for _ in range(2):
            cleanup.cancel()
            await asyncio.sleep(0)
            assert not cleanup.done(), "cleanup escaped before ingress and its tasks settled"
        release.set()
        await asyncio.wait_for(callback, 2)
        await asyncio.wait_for(stopping.wait(), 2)
        assert not tr._progress_task.done()
        # Another cancellation while joining the producer must not abandon its consumer either.
        cleanup.cancel()
        await asyncio.sleep(0)
        assert not cleanup.done()
        heartbeat_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, 2)
        assert all(task.done() for task in tasks)
        assert tr._progress_close_task is not None and tr._progress_close_task.done()
        assert ctx.progress_queue.empty()
        assert "too late" not in "\n".join(adapter.visible.values())

        sent, edits = list(adapter.sent), list(adapter.edits)

        def late_callbacks():
            tr._status_callback_sync("compacted", COMPACTION_DONE_STATUS)
            tr.progress_callback("tool.start", "web_search", "late", {"query": "late"})
            tr.enqueue_lifecycle_progress(("__upsert__", "heartbeat", "after cleanup"))

        await asyncio.to_thread(late_callbacks)
        await asyncio.sleep(0)
        assert ctx.progress_queue.empty()
        assert adapter.sent == sent and adapter.edits == edits
    finally:
        release.set()
        heartbeat_release.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 3)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["stale", "adapter_cancel"])
async def test_inflight_terminal_failure_resolves_false_without_recreated_send(failure):
    class Adapter(Capture):
        async def send(self, *args, **kwargs):
            result = await super().send(*args, **kwargs)
            if failure == "adapter_cancel":
                raise asyncio.CancelledError
            return result

    _, tr, ctx, adapter = rail(Adapter())
    adapter.release = asyncio.Event()
    ack = asyncio.get_running_loop().create_future()
    ctx.progress_queue.put(("__terminal__", "one attempt", ack))
    task = asyncio.create_task(tr.send_progress_messages())
    await asyncio.wait_for(adapter.entered.wait(), 1)
    if failure == "stale":
        ctx._run_still_current = lambda: False
    adapter.release.set()
    assert await asyncio.wait_for(ack, 1) is False
    await stop(task)
    assert len(adapter.sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "hook_cancel", "merge_error", "handoff"])
async def test_queued_exhaustion_retains_descendant_through_ancestor_settlement(monkeypatch, tmp_path, outcome):
    captured, runners, owned_tasks = {}, [], []
    fence_entered, fence_release = threading.Event(), threading.Event()
    close_entered = [asyncio.Event(), asyncio.Event()]
    close_release = [asyncio.Event(), asyncio.Event()]
    native_make = progress._make_runner
    native_close = TurnRunner._close_lifecycle_progress
    hook_entered = asyncio.Event()

    class HookCapture(Capture):
        async def on_processing_start(self, event):
            return None

        async def on_processing_complete(self, event, hook_outcome):
            from gateway.platforms.base import ProcessingOutcome
            assert hook_outcome is ProcessingOutcome.SUCCESS
            hook_entered.set()
            await asyncio.Event().wait()

    async def close(tr, terminal_text):
        depth = tr._ctx._interrupt_depth
        close_entered[depth].set()
        await close_release[depth].wait()
        return await native_close(tr, terminal_text)

    def make(adapter):
        owner = native_make(adapter)
        captured.update(owner=owner, adapter=adapter)
        native_run = owner._run_agent
        native_build = owner._run_agent_build_turn_context
        native_settle = owner._run_agent_settle_turn_tasks

        def build(*args, **kwargs):
            ctx, tr, cleanup_adapter = native_build(*args, **kwargs)
            runners.append(tr)
            if ctx._interrupt_depth == 0:
                native_fence = ctx._close_progress_ingress

                def fence():
                    fence_entered.set()
                    assert fence_release.wait(5), "ancestor ingress fence was not released"
                    native_fence()

                ctx._close_progress_ingress = fence
            return ctx, tr, cleanup_adapter

        async def settle(ctx, **tasks):
            owned_tasks.append(asyncio.current_task())
            owned_tasks.extend(task for task in tasks.values() if task is not None)
            await native_settle(ctx, **tasks)

        async def run(*args, **kwargs):
            if kwargs.get("_interrupt_depth", 0) == 0:
                kwargs["run_generation"] = owner._begin_session_run_generation(kwargs["session_key"])
                kwargs["defer_terminal_lifecycle_progress"] = True
            return await native_run(*args, **kwargs)

        owner._run_agent = run
        owner._run_agent_build_turn_context = build
        owner._run_agent_settle_turn_tasks = settle
        return owner

    class Agent(progress.FakeAgent):
        def run_conversation(self, message, *args, **kwargs):
            return {
                "final_response": "context exhausted" if message == "queued exhaustion" else "first response",
                "compression_exhausted": message == "queued exhaustion", "messages": [], "api_calls": 1,
            }

    if outcome == "merge_error":
        def fail_merge(*args):
            raise RuntimeError("queued merge failed after child return")
        monkeypatch.setattr(gateway_run, "_preserve_queued_followup_history_offset", fail_merge)
    monkeypatch.setattr(progress, "_make_runner", make)
    monkeypatch.setattr(TurnRunner, "_close_lifecycle_progress", close)
    turn = asyncio.create_task(progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id="queued-lifecycle", pending_text="queued exhaustion",
        adapter_cls=HookCapture if outcome == "hook_cancel" else Capture,
        config_data={"display": {"tool_progress": "off", "long_running_notifications": False}},
    ))
    descendant_wait = None
    try:
        if outcome == "hook_cancel":
            await asyncio.wait_for(hook_entered.wait(), 3)
            turn.cancel()
        assert await asyncio.to_thread(fence_entered.wait, 3)
        assert len(runners) == 2
        ancestor, descendant = runners
        assert descendant._ctx._defer_progress_completion
        assert not descendant._progress_task.done()
        assert descendant._progress_close_task is None
        if outcome in ("cancel", "hook_cancel"):
            for _ in range(2):
                turn.cancel()
                await asyncio.sleep(0)
                assert not turn.done()
        fence_release.set()
        await asyncio.wait_for(close_entered[0].wait(), 2)
        if outcome in ("cancel", "hook_cancel"):
            for _ in range(2):
                turn.cancel()
                await asyncio.sleep(0)
                assert not turn.done()
        close_release[0].set()
        if outcome == "handoff":
            adapter, result = await asyncio.wait_for(asyncio.shield(turn), 2)
            assert not descendant._progress_task.done()
            assert not close_entered[1].is_set(), "normal handoff closed the terminal consumer early"
            complete = result.get("_complete_lifecycle_progress")
            assert callable(complete)
            close_release[1].set()
            assert await complete("queued terminal notice") is True
            assert "queued terminal notice" in "\n".join(adapter.visible.values())
        else:
            descendant_wait = asyncio.create_task(close_entered[1].wait())
            await asyncio.wait({turn, descendant_wait}, timeout=2, return_when=asyncio.FIRST_COMPLETED)
            assert close_entered[1].is_set(), "ancestor escaped without closing its deferred descendant"
            assert not turn.done()
            if outcome in ("cancel", "hook_cancel"):
                for _ in range(2):
                    turn.cancel()
                    await asyncio.sleep(0)
                    assert not turn.done()
            close_release[1].set()
            expected = asyncio.CancelledError if outcome in ("cancel", "hook_cancel") else RuntimeError
            with pytest.raises(expected):
                await asyncio.wait_for(turn, 2)
        assert all(task.done() for task in owned_tasks)
        for tr in runners:
            assert tr._progress_task.done()
            assert tr._progress_close_task is not None and tr._progress_close_task.done()
            assert tr._ctx.progress_queue.empty()
            tr.enqueue_lifecycle_progress(("__append__", "too late"))
            tr._status_callback_sync("compacted", COMPACTION_DONE_STATUS)
            tr.progress_callback("tool.started", "web_search", "late", {"query": "late"})
            assert tr._ctx.progress_queue.empty()
    finally:
        fence_release.set()
        for release in close_release:
            release.set()
        if not turn.done():
            turn.cancel()
        await asyncio.wait_for(asyncio.gather(turn, return_exceptions=True), 3)
        if descendant_wait is not None:
            descendant_wait.cancel()
            await asyncio.gather(descendant_wait, return_exceptions=True)
        # A RED leak is asserted above; settle it here so it cannot pollute the
        # next test or manufacture an event-loop shutdown warning.
        for tr in runners:
            await tr.complete_lifecycle_progress()


@pytest.mark.asyncio
async def test_native_card_callbacks_cannot_enqueue_after_cleanup_fence():
    _, tr, ctx, _ = rail(progress.FailingNativeTaskCardAdapter(), mode="all")
    ctx._native_slack_task_cards = True
    tr.close_progress_ingress()
    tr.native_tool_start_callback("late", "web_search", {"query": "late"})
    tr.native_tool_complete_callback("late", "web_search", {}, {"result": "late"})
    assert ctx.progress_queue.empty()
