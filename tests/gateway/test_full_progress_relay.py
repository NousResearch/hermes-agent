"""Full progress must leave the native Slack relay answer stream open."""

import asyncio
import copy
import json
import threading

import pytest

from gateway import run_turn_runner
from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, MessageType
from gateway.relay.adapter import RelayAdapter
from tests.gateway import test_run_progress_topics as progress
from tests.gateway.relay.stub_connector import StubConnector
from tests.gateway.relay.test_relay_live_cards import make_desc


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking_first", [False, True])
async def test_full_progress_preserves_live_slack_draft(monkeypatch, tmp_path, thinking_first):
    loop = asyncio.get_running_loop()
    acknowledged = threading.Condition()
    acknowledged_frames = []
    observed = {}
    final = "Answer before progress. Answer after progress."
    arguments = {"question": "oversized nonsecret argument " * 45 + "END-LARGE"}
    desc = make_desc(max_message_length=320)

    class RecordingConnector(StubConnector):
        def __init__(self):
            super().__init__(desc)
            self.send_ids = []

        async def send_outbound(self, action, *, platform=None):
            if action["op"] == "send":
                message_id = f"progress-{len(self.send_ids) + 1}"
                self.send_ids.append(message_id)
                self.next_send_result = {"success": True, "message_id": message_id}
            if action["op"] == "draft":
                self.next_draft_result = {"success": True, "message_id": "answer-stream"}
            result = await super().send_outbound(copy.deepcopy(action), platform=platform)
            # Signal on the loop's next turn, after the real adapter consumes the ack.
            def notify():
                with acknowledged:
                    acknowledged_frames.append(action)
                    acknowledged.notify_all()
            loop.call_soon(notify)
            return result

    transport = RecordingConnector()
    adapter = RelayAdapter(PlatformConfig(), desc, transport=transport)
    native_make_runner = progress._make_runner

    def make_runner(actual_adapter):
        runner = native_make_runner(actual_adapter)
        native_run = runner._run_agent

        async def run_with_inbound(**kwargs):
            source = kwargs["source"]
            source.delivered_via_upstream_relay = True
            adapter._capture_scope(MessageEvent(
                text="hello", source=source, message_type=MessageType.TEXT,
                message_id="1700.0002",
            ))
            return await native_run(**kwargs, event_message_id="1700.0002")

        runner._run_agent = run_with_inbound
        return runner

    monkeypatch.setattr(progress, "_make_runner", make_runner)
    native_sender = run_turn_runner.TurnRunner.send_progress_messages

    async def observe_sender(self):
        observed["ctx"] = self._ctx
        observed["metadata"] = copy.deepcopy(self._ctx._progress_metadata)
        return await native_sender(self)

    monkeypatch.setattr(run_turn_runner.TurnRunner, "send_progress_messages", observe_sender)
    native_explicit_send = adapter.send_for_platform
    explicit_calls = []

    async def observe_explicit_send(*args, **kwargs):
        explicit_calls.append((args, kwargs))
        return await native_explicit_send(*args, **kwargs)

    monkeypatch.setattr(adapter, "send_for_platform", observe_explicit_send)

    def sealed():
        return any(f["op"] == "draft" and f.get("final") for f in acknowledged_frames)

    def wait_for(predicate):
        with acknowledged:
            assert acknowledged.wait_for(lambda: predicate() or sealed(), timeout=4), (
                "missing transport acknowledgment", transport.sent
            )
        return not sealed()

    def contains(text):
        return any(text in f.get("content", "") for f in acknowledged_frames)

    class Agent(progress.FakeAgent):
        def run_conversation(self, message, conversation_history=None, task_id=None, **kwargs):
            self.stream_delta_callback("Answer before progress.")
            assert wait_for(lambda: any(f["op"] == "draft" for f in acknowledged_frames))
            cb = self.tool_progress_callback

            def thinking():
                cb("_thinking", "raw thinking progress", "", {})
                return wait_for(lambda: contains("raw thinking progress"))

            def entries():
                cb("tool.started", "full_small_one", "preview", {"value": "one"})
                if not wait_for(lambda: contains("full_small_one")):
                    return False
                cb("tool.started", "full_small_two", "preview", {"value": "two"})
                if not wait_for(lambda: contains("full_small_two")):
                    return False
                observed["large_start"] = len(transport.sent)
                cb("tool.started", "full_large", "preview", arguments)
                if not wait_for(lambda: contains("END-LARGE")):
                    return False
                observed["large_end"] = len(transport.sent)
                return True

            steps = (thinking, entries) if thinking_first else (entries, thinking)
            for step in steps:
                if not step():
                    # Let native finalization complete so RED reports the actual bad seal.
                    return {"final_response": final, "messages": [], "api_calls": 1}
            self.stream_delta_callback(" Answer after progress.")
            assert wait_for(lambda: any(
                f["op"] == "draft" and not f.get("final") and f["content"] == final
                for f in acknowledged_frames
            ))
            return {"final_response": final, "messages": [], "api_calls": 1}

    actual_adapter, result = await progress._run_with_agent(
        monkeypatch, tmp_path, Agent, session_id=f"full-relay-{thinking_first}",
        platform=Platform.SLACK, chat_id="C1", chat_type="channel",
        thread_id="1699.9000", user_id="U1", scope_id="T1",
        adapter_cls=lambda platform: adapter,
        config_data={
            "display": {"platforms": {"slack": {
                "tool_progress": "full", "thinking_progress": True,
                "cleanup_progress": True, "streaming": True,
                "tool_progress_grouping": "accumulate",
            }}},
            "streaming": {"enabled": True, "transport": "draft",
                          "edit_interval": 0.0, "buffer_threshold": 1},
        },
    )
    assert actual_adapter is adapter
    seals = [f for f in transport.sent if f["op"] == "draft" and f.get("final")]
    assert [f["content"] for f in seals] == [final], transport.sent
    assert result["final_response"] == final
    assert not explicit_calls, "native progress entered the inherited second absorption door"
    assert observed["ctx"]._progress_metadata == observed["metadata"] == {
        "thread_id": "1699.9000", "message_id": "1700.0002",
        "scope_id": "T1", "slack_team_id": "T1", "user_id": "U1",
    }
    assert "_relay_logical_platform" not in observed["metadata"]
    frames = [f for f in transport.sent if f["op"] in {"send", "edit", "draft"}]
    for frame in frames:
        assert frame["chat_id"] == "C1"
        assert frame["metadata"]["thread_id"] == "1699.9000"
        assert frame["metadata"]["message_id"] == "1700.0002"
        assert frame["metadata"]["scope_id"] == "T1"
        assert frame["metadata"]["slack_team_id"] == "T1"
        assert frame["metadata"]["user_id"] == "U1"
        assert "_interim_send" not in frame["metadata"]
    assert all(p == "slack" for p in transport.sent_platforms)
    sends = [f for f in frames if f["op"] == "send"]
    assert sends and all(f["reply_to"] is None for f in sends)
    edits = [f for f in frames if f["op"] == "edit"]
    assert any("full_small_one" in f["content"] and "full_small_two" in f["content"] for f in edits)
    assert {f["message_id"] for f in edits} <= set(transport.send_ids)
    assert set(observed["ctx"]._cleanup_msg_ids) == set(transport.send_ids)
    large = [f for f in transport.sent[observed["large_start"]:observed["large_end"]]
             if f["op"] == "send"]
    assert len(large) > 1
    header, _, body = "".join(f["content"] for f in large).partition("\n")
    assert header.endswith("full_large")
    assert json.loads(body) == arguments
    drafts = [f for f in frames if f["op"] == "draft"]
    assert {f["draft_id"] for f in drafts} == {seals[0]["draft_id"]}
    assert any(not f["final"] and f["content"] == final for f in drafts)
    assert all(final.startswith(f["content"]) for f in drafts)
