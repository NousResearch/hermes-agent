"""Opt-in public speech on the real /v1/responses aiohttp/SSE boundary.

The model is synthetic; callbacks, route, writer, and response store are production code.
"""
import asyncio
import json
from unittest.mock import patch
from contextlib import ExitStack

from tests.gateway.test_api_server_transform_llm_output import _app_with_fake_turn, TRANSFORMS, RAW
from gateway.platforms import api_server_public_speech as public_speech

import pytest
from aiohttp.test_utils import TestClient, TestServer

from agent.stream_delivery import StreamDeliveryMixin
from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.test_api_server import _create_app


@pytest.fixture
def adapter():
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


class _StreamAgent(StreamDeliveryMixin):
    """Exercise the real delta/interim classifier without calling a provider."""

    def __init__(self, kw):
        self.stream_delta_callback = kw["stream_delta_callback"]
        self.interim_assistant_callback = kw["interim_assistant_callback"]
        self._stream_callback = None
        self.show_commentary = True
        self._strip_think_blocks = lambda text: text

    def _enqueue_stream_hook(self, *a, **kw):
        pass


def _events(chunk):
    frames = []
    for raw in chunk.split(b"\n\n"):
        event = next((line[7:] for line in raw.decode().splitlines() if line.startswith("event: ")), None)
        data = next((line[6:] for line in raw.decode().splitlines() if line.startswith("data: ")), None)
        if event and data:
            frames.append((event, json.loads(data)))
    return frames


async def _read_until(response, event):
    chunks = bytearray()
    async for line in response.content:
        chunks.extend(line)
        if any(name == event for name, _ in _events(bytes(chunks))):
            return bytes(chunks)
    raise AssertionError(f"stream ended before {event}")


def _messages(envelope):
    return [(item.get("phase"), item["content"][0]["text"])
            for item in envelope.get("output", []) if item["type"] == "message"]


@pytest.mark.asyncio
async def test_commentary_reaches_client_before_held_final_and_private_protocol_never_enters_queue(adapter):
    started, release = asyncio.Event(), asyncio.Event()
    secret = "PRIVATE_TOOL_ID_arg_result_reason_status_" + "x" * 2500
    queued = []

    async def run(**kw):
        agent = _StreamAgent(kw)
        kw["agent_ref"][0] = agent
        kw["tool_start_callback"](secret, secret, {"argument": secret})
        kw["reasoning_callback"](secret)
        kw["status_callback"]("lifecycle", secret)
        # Ordinary providers send an interim preamble through _fire_stream_delta first.
        agent._fire_stream_delta("I'll check now.")
        agent._emit_interim_assistant_message({"role": "assistant", "content": "I'll check now.",
                                               "tool_calls": [{"id": secret}]})
        kw["tool_complete_callback"](secret, secret, {}, secret)
        started.set()
        await release.wait()
        return {"final_response": "The answer is 42.", "completed": True,
                "messages": [{"role": "assistant", "content": "I'll check now.",
                              "tool_calls": [{"id": secret, "function": {"name": secret, "arguments": secret}}]},
                             {"role": "tool", "tool_call_id": secret, "content": secret},
                             {"role": "assistant", "content": "The answer is 42."}]}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch(
            "gateway.platforms.api_server.ThreadSafeAsyncQueue.put_threadsafe",
            autospec=True,
            side_effect=lambda self, item: (queued.append(item), self.put_nowait(item)),
        ):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            assert response.status == 200
            await asyncio.wait_for(started.wait(), 3)
            early = await asyncio.wait_for(_read_until(response, "response.output_item.done"), 3)
            early_frames = _events(early)
            assert any(e == "response.output_item.done" and d["item"].get("phase") == "commentary"
                       and d["item"]["content"][0]["text"] == "I'll check now." for e, d in early_frames)
            assert "response.completed" not in [e for e, _ in early_frames]
            assert secret not in early.decode()
            assert secret not in repr(queued)
            release.set()
            wire = early + await response.read()
            frames = _events(wire)
            terminal = [d["response"] for e, d in frames if e == "response.completed"]
            assert len(terminal) == 1
            assert _messages(terminal[0]) == [("commentary", "I'll check now."), (None, "The answer is 42.")]
            assert "".join(d["delta"] for e, d in frames if e == "response.output_text.delta") == "The answer is 42."
            assert secret not in wire.decode()
            ids = [(d["output_index"], d["item"]["id"]) for e, d in frames if e == "response.output_item.done"]
            assert [idx for idx, _ in ids] == [0, 1]
            assert ids[0][1] == terminal[0]["output"][0]["id"]
            assert ids[1][1] == terminal[0]["output"][1]["id"]
            stored = await client.get("/v1/responses/" + terminal[0]["id"])
            assert _messages(await stored.json()) == _messages(terminal[0])
            assert secret not in await stored.text()
            history = adapter._current_response_store().get(terminal[0]["id"])["conversation_history"]
            assert secret in repr(history)  # only the public projection is narrowed


@pytest.mark.asyncio
@pytest.mark.parametrize("result,expected", [
    ({"final_response": "SECRET_DIAGNOSTIC", "failed": True, "error": "SECRET_DIAGNOSTIC"}, "failed"),
    ({"final_response": "SECRET_DIAGNOSTIC", "completed": False, "partial": True}, "incomplete"),
    ({"final_response": "SECRET_DIAGNOSTIC", "interrupted": True}, "incomplete"),
    ({"final_response": "", "error": "SECRET_DIAGNOSTIC"}, "failed"),
    (RuntimeError("SECRET_DIAGNOSTIC"), "failed"),
])
@pytest.mark.parametrize("commentary", [False, True])
async def test_unsuccessful_result_never_promotes_diagnostic_to_speech(adapter, result, expected, commentary):
    async def run(**kw):
        if commentary:
            kw["interim_assistant_callback"]("Safe progress.", already_streamed=False)
        if isinstance(result, Exception):
            raise result
        return result, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            wire = await response.read()
        assert response.status == 200
        assert "SECRET_DIAGNOSTIC" not in wire.decode()
        frames = _events(wire)
        terminal = [(e, d["response"]) for e, d in frames if e in
                    ("response.completed", "response.failed", "response.incomplete")]
        assert len(terminal) == 1
        event, envelope = terminal[0]
        assert (event, envelope["status"]) == ("response." + expected, expected)
        assert _messages(envelope) == ([("commentary", "Safe progress.")] if commentary else [])
        assert envelope["hermes"]["partial"] is commentary
        assert not any(e == "response.output_text.delta" for e, _ in frames)
        stored = await client.get("/v1/responses/" + envelope["id"])
        assert "SECRET_DIAGNOSTIC" not in await stored.text()


@pytest.mark.asyncio
async def test_provider_resolution_guidance_without_failure_flags_is_not_speech(adapter):
    """_run_agent's provider-resolution catch returns api_calls=0 and a guidance final."""
    guidance = "PRIVATE_PROVIDER_AUTH_GUIDANCE"

    async def run(**kw):
        return {"final_response": guidance, "messages": [], "api_calls": 0, "tools": []}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            wire = await response.read()
        frames = _events(wire)
        assert guidance not in wire.decode()
        terminal = next(d["response"] for e, d in frames if e == "response.failed")
        assert _messages(terminal) == []
        stored = await client.get("/v1/responses/" + terminal["id"])
        assert guidance not in await stored.text()


@pytest.mark.asyncio
async def test_failure_after_delivered_delta_keeps_only_delivered_public_text(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("Safe spoken prefix.")
        return {"final_response": "SECRET_DIAGNOSTIC", "completed": False,
                "failed": True, "error": "SECRET_DIAGNOSTIC"}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "question", "stream": True, "public_commentary": True})
            wire = await response.read()
        frames = _events(wire)
        assert "SECRET_DIAGNOSTIC" not in wire.decode()
        terminal = next(d["response"] for e, d in frames if e == "response.failed")
        assert _messages(terminal) == []
        assert terminal["hermes"]["partial"] is False
        assert not any(e == "response.output_text.delta" for e, _ in frames)
        assert (await client.get("/v1/responses/" + terminal["id"])).status == 200


@pytest.mark.asyncio
async def test_uncategorized_final_is_held_until_agent_completes(adapter):
    entered, finish = asyncio.Event(), asyncio.Event()

    async def run(**kw):
        kw["stream_delta_callback"]("A delayed answer.")
        entered.set()
        await finish.wait()
        return {"final_response": "A delayed answer.", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            await asyncio.wait_for(entered.wait(), 3)
            early = await _read_until(response, "response.created")
            assert [e for e, _ in _events(early)] == ["response.created"]
            # No extra SSE frame while the task is blocked: inspect the transport's
            # already-buffered bytes without a clock-based negative wait.
            assert not any(e == "response.output_text.delta" for raw in response.content._buffer
                           for e, _ in _events(raw))
            finish.set()
            frames = _events(early + await response.read())
            assert [e for e, _ in frames if e == "response.output_text.delta"] == ["response.output_text.delta"]
            assert [e for e, _ in frames if e == "response.completed"] == ["response.completed"]


@pytest.mark.asyncio
async def test_public_opt_in_requires_stream_and_does_not_start_agent(adapter):
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent") as run:
            response = await client.post("/v1/responses", json={
                "input": "question", "public_commentary": True, "stream": False})
        assert response.status == 400
        run.assert_not_called()
        capability = await (await client.get("/v1/capabilities")).json()
        assert capability["features"]["responses_public_speech_streaming"] is True


@pytest.mark.asyncio
async def test_public_store_false_and_generic_stream_are_independent(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("ordinary delta")
        kw["tool_start_callback"]("call-private", "terminal", {"x": 1})
        return {"final_response": "ordinary delta", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            public = await client.post("/v1/responses", json={
                "input": "question", "public_commentary": True, "stream": True, "store": False})
            public_frames = _events(await public.read())
            terminal = next(d["response"] for e, d in public_frames if e == "response.completed")
            assert _messages(terminal) == [(None, "ordinary delta")]
            assert not any(item["type"] == "function_call" for item in terminal["output"])
            assert (await client.get("/v1/responses/" + terminal["id"])).status == 404
            generic = await client.post("/v1/responses", json={"input": "question", "stream": True})
            generic_frames = _events(await generic.read())
            assert any(e == "response.output_item.added" and d["item"]["type"] == "function_call"
                       for e, d in generic_frames)
            assert any(e == "response.output_text.delta" for e, _ in generic_frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", sorted(TRANSFORMS))
async def test_public_final_uses_real_post_stream_transform(adapter, kind):
    with ExitStack() as stack:
        app = _app_with_fake_turn(adapter, TRANSFORMS[kind], stack)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/v1/responses", json={
                "model": "m", "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
            terminal = next(d["response"] for e, d in frames if e == "response.completed")
            expected = TRANSFORMS[kind](RAW)
            assert _messages(terminal) == [(None, expected)]
            assert [d["text"] for e, d in frames if e == "response.output_text.done"] == [expected]
            stored = adapter._current_response_store().get(terminal["id"])
            assert _messages(stored["response"]) == [(None, expected)]


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix,interim", [("same inter", "same interim"),
                                            ("preface plus interim", "interim")])
async def test_real_classifier_interim_never_promotes_held_prefix(adapter, prefix, interim):
    async def run(**kw):
        agent = _StreamAgent(kw)
        agent._fire_stream_delta(prefix)
        agent._emit_interim_assistant_message({"role": "assistant", "content": interim,
                                               "tool_calls": [{"id": "private"}]})
        return {"final_response": "final answer", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.completed")
        assert _messages(terminal) == [("commentary", interim), (None, "final answer")]
        assert [d["delta"] for e, d in frames if e == "response.output_text.delta"] == ["final answer"]


@pytest.mark.asyncio
@pytest.mark.parametrize("sid,valid", [("rotated-child", True), ("../private", False),
                                           ("x" * 257, False), ("é" * 129, False),
                                           ("line\nbreak", False)])
async def test_effective_session_id_only_valid_success_and_next_chain(adapter, sid, valid):
    seen = []
    async def run(**kw):
        seen.append(kw["session_id"])
        return {"final_response": "reply", "completed": True, "session_id": sid,
                "_compressed": True, "messages": [{"role": "assistant", "content": "reply"}]}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            first = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await first.read())
            terminal = next(d["response"] for e, d in frames if e == "response.completed")
            expected = sid if valid else first.headers["X-Hermes-Session-Id"]
            assert terminal["hermes"]["session_id"] == expected
            if not valid:
                assert sid not in repr(terminal)
            second = await client.post("/v1/responses", json={"input": "next",
                "previous_response_id": terminal["id"], "stream": True, "public_commentary": True})
            await second.read()
            assert seen[1] == expected


@pytest.mark.asyncio
async def test_history_preflight_failure_stores_matching_failed_terminal_without_final(adapter):
    async def run(**kw):
        kw["interim_assistant_callback"]("Safe progress", already_streamed=False)
        return {"final_response": "unpublished final", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            adapter, "_build_response_conversation_history", side_effect=RuntimeError("PRIVATE_HISTORY_ERROR")):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.failed")
        stored = await (await client.get("/v1/responses/" + terminal["id"])).json()
        assert stored == terminal
        assert terminal["hermes"]["partial"] is True
        assert _messages(terminal) == [("commentary", "Safe progress")]
        assert "unpublished final" not in repr(frames)
        assert "PRIVATE_HISTORY_ERROR" not in repr(terminal)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_event,expected", [
    ("response.output_item.added", [("commentary", "Delivered")]),
    ("response.output_item.done", [("commentary", "Delivered"),
                                   ("commentary", "Partly delivered")]),
])
async def test_disconnect_snapshot_only_credits_written_commentary(adapter, failure_event, expected):
    async def run(**kw):
        kw["interim_assistant_callback"]("Delivered", already_streamed=False)
        kw["interim_assistant_callback"]("Partly delivered", already_streamed=False)
        return {"final_response": "not sent", "completed": True}, None

    original = public_speech.PublicSpeechStream.write_event
    async def disconnect(self, event, data):
        if (event == failure_event and data.get("item", {}).get("phase") == "commentary"
                and self.output_index == 2):
            raise ConnectionResetError("transport")
        return await original(self, event, data)

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            public_speech.PublicSpeechStream, "write_event", disconnect):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        response_id = next(d["response"]["id"] for e, d in frames if e == "response.created")
        stored = await (await client.get("/v1/responses/" + response_id)).json()
        assert stored["status"] == "incomplete"
        assert stored["hermes"]["partial"] is True
        assert _messages(stored) == expected
        if failure_event == "response.output_item.done":
            assert stored["output"][-1]["status"] == "in_progress"


@pytest.mark.asyncio
async def test_no_authoritative_final_does_not_promote_unclassified_deltas(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("unclassified provider text")
        return {"completed": True, "messages": []}, None
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.completed")
        assert not _messages(terminal)
        assert "unclassified provider text" not in repr(frames)


@pytest.mark.asyncio
async def test_empty_authoritative_transform_never_falls_back_to_raw(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("raw to discard")
        return {"final_response": "", "response_transformed": True, "completed": True}, None
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.completed")
        assert not _messages(terminal)
        assert "raw to discard" not in repr(frames)


@pytest.mark.asyncio
async def test_muted_result_does_not_publish_raw_or_final(adapter):
    async def run(**kw):
        kw["stream_delta_callback"]("raw muted")
        return {"final_response": "muted final", "completed": True,
                "_notification_presentation_suppressed": True}, None
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.completed")
        assert not _messages(terminal)
        assert "raw muted" not in repr(frames) and "muted final" not in repr(frames)


@pytest.mark.asyncio
async def test_empty_agent_result_fails_closed(adapter):
    async def run(**kw):
        return {}, None
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        assert any(e == "response.failed" for e, _ in frames)
        assert not any(e == "response.completed" for e, _ in frames)


@pytest.mark.asyncio
async def test_store_failure_before_final_reports_failed_without_publishing_final(adapter):
    async def run(**kw):
        kw["interim_assistant_callback"]("Visible progress", already_streamed=False)
        return {"final_response": "must not publish", "completed": True}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        store = adapter._current_response_store()
        original = store.put
        writes = 0
        def reject_preflight(response_id, snapshot):
            nonlocal writes
            writes += 1
            if writes == 2:  # created succeeded; safe preflight before any final write fails
                raise RuntimeError("PRIVATE_STORE_FAILURE")
            return original(response_id, snapshot)
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            store, "put", side_effect=reject_preflight):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        terminal = next(d["response"] for e, d in frames if e == "response.failed")
        assert terminal == (await (await client.get("/v1/responses/" + terminal["id"])).json())
        assert terminal["hermes"]["partial"] is True
        assert "must not publish" not in repr(frames)
        assert "PRIVATE_STORE_FAILURE" not in repr(terminal)


@pytest.mark.asyncio
async def test_final_delivery_disconnect_stores_only_delivered_delta(adapter):
    async def run(**kw):
        return {"final_response": "delivered text", "completed": True}, None
    original = public_speech.PublicSpeechStream.write_event
    async def disconnect(self, event, data):
        if event == "response.output_text.done":
            raise ConnectionResetError("transport")
        return await original(self, event, data)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            public_speech.PublicSpeechStream, "write_event", disconnect):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        response_id = next(d["response"]["id"] for e, d in frames if e == "response.created")
        stored = await (await client.get("/v1/responses/" + response_id)).json()
        assert stored["status"] == "incomplete"
        assert stored["hermes"]["partial"] is True
        assert _messages(stored) == [(None, "delivered text")]
        assert "session_id" not in stored.get("hermes", {})


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", ["response.output_text.delta", "response.completed"])
async def test_pending_final_get_and_followup_cannot_publish_future_speech_or_rotation(adapter, pause_at):
    reached, release = asyncio.Event(), asyncio.Event()
    seen = []
    original = public_speech.PublicSpeechStream.write_event

    async def hold(self, event, data):
        if event == pause_at and self.user_message == "q":
            reached.set()
            await release.wait()
        return await original(self, event, data)

    async def run(**kw):
        seen.append(kw["session_id"])
        return {"final_response": "future speech" if len(seen) == 1 else "followup",
                "completed": True, "session_id": "rotated-child" if len(seen) == 1 else kw["session_id"],
                "messages": [{"role": "assistant", "content": "future speech"}]}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            public_speech.PublicSpeechStream, "write_event", hold):
            first = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            created = await first.content.readuntil(b"\n\n")
            response_id = _events(created)[0][1]["response"]["id"]
            try:
                await asyncio.wait_for(reached.wait(), 3)
                before = await (await client.get("/v1/responses/" + response_id)).json()
                follow = await client.post("/v1/responses", json={
                    "input": "next", "previous_response_id": response_id,
                    "stream": True, "public_commentary": True})
                await follow.read()
                assert seen[1] == first.headers["X-Hermes-Session-Id"]
                assert before["status"] == "in_progress"
                assert "future speech" not in repr(before)
                assert before.get("hermes", {}).get("session_id") != "rotated-child"
            finally:
                release.set()
            terminal = next(d["response"] for e, d in _events(created + await first.read())
                            if e == "response.completed")
            assert (await (await client.get("/v1/responses/" + response_id)).json()) == terminal
            assert terminal["hermes"]["session_id"] == "rotated-child"


@pytest.mark.asyncio
@pytest.mark.parametrize("sid", [f"abc{chr(n)}xyz" for n in (*range(32), 127)],
                         ids=[f"U+{n:04X}" for n in (*range(32), 127)])
async def test_c0_del_effective_id_falls_back_on_terminal_store_and_real_next_post(adapter, sid):
    seen = []
    async def run(**kw):
        seen.append(kw["session_id"])
        return {"final_response": "speech", "completed": True, "session_id": sid,
                "messages": [{"role": "assistant", "content": "speech"}]}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run):
            first = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await first.read())
            terminal = next(d["response"] for e, d in frames if e == "response.completed")
            expected = first.headers["X-Hermes-Session-Id"]
            assert terminal["hermes"]["session_id"] == expected
            assert adapter._current_response_store().get(terminal["id"])["session_id"] == expected
            follow = await client.post("/v1/responses", json={
                "input": "next", "previous_response_id": terminal["id"],
                "stream": True, "public_commentary": True})
            assert follow.status == 200
            assert any(e == "response.completed" for e, _ in _events(await follow.read()))
            assert seen == [expected, expected]


@pytest.mark.asyncio
async def test_late_completed_store_failure_never_writes_second_terminal(adapter):
    async def run(**kw):
        return {"final_response": "delivered", "completed": True,
                "session_id": "rotated-child"}, None

    async with TestClient(TestServer(_create_app(adapter))) as client:
        store = adapter._current_response_store()
        original = store.put
        def fail_commit(response_id, snapshot):
            if snapshot["response"]["status"] == "completed":
                raise RuntimeError("PRIVATE_LATE_STORE_FAILURE")
            return original(response_id, snapshot)
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            store, "put", side_effect=fail_commit):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
            terminal = next(d["response"] for e, d in frames if e == "response.completed")
            assert [e for e, _ in frames if e in ("response.completed", "response.failed")] == ["response.completed"]
            pending = await (await client.get("/v1/responses/" + terminal["id"])).json()
            assert pending["status"] == "in_progress"
            assert "delivered" not in repr(pending)
            assert "rotated-child" not in repr(pending)


@pytest.mark.asyncio
async def test_terminal_write_disconnect_does_not_leave_completed_snapshot(adapter):
    async def run(**kw):
        return {"final_response": "written final", "completed": True,
                "session_id": "rotated-child"}, None
    original = public_speech.PublicSpeechStream.write_event
    async def disconnect(self, event, data):
        if event == "response.completed":
            raise ConnectionResetError("transport")
        return await original(self, event, data)
    async with TestClient(TestServer(_create_app(adapter))) as client:
        with patch.object(adapter, "_run_agent", side_effect=run), patch.object(
            public_speech.PublicSpeechStream, "write_event", disconnect):
            response = await client.post("/v1/responses", json={
                "input": "q", "stream": True, "public_commentary": True})
            frames = _events(await response.read())
        response_id = next(d["response"]["id"] for e, d in frames if e == "response.created")
        stored = adapter._current_response_store().get(response_id)
        assert stored["response"]["status"] == "incomplete"
        assert stored["response"]["hermes"]["partial"] is True
        assert "session_id" not in stored["response"]["hermes"]
        assert stored["session_id"] == response.headers["X-Hermes-Session-Id"]
