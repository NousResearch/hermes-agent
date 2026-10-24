"""Credential-free full _StreamingCall driver regressions (fake SDK, real Relay).
Run directly with Python; network is denied before candidate imports.
Not provider/Gateway end-to-end acceptance.
"""
import base64
import contextlib
import io
import os
from pathlib import Path
import socket
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

HOME = tempfile.TemporaryDirectory(prefix="image-worker-", dir=os.environ.get("TMPDIR"))
os.environ["HERMES_HOME"] = HOME.name
os.environ["HERMES_DISABLE_LAZY_INSTALLS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
DENY = patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden"))
DENY_EX = patch.object(socket.socket, "connect_ex", side_effect=AssertionError("network forbidden"))
DENY.start()
DENY_EX.start()
import httpx
from PIL import Image
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from agent import chat_completion_helpers as cch
from agent.stream_delivery import StreamDeliveryMixin

buf = io.BytesIO()
Image.new("RGB", (2, 2), (42, 90, 150)).save(buf, format="PNG")
PNG = buf.getvalue()
URL = "data:image/png;base64," + base64.b64encode(PNG).decode()


def chunk(text=None, image=False, tools=None, finish=None, reasoning=None):
    delta = {"content": text, "tool_calls": tools, "reasoning_content": reasoning}
    if image:
        delta["images"] = [{"image_url": {"url": URL}}]
    return ChatCompletionChunk.model_validate({"id": "synthetic", "object": "chat.completion.chunk", "created": 0,
        "model": "fixture", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})


def tool(args):
    return [{"index": 0, "id": "synthetic-tool", "type": "function", "function": {"name": "example", "arguments": args}}]


class SDKStream:
    def __init__(self, events):
        self.events = events
        self.closed = False
    def __iter__(self):
        for value in self.events:
            if isinstance(value, Exception):
                raise value
            yield value
    def close(self):
        self.closed = True


class Agent:
    # Actual production writer logic; no hand-implemented writer decision.
    _ensure_stream_writer_state = StreamDeliveryMixin._ensure_stream_writer_state
    _claim_stream_writer = StreamDeliveryMixin._claim_stream_writer
    _stream_writer_is_current = StreamDeliveryMixin._stream_writer_is_current
    def __init__(self, responses, callback=True):
        self.model, self.provider, self.base_url, self.api_mode = "fixture", "fixture", "https://fixture.invalid/v1", "chat_completions"
        self._interrupt_requested = False
        self._current_streamed_assistant_text = ""
        self.stream_delta_callback = (lambda text: None) if callback else None
        self.dispatches = 0
        self.responses = iter(responses)
        self.closed = []
        self.aborted = []
        self._stream_diag_init = lambda: {}
        self._is_provider_stream_parse_error = lambda e: False
        self._warning_presentation_enabled = lambda: False
        for name in ("_touch_activity", "_capture_rate_limits", "_capture_credits", "_capture_nous_model_switch",
                     "_stream_diag_capture_response", "_check_openrouter_cache_status", "_fire_reasoning_delta", "_fire_tool_gen_started",
                     "_emit_stream_drop", "_log_stream_retry", "_buffer_diagnostic_status", "_emit_diagnostic_wait", "_emit_wait_notice"):
            setattr(self, name, Mock())
    def _record_streamed_assistant_text(self, text):
        self._current_streamed_assistant_text += text
    def _fire_stream_delta(self, text):
        if self.stream_delta_callback:
            self.stream_delta_callback(text)
            self._record_streamed_assistant_text(text)
    def _reset_stream_delivery_tracking(self):
        self._current_streamed_assistant_text = ""
    def _create_request_openai_client(self, **kwargs):
        def create(**kwargs):
            self.dispatches += 1
            value = next(self.responses)
            return value() if callable(value) else value
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    def _close_request_openai_client(self, client, **kwargs):
        self.closed.append(threading.get_ident())
    def _abort_request_openai_client(self, client, **kwargs):
        self.aborted.append(kwargs.get("reason"))


DENY.stop()
DENY_EX.stop()


class WorkerLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        DENY.start()
        DENY_EX.start()

    @classmethod
    def tearDownClass(cls):
        DENY.stop()
        DENY_EX.stop()

    def setUp(self):
        self.stack = contextlib.ExitStack()
        # Only environment scheduling/backoff is replaced, not worker/retry/Relay logic.
        self.stack.enter_context(patch.object(cch, "should_use_direct_api_call", return_value=False))
        self.stack.enter_context(patch.object(cch, "_wait_stream_retry_backoff", return_value=True))
        self.addCleanup(self.stack.close)
    def run_call(self, events, callback=True):
        stream = SDKStream(events)
        agent = Agent([stream], callback)
        call = cch._StreamingCall(agent, {"model": "fixture", "messages": []}, None)
        result = call.run()
        self.assertIsNotNone(call.worker)
        call.worker.join(2)
        self.assertFalse(call.worker.is_alive())
        self.assertTrue(stream.closed)
        return agent, call, result
    def image_preserved(self, agent, call, result, count=1):
        self.assertEqual(agent.dispatches, count)
        self.assertEqual(len(call._chat_image_markers), 1)
        marker = call._chat_image_markers[0]
        self.assertEqual(Path(marker[6:]).read_bytes(), PNG)
        self.assertIn(marker, result.choices[0].message.content)
        self.assertNotIn("base64", str(result))
    def test_clean_eof_image_text_reasoning_without_callback(self):
        a, c, r = self.run_call([chunk("before", True, reasoning="thought"), chunk("after")], False)
        self.image_preserved(a, c, r)
        self.assertIn("before", r.choices[0].message.content)
        self.assertIn("after", r.choices[0].message.content)
        self.assertEqual(r.choices[0].message.reasoning_content, "thought")
        self.assertEqual(r.choices[0].finish_reason, "length")
        self.assertIsNone(c.result["error"])
    def test_clean_eof_parseable_tool_is_not_executable(self):
        a, c, r = self.run_call([chunk(image=True, tools=tool('{}'))])
        self.image_preserved(a, c, r)
        self.assertIsNone(r.choices[0].message.tool_calls)
        self.assertEqual(r.choices[0].finish_reason, "length")
    def test_clean_eof_incomplete_tool_is_not_executable(self):
        a, c, r = self.run_call([chunk(image=True, tools=tool('{"path":'))])
        self.image_preserved(a, c, r)
        self.assertIsNone(r.choices[0].message.tool_calls)
    def test_transport_after_image_no_callback_no_redispatch(self):
        a, c, r = self.run_call([chunk(image=True), httpx.ReadError("synthetic reset")], False)
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].finish_reason, "length")
        self.assertIsNone(r.choices[0].message.tool_calls)
    def test_transport_after_image_preserves_accumulated_text_reasoning(self):
        a, c, r = self.run_call([chunk("before", True, reasoning="thought"), chunk("after"), httpx.ReadError("synthetic reset")], False)
        self.image_preserved(a, c, r)
        self.assertIn("before", r.choices[0].message.content)
        self.assertIn("after", r.choices[0].message.content)
        self.assertEqual(r.choices[0].message.reasoning_content, "thought")

    def test_transport_after_suppressed_tool_image_no_redispatch(self):
        a, c, r = self.run_call([chunk(tools=tool('{')), chunk(image=True), httpx.ReadError("synthetic reset")], False)
        self.image_preserved(a, c, r)
        self.assertIsNone(r.choices[0].message.tool_calls)
    def test_transport_before_image_retries_then_succeeds(self):
        first, second = SDKStream([httpx.ReadError("synthetic reset")]), SDKStream([chunk(image=True), chunk(finish="stop")])
        a = Agent([first, second])
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        r = c.run()
        self.image_preserved(a, c, r, 2)
        self.assertEqual(r.choices[0].finish_reason, "stop")
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(c.stream_attempt_state["current"], 2)
    def test_length_stays_length(self):
        a, c, r = self.run_call([chunk(image=True), chunk("partial", finish="length")])
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].finish_reason, "length")
    def test_image_next_text_boundary_and_duplicate(self):
        a, c, r = self.run_call([chunk(image=True), chunk(image=True), chunk("following", finish="stop")])
        self.image_preserved(a, c, r)
        text = r.choices[0].message.content
        self.assertEqual(text.count("MEDIA:"), 1)
        self.assertIn(c._chat_image_markers[0] + "\nfollowing", text)
    def test_pending_text_same_chunk_tool_not_skipped(self):
        a, c, r = self.run_call([chunk("data:", True, tool('{}')), chunk(finish="tool_calls")])
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].message.tool_calls[0].function.name, "example")
    def test_runaway_image_retains_terminal_guard(self):
        a, c, r = self.run_call([chunk(image=True)] + [chunk("repeated phrase " * 400)] * 50)
        self.image_preserved(a, c, r)
        self.assertTrue(r._runaway_repetition)
        self.assertEqual(r.choices[0].finish_reason, "length")
    def test_interrupt_while_create_pending_completed_response_no_write(self):
        entered, release = threading.Event(), threading.Event()
        response = ChatCompletion.model_validate({"id": "synthetic", "object": "chat.completion", "created": 0, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": None, "images": [{"image_url": {"url": URL}}]}}]})
        def slow():
            entered.set()
            if not release.wait(3):
                raise AssertionError("create release missing")
            return response
        a = Agent([slow])
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        c._stream_stale_timeout = 180
        c._call_done, c._monitor_interrupted = threading.Event(), {"yes": False}
        c.worker = threading.Thread(target=c._run_call)
        c.worker.start()
        try:
            self.assertTrue(entered.wait(3))
            a._interrupt_requested = True
            c._abort_for_interrupt(0)
        finally:
            release.set()
            c.worker.join(3)
        self.assertFalse(c.worker.is_alive())
        self.assertEqual(a.dispatches, 1)
        self.assertEqual(c._chat_image_markers, [])
        self.assertNotIn("base64", response.model_dump_json())
        self.assertIn("cancelled_attempt_late_connect", a.aborted)
    def test_stale_watchdog_after_image_preserves_artifact_without_retry(self):
        entered, release = threading.Event(), threading.Event()
        class PausedStream(SDKStream):
            def __iter__(self):
                yield chunk("caption", image=True)
                entered.set()
                if not release.wait(3):
                    raise AssertionError("stream release missing")
                raise httpx.ReadError("synthetic watchdog reset")
        a = Agent([PausedStream([])], False)
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        c._stream_stale_timeout = 180
        c._call_done = threading.Event()
        worker = threading.Thread(target=c._run_call)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            c._kill_stale_stream(181)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        r = c._partial_stream_stub()
        self.image_preserved(a, c, r)
        self.assertIn("caption", r.choices[0].message.content)
        self.assertIsNone(r.choices[0].message.tool_calls)

    def test_inline_driver_image_eof(self):
        a = Agent([SDKStream([chunk("caption", image=True)])], False)
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        with patch.object(cch, "should_use_direct_api_call", return_value=True):
            r = c.run()
        self.assertIsNone(c.worker)
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].finish_reason, "length")

    def test_clean_eof_usage_does_not_complete_image_tool(self):
        usage = ChatCompletionChunk.model_validate({"id": "synthetic", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture", "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        a, c, r = self.run_call([chunk(image=True, tools=tool('{}')), usage])
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].finish_reason, "length")
        self.assertIsNone(r.choices[0].message.tool_calls)

    def test_completed_response_materializes_through_worker(self):
        response = ChatCompletion.model_validate({"id": "synthetic", "object": "chat.completion", "created": 0, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "caption", "images": [{"image_url": {"url": URL}}]}}]})
        a = Agent([response], False)
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        r = c.run()
        self.image_preserved(a, c, r)
        self.assertEqual(r.choices[0].finish_reason, "stop")
        self.assertTrue(a._disable_streaming)

    def test_stale_watchdog_while_create_pending_fences_image_and_retries(self):
        entered, release = threading.Event(), threading.Event()
        late = SDKStream([chunk(image=True), chunk(finish="stop")])
        def slow():
            entered.set()
            if not release.wait(3):
                raise AssertionError("create release missing")
            return late
        a = Agent([slow, SDKStream([chunk("recovered", finish="stop")])])
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        c._stream_stale_timeout = 180
        c._call_done = threading.Event()
        worker = threading.Thread(target=c._run_call)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            c._kill_stale_stream(181)
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(c._chat_image_markers, [])
        self.assertEqual(a.dispatches, 2)
        self.assertEqual(c.result["response"].choices[0].message.content, "recovered")
        self.assertIn("cancelled_attempt_late_connect", a.aborted)
        self.assertTrue(late.closed)

    def test_slow_old_stream_cannot_materialize_after_new_writer(self):
        entered, release = threading.Event(), threading.Event()
        late = SDKStream([chunk(image=True), chunk(finish="stop")])
        def slow():
            entered.set()
            if not release.wait(3):
                raise AssertionError("create release missing")
            return late
        a = Agent([slow, SDKStream([chunk("new", True, finish="stop")])])
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        c._call_done = threading.Event()
        worker = threading.Thread(target=c._run_call)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            newer = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
            r = newer.run()
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(c._chat_image_markers, [])
        self.image_preserved(a, newer, r, 2)
        self.assertTrue(late.closed)

    def test_slow_old_completed_response_cannot_steal_new_writer(self):
        entered, release = threading.Event(), threading.Event()
        old = ChatCompletion.model_validate({"id": "old", "object": "chat.completion", "created": 0, "model": "fixture",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "images": [{"image_url": {"url": URL}}]}}]})
        def slow():
            entered.set()
            if not release.wait(3):
                raise AssertionError("create release missing")
            return old
        a = Agent([slow, SDKStream([chunk("new", True, finish="stop")])])
        c = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
        c._call_done = threading.Event()
        worker = threading.Thread(target=c._run_call)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            newer = cch._StreamingCall(a, {"model": "fixture", "messages": []}, None)
            r = newer.run()
        finally:
            release.set()
            worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(c._chat_image_markers, [])
        self.assertIsInstance(c.result["error"], InterruptedError)
        self.image_preserved(a, newer, r, 2)
        self.assertNotIn("base64", old.model_dump_json())

if __name__ == "__main__":
    unittest.main(verbosity=2)
