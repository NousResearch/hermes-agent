"""Terminal structured model-output errors, exercised through the real SDK and turn loop.

The wire fixture is self-contained: no model, external endpoint or backend package.
Unittest assertions also permit a dependency-light ``python -m unittest`` run.
"""

import contextlib
import json
import os
from pathlib import Path
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import Mock, patch

import httpx
import openai


NATIVE = {
    "code": "unavailable_tool_call",
    "type": "model_output_error",
    "source": "tensorfold.tool_output_guard",
    "message": "fixture rejection",
}
TOOL = {
    "type": "function",
    "function": {
        "name": "native_output_fixture_echo",
        "description": "Echo fixture text",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}},
                       "required": ["text"]},
    },
}
TOOL_CALL = {"id": "call_fixture_exact", "type": "function",
             "function": {"name": TOOL["function"]["name"], "arguments": '{"text":"fixture"}'}}


def _chunk(delta, finish=None):
    return {"id": "fixture", "object": "chat.completion.chunk", "created": 0,
            "model": "fixture", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


@contextlib.contextmanager
def _wire(mode):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            requests.append(payload)
            # A forbidden second attempt succeeds, making concealed recovery observable.
            rejection = mode in ("json", "handshake", "sse", "partial") and len(requests) == 1
            if mode == "ordinary500" and len(requests) == 1:
                status, kind = 500, "application/json"
                body = json.dumps({"error": {"code": "backend_fault", "type": "server_error",
                                             "message": "ordinary backend failure"}})
            elif rejection and mode in ("json", "handshake"):
                status, kind, body = 500, "application/json", json.dumps({"error": NATIVE})
            elif rejection:
                events = []
                if mode == "partial":
                    events = [_chunk({"role": "assistant", "content": "Visible partial text."}),
                              _chunk({"tool_calls": [{"index": 0, **TOOL_CALL}]})]
                events.append({"error": NATIVE})
                status, kind = 200, "text/event-stream"
                body = "".join("data: " + json.dumps(event) + "\n\n" for event in events)
            else:
                tool_turn = mode == "tool" and len(requests) == 1
                content = None if tool_turn else "fixture answer"
                calls = [TOOL_CALL] if tool_turn else None
                finish = "tool_calls" if tool_turn else "stop"
                status = 200
                if payload.get("stream"):
                    delta = {"role": "assistant", "content": content}
                    if calls:
                        delta["tool_calls"] = [{"index": 0, **TOOL_CALL}]
                    kind = "text/event-stream"
                    body = "".join("data: " + json.dumps(event) + "\n\n"
                                   for event in [_chunk(delta), _chunk({}, finish)]) + "data: [DONE]\n\n"
                else:
                    kind = "application/json"
                    body = json.dumps({"id": "fixture", "object": "chat.completion", "created": 0,
                                       "model": "fixture", "choices": [{"index": 0,
                                       "message": {"role": "assistant", "content": content, "tool_calls": calls},
                                       "finish_reason": finish}]})
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class NativeToolOutputTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.root = root
        self.stack.enter_context(patch.dict(os.environ, {
            "HOME": str(root), "HERMES_HOME": str(root), "HERMES_RUNTIME_DIR": str(root / "runtime"),
        }))
        self.stack.enter_context(patch.object(Path, "home", return_value=root))
        connect = socket.socket.connect

        def loopback_only(sock, address):
            if not isinstance(address, tuple) or address[0] not in ("127.0.0.1", "::1"):
                raise AssertionError(f"Non-loopback connection attempted: {address!r}")
            return connect(sock, address)

        self.stack.enter_context(patch.object(socket.socket, "connect", loopback_only))
        # Match tests/agent/conftest.py when run directly with unittest too.
        self.stack.enter_context(patch("agent.retry_utils.jittered_backoff", return_value=0.0))

    def _agent(self, url, stream, *, with_tool=False):
        from run_agent import AIAgent
        from tools.registry import registry

        if with_tool:
            registry.register(name=TOOL["function"]["name"], toolset="native_output_fixture",
                              schema=TOOL["function"], handler=lambda args, **kw: json.dumps({"echo": args["text"]}),
                              check_fn=lambda: True)
            self.addCleanup(registry.deregister, TOOL["function"]["name"])

        (self.root / "config.yaml").write_text(json.dumps({
            "model": {"streaming": stream}, "tools": {"tool_search": {"enabled": "off"}},
            "memory": {"memory_enabled": False, "user_profile_enabled": False},
        }), encoding="utf-8")
        agent = AIAgent(
            base_url=url, api_key="fixture-only", provider="custom", api_mode="chat_completions",
            model="fixture", max_iterations=3,
            enabled_toolsets=["native_output_fixture"] if with_tool else [], quiet_mode=True,
            skip_context_files=True, skip_memory=True, skip_background_review=True,
            load_soul_identity=False, save_trajectories=False, checkpoints_enabled=False,
            cwd=str(self.root), stream_delta_callback=(lambda _delta: None) if stream else None,
            fallback_model={"provider": "custom", "model": "fixture-fallback",
                            "base_url": url, "api_key": "fixture-fallback-only"},
        )
        self.addCleanup(agent.client.close)
        return agent

    def test_exact_structured_contract_and_ordinary_errors(self):
        from agent.error_classifier import classify_api_error

        request = httpx.Request("POST", "http://127.0.0.1/v1/chat/completions")
        for body in (NATIVE, {"error": NATIVE}):
            for error in (openai.APIError("fixture", request=request, body=body),
                          openai.InternalServerError("fixture", response=httpx.Response(500, request=request), body=body)):
                verdict = classify_api_error(error, provider="custom")
                self.assertEqual(verdict.reason.value, "native_tool_output")
                self.assertFalse(verdict.retryable or verdict.should_compress or
                                 verdict.should_rotate_credential or verdict.should_fallback)
        bodies = [None, "unavailable_tool_call", [], {"error": None}]
        for key in ("code", "type", "source"):
            for value in (None, "other", NATIVE[key].upper()):
                bodies.append({**NATIVE, key: value})
            bodies.append({k: v for k, v in NATIVE.items() if k != key})
        for body in bodies:
            with self.subTest(body=body):
                error = openai.InternalServerError(
                    "unavailable_tool_call model_output_error tensorfold.tool_output_guard",
                    response=httpx.Response(500, request=request), body=body)
                verdict = classify_api_error(error, provider="custom")
                self.assertEqual(verdict.reason.value, "server_error")
                self.assertTrue(verdict.retryable)
        verdict = classify_api_error(Exception("ordinary unknown error"), provider="custom")
        self.assertEqual(verdict.reason.value, "unknown")
        self.assertTrue(verdict.retryable)

    def test_terminal_wire_errors_notify_once_without_recovery(self):
        from agent.chat_completion_helpers import _StreamingCall
        from hermes_cli import lifecycle
        from tools.registry import registry

        for mode in ("json", "handshake", "sse", "partial"):
            for hook_return in (None, True):
                with self.subTest(mode=mode, hook_return=hook_return), _wire(mode) as (url, requests):
                    agent = self._agent(url, mode != "json")
                    visible = []
                    agent.stream_delta_callback = visible.append if mode != "json" else None
                    original_hook = agent._invoke_api_request_error_hook

                    def notify(**kwargs):
                        original_hook(**kwargs)
                        return hook_return

                    with contextlib.ExitStack() as stack:
                        stack.enter_context(patch.object(lifecycle, "has_hook", side_effect=lambda name: name == "api_request_error"))
                        dispatch = stack.enter_context(patch.object(lifecycle, "invoke_hook", return_value=[]))
                        notification = stack.enter_context(patch.object(agent, "_invoke_api_request_error_hook", side_effect=notify))
                        blocked = [stack.enter_context(patch.object(agent, name, side_effect=AssertionError(name)))
                                   for name in ("_compress_context", "_recover_with_credential_pool", "_try_activate_fallback")]
                        pool = Mock()
                        stack.enter_context(patch.object(agent, "_credential_pool", pool))
                        tool = stack.enter_context(patch.object(registry, "dispatch", side_effect=AssertionError("tool executed")))
                        stub = stack.enter_context(patch.object(
                            _StreamingCall, "_partial_stream_stub", autospec=True,
                            side_effect=_StreamingCall._partial_stream_stub))
                        result = agent.run_conversation("Fixture request")
                    self.assertEqual(len(requests), 1, "retry/fallback/unmask probe occurred")
                    self.assertEqual(requests[0]["model"], "fixture")
                    self.assertEqual(bool(requests[0].get("stream")), mode != "json")
                    self.assertTrue(result.get("failed"), result)
                    self.assertFalse(result.get("completed"), result)
                    self.assertEqual(result.get("failure_reason"), "native_tool_output")
                    self.assertIs(result.get("failure_retryable"), False)
                    self.assertIn("Native tool output rejected", result["final_response"])
                    notification.assert_called_once()
                    kwargs = notification.call_args.kwargs
                    self.assertEqual(kwargs["reason"], "native_tool_output")
                    self.assertIs(kwargs["retryable"], False)
                    self.assertEqual(kwargs["retry_count"], 0)
                    self.assertEqual(kwargs["api_call_count"], 1)
                    self.assertEqual(kwargs["status_code"], 500 if mode in ("json", "handshake") else None)
                    for key in ("task_id", "turn_id", "api_request_id", "api_start_time", "api_kwargs", "error_type", "error_message", "max_retries"):
                        self.assertIn(key, kwargs)
                    errors = [call for call in dispatch.call_args_list if call.args == ("api_request_error",)]
                    self.assertEqual(len(errors), 1)
                    self.assertIs(errors[0].kwargs["retryable"], False)
                    for spy in (*blocked, tool, stub):
                        spy.assert_not_called()
                    pool.mark_exhausted_and_rotate.assert_not_called()
                    self.assertEqual(agent._fallback_index, 0)
                    self.assertFalse(agent._fallback_activated)
                    if mode == "partial":
                        self.assertIn("Visible partial text.", "".join(visible))

    def test_normal_and_tool_id_flows_remain(self):
        from tools.registry import registry

        for stream in (False, True):
            for mode in ("normal", "tool", "ordinary500"):
                with self.subTest(stream=stream, mode=mode), _wire(mode) as (url, requests):
                    agent = self._agent(url, stream, with_tool=mode == "tool")
                    with patch.object(registry, "dispatch", wraps=registry.dispatch) as tool:
                        result = agent.run_conversation("Fixture request")
                    self.assertFalse(result.get("failed"), result)
                    self.assertEqual(result["final_response"], "fixture answer")
                    self.assertEqual(len(requests), 1 if mode == "normal" else 2)
                    if mode == "tool":
                        tool.assert_called_once()
                        rows = [m for m in requests[-1]["messages"] if m["role"] == "tool"]
                        self.assertEqual(rows[0]["tool_call_id"], "call_fixture_exact")
                        self.assertEqual(json.loads(rows[0]["content"]), {"echo": "fixture"})
                    else:
                        tool.assert_not_called()


if __name__ == "__main__":
    unittest.main()
