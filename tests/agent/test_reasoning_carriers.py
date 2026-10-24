"""Gemini text-turn signatures and Copilot reasoning carriers survive capture and reach only their route."""

import copy
from types import SimpleNamespace

from openai.types.chat import ChatCompletion

from agent.chat_completion_helpers import build_assistant_message
from agent.gemini_native_adapter import _build_gemini_contents, translate_gemini_response, translate_stream_event
from agent.reasoning_carriers import shape_wire_carriers
from agent.transports.chat_completions import ChatCompletionsTransport


class _Agent:
    stream_delta_callback = _stream_callback = reasoning_callback = None
    verbose_logging = False

    def _extract_reasoning(self, msg):
        from agent.agent_runtime_helpers import extract_reasoning
        return extract_reasoning(self, msg)

    def _strip_think_blocks(self, text):
        return text

    def _needs_thinking_reasoning_pad(self):
        return False

    def _split_responses_tool_id(self, raw):
        return (None, None)

    def _derive_responses_function_call_id(self, call_id, item_id):
        return item_id

    def _deterministic_call_id(self, *a):
        return "call_x"


def _stored(response):
    return build_assistant_message(_Agent(), ChatCompletionsTransport().normalize_response(response), "stop")


def test_gemini_native_text_turn_signature_is_captured_and_replayed_on_the_last_part():
    resp = {"candidates": [{"content": {"parts": [{"text": "thinking", "thought": True},
                                                  {"text": "391", "thoughtSignature": "SIG_TEXT"}]},
                            "finishReason": "STOP"}]}
    stored = _stored(translate_gemini_response(resp, "gemini-3.6-flash"))
    # Streams deliver the signature on a trailing empty-text part.
    chunks = translate_stream_event({"candidates": [{"content": {"parts": [{"text": "", "thoughtSignature": "SIG_S"}]}}]},
                                    "gemini-3.6-flash", {})
    assert [c.choices[0].delta.extra_content for c in chunks] == [{"google": {"thought_signature": "SIG_S"}}]

    history = [{"role": "user", "content": "q"}, stored, {"role": "user", "content": "again"}]
    resumed = [{k: v for k, v in m.items() if k != "extra_content"} for m in history]  # state.db keeps only the record
    for msgs in (history, resumed):
        wire = shape_wire_carriers(msgs, model="gemini-3.6-flash", base_url="https://generativelanguage.googleapis.com/v1beta")
        contents, _ = _build_gemini_contents(wire, is_gemini3=True)
        assert contents[1]["parts"][-1] == {"text": "391", "thoughtSignature": "SIG_TEXT"}


def test_copilot_reasoning_and_function_signature_reach_only_copilot():
    resp = ChatCompletion.model_validate({
        "id": "c", "object": "chat.completion", "created": 0, "model": "gemini-3.6-pro",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None, "reasoning_text": "plan it", "reasoning_opaque": "OPAQUE",
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "f", "arguments": "{}", "thought_signature": "FSIG"}}]}}]})
    stored = _stored(resp)
    assert stored["reasoning"] == "plan it"
    snapshot = copy.deepcopy(stored)

    copilot = shape_wire_carriers([stored], model="gemini-3.6-pro", base_url="https://api.githubcopilot.com")[0]
    assert (copilot["reasoning_opaque"], copilot["reasoning_text"]) == ("OPAQUE", "plan it")
    assert copilot["tool_calls"][0]["function"]["thought_signature"] == "FSIG"
    assert "reasoning_details" not in copilot

    for base_url in ("https://api.fireworks.ai/inference/v1", "https://openrouter.ai/api/v1"):
        other = ChatCompletionsTransport().convert_messages(
            shape_wire_carriers([stored], model="claude-sonnet-5", base_url=base_url), model="claude-sonnet-5",
            base_url=base_url)[0]
        assert not {"reasoning_opaque", "reasoning_text", "extra_content"} & other.keys()
        assert "thought_signature" not in other["tool_calls"][0]["function"]
        assert all(d.get("type") != "hermes.native_assistant" for d in other.get("reasoning_details") or ())
    assert stored == snapshot


def test_vertex_openai_compat_streamed_google_thought_routes_to_reasoning_not_content():
    from openai.types.chat import ChatCompletionChunk

    from agent.auxiliary_client import _ChatStreamAccumulator
    from agent.chat_completion_helpers_relay import RelayChatAccumulator
    from agent.reasoning_carriers import StreamCarriers, delta_is_google_thought

    raw_chunks = [
        {
            "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "google/gemini-4-argon",
            "choices": [{
                "index": 0, "finish_reason": None,
                "delta": {
                    "role": "assistant",
                    "content": "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n",
                    "extra_content": {"google": {"thought": True}},
                },
            }],
        },
        {
            "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "google/gemini-4-argon",
            "choices": [{"index": 0, "finish_reason": None, "delta": {"role": "assistant", "content": "pong"}}],
        },
        {
            "id": "c1", "object": "chat.completion.chunk", "created": 1, "model": "google/gemini-4-argon",
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "delta": {"content": "", "extra_content": {"google": {"thought_signature": "SIG_VERTEX"}}},
            }],
        },
    ]
    sdk_chunks = [ChatCompletionChunk.model_validate(c) for c in raw_chunks]

    # 1. StreamCarriers + delta_is_google_thought
    carriers = StreamCarriers()
    content_parts, reasoning_parts = [], []
    for chunk in sdk_chunks:
        delta = chunk.choices[0].delta
        r_text = carriers.feed(delta, None)
        if r_text:
            reasoning_parts.append(r_text)
        c_text = "" if delta_is_google_thought(delta) else (delta.content or "")
        if c_text:
            content_parts.append(c_text)
    assert "".join(content_parts) == "pong"
    assert "".join(reasoning_parts) == "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n"
    assert carriers.signature == "SIG_VERTEX"
    assert carriers.text_parts == []

    # 2. RelayChatAccumulator
    relay = RelayChatAccumulator()
    for raw in raw_chunks:
        relay.observe(raw)
    finalized = relay.finalize()["choices"][0]["message"]
    assert finalized["content"] == "pong"
    assert finalized["reasoning_content"] == "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n"

    # 3. Auxiliary _ChatStreamAccumulator
    aux = _ChatStreamAccumulator(model="google/gemini-4-argon")
    for chunk in sdk_chunks:
        aux.feed(chunk)
    aux_msg = aux.finish().choices[0].message
    assert aux_msg.content == "pong"
    assert aux_msg.reasoning == "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n"

    # 4. Full interruptible_streaming_api_call + build_assistant_message end-to-end
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from agent.chat_completion_helpers import interruptible_streaming_api_call
    from run_agent import AIAgent

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for raw in raw_chunks:
                self.wfile.write(f"data: {json.dumps(raw)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        agent = AIAgent(
            api_key="local-test", base_url=f"http://127.0.0.1:{server.server_port}/v1",
            provider="vertex", model="google/gemini-4-argon",
            platform="subagent", quiet_mode=True, skip_memory=True, skip_context_files=True,
            enabled_toolsets=[], save_trajectories=False,
        )
        streamed_deltas = []
        agent.stream_delta_callback = streamed_deltas.append
        response = interruptible_streaming_api_call(
            agent, {"model": "google/gemini-4-argon", "messages": [{"role": "user", "content": "ping"}]},
        )
        stored_msg = _stored(response)
        assert "".join(streamed_deltas) == "pong"
        assert stored_msg["content"] == "pong"
        assert stored_msg["reasoning"] == "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n"
        assert stored_msg["reasoning_content"] == "**Responding with Pong**\n\nMy current thinking is to reply concisely.\n\n"
        assert stored_msg["extra_content"] == {"google": {"thought_signature": "SIG_VERTEX"}}
    finally:
        server.shutdown()
