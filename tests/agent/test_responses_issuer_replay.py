"""Issuer provenance must protect provider switches without rewriting history."""

from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from agent.codex_responses_adapter import (
    _chat_messages_to_responses_input,
    _classify_responses_issuer,
    _normalize_codex_response,
    _preflight_codex_input_items,
)


def _history(item_id, issuer):
    message = {
        "type": "message", "role": "assistant", "id": item_id,
        "status": "completed", "phase": "final_answer",
        "content": [{"type": "output_text", "text": "Previous answer"}],
    }
    reasoning = {"type": "reasoning", "id": "rs_previous", "encrypted_content": "sealed", "summary": []}
    if issuer is not None:
        message["_issuer_kind"] = reasoning["_issuer_kind"] = issuer
    return [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Previous answer", "codex_message_items": [message],
         "codex_reasoning_items": [reasoning]},
        {"role": "user", "content": "Follow-up"},
    ]


@pytest.mark.parametrize("current,stored,item_id,keep_id,keep_reasoning", [
    ("codex_backend", None, "a7ca4fd3c0b6ffcf", False, True),
    ("codex_backend", None, "msg_legacy", True, True),
    ("codex_backend", "codex_backend", "msg_native", True, True),
    ("codex_backend", "codex_backend", "msgWithoutUnderscore", True, True),
    ("codex_backend", "codex_backend", "foreign_short_id", False, True),
    ("codex_backend", "codex_backend", "msg" + "x" * 62, False, True),
    ("codex_backend", "xai_responses", "msg_foreign", False, False),
    ("xai_responses", "codex_backend", "msg_foreign", False, False),
    ("xai_responses", "xai_responses", "native_uuid", True, True),
    ("xai_responses", None, "legacy_uuid", True, True),
    ("github_responses", "github_responses", "msg_connection_scoped", False, True),
    (None, "xai_responses", "legacy_uuid", True, True),
    ("other:https://responses.example/v1", "other:https://RESPONSES.EXAMPLE:443/v1/", "native_uuid", True, True),
    ("other:http://responses.example/v1", "other:http://RESPONSES.EXAMPLE:80/v1/", "native_uuid", True, True),
    ("other:https://responses.example/v1", "other:https://different.example/v1", "msg_foreign", False, False),
    ("other:https://responses.example/v2", "other:https://responses.example/v1", "msg_foreign", False, False),
    ("other:https://api.x.ai/v1", "other:https://API.X.AI:443/v1/", "native_uuid", True, True),
    ("other:https://responses.example:invalid/v1", "other:https://responses.example:invalid/v1/", "native_uuid", True, True),
])
def test_replay_preserves_content_and_only_reuses_current_issuer_ids(current, stored, item_id, keep_id, keep_reasoning):
    history = _history(item_id, stored)
    before = deepcopy(history)
    output = _chat_messages_to_responses_input(
        history, current_issuer_kind=current, is_github_responses=current == "github_responses",
    )
    output = _preflight_codex_input_items(output, is_github_responses=current == "github_responses")
    message = next(item for item in output if item.get("type") == "message")
    assert ("id" in message) is keep_id
    if keep_id:
        assert message["id"] == item_id
    assert message["content"] == before[1]["codex_message_items"][0]["content"]
    assert message["phase"] == "final_answer"
    assert message["status"] == "completed"
    assert any(item.get("type") == "reasoning" for item in output) is keep_reasoning
    assert all("_issuer_kind" not in item for item in output)
    assert history == before

    # Provider identity stays flag-based: normalizing a URL must not promote it to a named issuer.
    if current and current.startswith("other:"):
        configured = current[len("other:"):]
        try:
            httpx.URL(configured)
        except httpx.InvalidURL:
            assert _classify_responses_issuer(base_url=configured + "/") == current
        else:
            with OpenAI(api_key="test-only", base_url=configured) as client:
                assert _classify_responses_issuer(base_url=configured) == _classify_responses_issuer(base_url=client.base_url)
                assert _classify_responses_issuer(base_url=configured).startswith("other:")

    response = SimpleNamespace(status="completed", output=[SimpleNamespace(
        type="message", role="assistant", id=item_id, status="completed", phase="final_answer",
        content=[SimpleNamespace(type="output_text", text="Previous answer")],
    )])
    normalized, _ = _normalize_codex_response(response, issuer_kind=stored)
    assert normalized.codex_message_items[0].get("_issuer_kind") == stored


@pytest.fixture
def responses_endpoint():
    received = []
    output_item = {
        "type": "message", "id": "msg_reply", "role": "assistant", "status": "completed",
        "content": [{"type": "output_text", "text": "Reply", "annotations": []}],
    }
    response = {
        "id": "resp_reply", "object": "response", "created_at": 0, "model": "gpt-test",
        "status": "completed", "output": [output_item],
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            received.append(payload)
            if payload.get("stream"):
                events = [
                    {"type": "response.output_item.done", "output_index": 0, "item": output_item},
                    {"type": "response.completed", "response": response},
                ]
                body = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
                content_type = "text/event-stream"
            else:
                body = json.dumps(response).encode()
                content_type = "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.05), daemon=True)
    thread.start()
    try:
        yield f"http://LOCALHOST:{server.server_port}/v1", received
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("route,stored,item_id,keep_id,keep_reasoning", [
    ("codex", None, "foreign_short_id", False, True),
    ("codex", "codex_backend", "msg_native", True, True),
    ("codex", "xai_responses", "msg_foreign", False, False),
    ("xai", "xai_responses", "native_uuid", True, True),
    ("xai", "codex_backend", "msg_foreign", False, False),
    ("custom", "same-custom", "native_uuid", True, True),
    ("custom", "codex_backend", "msg_foreign", False, False),
])
def test_auxiliary_and_main_transports_apply_same_replay_policy_on_wire(
    route, stored, item_id, keep_id, keep_reasoning, responses_endpoint, monkeypatch,
):
    from agent import auxiliary_client as auxiliary
    from agent.transports.codex import ResponsesApiTransport

    configured, received = responses_endpoint
    if stored == "same-custom":
        stored = "other:" + configured + "/"
    history = _history(item_id, stored)
    before = deepcopy(history)
    # Only the credential source is replaced; builders, client wrappers, SDK serialization,
    # transport conversion and the HTTP/SSE exchange all run for real against a local server.
    if route == "codex":
        monkeypatch.setattr(auxiliary, "_select_pool_entry", lambda _provider: (False, None))
        monkeypatch.setattr(auxiliary, "_read_codex_access_token", lambda: "test-only")
        monkeypatch.setattr(auxiliary, "_CODEX_AUX_BASE_URL", configured)
        client, _ = auxiliary._build_codex_client("gpt-test")
    elif route == "xai":
        monkeypatch.setattr(auxiliary, "_resolve_xai_oauth_for_aux", lambda: ("test-only", configured))
        client, _ = auxiliary._build_xai_oauth_aux_client("gpt-test")
    else:
        client = auxiliary.CodexAuxiliaryClient(OpenAI(api_key="test-only", base_url=configured), "gpt-test")
    try:
        answer = client.chat.completions.create(messages=history, timeout=5)
        assert answer.choices[0].message.content == "Reply"
    finally:
        client.close()

    kwargs = ResponsesApiTransport().build_kwargs(
        model="gpt-test", messages=history, tools=[], base_url=configured,
        is_codex_backend=route == "codex", is_xai_responses=route == "xai",
    )
    with OpenAI(api_key="test-only", base_url=configured, timeout=5, max_retries=0) as main_client:
        main_client.responses.create(**kwargs)
    assert len(received) == 2
    assert received[0]["input"] == received[1]["input"]
    for payload in received:
        message = next(item for item in payload["input"] if item.get("type") == "message")
        assert ("id" in message) is keep_id
        if keep_id:
            assert message["id"] == item_id
        assert message["content"] == before[1]["codex_message_items"][0]["content"]
        assert message["phase"] == "final_answer"
        assert message["status"] == "completed"
        assert any(item.get("type") == "reasoning" for item in payload["input"]) is keep_reasoning
        assert all("_issuer_kind" not in item for item in payload["input"])
    assert history == before
