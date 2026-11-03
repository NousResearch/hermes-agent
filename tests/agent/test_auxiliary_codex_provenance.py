"""Auxiliary Responses results retain provider-reported identity and usage.

The request alias is not evidence of the model that served the response, and an
omitted token count must remain distinguishable from an explicit zero.
"""

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from types import SimpleNamespace

import pytest
from openai import OpenAI

from agent.auxiliary_client import _CodexCompletionsAdapter


@contextmanager
def _provider(response, streamed, *, terminal=True):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if streamed:
                events = [dict(type="response.output_item.done", item=response["output"][0])]
                if terminal:
                    events.append(dict(type="response." + response["status"], response=response))
                body = "".join("data: " + json.dumps(event) + "\n\n" for event in events) + "data: [DONE]\n\n"
                content_type = "text/event-stream"
            else:
                body, content_type = json.dumps(response), "application/json"
            raw = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *_args):
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}/v1"
        finally:
            server.shutdown()
            thread.join()


def _response(**fields):
    return dict(
        id="resp_provenance123", object="response", created_at=1,
        status="completed", model="served-model-2026", error=None,
        incomplete_details=None,
        output=[dict(id="msg_provenance123", type="message", role="assistant", status="completed",
                     phase="final_answer", content=[dict(type="output_text", text="Provider answer.", annotations=[])])],
        **fields,
    )


def _complete(response, streamed, *, terminal=True):
    with _provider(response, streamed, terminal=terminal) as url:
        with OpenAI(api_key="local-test-key", base_url=url, max_retries=0) as client:
            # Compatible hosts may return a Response object even when the caller
            # asks for streaming. Keep SDK parsing real on both paths.
            host = SimpleNamespace(base_url=url, responses=SimpleNamespace(
                create=lambda **kw: client.responses.create(**{**kw, "stream": streamed}),
            ))
            return _CodexCompletionsAdapter(host, "requested-alias").create(
                messages=[dict(role="user", content="Answer the question.")],
            )


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
def test_provider_identity_and_detailed_usage_survive_the_actual_sdk(streamed):
    response = _response(usage=dict(
        input_tokens=11, input_tokens_details=dict(cached_tokens=7), output_tokens=3,
        output_tokens_details=dict(reasoning_tokens=1), total_tokens=14,
    ))
    result = _complete(response, streamed)

    assert getattr(result, "id", None) == "resp_provenance123"
    assert result.model == "served-model-2026"
    assert getattr(result, "status", None) == "completed"
    assert result.choices[0].message.content == "Provider answer."
    assert result.choices[0].finish_reason == "stop"
    assert (result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens) == (11, 3, 14)
    assert result.usage.input_tokens_details["cached_tokens"] == 7
    assert result.usage.output_tokens_details["reasoning_tokens"] == 1


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
def test_incomplete_details_are_reported_without_changing_partial_answer_gates(streamed):
    response = _response(usage=dict(input_tokens=11, output_tokens=3, total_tokens=14))
    response.update(status="incomplete", incomplete_details=dict(reason="max_output_tokens"))
    result = _complete(response, streamed)

    assert getattr(result, "status", None) == "incomplete"
    assert result.incomplete_details["reason"] == "max_output_tokens"
    assert result.choices[0].finish_reason == "length"


@pytest.mark.parametrize("streamed", [False, True], ids=["response-object", "sse"])
def test_absent_provider_fields_are_not_filled_from_the_request_or_zero(streamed):
    response = _response(usage=dict(output_tokens=0))
    del response["model"]
    result = _complete(response, streamed)

    assert result.model is None
    assert getattr(result.usage, "prompt_tokens", None) is None
    assert result.usage.completion_tokens == 0
    assert getattr(result.usage, "total_tokens", None) is None


def test_stream_without_a_terminal_frame_does_not_invent_provider_identity():
    result = _complete(_response(), True, terminal=False)

    assert getattr(result, "id", None) is None
    assert result.model is None
    assert getattr(result, "status", None) is None
    assert result.usage is None
