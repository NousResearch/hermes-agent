"""Chat-completions request-transform bypass (#93650 extended to chat.completions).

``chat.completions.create`` re-walks the whole request body against the
``CompletionCreateParams`` union client-side, GIL held, before any byte leaves
the process. The bypass moves the already-wire-format bulk fields into
``extra_body``; its whole safety argument is that the server receives the same
bytes, so that is what these tests pin.
"""

import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import httpx
import openai

from agent.sdk_transform_bypass import ESCAPE_HATCH_ENV, bypass_chat_sdk_request_transform
from openai.resources.chat import completions as _sdk_completions

_SSE = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)


def _wire_body() -> dict:
    """Production-shaped chat body: content parts incl. an image, a tool_calls turn, a tool
    result, function tool schemas, and a caller-populated extra_body (reasoning/provider)."""
    return {
        "model": "hermes-4-70b",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {"role": "user", "content": [
                {"type": "text", "text": "look at this"},
                {"type": "image_url", "image_url": {"url": "https://e.example/i.png", "detail": "low"}},
            ]},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'}},
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "total 0"},
        ],
        "tools": [
            {"type": "function", "function": {"name": f"tool_{i}", "description": "d",
             "parameters": {"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]}}}
            for i in range(3)
        ],
        "tool_choice": "auto",
        "stream": True,
        "temperature": 0.7,
        "stream_options": {"include_usage": True},
        "extra_body": {"reasoning": {"effort": "high"}, "provider": {"order": ["nous"]}},
    }


class _Recorder:
    """A real openai.OpenAI client whose transport records the request bytes."""

    def __init__(self):
        self.content: bytes | None = None
        self.client = openai.OpenAI(
            api_key="k", base_url="https://chat.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.content = request.content
        return httpx.Response(200, content=_SSE, headers={"content-type": "text/event-stream"})

    def send(self, kwargs: dict) -> bytes:
        for _ in self.client.chat.completions.create(**kwargs):
            pass
        assert self.content is not None
        return self.content


def test_bulk_fields_ride_in_extra_body_and_the_wire_bytes_are_identical():
    """Same bytes → same server behaviour and the same byte-keyed prompt-cache prefix."""
    recorder = _Recorder()
    body = _wire_body()

    moved = bypass_chat_sdk_request_transform(dict(body), recorder.client)

    assert moved["messages"] == [] and moved["tools"] == []
    assert moved["extra_body"]["messages"] == body["messages"]
    assert moved["extra_body"]["tools"] == body["tools"]
    assert moved["extra_body"]["reasoning"] == body["extra_body"]["reasoning"]
    assert recorder.send(moved) == recorder.send(dict(body))


def test_escape_hatch_and_non_sdk_facades_keep_the_typed_path(monkeypatch):
    """Both rails hand the kwargs back untouched: the env hatch, and a chat-shaped facade
    that is not the SDK (MoA aggregator, test stand-ins) — it never merges ``extra_body``."""
    recorder = _Recorder()
    kwargs = _wire_body()

    facade = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: kw)))
    assert bypass_chat_sdk_request_transform(kwargs, facade) is kwargs

    monkeypatch.setenv(ESCAPE_HATCH_ENV, "1")
    assert bypass_chat_sdk_request_transform(kwargs, recorder.client) is kwargs


def _capture_sdk_create(monkeypatch) -> list[dict]:
    """Record the kwargs that reach the SDK's ``Completions.create`` (the transform boundary)."""
    seen: list[dict] = []
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok", tool_calls=None), finish_reason="stop")],
        usage=None, model="m", id="1")
    monkeypatch.setattr(_sdk_completions.Completions, "create", lambda self, **kw: seen.append(kw) or response)
    return seen


def test_auxiliary_completion_path_hands_the_sdk_only_the_placeholder(monkeypatch):
    """Aux tasks (compression, summaries) reach the SDK via ``_create_with_progress_once``;
    the bulk conversation must ride in ``extra_body`` so the SDK's typed walk sees only ``[]``."""
    from agent import auxiliary_client

    seen = _capture_sdk_create(monkeypatch)
    client = _Recorder().client
    body = {"model": "m", "messages": [{"role": "user", "content": "x" * 4096}], "max_tokens": 8}

    auxiliary_client._relay_sync_completion(client, dict(body))

    assert len(seen) == 1
    assert seen[0]["messages"] == []
    assert seen[0]["extra_body"]["messages"] == body["messages"]


def test_iteration_summary_path_hands_the_sdk_only_the_placeholder(monkeypatch):
    """The iteration-limit summary builds the full main-loop kwargs (``_build_api_kwargs``) and calls
    ``chat.completions.create`` itself — the same multi-MB payload, so the same bypass."""
    from agent import chat_completion_helpers

    seen = _capture_sdk_create(monkeypatch)
    client = _Recorder().client
    body = {"model": "m", "messages": [{"role": "user", "content": "x" * 4096}], "tools": _wire_body()["tools"]}
    transport = types.SimpleNamespace(normalize_response=lambda response, **kw: types.SimpleNamespace(content="ok", tool_calls=None))
    agent = types.SimpleNamespace(
        provider="p", model="m", api_mode="chat_completions", _force_ascii_payload=False,
        _build_api_kwargs=lambda messages: dict(body), _ensure_primary_openai_client=lambda reason: client,
        _get_transport=lambda: transport)

    assert chat_completion_helpers._chat_summary_attempt(agent, body["messages"], "req-1")(0) == "ok"
    assert len(seen) == 1
    assert seen[0]["messages"] == [] and seen[0]["tools"] == []
    assert seen[0]["extra_body"]["messages"] == body["messages"]
    assert seen[0]["extra_body"]["tools"] == body["tools"]


def test_tool_budget_summary_disables_tools_before_sdk_dispatch(monkeypatch):
    from agent import chat_completion_helpers

    seen = _capture_sdk_create(monkeypatch)
    client = _Recorder().client
    body = {"model": "m", "messages": [{"role": "user", "content": "evidence"}]}
    build_calls = []

    def build(messages, tools_for_api=None):
        build_calls.append(tools_for_api)
        return {
            **body,
            "messages": messages,
            "tools": _wire_body()["tools"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

    transport = types.SimpleNamespace(
        normalize_response=lambda response, **kw: types.SimpleNamespace(
            content="bounded answer", tool_calls=None
        )
    )
    agent = types.SimpleNamespace(
        provider="p", model="m", api_mode="chat_completions",
        _force_ascii_payload=False, _build_api_kwargs=build,
        _ensure_primary_openai_client=lambda reason: client,
        _get_transport=lambda: transport,
    )

    result = chat_completion_helpers._chat_summary_attempt(
        agent, body["messages"], "req-budget", disable_tools=True
    )(0)

    assert result == "bounded answer"
    assert build_calls == [[]]
    assert len(seen) == 1
    assert "tools" not in seen[0]
    assert "tool_choice" not in seen[0]
    assert "parallel_tool_calls" not in seen[0]


def test_relay_stream_path_shows_relay_the_full_conversation(monkeypatch):
    """On the Relay-managed stream path the bypass must run INSIDE the provider callback: Relay's
    tracing/intercepts see the real ``messages`` while the SDK still gets only the placeholder."""
    from agent import auxiliary_client, relay_llm

    seen = _capture_sdk_create(monkeypatch)
    client = _Recorder().client
    body = {"model": "m", "messages": [{"role": "user", "content": "x" * 4096}], "stream": True}
    relay_saw: list[dict] = []

    def fake_stream_current(request, provider_call, **_kw):
        relay_saw.append(dict(request))
        return provider_call(request)

    monkeypatch.setattr(relay_llm, "stream_current", fake_stream_current)
    monkeypatch.setattr(
        auxiliary_client, "_relay_auxiliary_metadata", lambda **_kw: ("openrouter", "m", {}))

    auxiliary_client._relay_sync_stream(client, dict(body))

    assert relay_saw[0]["messages"] == body["messages"]
    assert seen[0]["messages"] == []
    assert seen[0]["extra_body"]["messages"] == body["messages"]
