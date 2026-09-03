"""Chat-completions request-transform bypass (#93650).

``chat.completions.create`` re-walks the whole request body against the
``CompletionCreateParams`` union graph client-side, with the GIL held, before
any byte leaves the process. #93650 documents that class of walk wedging for
12+ hours on a ~1.4 MB conversation, where no in-process watchdog can fire and
no socket kill helps because the hang is pre-network. #93773 fixed it for
``responses.create`` only; these tests cover extending the same, already
reviewed mechanism to the default chat path.

The safety argument is that the request the server receives is unchanged, so
the byte-identity test below is the important one.
"""

import json
import sys
import types

sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())

import httpx
import openai
import pytest

from agent.sdk_transform_bypass import (
    bypass_chat_sdk_request_transform,
    is_openai_sdk_completions,
)

_SSE = (
    b'data: {"id":"1","object":"chat.completion.chunk","created":1,"model":"m",'
    b'"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    b"data: [DONE]\n\n"
)


def _wire_body(tools: bool = True) -> dict:
    """A production-shaped chat body: list content parts, an image part, a
    tool_calls turn, a tool result, and function tool schemas."""
    body = {
        "model": "hermes-4-70b",
        "messages": [
            {"role": "system", "content": "You are Hermes."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look at this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://e.example/i.png", "detail": "low"},
                    },
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "terminal", "arguments": '{"cmd":"ls"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "c1", "content": "total 0"},
        ],
        "stream": True,
        "temperature": 0.7,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": f"tool_{i}",
                    "description": "d",
                    "parameters": {
                        "type": "object",
                        "properties": {"p": {"type": "string"}},
                        "required": ["p"],
                    },
                },
            }
            for i in range(30)
        ]
    return body


class _Recorder:
    """A real openai.OpenAI client whose transport records the request body."""

    def __init__(self):
        self.content: bytes | None = None
        self.client = openai.OpenAI(
            api_key="k",
            base_url="https://chat.invalid/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._handle)),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.content = request.content
        return httpx.Response(
            200, content=_SSE, headers={"content-type": "text/event-stream"}
        )

    def send(self, kwargs: dict) -> bytes:
        stream = self.client.chat.completions.create(**kwargs)
        for _ in stream:
            pass
        assert self.content is not None
        return self.content


class _FacadeCompletions:
    """Stands in for the MoA aggregator: chat-completions shaped, not the SDK."""

    def create(self, **kwargs):  # pragma: no cover - never called here
        raise AssertionError("not exercised")


class _FacadeClient:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=_FacadeCompletions())


class TestByteIdentity:
    def test_request_body_is_byte_identical_with_and_without_the_bypass(self):
        """The whole safety argument: the server sees the same request."""
        recorder = _Recorder()
        body = _wire_body()

        plain = recorder.send(dict(body))
        bypassed = recorder.send(bypass_chat_sdk_request_transform(dict(body), recorder.client))

        assert bypassed == plain
        # And it really did take the bypass, rather than silently no-opping.
        moved = bypass_chat_sdk_request_transform(dict(body), recorder.client)
        assert moved["messages"] == []
        assert moved["extra_body"]["messages"] == body["messages"]
        assert moved["extra_body"]["tools"] == body["tools"]

    def test_the_wire_payload_survives_a_json_round_trip_unchanged(self):
        recorder = _Recorder()
        body = _wire_body()
        sent = json.loads(recorder.send(bypass_chat_sdk_request_transform(dict(body), recorder.client)))

        assert sent["messages"] == body["messages"]
        assert sent["tools"] == body["tools"]
        assert sent["temperature"] == 0.7
        assert sent["stream_options"] == {"include_usage": True}


class TestGuards:
    def test_a_non_sdk_client_is_left_completely_alone(self):
        """MoA's facade never merges extra_body, so moving the conversation
        there would silently send an empty message list."""
        kwargs = _wire_body()

        result = bypass_chat_sdk_request_transform(kwargs, _FacadeClient())

        assert result is kwargs
        assert "extra_body" not in result
        assert result["messages"][0]["role"] == "system"

    def test_is_openai_sdk_completions_discriminates(self):
        assert is_openai_sdk_completions(_Recorder().client)
        assert not is_openai_sdk_completions(_FacadeClient())
        assert not is_openai_sdk_completions(object())

    def test_non_plain_json_messages_stay_on_the_typed_path(self):
        """The transform exists to convert typed params; anything that is not
        already wire data still needs it."""
        recorder = _Recorder()
        kwargs = _wire_body()
        kwargs["messages"][1]["content"] = object()

        result = bypass_chat_sdk_request_transform(kwargs, recorder.client)

        assert result["messages"] is kwargs["messages"]
        assert "extra_body" not in result or "messages" not in result["extra_body"]

    def test_a_tool_free_request_does_not_gain_an_empty_tools_key(self):
        recorder = _Recorder()
        body = _wire_body(tools=False)

        sent = json.loads(recorder.send(bypass_chat_sdk_request_transform(dict(body), recorder.client)))

        assert "tools" not in sent

    def test_caller_supplied_extra_body_keeps_precedence(self):
        """The chat path already populates extra_body from custom providers,
        reasoning config and Nous Portal; those entries must win."""
        recorder = _Recorder()
        body = _wire_body()
        body["extra_body"] = {"provider": {"order": ["nous"]}, "messages": ["caller wins"]}

        result = bypass_chat_sdk_request_transform(dict(body), recorder.client)

        assert result["extra_body"]["messages"] == ["caller wins"]
        assert result["extra_body"]["provider"] == {"order": ["nous"]}

    def test_env_escape_hatch_restores_the_pre_fix_kwargs(self, monkeypatch):
        monkeypatch.setenv("HERMES_CHAT_SDK_TRANSFORM", "1")
        recorder = _Recorder()
        kwargs = _wire_body()

        assert bypass_chat_sdk_request_transform(kwargs, recorder.client) is kwargs

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_escape_hatch_stays_off_for_falsey_values(self, monkeypatch, value):
        monkeypatch.setenv("HERMES_CHAT_SDK_TRANSFORM", value)
        recorder = _Recorder()

        result = bypass_chat_sdk_request_transform(_wire_body(), recorder.client)

        assert result["messages"] == []


class TestResponsesPathUnchanged:
    def test_the_codex_import_path_and_behaviour_are_preserved(self):
        """agent/auxiliary_client.py and the existing codex test import these
        names from agent.codex_runtime; the move must not break them."""
        from agent.codex_runtime import (
            _SDK_TRANSFORM_BYPASS_FIELDS,
            _bypass_sdk_request_transform,
            _is_plain_json_data,
        )

        assert _SDK_TRANSFORM_BYPASS_FIELDS == ("input", "tools")
        assert _is_plain_json_data([{"role": "user", "content": "hi"}])
        assert not _is_plain_json_data([{"role": "user", "content": object()}])

        kwargs = {
            "model": "gpt-5.6-sol",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": [{"type": "function", "name": "terminal", "parameters": {}}],
            "stream": True,
        }
        bypassed = _bypass_sdk_request_transform(kwargs)

        # Responses keeps its historical shape: the fields are REMOVED from the
        # typed kwargs entirely (input is not @required_args there).
        assert "input" not in bypassed
        assert "tools" not in bypassed
        assert bypassed["extra_body"]["input"] == kwargs["input"]
