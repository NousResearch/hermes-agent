"""Bedrock Converse recovery for Claude thinking-block prefix-binding rejections.

Bedrock rejects signed thinking when its earlier conversation prefix changes, either on a provider
switch or within a Bedrock-only session. The recovery resends once with
the documented ``block_binding.prefix_mismatch_behavior=drop_block`` opt-in. The contract pinned here:
only that rejection triggers it, healthy requests never change, history and tool results are replayed
verbatim (no tool re-run), and the retry happens once on every production call site.
"""

import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent import chat_completion_helpers as cch
from agent.bedrock_adapter import build_converse_kwargs, call_converse
from agent.transports.bedrock import BedrockTransport

ClientError = pytest.importorskip("botocore.exceptions").ClientError

ADAPTIVE = "global.anthropic.claude-opus-5-5"
BETA = "thinking-binding-controls-2026-08-01"
BINDING_MESSAGES = (
    "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation. "
    "Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \"drop_block\". "
    "That setting requires the `thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header.",
    "thinking block signature does not match; set thinking.block_binding.prefix_mismatch_behavior",
)
OK_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
    "stopReason": "end_turn", "usage": {"inputTokens": 1, "outputTokens": 1},
}


def _client_error(code, message, op="Converse"):
    return ClientError({"Error": {"Code": code, "Message": message}}, op)


def _binding_error(op="Converse", variant=0):
    return _client_error("ValidationException", BINDING_MESSAGES[variant], op)


class _FakeRuntime:
    """boto3 bedrock-runtime stand-in: records each request (deep-copied) and plays scripted outcomes."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def _next(self, op, kwargs):
        self.calls.append((op, copy.deepcopy(kwargs)))
        outcome = self.outcomes.pop(0) if self.outcomes else OK_RESPONSE
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def converse(self, **kwargs):
        return self._next("converse", kwargs)

    def converse_stream(self, **kwargs):
        return self._next("converse_stream", kwargs)


class _StreamAgent:
    api_mode = "bedrock_converse"
    _interrupt_requested = False
    _disable_streaming = False
    reasoning_callback = None
    stream_delta_callback = None
    provider = "bedrock"
    model = ADAPTIVE
    _consecutive_stale_streams = 0

    def _has_stream_consumers(self):
        return False

    def _buffer_status(self, *a, **k):
        pass

    def _buffer_diagnostic_status(self, *a, **k):
        pass

    def _fire_stream_delta(self, text):
        pass

    def _fire_tool_gen_started(self, name):
        pass

    def _fire_reasoning_delta(self, text):
        pass

    def _safe_print(self, *a, **k):
        pass


class _ProviderStream:
    def __iter__(self):
        return iter(())

    def close(self):
        pass


STREAM_RESPONSE = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="done"))],
                                  usage=None, stop_reason="end_turn")


def _history():
    """Synthetic signed Claude tool round: user -> assistant(thinking + tool_call) -> tool result."""
    return [
        {"role": "system", "content": "You are Claude Code, Anthropic's official CLI."},
        {"role": "user", "content": "list the files"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "toolu_synthetic_1", "type": "function",
             "function": {"name": "mcp__terminal", "arguments": "{\"command\": \"ls\"}"}}],
         "reasoning_details": [{"type": "thinking", "thinking": "run ls", "signature": "sig_synthetic"}]},
        {"role": "tool", "tool_call_id": "toolu_synthetic_1", "content": "a.txt"},
    ]


def _main_kwargs(reasoning_config=None, model=ADAPTIVE, messages=None):
    return BedrockTransport().build_kwargs(model=model, messages=messages or _history(),
        tools=[{"type": "function", "function": {"name": "mcp__terminal", "description": "run",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}}}}],
        region="us-east-1", reasoning_config=reasoning_config)


def _wire(kwargs):
    return {k: v for k, v in kwargs.items() if not k.startswith("__bedrock")}


def _run(route, client, kwargs=None, model=ADAPTIVE):
    """Drive one production call site against ``client``; returns the request that was built."""
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
        if route == "aux":
            call_converse("us-east-1", model, _history())
            return build_converse_kwargs(model, _history())
        kwargs = kwargs if kwargs is not None else _main_kwargs({"effort": "high"}, model=model)
        built = copy.deepcopy(_wire(kwargs))
        if route == "stream":
            client.outcomes = [o if not isinstance(o, dict) else {"stream": _ProviderStream()} for o in client.outcomes]
            with patch("agent.bedrock_adapter.stream_converse_with_callbacks", return_value=STREAM_RESPONSE):
                cch.interruptible_streaming_api_call(_StreamAgent(), kwargs)
        else:
            cch._dispatch_nonstreaming_api_request(SimpleNamespace(api_mode="bedrock_converse"), kwargs,
                                                   make_client=None)
        return built


ROUTES = ("stream", "nonstream", "aux")


def _op(route):
    return "ConverseStream" if route == "stream" else "Converse"


@pytest.mark.parametrize("variant", [0, 1])
@pytest.mark.parametrize("route", ROUTES)
def test_binding_rejection_resends_once_with_opt_in_on_every_call_site(route, variant):
    """C1/C3/R2: each production site retries once with the documented nested opt-in; nothing else on
    the request changes. ``aux`` sends no thinking field, the adaptive-default case."""
    client = _FakeRuntime(_binding_error(_op(route), variant), OK_RESPONSE)
    base = _run(route, client)

    assert len(client.calls) == 2
    first, retry = client.calls[0][1], client.calls[1][1]
    assert first == base
    fields = retry["additionalModelRequestFields"]
    assert BETA in fields["anthropic_beta"]
    assert fields["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    base_thinking = (base.get("additionalModelRequestFields") or {}).get("thinking") or {"type": "adaptive"}
    assert {k: v for k, v in fields["thinking"].items() if k != "block_binding"} == base_thinking
    strip = lambda kw: {k: v for k, v in kw.items() if k != "additionalModelRequestFields"}
    assert strip(retry) == strip(first)
    base_fields = {k: v for k, v in (base.get("additionalModelRequestFields") or {}).items()
                   if k not in ("anthropic_beta", "thinking")}
    assert {k: v for k, v in fields.items() if k not in ("anthropic_beta", "thinking")} == base_fields


@pytest.mark.parametrize("retry_fails", [False, True])
def test_iam_denied_stream_fallback_recovers_binding_once(retry_fails):
    """The IAM fallback retains its region and recovers a binding rejection on Converse."""
    denied = _client_error("AccessDeniedException", "Not authorized: bedrock:InvokeModelWithResponseStream", "ConverseStream")
    last = _binding_error() if retry_fails else OK_RESPONSE
    client = _FakeRuntime(denied, _binding_error(), last)
    kwargs = _main_kwargs({"effort": "high"})
    kwargs["__bedrock_region__"] = "eu-west-1"
    original = copy.deepcopy(_wire(kwargs))
    agent = _StreamAgent()
    regions = []

    def runtime(region):
        regions.append(region)
        return client

    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", side_effect=runtime):
        stream = cch._BedrockStream(agent, kwargs, None)
        if retry_fails:
            with pytest.raises(ClientError) as raised:
                stream._open_stream(kwargs)
            assert raised.value is last
        else:
            response = stream._open_stream(kwargs)
            assert response.choices[0].message.content == "done"
    assert agent._disable_streaming is True
    assert regions and all(region == "eu-west-1" for region in regions)
    assert [op for op, _ in client.calls] == ["converse_stream", "converse", "converse"]
    assert client.calls[0][1] == client.calls[1][1] == original
    retried = client.calls[2][1]
    assert retried["messages"] == original["messages"]
    assert retried["system"] == original["system"]
    assert retried["toolConfig"] == original["toolConfig"]
    fields = retried["additionalModelRequestFields"]
    assert fields["thinking"]["block_binding"] == {"prefix_mismatch_behavior": "drop_block"}
    assert BETA in fields["anthropic_beta"]


def test_existing_betas_are_kept_and_input_kwargs_are_not_mutated():
    """C1/R2: the beta merges into an existing list, and the caller's kwargs stay byte-identical."""
    kwargs = _main_kwargs({"effort": "high"})
    kwargs["additionalModelRequestFields"]["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]
    kwargs["additionalModelRequestFields"]["thinking"]["block_binding"] = {"other": 1}
    before = json.dumps(_wire(kwargs), sort_keys=True, default=repr)
    client = _FakeRuntime(_binding_error(), OK_RESPONSE)
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client):
        cch._bedrock_converse_call(kwargs, stream=False)

    assert json.dumps(_wire(kwargs), sort_keys=True, default=repr) == before
    fields = client.calls[1][1]["additionalModelRequestFields"]
    assert fields["anthropic_beta"] == ["interleaved-thinking-2025-05-14", BETA]
    assert fields["thinking"]["block_binding"] == {"other": 1, "prefix_mismatch_behavior": "drop_block"}


@pytest.mark.parametrize("error", [
    _client_error("ValidationException", "Invalid signature in thinking block"),
    _client_error("ValidationException", "messages: roles must alternate between user and assistant"),
    _client_error("AccessDeniedException", "not authorized: bound to a different conversation"),
    _client_error("ThrottlingException", "Too many requests"),
], ids=["signature", "other-400", "auth", "rate-limit"])
@pytest.mark.parametrize("route", ROUTES)
def test_unrelated_errors_propagate_without_binding_retry(route, error):
    """C2/R1: generic signature errors, other 400s, auth and throttling keep their existing behavior."""
    client = _FakeRuntime(error)
    with pytest.raises(ClientError) as raised:
        _run(route, client)
    assert raised.value is error
    assert len(client.calls) == 1


@pytest.mark.parametrize("route", ROUTES)
def test_second_binding_rejection_propagates_without_recursion(route):
    """C2/R3: the recovery request's own failure surfaces; there is no third attempt."""
    second = _binding_error(_op(route))
    client = _FakeRuntime(_binding_error(_op(route)), second, OK_RESPONSE)
    with pytest.raises(ClientError) as raised:
        _run(route, client)
    assert raised.value is second
    assert len(client.calls) == 2


@pytest.mark.parametrize(("model", "reasoning_config"), [
    ("us.anthropic.claude-sonnet-4-6", {"enabled": False}),
    ("anthropic.claude-3-7-sonnet-20250219-v1:0", None),
    ("amazon.nova-pro-v1:0", None),
], ids=["explicitly-disabled", "non-adaptive-claude-omitted", "non-claude"])
def test_no_opt_in_where_thinking_is_off_or_unsupported(model, reasoning_config):
    """C2/R2: disabled thinking is never turned on and other models get no fabricated thinking field.
    (Mandatory-thinking models such as Opus 5.5 cannot send a disable, so they omit thinking and keep
    the adaptive default; those are covered by the aux rows above.)"""
    kwargs = _main_kwargs(reasoning_config, model=model)
    thinking = (kwargs.get("additionalModelRequestFields") or {}).get("thinking")
    assert thinking == ({"type": "disabled"} if reasoning_config else None)
    client = _FakeRuntime(_binding_error())
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client), \
         pytest.raises(ClientError):
        cch._bedrock_converse_call(kwargs, stream=False)
    assert len(client.calls) == 1


@pytest.mark.parametrize("route", ROUTES)
def test_healthy_request_is_sent_once_and_byte_identical(route):
    """C2/R4: without a rejection the request is exactly what the builder produced, sent once."""
    client = _FakeRuntime(OK_RESPONSE)
    base = _run(route, client)
    assert len(client.calls) == 1
    assert json.dumps(client.calls[0][1], sort_keys=True, default=repr) == json.dumps(base, sort_keys=True, default=repr)
    assert BETA not in repr(client.calls[0][1])


def test_binding_error_raised_while_reading_stream_is_not_retried():
    """R3: once the stream is open (events may already be shown), a failure is not resent."""
    client = _FakeRuntime({"stream": _ProviderStream()})
    with patch("agent.bedrock_adapter._get_bedrock_runtime_client", return_value=client), \
         patch("agent.bedrock_adapter.stream_converse_with_callbacks",
               side_effect=_binding_error("ConverseStream")), \
         pytest.raises(ClientError):
        cch.interruptible_streaming_api_call(_StreamAgent(), _main_kwargs({"effort": "high"}))
    assert len(client.calls) == 1


@pytest.mark.parametrize("route", ("stream", "nonstream"))
def test_signed_tool_history_survives_recovery_unchanged(route):
    """C4/R4: a completed tool round converts through the production transport and replays verbatim.
    The request keeps the signature, one toolUse and one toolResult for the call id."""
    history = _history()
    snapshot = copy.deepcopy(history)
    kwargs = _main_kwargs({"effort": "high"}, messages=history)
    client = _FakeRuntime(_binding_error(_op(route)), OK_RESPONSE)
    _run(route, client, kwargs=kwargs)

    assert history == snapshot
    first, retry = client.calls[0][1], client.calls[1][1]
    assert retry["messages"] == first["messages"]
    wire = json.dumps(retry["messages"])
    assert wire.count('"toolUseId": "toolu_synthetic_1"') == 2  # toolUse + toolResult
    assert wire.count('"toolUse"') == 1 and wire.count('"toolResult"') == 1
    assert wire.count("sig_synthetic") == 1
    assert retry["system"] == first["system"] and retry["toolConfig"] == first["toolConfig"]
