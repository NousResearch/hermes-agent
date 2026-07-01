"""Tests for MoA aggregator streaming.

MoAChatCompletions.create() honors stream=True by running the references first
and then returning the aggregator's raw streaming iterator (from call_llm), so
the acting model's output can stream to the user. stream=False is the original
complete-response path and must stay byte-identical.
"""
from types import SimpleNamespace

import pytest


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _codex_completed_response(
    content="codex acted",
    *,
    tool_calls=None,
    usage=None,
    model="gpt-5.5",
):
    """A completed chat response shaped exactly like Hermes' Codex auxiliary
    adapter returns from ``_CodexCompletionsAdapter.create()`` — a bare
    ``SimpleNamespace`` with ``choices[0].message`` and NO ``__iter__``.

    This is the object that crashed the live MoA streaming turn with
    ``'types.SimpleNamespace' object is not iterable`` when handed straight to
    the outer streaming consumer. Mirrors agent/auxiliary_client.py: tool_calls
    are ``SimpleNamespace(id, type='function', function=SimpleNamespace(name,
    arguments))`` WITHOUT a stream ``.index`` field, and finish_reason is
    ``"tool_calls"`` when tools are present, else ``"stop"``.
    """
    message = SimpleNamespace(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )
    choice = SimpleNamespace(
        index=0,
        message=message,
        finish_reason="stop" if not tool_calls else "tool_calls",
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage)


def _codex_tool_call(name="do_thing", arguments='{"x": 1}', call_id="call_abc"):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _drain(stream):
    """Consume ``stream`` exactly the way the real streaming consumer does
    (agent/chat_completion_helpers.py) and return the reassembled result.

    Reads only the chunk fields the production consumer reads: ``chunk.model``,
    ``chunk.usage``, ``chunk.choices[0].delta.content``, ``.delta.tool_calls``
    (with ``.index``/``.id``/``.function.name``/``.function.arguments``), and
    ``chunk.choices[0].finish_reason``. If ``stream`` is not iterable this
    raises the same ``TypeError`` the live turn hit.
    """
    content = ""
    reasoning = ""
    tool_calls: dict = {}
    finish_reason = None
    model = None
    usage = None
    roles: list = []
    for chunk in stream:  # TypeError here == the historical crash
        if getattr(chunk, "model", None):
            model = chunk.model
        if not chunk.choices:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            continue
        delta = chunk.choices[0].delta
        if getattr(delta, "role", None):
            roles.append(delta.role)
        reasoning_text = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if reasoning_text:
            reasoning += reasoning_text
        if getattr(delta, "content", None):
            content += delta.content
        for tc in getattr(delta, "tool_calls", None) or []:
            idx = tc.index if tc.index is not None else 0
            entry = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                entry["id"] = tc.id
            if tc.function and tc.function.name:
                entry["name"] = tc.function.name
            if tc.function and tc.function.arguments:
                entry["arguments"] += tc.function.arguments  # TypeError if args not a str
        if chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason
        if getattr(chunk, "usage", None):
            usage = chunk.usage
    return SimpleNamespace(
        content=content,
        reasoning=reasoning,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        model=model,
        usage=usage,
        roles=roles,
    )


def _write_cfg(home):
    home.mkdir()
    (home / "config.yaml").write_text(
        """
moa:
  default_preset: review
  presets:
    review:
      reference_models:
        - provider: openai-codex
          model: gpt-5.5
      aggregator:
        provider: openrouter
        model: anthropic/claude-opus-4.8
""".strip(),
        encoding="utf-8",
    )


def _facade(monkeypatch, tmp_path, on_call=None):
    home = tmp_path / ".hermes"
    _write_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))
    calls = []

    def fake_call_llm(**kwargs):
        calls.append(kwargs)
        if on_call is not None:
            r = on_call(kwargs)
            if r is not None:
                return r
        if kwargs["task"] == "moa_reference":
            return _response("reference advice")
        return _response("aggregator acted")

    monkeypatch.setattr("agent.moa_loop.call_llm", fake_call_llm)
    from agent.moa_loop import MoAChatCompletions

    return MoAChatCompletions("review"), calls


# --------------------------------------------------------------------------
# Facade-level: create() stream branch
# --------------------------------------------------------------------------

def test_create_streams_aggregator_when_requested(monkeypatch, tmp_path):
    """stream=True: references still run, aggregator is called with stream=True
    and stream_options, and create() returns the aggregator call's result
    (the raw stream) verbatim."""
    sentinel = object()

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return sentinel
        return None

    facade, calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )

    # create() returns the aggregator's streaming result untouched.
    assert out is sentinel
    # References still ran (MoA not bypassed).
    assert any(c["task"] == "moa_reference" for c in calls)
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["stream"] is True
    assert agg["stream_options"] == {"include_usage": True}
    # Tools still flow to the (streaming) aggregator.
    assert agg["tools"] is not None


def test_create_non_stream_path_unchanged(monkeypatch, tmp_path):
    """Default (no stream): the aggregator call carries NO stream/stream_options
    keys, so the non-streaming path is byte-identical to before."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(messages=[{"role": "user", "content": "q"}], tools=[])

    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert "stream" not in agg
    assert "stream_options" not in agg
    assert "timeout" not in agg


def test_create_forwards_stream_read_timeout(monkeypatch, tmp_path):
    """The consumer's per-request (stream read) timeout is forwarded to the
    aggregator so it actually governs the stream."""
    timeout_sentinel = object()
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
        timeout=timeout_sentinel,
    )
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["timeout"] is timeout_sentinel


def test_create_respects_caller_stream_options(monkeypatch, tmp_path):
    """A caller-provided stream_options is forwarded as-is (not overwritten)."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
        stream_options={"include_usage": False, "extra": 1},
    )
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["stream_options"] == {"include_usage": False, "extra": 1}


def test_create_does_not_forward_timeout_when_not_streaming(monkeypatch, tmp_path):
    """A stray timeout on a non-streaming call is NOT forwarded — the non-stream
    path must remain unchanged regardless of incidental kwargs."""
    facade, calls = _facade(monkeypatch, tmp_path)
    facade.create(messages=[{"role": "user", "content": "q"}], tools=[], timeout=object())
    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert "timeout" not in agg
    assert "stream" not in agg


# --------------------------------------------------------------------------
# call_llm-level: stream branch returns the raw SDK stream
# --------------------------------------------------------------------------

def test_call_llm_stream_returns_raw_stream_and_skips_validation(monkeypatch):
    """call_llm(stream=True) returns the client's raw stream object directly,
    attaches stream/stream_options to the request, and does NOT run response
    validation (which assumes a complete response)."""
    from agent import auxiliary_client as ac

    captured = {}

    class _Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return "RAW_STREAM"

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="http://localhost:8001/v1",
    )

    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("custom", "m", "http://localhost:8001/v1", "key", "chat_completions"),
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "m"))

    def _no_validate(*a, **k):
        raise AssertionError("streaming must not go through _validate_llm_response")

    monkeypatch.setattr(ac, "_validate_llm_response", _no_validate)

    out = ac.call_llm(
        provider="custom",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        stream_options={"include_usage": True},
    )

    assert out == "RAW_STREAM"
    assert captured.get("stream") is True
    assert captured.get("stream_options") == {"include_usage": True}


def test_call_llm_non_stream_still_validates(monkeypatch):
    """Sanity: stream=False keeps the validated path (regression guard for the
    early-return not leaking into normal calls)."""
    from agent import auxiliary_client as ac

    class _Completions:
        def create(self, **kwargs):
            return _response("ok")

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions()),
        base_url="http://localhost:8001/v1",
    )
    monkeypatch.setattr(
        ac, "_resolve_task_provider_model",
        lambda *a, **k: ("custom", "m", "http://localhost:8001/v1", "key", "chat_completions"),
    )
    monkeypatch.setattr(ac, "_get_cached_client", lambda *a, **k: (fake_client, "m"))

    validated = {"called": False}

    def _validate(resp, task):
        validated["called"] = True
        return resp

    monkeypatch.setattr(ac, "_validate_llm_response", _validate)

    ac.call_llm(
        provider="custom",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert validated["called"] is True


# --------------------------------------------------------------------------
# U1: stream-contract normalization at the MoA boundary
#
# When the aggregator adapter returns a COMPLETED chat response despite
# stream=True (Hermes' Codex auxiliary shim does exactly this), create() must
# hand the outer streaming consumer something iterable — never the raw
# non-iterable response. Raw provider streams still pass through untouched.
# --------------------------------------------------------------------------

def test_stream_wraps_completed_text_response_into_iterator(monkeypatch, tmp_path):
    """A completed text response from the aggregator (stream=True) is wrapped
    into a one-shot iterator whose single content delta is the message text."""
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response("hello from codex")
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)

    # Must be iterable (not the raw completed response) and reassemble the text.
    result = _drain(out)
    assert result.content == "hello from codex"
    assert result.finish_reason == "stop"


def test_stream_wraps_completed_tool_call_response(monkeypatch, tmp_path):
    """A completed tool-call response is wrapped so the consumer can still
    reassemble tool calls from delta.tool_calls (with a stream .index)."""
    tc = _codex_tool_call(name="search", arguments='{"q": "cats"}', call_id="c1")

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response(content=None, tool_calls=[tc])
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )

    result = _drain(out)
    assert result.finish_reason == "tool_calls"
    assert list(result.tool_calls.values()) == [
        {"id": "c1", "name": "search", "arguments": '{"q": "cats"}'}
    ]


def test_stream_raw_iterable_passthrough_unchanged(monkeypatch, tmp_path):
    """A genuine raw stream iterable is returned verbatim — not wrapped or
    consumed early — preserving the upstream streaming feature."""
    raw_stream = iter(["real", "provider", "chunks"])

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return raw_stream
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)
    assert out is raw_stream


def test_stream_string_like_not_treated_as_char_stream(monkeypatch, tmp_path):
    """A plain string return is technically iterable but is NOT a provider
    chunk stream — the facade must not hand it over for character iteration."""
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return "a bare string that must not char-iterate"
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)

    # Not returned as the raw string (which would char-iterate in the consumer).
    assert out != "a bare string that must not char-iterate"
    # Draining yields chunk objects, never single characters.
    result = _drain(out)
    assert result.content == "a bare string that must not char-iterate"


def test_stream_empty_completed_response_yields_terminal_chunk(monkeypatch, tmp_path):
    """A completed response with no choices yields a terminal chunk rather than
    crashing the consumer's zero-chunk guard."""
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return SimpleNamespace(choices=[], model="gpt-5.5", usage=None)
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)

    result = _drain(out)
    assert result.content == ""
    assert result.tool_calls == {}
    # A non-None finish_reason so the consumer's empty-stream guard does not fire.
    assert result.finish_reason is not None


# --------------------------------------------------------------------------
# U3: integration-shaped regression around the historical live failure
# --------------------------------------------------------------------------

def test_stream_simplenamespace_not_iterable_regression(monkeypatch, tmp_path):
    """Historical crash repro: iterating the MoA stream return value no longer
    raises ``'types.SimpleNamespace' object is not iterable``.

    Before the fix, create(stream=True) returned the Codex adapter's completed
    SimpleNamespace directly; the outer consumer's ``for chunk in stream`` then
    raised TypeError. The wrapper makes the return iterable.
    """
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response("recovered")
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)

    # The exact operation that crashed live — must not raise TypeError.
    result = _drain(out)
    assert result.content == "recovered"


def test_stream_wrapped_response_preserves_usage(monkeypatch, tmp_path):
    """When the completed response carries usage, the wrapped stream preserves
    it in the shape session accounting reads (final chunk's ``usage``)."""
    usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18)

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response("with usage", usage=usage)
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)

    result = _drain(out)
    assert result.usage is usage


def test_stream_wrapped_tool_calls_finish_reason_distinguishable(monkeypatch, tmp_path):
    """A completed tool_calls finish_reason stays distinguishable from a normal
    stop completion after wrapping."""
    tc = _codex_tool_call()

    def on_call(kwargs):
        task = kwargs["task"]
        if task == "moa_aggregator":
            # First aggregator call returns tools, distinct facade instances
            # would be needed for two turns; here we assert a single tool turn.
            return _codex_completed_response(content=None, tool_calls=[tc])
        return None

    tool_dir = tmp_path / "tool_turn"
    tool_dir.mkdir()
    facade, _calls = _facade(monkeypatch, tool_dir, on_call=on_call)
    out = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )
    assert _drain(out).finish_reason == "tool_calls"

    # A plain stop completion (separate facade/turn) stays "stop".
    def on_call_stop(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response("just text")
        return None

    stop_dir = tmp_path / "stop_turn"
    stop_dir.mkdir()
    facade2, _calls2 = _facade(monkeypatch, stop_dir, on_call=on_call_stop)
    out2 = facade2.create(messages=[{"role": "user", "content": "q2"}], stream=True)
    assert _drain(out2).finish_reason == "stop"


# --------------------------------------------------------------------------
# Hardening (from code review): contract fidelity of the wrapped chunks
# --------------------------------------------------------------------------

def test_stream_wrapped_delta_sets_assistant_role(monkeypatch, tmp_path):
    """The wrapped delta sets role="assistant", matching every other provider's
    stream delta (copilot/gemini shims) so the consumer's message reassembly
    is not a silent divergence for the Codex one-shot path."""
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response("hi")
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)
    assert _drain(out).roles == ["assistant"]


def test_stream_wrapped_carries_reasoning(monkeypatch, tmp_path):
    """A completed response carrying reasoning_content is not dropped — the
    wrapped delta exposes it so reasoning display/accumulation survives."""
    resp = _codex_completed_response("answer")
    resp.choices[0].message.reasoning_content = "thinking step"

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return resp
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)
    result = _drain(out)
    assert result.reasoning == "thinking step"
    assert result.content == "answer"


def test_stream_tool_call_dict_arguments_are_json_stringified(monkeypatch, tmp_path):
    """A tool call whose arguments arrive already-parsed (dict) is serialized to
    a JSON string so the consumer's ``arguments +=`` concatenation cannot raise
    TypeError."""
    tc = _codex_tool_call(name="search", arguments={"q": "cats"}, call_id="c1")

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return _codex_completed_response(content=None, tool_calls=[tc])
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )
    # _drain concatenates arguments as strings; a dict would have raised.
    reassembled = _drain(out).tool_calls[0]["arguments"]
    import json

    assert json.loads(reassembled) == {"q": "cats"}


def test_stream_completed_response_with_none_choices_is_wrapped(monkeypatch, tmp_path):
    """A completed response whose ``choices`` is None (error/content-filter
    frame) is still recognized as a whole response and wrapped, not passed
    through to crash ``for chunk in stream``."""
    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return SimpleNamespace(choices=None, model="gpt-5.5", usage=None)
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    out = facade.create(messages=[{"role": "user", "content": "q"}], stream=True)
    # Must not raise 'SimpleNamespace object is not iterable'.
    result = _drain(out)
    assert result.finish_reason is not None
