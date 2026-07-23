"""Tests for MoA aggregator streaming.

MoAChatCompletions.create() honors stream=True by running the references first
and then returning the aggregator's raw streaming iterator (from call_llm), so
the acting model's output can stream to the user. stream=False is the original
complete-response path and must stay byte-identical.
"""
from types import SimpleNamespace

import pytest


_TOOL_BETA_ERROR = "Client-side tools for multi-agent models require beta access"


class _ProviderError(RuntimeError):
    def __init__(self, message, *, status_code, body=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = response


def _response(content="done", *, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls or [])
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


def _beta_error(*, status_code=400, body=None, message="opaque provider failure"):
    return _ProviderError(
        message,
        status_code=status_code,
        body={"error": _TOOL_BETA_ERROR} if body is None else body,
    )


def _stream_chunk(*, content=None, tool_calls=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(index=0, delta=delta, finish_reason=None)
    return SimpleNamespace(choices=[choice], usage=None, model="fake-model")


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


def test_exact_structured_400_tool_beta_rejection_retries_without_tools_once(
    monkeypatch, tmp_path
):
    """A provider's exact multi-agent tool entitlement rejection is a
    capability negotiation, not a reason to abandon the whole MoA preset.

    The first request keeps tools for accounts that have beta access.  After
    the exact rejection, the same facade retries without tools, reuses the
    already-computed references, and refuses to downgrade a second time.
    """
    facade, calls = _facade(monkeypatch, tmp_path)
    messages = [{"role": "user", "content": "q"}]
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    facade.create(messages=messages, tools=tools, stream=True)
    error = _beta_error()

    assert facade.retry_without_aggregator_tools(error) is True
    facade.create(messages=messages, tools=tools, stream=True)

    reference_calls = [c for c in calls if c["task"] == "moa_reference"]
    aggregator_calls = [c for c in calls if c["task"] == "moa_aggregator"]
    assert len(reference_calls) == 1
    assert len(aggregator_calls) == 2
    assert aggregator_calls[0]["tools"] == tools
    assert aggregator_calls[1]["tools"] is None
    retry_guidance = str(aggregator_calls[1]["messages"][-1]["content"])
    assert "Client-side tools are unavailable" in retry_guidance
    assert "authoritative runtime configuration" in retry_guidance
    assert "answer from those labels" in retry_guidance
    assert facade.retry_without_aggregator_tools(error) is False


@pytest.mark.parametrize(
    ("status_code", "body", "message"),
    [
        (403, {"error": _TOOL_BETA_ERROR}, "opaque provider failure"),
        (400, {"error": f"{_TOOL_BETA_ERROR}. Contact support."}, "opaque provider failure"),
        (400, {"error": f"ProviderError: {_TOOL_BETA_ERROR}"}, "opaque provider failure"),
        (400, {"wrapper": {"error": _TOOL_BETA_ERROR}}, "opaque provider failure"),
        (400, {"error": "another provider error", "message": _TOOL_BETA_ERROR}, "opaque provider failure"),
        (400, {"error": "another provider error"}, _TOOL_BETA_ERROR),
    ],
)
def test_tool_beta_downgrade_requires_exact_structured_400_error_field(
    monkeypatch, tmp_path, status_code, body, message
):
    facade, _calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
    )

    error = _beta_error(status_code=status_code, body=body, message=message)
    assert facade.retry_without_aggregator_tools(error) is False


def test_tool_beta_rejection_reads_structured_response_json(monkeypatch, tmp_path):
    facade, _calls = _facade(monkeypatch, tmp_path)
    facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
    )

    class Response:
        @staticmethod
        def json():
            return {"error": _TOOL_BETA_ERROR}

    error = _ProviderError(
        "opaque provider failure", status_code=400, response=Response()
    )

    assert facade.retry_without_aggregator_tools(error) is True


def test_tool_beta_downgrade_requires_tools(monkeypatch, tmp_path):
    facade, _calls = _facade(monkeypatch, tmp_path)
    facade_without_tools = facade.__class__("review")
    facade_without_tools.create(messages=[{"role": "user", "content": "q"}], tools=[])
    exact = _beta_error()
    assert facade_without_tools.retry_without_aggregator_tools(exact) is False


def test_tool_beta_error_after_visible_content_is_propagated_without_downgrade(
    monkeypatch, tmp_path
):
    error = _beta_error()

    def failing_stream():
        yield _stream_chunk(content="already visible")
        raise error

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return failing_stream()
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    stream = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )

    assert next(stream).choices[0].delta.content == "already visible"
    assert facade.aggregator_attempt_emitted_output() is True
    with pytest.raises(RuntimeError) as caught:
        next(stream)

    assert caught.value is error
    assert facade.retry_without_aggregator_tools(caught.value) is False


def test_tool_beta_error_after_tool_call_delta_is_propagated_without_downgrade(
    monkeypatch, tmp_path
):
    error = _beta_error()
    tool_delta = SimpleNamespace(
        index=0,
        id="call-1",
        function=SimpleNamespace(name="read_file", arguments=""),
    )

    def failing_stream():
        yield _stream_chunk(tool_calls=[tool_delta])
        raise error

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return failing_stream()
        return None

    facade, _calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    stream = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[{"type": "function"}],
        stream=True,
    )

    assert next(stream).choices[0].delta.tool_calls == [tool_delta]
    assert facade.aggregator_attempt_emitted_output() is True
    with pytest.raises(RuntimeError) as caught:
        next(stream)

    assert caught.value is error
    assert facade.retry_without_aggregator_tools(caught.value) is False


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

    def _validate(resp, task, provider=None, base_url=None):
        validated["called"] = True
        return resp

    monkeypatch.setattr(ac, "_validate_llm_response", _validate)

    ac.call_llm(
        provider="custom",
        model="m",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert validated["called"] is True
