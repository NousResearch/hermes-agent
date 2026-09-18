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


def test_build_moa_facade_ignores_fallback_model_name_when_restoring(monkeypatch, tmp_path):
    """A fallback restore must not turn the temporary fallback model name into
    a MoA preset. Sessions that drifted to e.g. deepseek-v4-flash previously
    crashed on restore with MoAPresetNotFoundError because build_moa_facade()
    reused agent.model as the preset (#MoA restore path).
    """
    home = tmp_path / ".hermes"
    _write_cfg(home)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from agent.moa_loop import build_moa_facade

    agent = SimpleNamespace(
        provider="moa",
        model="deepseek-v4-flash",
        tool_progress_callback=None,
    )

    client = build_moa_facade(agent, None)

    assert client.chat.completions.preset_name == "review"


def test_create_wraps_completed_aggregator_response_as_delta_chunk(monkeypatch, tmp_path):
    """When an aggregator adapter returns a completed response despite
    stream=True (Codex Responses compatibility shape), MoA must return a
    one-chunk delta iterator for the outer streaming accumulator instead of
    the raw non-iterable response object (#55933).
    """

    completed = _response("aggregator acted")
    completed.choices[0].message.tool_calls = [
        SimpleNamespace(
            id="call_1",
            type="function",
            function=SimpleNamespace(name="read_file", arguments='{"path":"x"}'),
        )
    ]

    def on_call(kwargs):
        if kwargs["task"] == "moa_aggregator":
            return completed
        return None

    facade, calls = _facade(monkeypatch, tmp_path, on_call=on_call)
    stream = facade.create(
        messages=[{"role": "user", "content": "q"}],
        tools=[],
        stream=True,
    )

    chunk = next(iter(stream))
    assert chunk.choices[0].delta.content == "aggregator acted"
    assert chunk.choices[0].delta.tool_calls[0].index == 0
    assert chunk.choices[0].delta.tool_calls[0].function.name == "read_file"
    assert chunk.choices[0].finish_reason == "stop"
    with pytest.raises(StopIteration):
        next(stream)

    agg = next(c for c in calls if c["task"] == "moa_aggregator")
    assert agg["stream"] is True


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


