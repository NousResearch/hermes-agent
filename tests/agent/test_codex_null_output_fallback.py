"""Regression tests for chatgpt.com/backend-api/codex null-output handling.

The ChatGPT Codex backend (https://chatgpt.com/backend-api/codex) emits
``response.completed`` events whose ``response.output`` field is ``null``
rather than ``[]``.  The OpenAI SDK's stream accumulator calls
``parse_response`` on every snapshot, and ``parse_response`` iterates
``response.output`` without a None guard
(``openai/lib/_parsing/_responses.py``), so the terminal event raises
``TypeError: 'NoneType' object is not iterable`` from inside the
``for event in stream`` loop in ``run_codex_stream`` — before our
``get_final_response()``-time output-backfill can run.

These tests pin two behaviours:

1. ``run_codex_stream`` catches the SDK-parser TypeError and routes to
   ``run_codex_create_stream_fallback`` (which iterates the raw SSE stream
   and never calls ``parse_response``).
2. ``run_codex_create_stream_fallback`` treats ``output=None`` the same as
   ``output=[]`` when deciding whether to synthesise output from collected
   stream events.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

# Stub optional heavy imports so run_agent imports cleanly in isolation.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


def _make_codex_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / "config.yaml").write_text("{}\n", encoding="utf-8")
    from run_agent import AIAgent

    agent = AIAgent(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    agent.api_mode = "codex_responses"
    monkeypatch.setattr(agent, "_emit_status", lambda *a, **k: None)
    return agent


@pytest.fixture
def sdk_parser_typeerror_raiser(tmp_path_factory):
    """Yield a zero-arg callable that raises a TypeError whose traceback
    includes a frame whose ``co_filename`` ends with
    ``openai/lib/_parsing/_responses.py`` — mirroring the SDK's actual
    crash.  The stub module is written under tmp_path_factory so nothing
    leaks into the source tree."""
    import importlib.util

    stub_root = tmp_path_factory.mktemp("openai_sdk_frame_stub")
    parsing_dir = stub_root / "openai" / "lib" / "_parsing"
    parsing_dir.mkdir(parents=True, exist_ok=True)
    stub_path = parsing_dir / "_responses.py"
    stub_path.write_text(
        "def parse_response():\n"
        "    response_output = None\n"
        "    for _ in response_output:  # noqa: B007\n"
        "        pass\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        "openai_sdk_frame_stub._responses", str(stub_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.parse_response


class _FakeStream:
    """Stand-in for the SDK's ResponsesStreamManager.

    Yields one event so the loop body executes once, then on the next
    iteration invokes ``raiser`` to raise the SDK parser TypeError —
    exactly the shape the real bug presents (the terminal
    ``response.completed`` event is what trips ``parse_response``)."""

    def __init__(self, raiser):
        self._yielded = False
        self._raiser = raiser

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return self

    def __next__(self):
        if not self._yielded:
            self._yielded = True
            return SimpleNamespace(
                type="response.output_text.delta",
                delta="hi",
            )
        self._raiser()  # never returns
        raise StopIteration  # pragma: no cover

    def get_final_response(self):  # pragma: no cover - never reached
        raise AssertionError("should not be reached: stream crashes first")


def test_run_codex_stream_falls_back_on_sdk_null_output_typeerror(
    tmp_path, monkeypatch, sdk_parser_typeerror_raiser
):
    """The SDK parser TypeError from a null ``response.output`` field
    triggers the create(stream=True) fallback, which returns a valid
    response synthesised from the streamed text deltas."""
    from agent import codex_runtime

    agent = _make_codex_agent(tmp_path, monkeypatch)

    # Fake OpenAI client whose ``responses.stream`` returns our crashing manager.
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(
            stream=lambda **k: _FakeStream(sdk_parser_typeerror_raiser)
        )
    )

    # Capture whether the fallback fired and with what args.
    fallback_called = {"count": 0}

    def fake_fallback(self_agent, api_kwargs, client=None):
        fallback_called["count"] += 1
        return SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text="recovered")],
                )
            ]
        )

    monkeypatch.setattr(agent, "_run_codex_create_stream_fallback",
                        lambda kw, client=None: fake_fallback(agent, kw, client))

    resp = codex_runtime.run_codex_stream(
        agent,
        {"model": "gpt-5.5", "input": "hi", "instructions": "be brief"},
        client=fake_client,
    )

    assert fallback_called["count"] == 1, "fallback should have been invoked exactly once"
    assert isinstance(resp.output, list) and len(resp.output) == 1
    assert resp.output[0].content[0].text == "recovered"


def test_run_codex_stream_reraises_unrelated_typeerror(tmp_path, monkeypatch):
    """A TypeError that does NOT originate from the SDK parser is a real bug
    and must surface — the catch is narrow, not a blanket swallow."""
    from agent import codex_runtime

    agent = _make_codex_agent(tmp_path, monkeypatch)

    class _UnrelatedTypeErrorStream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return self
        def __next__(self):
            raise TypeError("unrelated: int() argument must be a string")

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(stream=lambda **k: _UnrelatedTypeErrorStream())
    )

    # If the fallback fires here, the test fails — the catch is too broad.
    monkeypatch.setattr(agent, "_run_codex_create_stream_fallback",
                        lambda kw, client=None: pytest.fail(
                            "fallback should not fire for unrelated TypeErrors"))

    with pytest.raises(TypeError, match="unrelated"):
        codex_runtime.run_codex_stream(
            agent,
            {"model": "gpt-5.5", "input": "hi"},
            client=fake_client,
        )


def test_fallback_synthesises_output_when_terminal_event_has_null_output(tmp_path, monkeypatch):
    """``run_codex_create_stream_fallback`` must treat ``output=None`` the
    same as ``output=[]`` when deciding to synthesise from text deltas —
    chatgpt.com/backend-api/codex emits the former on response.completed."""
    from agent import codex_runtime

    agent = _make_codex_agent(tmp_path, monkeypatch)

    # Raw SSE iterator: two text-delta events, then a response.completed
    # whose response.output is None (the actual backend behaviour).
    terminal_response = SimpleNamespace(output=None)
    events = [
        SimpleNamespace(type="response.output_text.delta", delta="hello "),
        SimpleNamespace(type="response.output_text.delta", delta="world"),
        SimpleNamespace(type="response.completed", response=terminal_response),
    ]

    class _RawStream:
        def __init__(self, evs):
            self._it = iter(evs)
        def __iter__(self): return self
        def __next__(self): return next(self._it)
        def close(self): pass

    raw_stream = _RawStream(events)
    fake_client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **k: raw_stream)
    )

    # Avoid touching the real transport-preflight machinery for this unit test.
    class _NoopTransport:
        def preflight_kwargs(self, kw, allow_stream=False):
            return kw
    monkeypatch.setattr(agent, "_get_transport", lambda: _NoopTransport())

    resp = codex_runtime.run_codex_create_stream_fallback(
        agent,
        {"model": "gpt-5.5", "input": "hi"},
        client=fake_client,
    )

    # Output must have been synthesised from the two collected deltas.
    assert resp is terminal_response
    assert isinstance(resp.output, list) and len(resp.output) == 1
    item = resp.output[0]
    assert item.type == "message"
    assert item.content[0].text == "hello world"
