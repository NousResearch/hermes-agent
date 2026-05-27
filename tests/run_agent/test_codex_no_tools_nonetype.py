"""Regression coverage for #32892.

The openai SDK's ``responses.stream()`` / ``responses.parse()`` eagerly
call ``_make_tools(tools)``, which iterates ``tools`` *without* a None
guard.  Passing ``tools=None`` therefore raises::

    TypeError: 'NoneType' object is not iterable

…before any HTTP request is issued.  This trips the
``openai-codex`` / ``gpt-5.5`` combo on ``chatgpt.com/backend-api/codex``
whenever the user runs Hermes without external tools registered: the
agent loop catches the TypeError, sees no HTTP status, classifies it as
non-retryable, and aborts.

These tests pin the two defences that together prevent the regression:

1.  :func:`agent.transports.codex.ResponsesApiTransport.build_kwargs`
    must never emit ``tools=None`` (only add the key when there are
    function tools to expose).
2.  :func:`agent.codex_runtime._strip_sdk_none_iterables` must strip
    ``tools=None`` (and the meaningless ``tool_choice`` /
    ``parallel_tool_calls`` partners) at the SDK boundary, so any code
    path that bypasses preflight or re-injects the key still survives.
"""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest


# Stub optional deps the parent module imports at top level — keeps this
# test file runnable in the same environment as the existing Codex tests.
sys.modules.setdefault("fire", types.SimpleNamespace(Fire=lambda *a, **k: None))
sys.modules.setdefault("firecrawl", types.SimpleNamespace(Firecrawl=object))
sys.modules.setdefault("fal_client", types.SimpleNamespace())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def transport():
    """Fresh ``ResponsesApiTransport`` per test (it is stateless but
    the import has side-effects on a global transport registry)."""
    from agent.transports.codex import ResponsesApiTransport

    return ResponsesApiTransport()


@pytest.fixture
def codex_messages() -> List[Dict[str, Any]]:
    """Minimal Codex-shaped chat history mirroring the #32892 reproducer:
    one system + one short user message, with no tool calls in history."""
    return [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Hey! What can I help you with?"},
    ]


def _build_kwargs_no_tools(transport, messages) -> Dict[str, Any]:
    """Exercise the real ``build_kwargs`` for the codex backend with no tools."""
    return transport.build_kwargs(
        model="gpt-5.5",
        messages=messages,
        tools=None,
        is_codex_backend=True,
    )


# ---------------------------------------------------------------------------
# build_kwargs: the "tools=None" key must never appear
# ---------------------------------------------------------------------------


def test_build_kwargs_omits_tools_key_when_no_tools(transport, codex_messages):
    """``build_kwargs`` must not place ``tools=None`` in the outgoing dict.

    Putting ``tools=None`` reaches ``responses.stream()`` which calls
    ``_make_tools(None)`` and crashes with the #32892 TypeError before any
    request is sent.
    """
    kwargs = _build_kwargs_no_tools(transport, codex_messages)

    assert "tools" not in kwargs, (
        f"tools key must be omitted entirely when no tools are registered, "
        f"got kwargs={sorted(kwargs)}"
    )


def test_build_kwargs_omits_tool_choice_and_parallel_when_no_tools(transport, codex_messages):
    """``tool_choice`` / ``parallel_tool_calls`` are meaningless without
    tools — and some backends 400 on them.  Confirm we never set them."""
    kwargs = _build_kwargs_no_tools(transport, codex_messages)

    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs


def test_build_kwargs_keeps_required_codex_fields_without_tools(transport, codex_messages):
    """The toolless build must still emit the non-negotiable Codex fields
    (model / instructions / input / store) — otherwise we'd just be moving
    the bug from the SDK to preflight."""
    kwargs = _build_kwargs_no_tools(transport, codex_messages)

    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "You are Hermes."
    assert kwargs["store"] is False
    assert isinstance(kwargs["input"], list)
    assert kwargs["input"] and kwargs["input"][0]["role"] == "user"


def test_build_kwargs_emits_tools_when_tools_present(transport, codex_messages):
    """Sanity check the inverse: when tools ARE provided, they MUST appear
    in the outgoing kwargs along with the related ``tool_choice`` /
    ``parallel_tool_calls`` switches."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    kwargs = transport.build_kwargs(
        model="gpt-5.5",
        messages=codex_messages,
        tools=tools,
        is_codex_backend=True,
    )

    assert "tools" in kwargs and kwargs["tools"], "tools must be present when registered"
    assert kwargs["tools"][0]["name"] == "terminal"
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_build_kwargs_drops_empty_tools_list(transport, codex_messages):
    """``tools=[]`` collapses to ``None`` inside ``_responses_tools`` —
    the resulting kwargs must therefore also omit the key."""
    kwargs = transport.build_kwargs(
        model="gpt-5.5",
        messages=codex_messages,
        tools=[],
        is_codex_backend=True,
    )

    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs
    assert "parallel_tool_calls" not in kwargs


# ---------------------------------------------------------------------------
# _strip_sdk_none_iterables: belt-and-braces guard at the SDK boundary
# ---------------------------------------------------------------------------


def test_strip_sdk_none_iterables_removes_tools_none():
    """The defensive shim must drop ``tools=None`` before the SDK iterates it."""
    from agent.codex_runtime import _strip_sdk_none_iterables

    kwargs = {
        "model": "gpt-5.5",
        "instructions": "be helpful",
        "input": [{"role": "user", "content": "hi"}],
        "tools": None,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
    }
    result = _strip_sdk_none_iterables(kwargs)

    # In-place mutation is documented in the helper's docstring so the
    # caller's dict reflects what was actually sent.
    assert result is kwargs
    assert "tools" not in result
    assert "tool_choice" not in result, (
        "tool_choice is meaningless without tools and 400s on some backends"
    )
    assert "parallel_tool_calls" not in result
    # Untouched fields stay put.
    assert result["model"] == "gpt-5.5"
    assert result["store"] is False


def test_strip_sdk_none_iterables_preserves_real_tools():
    """A populated ``tools`` list must pass through untouched."""
    from agent.codex_runtime import _strip_sdk_none_iterables

    real_tools = [{"type": "function", "name": "terminal"}]
    kwargs = {
        "model": "gpt-5.5",
        "tools": real_tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    _strip_sdk_none_iterables(kwargs)

    assert kwargs["tools"] is real_tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_strip_sdk_none_iterables_noop_without_tools_key():
    """When the key was never set, the helper must NOT add it."""
    from agent.codex_runtime import _strip_sdk_none_iterables

    kwargs = {"model": "gpt-5.5", "instructions": "x"}
    _strip_sdk_none_iterables(kwargs)
    assert "tools" not in kwargs
    assert sorted(kwargs) == ["instructions", "model"]


def test_strip_sdk_none_iterables_handles_non_dict():
    """Defensive: a non-dict input must not raise."""
    from agent.codex_runtime import _strip_sdk_none_iterables

    assert _strip_sdk_none_iterables(None) is None
    assert _strip_sdk_none_iterables("not a dict") == "not a dict"


# ---------------------------------------------------------------------------
# End-to-end: run_codex_stream must not crash when api_kwargs has tools=None
# ---------------------------------------------------------------------------


class _RecordingResponsesStream:
    """Stand-in for the openai SDK's stream manager that records the kwargs
    it was invoked with, so the test can assert ``tools=None`` was scrubbed
    before reaching the SDK."""

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self._final = SimpleNamespace(
            output=[
                SimpleNamespace(
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[SimpleNamespace(type="output_text", text="ok")],
                )
            ],
            status="completed",
            model="gpt-5.5",
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def __iter__(self):
        return iter(())

    def get_final_response(self):
        return self._final


def _fake_codex_agent():
    """Tiny stand-in for ``AIAgent`` exposing exactly the surface that
    ``run_codex_stream`` touches.  Avoids the full bootstrap (~25 imports)
    so the regression remains a unit test."""
    agent = SimpleNamespace()
    agent._interrupt_requested = False
    agent._codex_streamed_text_parts = []
    agent._codex_stream_last_event_ts = None
    agent._client_log_context = lambda: ""
    agent._touch_activity = lambda *_a, **_k: None
    agent._fire_stream_delta = lambda *_a, **_k: None
    agent._fire_reasoning_delta = lambda *_a, **_k: None
    return agent


def test_run_codex_stream_scrubs_tools_none_before_sdk_call():
    """Even if a caller bypasses preflight, ``run_codex_stream`` must drop
    ``tools=None`` so the openai SDK's ``_make_tools(None)`` never sees it.

    This is the actual #32892 reproducer: ``api_kwargs`` contains
    ``tools=None``, ``run_codex_stream`` is called, and the call must
    succeed instead of raising ``TypeError: 'NoneType' object is not iterable``.
    """
    from agent.codex_runtime import run_codex_stream

    captured: Dict[str, Any] = {}

    def _fake_stream(**kwargs):
        captured.update(kwargs)
        return _RecordingResponsesStream(kwargs)

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(stream=_fake_stream)
    )
    agent = _fake_codex_agent()
    api_kwargs = {
        "model": "gpt-5.5",
        "instructions": "be helpful",
        "input": [{"role": "user", "content": "Hey! What can I help you with?"}],
        "tools": None,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
    }

    # Must NOT raise TypeError.
    response = run_codex_stream(agent, api_kwargs, client=fake_client)

    assert response is not None
    # The SDK call site never saw tools=None.
    assert "tools" not in captured, (
        f"tools=None leaked into responses.stream() kwargs: {captured}"
    )
    assert "tool_choice" not in captured
    assert "parallel_tool_calls" not in captured
    # And the caller's dict reflects the scrub (documented in-place
    # mutation), so debug dumps / retries see what was actually sent.
    assert "tools" not in api_kwargs


def test_run_codex_stream_preserves_real_tools():
    """The scrub must NOT remove a populated tools list — that would
    silently break every tool-calling turn."""
    from agent.codex_runtime import run_codex_stream

    captured: Dict[str, Any] = {}

    def _fake_stream(**kwargs):
        captured.update(kwargs)
        return _RecordingResponsesStream(kwargs)

    fake_client = SimpleNamespace(
        responses=SimpleNamespace(stream=_fake_stream)
    )
    agent = _fake_codex_agent()
    real_tools = [
        {"type": "function", "name": "terminal", "parameters": {"type": "object"}},
    ]
    api_kwargs = {
        "model": "gpt-5.5",
        "instructions": "be helpful",
        "input": [{"role": "user", "content": "ls"}],
        "tools": real_tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "store": False,
    }

    run_codex_stream(agent, api_kwargs, client=fake_client)

    assert captured.get("tools") is real_tools
    assert captured.get("tool_choice") == "auto"
    assert captured.get("parallel_tool_calls") is True


# ---------------------------------------------------------------------------
# Direct openai SDK assertion: the bug is real (sanity-checks the fix target)
# ---------------------------------------------------------------------------


def test_openai_sdk_raises_typeerror_on_tools_none():
    """Document the upstream behaviour the two defences guard against.

    If the SDK ever fixes ``_make_tools(None)`` to return ``omit``
    gracefully, this test will start failing — at which point the agent
    defences become belt-only and this test should be flipped to an
    ``xfail`` so we notice the upstream change.
    """
    from openai.resources.responses.responses import _make_tools

    with pytest.raises(TypeError, match="NoneType.*not iterable"):
        _make_tools(None)
