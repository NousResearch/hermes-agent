"""Regression coverage for #32892.

The openai SDK's ``responses.stream()`` / ``responses.parse()`` eagerly
call ``_make_tools(tools)``, which iterates ``tools`` *without* a None
guard.  Passing ``tools=None`` therefore raises::

    TypeError: 'NoneType' object is not iterable

…before any HTTP request is issued.  This trips the
``openai-codex`` / ``gpt-5.5`` combo on ``chatgpt.com/backend-api/codex``
whenever the user runs Hermes without external tools registered: the
agent loop catches the TypeError, sees no HTTP status, classifies it as
non-retryable, and aborts (#32892).

These tests pin the defence:
:func:`agent.transports.codex.ResponsesApiTransport.build_kwargs` must
never emit ``tools=None`` — only add the ``tools`` key when there are
function tools to expose.  When there are no tools, the entire ``tools``
key (plus ``tool_choice`` and ``parallel_tool_calls`` which are
meaningless without it) is omitted from the kwargs.

Note: #33042 separately removed the SDK's ``responses.stream()`` helper
from our own Codex call paths, so the specific iteration crash inside
``_make_tools`` is also structurally avoided in normal operation.  This
test class additionally pins the SDK's ``_make_tools(None)`` contract so
we notice if upstream ever changes it.
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