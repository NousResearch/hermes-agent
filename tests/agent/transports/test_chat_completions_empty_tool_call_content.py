"""Empty assistant ``content`` on pure tool-call turns must reach the wire as ``null``.

Strict OpenAI-compatible validators (Bedrock-backed Claude: "text content blocks must be
non-empty", Mistral, Fireworks) reject ``content: ""`` beside ``tool_calls``; ``null`` is the
schema-compatible form. History stores ``""`` for textless tool-call turns, so both outgoing
boundaries normalize: the transport (``convert_messages``) and the strict-API helper the
max-iteration summary request uses instead of the transport. Upstream PR #31615.
"""

import pytest

from agent.reasoning_params import ReasoningParamsMixin
from agent.transports import get_transport

_TC = [{"id": "call_1", "type": "function", "function": {"name": "terminal", "arguments": "{}"}}]

# (message, expected wire content) — sentinel ``...`` means the key must stay absent.
CASES = [
    ({"role": "assistant", "content": "", "tool_calls": _TC}, None),
    ({"role": "assistant", "content": "  \n", "tool_calls": _TC}, None),
    ({"role": "assistant", "tool_calls": _TC}, ...),
    ({"role": "assistant", "content": "running it", "tool_calls": _TC}, "running it"),
    ({"role": "assistant", "content": ""}, ""),
]


def _check(out: dict, expected) -> None:
    if expected is ...:
        assert "content" not in out
    else:
        assert out["content"] == expected


@pytest.mark.parametrize("msg, expected", CASES)
def test_transport_normalizes_empty_tool_call_content(msg, expected):
    import agent.transports.chat_completions  # noqa: F401 — registers the transport
    original = dict(msg)
    out = get_transport("chat_completions").convert_messages([msg])[0]
    _check(out, expected)
    assert msg == original, "history copy must not be mutated"
    if "tool_calls" in msg:
        assert out["tool_calls"] == _TC


@pytest.mark.parametrize("msg, expected", CASES)
def test_strict_api_helper_normalizes_empty_tool_call_content(msg, expected):
    out = ReasoningParamsMixin._sanitize_tool_calls_for_strict_api(dict(msg), model="mistral-large")
    _check(out, expected)
