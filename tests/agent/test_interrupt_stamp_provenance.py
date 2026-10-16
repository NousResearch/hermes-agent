"""The ``Operation interrupted.`` wire stamp is scoped to interruption provenance.

Review note on #63292: the sanitizer's tool-tail closure must fire only for
cancellation-originated tails. A ``tool → user`` adjacency that is a valid
redirect — assistant issued tool_calls, tool ran, user redirected — is the
documented "user jumped in mid-run" pattern and must reach the API copy
unstamped (see ``apply_pending_steer_to_tool_results``).
"""

from agent.agent_runtime_helpers import (
    apply_pending_steer_to_tool_results,
    sanitize_api_messages,
)

_STAMP = "Operation interrupted."


def _tool_tail():
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "c1", "function": {"name": "patch", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok edited"},
    ]


def test_valid_redirect_adjacency_is_not_stamped():
    """NEGATIVE: ``assistant(tool_calls) → tool → user`` is a valid redirect —
    the redirect user row must never acquire a synthetic cancellation turn."""
    canonical = _tool_tail() + [{"role": "user", "content": "actually use tabs"}]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert all(message.get("content") != _STAMP for message in wire)
    assert [message["role"] for message in wire] == ["user", "assistant", "tool", "user"]


def test_cancellation_originated_tail_is_stamped():
    """POSITIVE: a tool tail marked by a cancellation exit gets the API-only
    closure before the next user row; the canonical transcript keeps real rows."""
    tail = _tool_tail()
    tail[-1]["_interrupted_tool_tail"] = True
    canonical = tail + [{"role": "user", "content": "resume"}]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    stamps = [message for message in wire if message.get("content") == _STAMP]
    assert [message["role"] for message in stamps] == ["assistant"]
    assert [message["role"] for message in wire[-3:]] == ["tool", "assistant", "user"]
    assert wire[-2]["content"] == _STAMP
    assert not any(message.get("content") == _STAMP for message in canonical)
    assert [message["role"] for message in canonical[-2:]] == ["tool", "user"]


def test_steer_drained_redirect_row_is_not_stamped():
    """NEGATIVE through the real redirect producer: a /steer row drained after a
    tool batch lands as ``tool → user`` and stays unstamped end to end."""
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._pending_steer = None
    agent._pending_steer_lock = None
    assert agent.steer("switch to the staging config") is True
    messages = _tool_tail()
    apply_pending_steer_to_tool_results(agent, messages, 1)

    assert messages[-1]["role"] == "user"
    wire = sanitize_api_messages([dict(message) for message in messages])

    assert all(message.get("content") != _STAMP for message in wire)
    assert [message["role"] for message in wire] == ["user", "assistant", "tool", "user"]
