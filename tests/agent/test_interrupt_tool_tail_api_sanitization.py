"""Interrupted tool-tail closure stays API-only."""

from agent.agent_runtime_helpers import sanitize_api_messages


def _tool_tail(*, interrupted=False):
    tool_result = {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "ok edited",
    }
    if interrupted:
        tool_result["_interrupted_tool_tail"] = True
    return [
        {"role": "user", "content": "edit the file"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "function": {"name": "patch", "arguments": "{}"}}
            ],
        },
        tool_result,
    ]


def test_user_after_interrupted_tool_tail_is_closed_only_in_api_copy():
    canonical = _tool_tail(interrupted=True) + [
        {"role": "user", "content": "do something else"}
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert [message["role"] for message in canonical[-2:]] == ["tool", "user"]
    assert canonical[-2]["_interrupted_tool_tail"] is True
    assert [message["role"] for message in wire[-3:]] == [
        "tool",
        "assistant",
        "user",
    ]
    assert wire[-2]["content"] == "Operation interrupted."
    assert all("_interrupted_tool_tail" not in message for message in wire)


def test_normal_user_redirect_reaches_api_copy_unchanged():
    canonical = _tool_tail() + [{"role": "user", "content": "do something else"}]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical


def test_real_partial_assistant_text_remains_the_closure():
    canonical = _tool_tail() + [
        {"role": "assistant", "content": "Partial answer so far"},
        {"role": "user", "content": "do something else"},
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical
    assert wire[-2]["content"] == "Partial answer so far"


def test_legacy_synthetic_interrupt_sentinel_remains_compatible():
    canonical = _tool_tail() + [
        {"role": "assistant", "content": "Operation interrupted."},
        {"role": "user", "content": "do something else"},
    ]

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical


def test_tool_tail_without_followup_is_unchanged():
    canonical = _tool_tail()

    wire = sanitize_api_messages([dict(message) for message in canonical])

    assert wire == canonical
