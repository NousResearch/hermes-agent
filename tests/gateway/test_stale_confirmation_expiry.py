"""User-authored replay text is never interpreted as approval authority."""

import time

import pytest

from gateway.run import _build_gateway_agent_history


@pytest.mark.parametrize(
    "authored_text",
    [
        "confirm forced restart",
        "CONFIRM SHUTDOWN",
        "yes, delete everything",
        "confirm factory reset",
        "確認強制重開機",
    ],
)
def test_old_confirmation_like_prose_is_preserved_verbatim(authored_text):
    """Age and wording cannot rewrite an ordinary transcript row.

    Dangerous mutations are authorized by the explicit approval/passkey
    subsystem, not by scanning replay text for phrases that resemble consent.
    """
    history = [
        {
            "role": "user",
            "content": authored_text,
            "timestamp": time.time() - 86_400,
        },
        {"role": "assistant", "content": "I will use the approval boundary."},
    ]

    agent_history, _ = _build_gateway_agent_history(history)

    assert agent_history == [
        {"role": "user", "content": authored_text},
        {"role": "assistant", "content": "I will use the approval boundary."},
    ]


def test_replay_preserves_non_confirmation_text_too():
    history = [
        {"role": "user", "content": "can you help me with the docs?", "timestamp": 1},
        {"role": "assistant", "content": "Sure."},
    ]

    agent_history, _ = _build_gateway_agent_history(history)

    assert agent_history == [
        {"role": "user", "content": "can you help me with the docs?"},
        {"role": "assistant", "content": "Sure."},
    ]
