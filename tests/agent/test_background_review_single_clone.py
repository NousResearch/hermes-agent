"""Every review hand-off path structurally clones the live transcript exactly once.

``run_agent.AIAgent._spawn_background_review`` is the single chokepoint that owns the
isolation guarantee (the fork sanitizes its history in place, so the review must never share
list/dict objects with the live transcript). The bare branch of
``agent/turn_finalizer._hand_off_background_review`` (no gateway admission, no durable lease
holder: a CLI turn) calls that chokepoint synchronously, so it must hand over a shallow list
copy and let the chokepoint clone — a pre-clone there made every CLI turn that tripped a
review pay two O(transcript) structural clones. The gateway admission and the lease-holder
staging clone once at capture time instead, because their spawn runs after the live
transcript may already have moved on.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import agent.turn_finalizer as turn_finalizer_module
from agent.turn_finalizer import _hand_off_background_review
from gateway.run_turn_review import _GatewayReviewAdmission
from tests.agent.test_background_review_compute_overlap import (  # fixtures register by import
    _bare_agent,
    _config,
    _patch_config,
    clear_review_admission_state,
    review_forks,
)


def _count_clones(monkeypatch) -> list[list[dict]]:
    """Wrap the real clone helper so both the finalizer and the chokepoint (which imports it
    lazily by attribute at call time) are counted."""
    real = turn_finalizer_module._clone_background_review_messages
    calls: list[list[dict]] = []

    def counting(messages):
        calls.append(messages)
        return real(messages)

    monkeypatch.setattr(
        turn_finalizer_module, "_clone_background_review_messages", counting
    )
    return calls


def _live_messages() -> list[dict]:
    return [
        {"role": "user", "content": "please read x"},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "answer"}],
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {"name": "read_file", "arguments": '{"path":"x"}'},
                }
            ],
        },
    ]


def _assert_isolated(snapshot: list[dict], live: list[dict]) -> None:
    """Mutating the spawned snapshot never touches the live transcript."""
    assert snapshot is not live
    assert all(s is not l for s in snapshot for l in live)
    snapshot[-1]["content"][0]["text"] = "review mutation"
    snapshot[-1]["tool_calls"][0]["function"]["arguments"] = "{}"
    snapshot.append({"role": "user", "content": "review-appended"})
    assert live[-1]["content"][0]["text"] == "answer"
    assert live[-1]["tool_calls"][0]["function"]["arguments"] == '{"path":"x"}'
    assert len(live) == 2


def test_bare_hand_off_clones_the_transcript_exactly_once_at_the_chokepoint(
    review_forks,  # health: allow F811 -- pytest injects the imported fixture by parameter name
    monkeypatch,
):
    _patch_config(monkeypatch, _config())
    clones = _count_clones(monkeypatch)
    agent = _bare_agent()
    live = _live_messages()

    _hand_off_background_review(agent, live, review_memory=True, review_skills=False)

    assert len(review_forks) == 1, "the bare path must still spawn the review"
    assert len(clones) == 1, (
        f"expected one structural clone on the bare path, got {len(clones)}"
    )
    _assert_isolated(review_forks[0]["history"], live)


def test_lease_holder_hand_off_stages_one_isolated_snapshot(monkeypatch):
    clones = _count_clones(monkeypatch)
    agent = _bare_agent()
    agent._active_session_turn_lease_holder = object()
    agent._spawn_background_review = MagicMock(name="_spawn_background_review")
    live = _live_messages()

    _hand_off_background_review(agent, live, review_memory=True, review_skills=False)

    agent._spawn_background_review.assert_not_called()
    assert len(clones) == 1
    _assert_isolated(
        agent._post_turn_background_review_candidate["messages_snapshot"], live
    )


def test_gateway_hand_off_captures_one_isolated_snapshot(monkeypatch):
    clones = _count_clones(monkeypatch)
    agent = _bare_agent()
    admission = _GatewayReviewAdmission(agent.session_id, "profile", 1)
    agent._gateway_review_admission = admission
    agent._spawn_background_review = MagicMock(name="_spawn_background_review")
    live = _live_messages()

    _hand_off_background_review(agent, live, review_memory=True, review_skills=False)

    agent._spawn_background_review.assert_not_called()
    assert len(clones) == 1
    _assert_isolated(admission._candidate["messages_snapshot"], live)
