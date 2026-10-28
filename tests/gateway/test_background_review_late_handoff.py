"""A gateway turn whose review ownership completed before the finalizer's hand-off never
spawns its review through the facade staging or the direct path.

``_GatewayReviewAdmission.finish`` (``gateway/run_turn_review.py``) runs when the adapter task
completes: after the confirmed send, or in the task's ``finally`` with
``delivery_succeeded=False`` when a ``/stop``, ``/new`` or teardown cancel lands. The gateway
never joins the executor thread on cancel (``_run_agent_await_turn_worker`` propagates the
``CancelledError``), so that thread can still be inside ``finalize_turn`` — the final response
already produced (``interrupted`` is False), ``clear_interrupt()`` already run, the plugin
``post_llm_call`` hook and the side-question lease still ahead of the hand-off. The hand-off
(``agent/turn_finalizer.py::_hand_off_background_review``) must then meet the finished owner
and drop the candidate as ``delivery_unconfirmed``. Falling through to the durable-lease
staging (spawned by ``ForegroundReviewOwnership.finish``) or to the direct spawn would review a
turn whose reply was never delivered — the outcome the delivery gate exists to prevent.

Turns that never had a gateway owner keep their paths (``tests/agent/test_skip_background_review.py``
pins the staging and the direct spawn).
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest

from agent import review_admission
from agent.turn_finalizer import _hand_off_background_review
from gateway.run_turn_review import _GatewayReviewAdmission

_PRIVATE_BODY = "private reply body"


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "ask"},
        {"role": "assistant", "content": _PRIVATE_BODY},
    ]


def _bound_turn(session_id: str, *, durable_holder: bool):
    profile_key = review_admission.current_profile_key()
    token = review_admission.note_turn_started(session_id, profile_key)
    admission = _GatewayReviewAdmission(session_id, profile_key, token)
    agent = types.SimpleNamespace(
        session_id=session_id,
        _spawn_background_review=MagicMock(),
        _active_session_turn_lease_holder="durable-holder" if durable_holder else None,
        _active_turn_token=7,
        _active_turn_profile_key=profile_key,
    )
    admission.bind_agent(agent)
    return admission, agent, profile_key


def _slug_lines(caplog, slug: str) -> list[str]:
    return [
        record.getMessage() for record in caplog.records if slug in record.getMessage()
    ]


@pytest.mark.parametrize(
    "durable_holder", [True, False], ids=["durable-holder", "bare-agent"]
)
@pytest.mark.parametrize(
    "delivery_succeeded",
    [False, True],
    ids=["cancelled", "completed-without-candidate"],
)
def test_a_hand_off_after_the_owner_finished_is_dropped_with_the_slug(
    caplog, durable_holder, delivery_succeeded
):
    """Orders B and C: ``finish`` first (the cancel completed the owner, or the owner completed
    with no candidate yet), then the late hand-off. Nothing is staged, nothing spawns, one
    owner-tagged body-free line names the reason, and the token was released exactly once."""
    session_id = f"late-handoff-{int(durable_holder)}-{int(delivery_succeeded)}"
    admission, agent, profile_key = _bound_turn(
        session_id, durable_holder=durable_holder
    )
    messages = _messages()

    admission.finish(delivery_succeeded=delivery_succeeded)
    with caplog.at_level("INFO"):
        _hand_off_background_review(
            agent, messages, review_memory=True, review_skills=False
        )

    assert not hasattr(agent, "_post_turn_background_review_candidate")
    agent._spawn_background_review.assert_not_called()
    assert admission._candidate is None
    # The finished owner stays bound so the late hand-off met it; the next turn's bind_agent
    # supersedes it.
    assert agent._gateway_review_admission is admission
    lines = _slug_lines(caplog, review_admission.REASON_DELIVERY_UNCONFIRMED)
    assert len(lines) == 1, caplog.text
    assert review_admission.owner_tag(profile_key, session_id) in lines[0]
    assert session_id not in lines[0]
    assert _PRIVATE_BODY not in lines[0]
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
    # A second completion is the latch, never a second log line.
    assert admission.finish(delivery_succeeded=delivery_succeeded) is None
    assert len(_slug_lines(caplog, review_admission.REASON_DELIVERY_UNCONFIRMED)) == 1


def test_the_next_turns_bind_supersedes_the_finished_owner(caplog):
    """A cached agent's next gateway turn binds a fresh admission over the finished one: its
    hand-off is captured there (spawned once after ITS confirmed delivery), never dropped against
    the previous turn's completion."""
    session_id = "late-handoff-rebind"
    first, agent, profile_key = _bound_turn(session_id, durable_holder=True)
    first.finish(delivery_succeeded=False)

    second_token = review_admission.note_turn_started(session_id, profile_key)
    second = _GatewayReviewAdmission(session_id, profile_key, second_token)
    second.bind_agent(agent)
    with caplog.at_level("INFO"):
        _hand_off_background_review(
            agent, _messages(), review_memory=True, review_skills=True
        )

    assert agent._gateway_review_admission is second
    assert isinstance(second._candidate, dict)
    assert second._candidate["_spawning_turn_token"] == second_token
    assert first._candidate is None
    assert not hasattr(agent, "_post_turn_background_review_candidate")
    agent._spawn_background_review.assert_not_called()
    assert _slug_lines(caplog, review_admission.REASON_DELIVERY_UNCONFIRMED) == []
    # Release the second turn without a delivery: the captured candidate is dropped, not spawned.
    assert second.finish(delivery_succeeded=False) is None
    agent._spawn_background_review.assert_not_called()
    assert review_admission.other_live_turn(session_id, None, profile_key) is False
