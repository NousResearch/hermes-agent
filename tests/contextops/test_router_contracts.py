from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from contextops.models import Event, Thread
from contextops.router import RouteProposal, build_router_prompt, route_context_event


def _event(text: str) -> Event:
    return Event(id="evt-1", source="Devhub/#contextops", text=text)


def _pricing_thread() -> Thread:
    return Thread(
        id="thread:devhub:contextops:msg-7",
        anchor_event_ids=["evt-0"],
        stance="investigate why pricing drift keeps recurring after each release",
    )


def _model(payload: Any) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _call


def test_router_prompt_exposes_thread_stances_for_conceptual_matching() -> None:
    thread = _pricing_thread()
    prompt = build_router_prompt(_event("the cost figures keep sliding again"), [thread])
    assert thread.id in prompt
    # the stance concept must be available so a model can match conceptually
    assert "pricing drift" in prompt


def test_router_supports_conceptual_continuation_without_keyword_overlap() -> None:
    thread = _pricing_thread()
    # event shares stance/evidence concepts but never says "pricing"
    event = _event("the cost figures keep sliding right after we ship a release")
    proposal = route_context_event(
        event,
        [thread],
        _model(
            {
                "event_id": "evt-1",
                "decision": "continue",
                "thread_id": thread.id,
                "rationale": "cost sliding after each release continues the drift investigation",
                "confidence": 0.74,
                "evidence_refs": ["evt-1"],
            }
        ),
    )
    assert proposal.decision == "continue"
    assert proposal.thread_id == thread.id
    # prove this is a conceptual match, not keyword-only
    assert "pricing" not in event.text.lower()


def test_router_can_open_new_thread() -> None:
    proposal = route_context_event(
        _event("an unrelated infrastructure incident report"),
        [_pricing_thread()],
        _model(
            {
                "event_id": "evt-1",
                "decision": "new",
                "thread_id": None,
                "rationale": "no existing thread covers this incident",
                "confidence": 0.6,
                "evidence_refs": ["evt-1"],
            }
        ),
    )
    assert proposal.decision == "new"
    assert proposal.thread_id is None


def test_router_fails_closed_on_invalid_json() -> None:
    with pytest.raises(ValueError):
        route_context_event(_event("x"), [_pricing_thread()], _model("not json at all"))


def test_router_fails_closed_on_non_object_json() -> None:
    with pytest.raises(ValueError):
        route_context_event(_event("x"), [_pricing_thread()], _model([1, 2, 3]))


def test_router_fails_closed_on_invalid_model_shape() -> None:
    with pytest.raises((ValueError, ValidationError)):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model({"decision": "continue"}),
        )


def test_router_fails_closed_on_out_of_range_confidence() -> None:
    with pytest.raises((ValueError, ValidationError)):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-1",
                    "decision": "new",
                    "thread_id": None,
                    "rationale": "r",
                    "confidence": 1.5,
                    "evidence_refs": ["evt-1"],
                }
            ),
        )


def test_router_fails_closed_when_continue_targets_unknown_thread() -> None:
    with pytest.raises(ValueError):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-1",
                    "decision": "continue",
                    "thread_id": "thread:devhub:contextops:msg-999",
                    "rationale": "r",
                    "confidence": 0.8,
                    "evidence_refs": ["evt-1"],
                }
            ),
        )


def test_router_fails_closed_when_model_event_id_is_not_current_event() -> None:
    with pytest.raises(ValueError, match="current event"):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-other",
                    "decision": "new",
                    "thread_id": None,
                    "rationale": "r",
                    "confidence": 0.5,
                    "evidence_refs": ["evt-1"],
                }
            ),
        )


def test_router_fails_closed_when_evidence_ref_is_not_anchored_to_current_event() -> None:
    with pytest.raises(ValueError, match="current event"):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-1",
                    "decision": "new",
                    "thread_id": None,
                    "rationale": "r",
                    "confidence": 0.5,
                    "evidence_refs": ["evt-other"],
                }
            ),
        )


def test_router_fails_closed_when_continue_missing_thread_id() -> None:
    with pytest.raises((ValueError, ValidationError)):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-1",
                    "decision": "continue",
                    "thread_id": None,
                    "rationale": "r",
                    "confidence": 0.8,
                    "evidence_refs": ["evt-1"],
                }
            ),
        )


def test_router_requires_non_empty_evidence_refs() -> None:
    with pytest.raises((ValueError, ValidationError)):
        route_context_event(
            _event("x"),
            [_pricing_thread()],
            _model(
                {
                    "event_id": "evt-1",
                    "decision": "new",
                    "thread_id": None,
                    "rationale": "r",
                    "confidence": 0.5,
                    "evidence_refs": [],
                }
            ),
        )


def test_route_proposal_is_a_proposal_not_a_persisted_record() -> None:
    proposal = RouteProposal(
        event_id="evt-1",
        decision="new",
        rationale="opening a fresh thread",
        confidence=0.5,
        evidence_refs=["evt-1"],
    )
    # proposals carry confidence + evidence and are not transcripts/summaries
    assert 0.0 <= proposal.confidence <= 1.0
    assert proposal.evidence_refs == ["evt-1"]
