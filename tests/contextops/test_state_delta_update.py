from __future__ import annotations

import json
from typing import Any, Callable

import pytest
from pydantic import ValidationError

from contextops.extractor import StateDeltaProposal, extract_state_deltas
from contextops.models import Event, StateDelta, Thread


def _event(text: str = "operator corrected the heat-vs-recency assumption") -> Event:
    return Event(id="evt-1", source="Devhub/#contextops", text=text)


def _thread() -> Thread:
    return Thread(
        id="thread:devhub:contextops:msg-7",
        anchor_event_ids=["evt-0"],
        stance="separate heat from recency",
    )


def _model(payload: Any) -> Callable[[str], str]:
    def _call(prompt: str) -> str:
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _call


def test_extractor_emits_state_delta_proposals_with_evidence_and_confidence() -> None:
    proposals = extract_state_deltas(
        _event(),
        [_thread()],
        _model(
            [
                {
                    "thread_id": "thread:devhub:contextops:msg-7",
                    "kind": "belief_shift",
                    "description": "operator separated heat from recency",
                    "evidence_refs": ["evt-1"],
                    "confidence": 0.82,
                    "low_confidence": False,
                }
            ]
        ),
    )
    assert len(proposals) == 1
    proposal = proposals[0]
    assert isinstance(proposal, StateDeltaProposal)
    assert proposal.kind == "belief_shift"
    assert proposal.evidence_refs == ["evt-1"]
    assert 0.0 <= proposal.confidence <= 1.0
    assert proposal.low_confidence is False


def test_extractor_allows_low_confidence_proposal_without_evidence() -> None:
    proposals = extract_state_deltas(
        _event(),
        [_thread()],
        _model(
            [
                {
                    "thread_id": None,
                    "kind": "hypothesis",
                    "description": "a possible new tension may be forming",
                    "evidence_refs": [],
                    "confidence": 0.2,
                    "low_confidence": True,
                }
            ]
        ),
    )
    assert proposals[0].low_confidence is True
    assert proposals[0].evidence_refs == []


def test_extractor_rejects_summary_only_output() -> None:
    # a summary lacks delta kind + evidence; it is only prose
    with pytest.raises((ValueError, ValidationError)):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model([{"description": "here is a summary of the whole conversation"}]),
        )


def test_extractor_rejects_evidence_free_high_confidence_delta() -> None:
    with pytest.raises((ValueError, ValidationError)):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model(
                [
                    {
                        "thread_id": None,
                        "kind": "belief_shift",
                        "description": "a claimed shift with no evidence",
                        "evidence_refs": [],
                        "confidence": 0.9,
                        "low_confidence": False,
                    }
                ]
            ),
        )


def test_extractor_fails_closed_on_invalid_json() -> None:
    with pytest.raises(ValueError):
        extract_state_deltas(_event(), [_thread()], _model("not json"))


def test_extractor_fails_closed_when_output_is_not_a_list() -> None:
    with pytest.raises(ValueError):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model(
                {
                    "kind": "belief_shift",
                    "description": "d",
                    "evidence_refs": ["evt-1"],
                    "confidence": 0.5,
                }
            ),
        )


def test_extractor_rejects_out_of_range_confidence() -> None:
    with pytest.raises((ValueError, ValidationError)):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model(
                [
                    {
                        "thread_id": None,
                        "kind": "belief_shift",
                        "description": "d",
                        "evidence_refs": ["evt-1"],
                        "confidence": -0.1,
                        "low_confidence": False,
                    }
                ]
            ),
        )


def test_extractor_rejects_unknown_thread_reference() -> None:
    with pytest.raises(ValueError):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model(
                [
                    {
                        "thread_id": "thread:devhub:contextops:msg-404",
                        "kind": "belief_shift",
                        "description": "d",
                        "evidence_refs": ["evt-1"],
                        "confidence": 0.7,
                        "low_confidence": False,
                    }
                ]
            ),
        )


def test_extractor_rejects_evidence_ref_not_anchored_to_current_event() -> None:
    with pytest.raises(ValueError, match="current event"):
        extract_state_deltas(
            _event(),
            [_thread()],
            _model(
                [
                    {
                        "thread_id": "thread:devhub:contextops:msg-7",
                        "kind": "belief_shift",
                        "description": "a contaminated cross-event claim",
                        "evidence_refs": ["evt-other"],
                        "confidence": 0.82,
                        "low_confidence": False,
                    }
                ]
            ),
        )


def test_extractor_proposal_converts_to_state_delta() -> None:
    proposals = extract_state_deltas(
        _event(),
        [_thread()],
        _model(
            [
                {
                    "thread_id": "thread:devhub:contextops:msg-7",
                    "kind": "belief_shift",
                    "description": "operator separated heat from recency",
                    "evidence_refs": ["evt-1"],
                    "confidence": 0.82,
                    "low_confidence": False,
                }
            ]
        ),
    )
    delta = proposals[0].to_state_delta(id="delta-1")
    assert isinstance(delta, StateDelta)
    assert delta.kind == "belief_shift"
    assert delta.evidence_refs == ["evt-1"]
