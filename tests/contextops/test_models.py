from __future__ import annotations

import pytest
from pydantic import ValidationError

from contextops.models import ContextPack, Event, StateDelta, Tension, Thread


def test_thread_rejects_generic_topic_only_identifier() -> None:
    with pytest.raises(ValidationError, match="topic-only"):
        Thread(
            id="pricing",
            anchor_event_ids=["evt-1"],
            stance="investigate pricing drift",
        )


def test_thread_accepts_specific_non_topic_anchor() -> None:
    thread = Thread(
        id="thread:discord:contextops:msg-42",
        anchor_event_ids=["evt-1"],
        stance="investigate pricing drift",
        heat=0.7,
    )

    assert thread.id == "thread:discord:contextops:msg-42"
    assert thread.heat == pytest.approx(0.7)


def test_context_pack_requires_restore_and_avoid_contracts() -> None:
    with pytest.raises(ValidationError, match="restore"):
        ContextPack(
            id="pack-1",
            thread_ids=["thread:discord:contextops:msg-42"],
            restore=[],
            avoid=["do not treat pack as transcript"],
        )

    with pytest.raises(ValidationError, match="avoid"):
        ContextPack(
            id="pack-1",
            thread_ids=["thread:discord:contextops:msg-42"],
            restore=["restore unresolved tension"],
            avoid=[],
        )


def test_state_delta_needs_evidence_refs_unless_low_confidence() -> None:
    with pytest.raises(ValidationError, match="evidence"):
        StateDelta(
            id="delta-1",
            kind="belief_shift",
            description="Thread heat increased because operator corrected a mistaken assumption.",
        )

    low_confidence = StateDelta(
        id="delta-2",
        kind="hypothesis",
        description="Possible unresolved tension emerged.",
        low_confidence=True,
    )

    evidenced = StateDelta(
        id="delta-3",
        kind="belief_shift",
        description="Operator explicitly separated thread from topic.",
        evidence_refs=["evt-1"],
    )

    assert low_confidence.low_confidence is True
    assert evidenced.evidence_refs == ["evt-1"]


def test_event_tension_and_pack_preserve_epistemic_fields() -> None:
    event = Event(
        id="evt-1",
        source="Devhub/#contextops",
        text="Thread is not topic; heat is not recency.",
        refs=["msg-42"],
    )
    tension = Tension(
        id="tension-1",
        thread_id="thread:discord:contextops:msg-42",
        description="Need distinguish heat from recency.",
        evidence_refs=[event.id],
    )
    pack = ContextPack(
        id="pack-1",
        thread_ids=["thread:discord:contextops:msg-42"],
        restore=["Restore active tension, not transcript text."],
        avoid=["Do not collapse compaction into summary."],
        event_ids=[event.id],
        tension_ids=[tension.id],
    )

    assert event.refs == ["msg-42"]
    assert tension.evidence_refs == ["evt-1"]
    assert pack.restore == ["Restore active tension, not transcript text."]
