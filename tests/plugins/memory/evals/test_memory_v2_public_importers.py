"""Public benchmark importer skeleton tests for Memory v2 evals."""

from __future__ import annotations

from pathlib import Path

from plugins.memory.memory_v2.evals.datasets import EvalEvent, EvalQuery, load_locomo_sample

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_locomo_sample_maps_messages_to_events_and_qa_to_queries():
    dataset = load_locomo_sample(FIXTURES / "locomo_tiny_sample.json")

    assert dataset.name == "locomo_tiny_sample"
    assert "not real LoCoMo data" in dataset.description
    assert dataset.version == 1

    assert dataset.events == [
        EvalEvent(
            id="locomo_msg_001",
            session_id="locomo_conv_tiny_001",
            role="user",
            text="I moved to Seattle in 2024 and keep my coffee grinder in storage.",
        ),
        EvalEvent(
            id="locomo_msg_002",
            session_id="locomo_conv_tiny_001",
            role="assistant",
            text="Got it: Seattle move in 2024 and the coffee grinder is in storage.",
        ),
    ]
    assert dataset.queries == [
        EvalQuery(
            id="locomo_qa_001",
            route="past_conversation_exact",
            text="Where did the user move in 2024?",
            expected_answer_contains=["Seattle"],
            expected_source_refs=["locomo_msg_001"],
            should_retrieve=True,
        )
    ]


def test_load_locomo_sample_preserves_source_message_refs():
    dataset = load_locomo_sample(FIXTURES / "locomo_tiny_sample.json")

    query = dataset.query_by_id("locomo_qa_001")

    assert query.expected_source_refs == ["locomo_msg_001"]
    assert {event.id for event in dataset.events} >= set(query.expected_source_refs)
