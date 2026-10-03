"""Stacked handoff carriers keep projecting the live ask, not an older summary (#126102).

A force-user-leading carrier folds the summary in front of the live ask:

    [CONTEXT COMPACTION — REFERENCE ONLY] ... summary ...
    --- END OF CONTEXT SUMMARY — ... ---
    <the live user ask>

When such a carrier itself rides the protected tail into the NEXT compaction
and a new summary is folded in front again, the stored content stacks two
generations (two end markers). The display projections peel exactly one
generation: the "live view" they keep is the PREVIOUS generation's full
summary carrier, which then paints as the user's message — the reported
symptom: Desktop shows ``[CONTEXT COMPACTION SUMMARY]`` instead of the newest
user message.

The unwrap must repeat: keep peeling while the remainder is itself a summary
carrier; only the final non-summary remainder is the live view.
"""
import pytest

from agent.compaction_display import project_compaction_message_for_display
from agent.context_compressor import (
    COMPRESSED_SUMMARY_METADATA_KEY,
    ContextCompressor,
    SUMMARY_PREFIX,
    _SUMMARY_END_MARKER,
    is_compaction_summary_message,
)


ASK = "still freeze at the krita ."


def _carrier(*generations: str, ask: str = ASK) -> dict:
    """A user carrier whose content stacks *generations* summaries before *ask*."""
    parts = []
    for body in generations:
        parts.append(f"{SUMMARY_PREFIX}\n{body}\n\n{_SUMMARY_END_MARKER}\n\n")
    return {"role": "user", "content": "".join(parts) + ask, COMPRESSED_SUMMARY_METADATA_KEY: True}


def test_single_generation_carrier_projects_the_live_ask():
    projected = project_compaction_message_for_display(_carrier("gen1 summary"))
    assert projected is not None
    assert projected["content"] == ASK


def test_stacked_generation_carrier_projects_the_live_ask():
    """Two stacked generations: the live view must be the ask, not gen1's summary."""
    message = _carrier("gen2 summary", "gen1 summary")
    assert message["content"].count(_SUMMARY_END_MARKER) == 2
    projected = project_compaction_message_for_display(message)
    assert projected is not None
    assert projected["content"] == ASK, "the live view must not be a summary carrier"


def test_stacked_carrier_live_view_is_not_classified_as_a_summary():
    message = _carrier("gen2 summary", "gen1 summary")
    live = ContextCompressor._strip_context_summary_handoff_message(message)
    assert live is not None
    assert not is_compaction_summary_message(live), (
        "the unwrapped live view still carries the handoff prefix and would paint as the summary"
    )


def test_standalone_stacked_summary_without_live_ask_stays_hidden():
    """No ask after the last marker: a pure stacked handoff projects to None (hidden)."""
    content = (
        f"{SUMMARY_PREFIX}\ngen2 summary\n\n{_SUMMARY_END_MARKER}\n\n"
        f"{SUMMARY_PREFIX}\ngen1 summary\n\n{_SUMMARY_END_MARKER}"
    )
    message = {"role": "user", "content": content, COMPRESSED_SUMMARY_METADATA_KEY: True}
    assert project_compaction_message_for_display(message) is None
