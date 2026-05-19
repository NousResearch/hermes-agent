from __future__ import annotations

from pathlib import Path

import pytest

from contextops.context_pack import build_context_pack
from contextops.models import ContextPack

SEED_PATH = Path(__file__).parent / "fixtures" / "epistemic_state_engine_seed.yaml"

PRESSURED_THREAD = "thread:discord:contextops:msg-42"
RECENT_THREAD = "thread:cli:contextops:msg-99"

# A message carrying cognitive pressure, no topic label of the recent thread.
PRESSURE_MESSAGE = (
    "I'm still stuck on that unresolved coupling anomaly — the contradiction "
    "between the two systems has not gone away."
)


def test_pack_is_a_context_pack_with_restore_and_avoid_sections() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)

    assert isinstance(pack, ContextPack)
    assert pack.restore, "pack must carry a non-empty restore section"
    assert pack.avoid, "pack must carry a non-empty avoid section"


def test_pack_restores_the_pressured_thread_not_the_recent_topic_thread() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)

    assert pack.thread_ids[0] == PRESSURED_THREAD
    restore_text = " ".join(pack.restore).lower()
    assert "coupling" in restore_text
    assert "unresolved" in restore_text


def test_pack_favors_open_tension_over_recency_for_a_topic_only_message() -> None:
    """A bare topic-label hit on the recent thread must not outrank live pressure."""

    pack = build_context_pack(SEED_PATH, "quick question about pricing")
    scores = pack.metadata["scores"]

    assert scores[PRESSURED_THREAD] > scores[RECENT_THREAD]
    assert pack.thread_ids[0] == PRESSURED_THREAD


def test_pack_avoid_section_blocks_the_five_contamination_collapses() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)
    avoid = " ".join(pack.avoid).lower()

    # thread != topic, heat != recency, compaction != summary,
    # context pack != transcript, StateDelta != note-taking.
    assert "thread" in avoid and "topic" in avoid
    assert "heat" in avoid and "recency" in avoid
    assert "compaction" in avoid and "summary" in avoid
    assert "transcript" in avoid
    assert "statedelta" in avoid and "note-taking" in avoid


def test_pack_preserves_evidence_refs_in_ids_and_metadata() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)

    assert "tension-coupling" in pack.tension_ids
    assert "tension-pricing" not in pack.tension_ids  # resolved -> excluded
    assert "evt-coupling" in pack.event_ids
    assert "evt-coupling" in pack.metadata["evidence_refs"]


def test_pack_excludes_the_recency_only_thread_from_thread_ids() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)

    assert RECENT_THREAD not in pack.thread_ids


def test_pack_build_is_deterministic_for_path_and_dict_inputs() -> None:
    import yaml

    seed_dict = yaml.safe_load(SEED_PATH.read_text(encoding="utf-8"))

    from_path = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)
    from_dict = build_context_pack(seed_dict, PRESSURE_MESSAGE)
    again = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)

    assert from_path == from_dict == again


def test_pack_records_matched_pressure_words_over_topic_labels() -> None:
    pack = build_context_pack(SEED_PATH, PRESSURE_MESSAGE)
    matched = {word.lower() for word in pack.metadata["pressure_words_matched"]}

    assert {"unresolved", "coupling", "contradiction", "anomaly", "stuck"} <= matched
