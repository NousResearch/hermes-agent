"""Tests for gating the MEMORY.md/USER.md sentence in compaction notes.

When the built-in flat-file memory store is not loaded (``memory_enabled``
and ``user_profile_enabled`` both false — data-layer / external memory
providers), compaction handoffs must not tell the model that its
persistent memory lives in MEMORY.md/USER.md, or agents go looking for
markdown memory files that don't exist.
"""

from __future__ import annotations

import pytest

from agent.context_compressor import (
    SUMMARY_PREFIX,
    _MEMORY_AUTHORITY_SENTENCE,
    _SUMMARY_PREFIX_NO_MEMORY,
    _summary_prefix_for,
    ContextCompressor,
)


def test_summary_prefix_still_contains_memory_sentence_by_default():
    # The refactor must keep the live default prefix byte-identical.
    assert _MEMORY_AUTHORITY_SENTENCE in SUMMARY_PREFIX
    assert "MEMORY.md" in SUMMARY_PREFIX


def test_summary_prefix_for_false_omits_markdown_memory():
    no_memory = _summary_prefix_for(False)
    assert "MEMORY.md" not in no_memory
    assert "USER.md" not in no_memory
    # Still a recognizable compaction handoff prefix, sharing the intro with
    # the full variant (the sentence is removed from the middle).
    assert no_memory.startswith("[CONTEXT COMPACTION — REFERENCE ONLY]")
    intro = SUMMARY_PREFIX.split(_MEMORY_AUTHORITY_SENTENCE)[0]
    assert no_memory.startswith(intro)
    assert _MEMORY_AUTHORITY_SENTENCE not in no_memory


def test_with_summary_prefix_gates_memory_sentence():
    with_memory = ContextCompressor._with_summary_prefix("body")
    assert "MEMORY.md" in with_memory

    without_memory = ContextCompressor._with_summary_prefix(
        "body", flat_memory_active=False
    )
    assert "MEMORY.md" not in without_memory
    assert without_memory.startswith(_SUMMARY_PREFIX_NO_MEMORY)


def test_micro_marker_gates_memory_sentence():
    with_memory = ContextCompressor._render_micro_marker_content("sum")
    assert "MEMORY.md" in with_memory

    without_memory = ContextCompressor._render_micro_marker_content(
        "sum", flat_memory_active=False
    )
    assert "MEMORY.md" not in without_memory


def test_no_memory_variant_still_recognized_and_stripped():
    # Rehydration must recognize + strip handoffs generated with the
    # no-memory prefix, otherwise they leak on re-compaction.
    assert ContextCompressor._starts_with_summary_prefix(_SUMMARY_PREFIX_NO_MEMORY)
    stripped = ContextCompressor._strip_summary_prefix(
        _SUMMARY_PREFIX_NO_MEMORY + "\nbody"
    )
    assert stripped == "body"


def test_compressor_instance_flag_drives_prefix():
    c = ContextCompressor(model="test", quiet_mode=True, flat_memory_active=False)
    out = c._with_summary_prefix("body", flat_memory_active=c._flat_memory_active)
    assert "MEMORY.md" not in out

    c2 = ContextCompressor(model="test", quiet_mode=True)
    out2 = c2._with_summary_prefix("body", flat_memory_active=c2._flat_memory_active)
    assert "MEMORY.md" in out2
