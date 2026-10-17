"""Pytest wrapper for agent-hooks/tests/test-hub-memory-recall.py (PERS-117)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_HOOK_TEST = (
    Path(__file__).resolve().parents[2]
    / "agent-hooks"
    / "tests"
    / "test-hub-memory-recall.py"
)
_spec = importlib.util.spec_from_file_location("hub_memory_recall_pers117", _HOOK_TEST)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

test_overlapping_0_152_score_keeps_signal_query = (
    _mod.test_overlapping_0_152_score_keeps_signal_query
)
test_overlapping_0_152_score_ignores_nonsense_query = (
    _mod.test_overlapping_0_152_score_ignores_nonsense_query
)
test_below_floor_score_is_dropped_even_with_word_overlap = (
    _mod.test_below_floor_score_is_dropped_even_with_word_overlap
)
test_empty_prompt_does_not_wipe_the_tier = _mod.test_empty_prompt_does_not_wipe_the_tier
