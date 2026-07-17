# SPDX-License-Identifier: Apache-2.0
"""Tests for the self-evolution ledger.

Focus on the actual invariants:

* Actionable filtering via tags / event_type / mem_id regex.
* Round-trip through the config store preserves state shape.
* Recent set is bounded (MAX_RECENT), duplicates collapse, newest-
  first ordering survives updates.
* Prompt formatter obeys the age cutoff and MAX_RECENT bounds AND
  returns empty when the ledger is disabled or empty (the caller
  relies on that to skip injection entirely).
* Disabled ledger is a strict no-op for ingestion.
* Bad memory lookup does not block ingestion (best-effort enrichment).
"""
from __future__ import annotations

import json
import time
from typing import Any

from agent.self_evolution import (
    MAX_RECENT,
    PROMPT_MAX_AGE_MS,
    STATE_KEY,
    SelfEvolutionState,
    format_self_evolution_for_prompt,
    get_self_evolution_snapshot,
    get_self_evolution_state,
    is_self_evolution_memory,
    record_self_evolution_from_memories,
    reset_self_evolution_state,
)


# ── Fake stores ─────────────────────────────────────────────────────


class DictConfigStore:
    def __init__(self, initial: dict[str, Any] | None = None):
        self._data: dict[str, Any] = dict(initial or {})

    def get_config(self, key: str) -> Any:
        return self._data.get(key)

    def set_config(self, key: str, value: str) -> None:
        self._data[key] = value

    def raw(self) -> dict[str, Any]:
        return dict(self._data)


class MemoryLookupMap:
    def __init__(self, mapping: dict[str, dict[str, Any]] | None = None):
        self._map = dict(mapping or {})
        self.calls: list[str] = []

    def get_memory_by_mem_id(self, mem_id: str):
        self.calls.append(mem_id)
        return self._map.get(mem_id)


class FailingLookup:
    def get_memory_by_mem_id(self, mem_id: str):  # noqa: ARG002
        raise RuntimeError("db down")


# ── is_self_evolution_memory ────────────────────────────────────────


def test_is_self_evolution_memory_by_tag() -> None:
    assert is_self_evolution_memory({"mem_id": "x", "tags": ["kind:procedure"]})
    assert is_self_evolution_memory(
        {"mem_id": "x", "tags": json.dumps(["kind:policy"])}
    )


def test_is_self_evolution_memory_by_event_type() -> None:
    assert is_self_evolution_memory(
        {"mem_id": "x", "event_type": "self_constraint"}
    )
    assert is_self_evolution_memory({"mem_id": "x", "type": "self_constraint"})


def test_is_self_evolution_memory_by_mem_id_prefix() -> None:
    for prefix in ("procedure_", "constraint_", "policy_", "lesson_", "rule_"):
        assert is_self_evolution_memory({"mem_id": prefix + "abc"})


def test_is_self_evolution_memory_rejects_plain_fact() -> None:
    assert not is_self_evolution_memory(
        {"mem_id": "fact_1", "tags": ["kind:other"], "event_type": "fact"}
    )


def test_is_self_evolution_memory_rejects_non_mapping() -> None:
    assert not is_self_evolution_memory(None)  # type: ignore[arg-type]
    assert not is_self_evolution_memory("string")  # type: ignore[arg-type]


# ── State round-trip ────────────────────────────────────────────────


def test_get_state_starts_at_defaults() -> None:
    store = DictConfigStore()
    state = get_self_evolution_state(store)
    assert isinstance(state, SelfEvolutionState)
    assert state.enabled is True
    assert state.total_events == 0
    assert state.recent == []


def test_get_state_recovers_from_corrupt_config() -> None:
    store = DictConfigStore({STATE_KEY: "not-json"})
    state = get_self_evolution_state(store)
    assert state.enabled is True
    assert state.recent == []


def test_get_state_respects_disabled_flag_across_roundtrip() -> None:
    payload = {"enabled": False, "total_events": 5, "recent": []}
    store = DictConfigStore({STATE_KEY: json.dumps(payload)})
    state = get_self_evolution_state(store)
    assert state.enabled is False
    assert state.total_events == 5


def test_reset_wipes_state() -> None:
    store = DictConfigStore(
        {
            STATE_KEY: json.dumps(
                {"enabled": True, "total_events": 99, "recent": [{"mem_id": "x"}]}
            )
        }
    )
    reset = reset_self_evolution_state(store)
    assert reset.total_events == 0
    assert reset.recent == []


# ── Ingestion ───────────────────────────────────────────────────────


def test_record_ingests_actionable_and_skips_plain_facts() -> None:
    store = DictConfigStore()
    learned = record_self_evolution_from_memories(
        store,
        [
            {"mem_id": "procedure_deploy", "tags": ["kind:procedure"], "title": "deploy"},
            {"mem_id": "fact_1", "event_type": "fact", "title": "not tracked"},
            {"mem_id": "policy_ratelimit", "tags": ["kind:policy"]},
        ],
    )
    ids = [e["mem_id"] for e in learned]
    assert ids == ["procedure_deploy", "policy_ratelimit"]
    state = get_self_evolution_state(store)
    assert state.total_events == 2
    assert state.learned_count == 2


def test_record_deduplicates_by_mem_id_within_batch() -> None:
    store = DictConfigStore()
    learned = record_self_evolution_from_memories(
        store,
        [
            {"mem_id": "policy_x", "tags": ["kind:policy"]},
            {"mem_id": "policy_x", "tags": ["kind:policy"]},
            {"mem_id": "policy_y", "tags": ["kind:policy"]},
        ],
    )
    assert [e["mem_id"] for e in learned] == ["policy_x", "policy_y"]


def test_record_returns_empty_when_disabled() -> None:
    store = DictConfigStore(
        {STATE_KEY: json.dumps({"enabled": False, "recent": []})}
    )
    learned = record_self_evolution_from_memories(
        store, [{"mem_id": "policy_x", "tags": ["kind:policy"]}]
    )
    assert learned == []
    # Also does NOT flip the ledger back on as a side effect.
    state = get_self_evolution_state(store)
    assert state.enabled is False


def test_record_returns_empty_for_empty_batch() -> None:
    store = DictConfigStore()
    assert record_self_evolution_from_memories(store, []) == []


def test_record_bounds_recent_list_to_max() -> None:
    store = DictConfigStore()
    batch = [
        {"mem_id": f"policy_{i}", "tags": ["kind:policy"], "title": str(i)}
        for i in range(MAX_RECENT + 5)
    ]
    record_self_evolution_from_memories(store, batch)
    state = get_self_evolution_state(store)
    assert len(state.recent) == MAX_RECENT


def test_record_survives_broken_memory_lookup() -> None:
    """Best-effort enrichment: if the DB lookup blows up, ingestion
    still uses the caller-provided row and moves on.
    """
    store = DictConfigStore()
    learned = record_self_evolution_from_memories(
        store,
        [{"mem_id": "policy_x", "tags": ["kind:policy"], "title": "raw"}],
        memory_lookup=FailingLookup(),
    )
    assert len(learned) == 1
    assert learned[0]["title"] == "raw"


def test_record_uses_lookup_to_enrich_thin_reference() -> None:
    """If caller only provides an id-shaped reference, the lookup
    should be consulted for the full row.
    """
    lookup = MemoryLookupMap(
        {
            "procedure_ship": {
                "mem_id": "procedure_ship",
                "tags": ["kind:procedure"],
                "title": "Ship steps",
                "content": "1. build 2. test 3. push",
            }
        }
    )
    store = DictConfigStore()
    learned = record_self_evolution_from_memories(
        store,
        [{"mem_id": "procedure_ship"}],
        memory_lookup=lookup,
    )
    assert lookup.calls == ["procedure_ship"]
    assert len(learned) == 1
    assert "build" in learned[0]["content"]


def test_record_emit_event_receives_summary_and_entries() -> None:
    hits: list[tuple[str, dict]] = []
    store = DictConfigStore()
    record_self_evolution_from_memories(
        store,
        [{"mem_id": "policy_x", "tags": ["kind:policy"], "title": "t"}],
        emit_event=lambda name, payload: hits.append((name, payload)),
    )
    assert len(hits) == 1
    name, payload = hits[0]
    assert name == "self_evolution"
    assert payload["count"] == 1
    assert payload["entries"][0]["mem_id"] == "policy_x"
    assert payload["summary"]["total_events"] == 1


def test_record_emit_event_hook_error_does_not_lose_ledger_update() -> None:
    def failing(name, payload):  # noqa: ARG001
        raise RuntimeError("broken hook")

    store = DictConfigStore()
    learned = record_self_evolution_from_memories(
        store,
        [{"mem_id": "policy_x", "tags": ["kind:policy"]}],
        emit_event=failing,
    )
    assert len(learned) == 1
    # State was still updated.
    assert get_self_evolution_state(store).total_events == 1


def test_record_second_batch_promotes_new_entries_ahead_of_old() -> None:
    store = DictConfigStore()
    record_self_evolution_from_memories(
        store, [{"mem_id": "policy_old", "tags": ["kind:policy"]}]
    )
    time.sleep(0.005)
    record_self_evolution_from_memories(
        store, [{"mem_id": "policy_new", "tags": ["kind:policy"]}]
    )
    state = get_self_evolution_state(store)
    # newest-first ordering.
    assert state.recent[0]["mem_id"] == "policy_new"
    assert state.recent[1]["mem_id"] == "policy_old"
    assert state.total_events == 2


# ── Snapshot ────────────────────────────────────────────────────────


def test_snapshot_respects_max_recent_argument() -> None:
    store = DictConfigStore()
    for i in range(10):
        record_self_evolution_from_memories(
            store, [{"mem_id": f"policy_{i}", "tags": ["kind:policy"]}]
        )
    snap = get_self_evolution_snapshot(store, max_recent=3)
    assert len(snap["recent"]) == 3
    assert snap["total_events"] == 10


def test_snapshot_recent_upper_bound_is_max_recent() -> None:
    store = DictConfigStore()
    for i in range(MAX_RECENT + 5):
        record_self_evolution_from_memories(
            store, [{"mem_id": f"policy_{i}", "tags": ["kind:policy"]}]
        )
    snap = get_self_evolution_snapshot(store, max_recent=999)
    assert len(snap["recent"]) == MAX_RECENT


# ── Prompt formatter (with caching warning enforced by empty-when-disabled) ─


def test_format_returns_empty_when_ledger_is_disabled() -> None:
    """This is load-bearing: callers rely on empty-string to mean
    "skip injection entirely". If the formatter ever silently emits
    a placeholder for a disabled ledger, we lose the cache-safety
    escape hatch.
    """
    store = DictConfigStore(
        {STATE_KEY: json.dumps({"enabled": False, "recent": []})}
    )
    assert format_self_evolution_for_prompt(store) == ""


def test_format_returns_empty_when_recent_is_empty() -> None:
    store = DictConfigStore()
    assert format_self_evolution_for_prompt(store) == ""


def test_format_contains_kind_and_mem_id_lines() -> None:
    store = DictConfigStore()
    record_self_evolution_from_memories(
        store,
        [
            {
                "mem_id": "policy_x",
                "tags": ["kind:policy"],
                "title": "Rate limit rule",
                "content": "cap 3 rpm",
            }
        ],
    )
    text = format_self_evolution_for_prompt(store)
    assert "- [policy] policy_x:" in text
    assert "Rate limit rule" in text
    assert "cap 3 rpm" in text
    # Provenance line is always present so the model doesn't confuse
    # the ledger with active per-turn instructions.
    assert "Turn-specific guidance still comes from <active-policies>" in text


def test_format_filters_out_entries_older_than_max_age() -> None:
    """Entries older than max_age_ms are dropped. Simulate by feeding
    an explicit now_ms far in the future.
    """
    store = DictConfigStore()
    record_self_evolution_from_memories(
        store, [{"mem_id": "policy_x", "tags": ["kind:policy"]}]
    )
    # 30 days in the future — entry is older than 7-day cutoff.
    future_ms = (time.time() + 30 * 86400) * 1000
    assert format_self_evolution_for_prompt(store, now_ms=future_ms) == ""


def test_format_clamps_max_recent_to_range() -> None:
    """max_recent is clamped to [1, 8]. Confirm the ceiling."""
    store = DictConfigStore()
    for i in range(12):
        record_self_evolution_from_memories(
            store, [{"mem_id": f"policy_{i}", "tags": ["kind:policy"]}]
        )
    text = format_self_evolution_for_prompt(store, max_recent=999)
    # 8 bullet lines expected.
    bullet_lines = [ln for ln in text.splitlines() if ln.startswith("- [")]
    assert len(bullet_lines) == 8


def test_format_handles_none_learned_at_gracefully() -> None:
    """A recent entry with missing/broken learned_at should not crash
    the age filter — it counts as "in range".
    """
    state_payload = {
        "enabled": True,
        "version": 1,
        "total_events": 1,
        "learned_count": 1,
        "last_at": None,
        "recent": [
            {
                "mem_id": "policy_broken",
                "kind": "policy",
                "action": "observed",
                "title": "no time",
                "content": "",
                "salience": 3,
                "tags": ["kind:policy"],
                "learned_at": None,
            }
        ],
    }
    store = DictConfigStore({STATE_KEY: json.dumps(state_payload)})
    text = format_self_evolution_for_prompt(store)
    assert "policy_broken" in text
