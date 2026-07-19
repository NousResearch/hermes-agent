"""Production-path unit tests for gateway.turn_sidecar helpers."""

from __future__ import annotations

from gateway.turn_sidecar import (
    consume_pending_turn_sidecar_notes,
    join_turn_sidecar_notes,
    set_pending_turn_sidecar_notes,
)


def test_set_then_consume_once():
    store: dict[str, list[str]] = {}
    set_pending_turn_sidecar_notes(store, "sk", ["a", "b"])
    assert consume_pending_turn_sidecar_notes(store, "sk") == ["a", "b"]
    assert consume_pending_turn_sidecar_notes(store, "sk") == []
    assert "sk" not in store


def test_empty_inputs_are_noops():
    store: dict[str, list[str]] = {}
    set_pending_turn_sidecar_notes(store, "", ["x"])
    set_pending_turn_sidecar_notes(store, "sk", [])
    assert store == {}
    assert consume_pending_turn_sidecar_notes(None, "sk") == []
    assert consume_pending_turn_sidecar_notes(store, "") == []


def test_join_skips_empty_segments():
    assert join_turn_sidecar_notes(["a", "", "b"]) == "a\n\nb"
    assert join_turn_sidecar_notes([]) == ""
