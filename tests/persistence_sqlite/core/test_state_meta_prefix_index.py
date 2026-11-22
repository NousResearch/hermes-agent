"""Idle state-meta probes seek the key index without changing LIKE or stored-order semantics."""

import itertools
import sqlite3

import pytest

from persistence.sqlite._state.common import escape_like, key_prefix_range, key_prefix_ranges_casefold
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_holder import raw_holder


@pytest.mark.parametrize("prefix", ["heartbeat:", "loop:"])
def test_idle_prefix_probe_seeks_key_index(tmp_path, monkeypatch, prefix):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.state_meta.set(prefix + "z", "1")
        holder = raw_holder(store)
        read_all = holder._read_all
        captured = []

        def capture(sql, params=()):
            captured.append((sql, params))
            return read_all(sql, params)

        monkeypatch.setattr(holder, "_read_all", capture)
        assert holder.list_meta_prefix(prefix) == [(prefix + "z", "1")]
        (sql, params), = captured
        with holder._read_ctx() as conn:
            plan = "\n".join(row[3] for row in conn.execute("EXPLAIN QUERY PLAN " + sql, params))
        assert "SEARCH state_meta USING INDEX" in plan
        assert "SCAN state_meta" not in plan


def test_holder_prefix_matches_are_literal_casefolded_and_in_stored_order(tmp_path):
    entries = [
        ("loop:z", "old"), ("LOOP:b", "2"), ("Loop:a", "3"), ("loopx", "4"),
        ("loop_x", "5"), ("loop%z", "6"), ("LOOP_y", "7"), ("LOOP%a", "8"),
        ("loop\\z", "9"), ("lööp:x", "10"), ("LÖÖP:y", "11"),
    ]
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for key, value in entries:
            store.state_meta.set(key, value)
        store.state_meta.set("loop:z", "updated")
        holder = raw_holder(store)
        assert holder.list_meta_prefix("loop:") == [("loop:z", "updated"), *entries[1:3]]
        assert holder.list_meta_prefix("loop_") == [entries[4], entries[6]]
        assert holder.list_meta_prefix("loop%") == [entries[5], entries[7]]
        assert holder.list_meta_prefix("loop\\") == [entries[8]]
        assert holder.list_meta_prefix("lööp:") == [entries[9]]


@pytest.mark.parametrize("prefix", ["\U0010ffff", "x\U0010ffff", "\U0010ffffx", "\0x", "loop\0x"])
def test_unbounded_prefix_edges_preserve_the_old_like_filter(tmp_path, prefix):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for key in ["loop:z", "LOOP\0other", "x\U0010ffffz", "\U0010ffffx", "\U0010ffffz"]:
            store.state_meta.set(key, key)
        holder = raw_holder(store)
        expected = holder._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'", (escape_like(prefix) + "%",))
        assert holder.list_meta_prefix(prefix) == [(row[0], row[1]) for row in expected]


def test_key_prefix_range_bounds_and_unicode_successors():
    low, high = key_prefix_range("heartbeat:")
    assert (low, high) == ("heartbeat:", "heartbeat;")
    assert key_prefix_range("é") == ("é", "ê")
    for key in ["heartbeat:", "heartbeat:\U0010ffff"]:
        assert low <= key < high
    for key in ["heartbeat", "heartbeat;"]:
        assert not low <= key < high
    for prefix in ["", "x\U0010ffff"]:
        with pytest.raises(ValueError):
            key_prefix_range(prefix)
    low, high = key_prefix_range("x\ud7ff")
    assert (low, high) == ("x\ud7ff", "x\ue000")
    with sqlite3.connect(":memory:") as conn:
        for key, inside in [(low, True), (low + "\U0010ffff", True), (high, False)]:
            assert bool(conn.execute("SELECT ? >= ? AND ? < ?", (key, low, key, high)).fetchone()[0]) == inside


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 16])
def test_casefold_ranges_cover_ascii_matches_without_exceeding_limit(limit):
    fold = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
    for prefix in ["ab:", "abcdefghijklmnopqrstuvwxyz:", "éAb:", "\U0010ffffa:"]:
        ranges = key_prefix_ranges_casefold(prefix, limit=limit)
        assert len(ranges) <= limit
        keys = [prefix, prefix + "1", "unrelated"]
        keys += ["".join(chars) + prefix[4:] + "1" for chars in itertools.product(*(
            (char.lower(), char.upper()) if char.isascii() and char.isalpha() else (char,)
            for char in prefix[:4]))]
        for key in keys:
            if key.translate(fold).startswith(prefix.translate(fold)):
                assert not ranges or any(low <= key < high for low, high in ranges)
    ranges = key_prefix_ranges_casefold("ab:")
    for key in ["ab:1", "AB:1", "aB:1", "Ab:1"]:
        assert any(low <= key < high for low, high in ranges)
    assert key_prefix_ranges_casefold("\U0010ffff") == []
    assert key_prefix_ranges_casefold("") == []


def test_long_prefix_range_work_is_bounded():
    prefix = "loop" + "0" * 49000
    ranges = key_prefix_ranges_casefold(prefix)
    assert ranges
    for low, high in ranges:
        assert len(low) <= 64
        assert len(high) <= 64
    for key in [prefix, prefix.upper() + "x"]:
        assert any(low <= key < high for low, high in ranges)


def test_long_holder_prefix_retains_full_like_filter(tmp_path):
    prefix = "loop" + "0" * 49000
    with open_persistence(profile_scope(home=tmp_path)) as store:
        store.state_meta.set(prefix + "x", "match")
        store.state_meta.set(prefix[:64] + "1", "not a match")
        holder = raw_holder(store)
        assert holder.list_meta_prefix(prefix) == [(prefix + "x", "match")]


@pytest.mark.parametrize("leading", ["", "loop:"])
@pytest.mark.parametrize("replacement", ["\ufffd", "\ufffe", "\uffff"])
def test_replacement_character_prefix_preserves_old_like(tmp_path, leading, replacement):
    with open_persistence(profile_scope(home=tmp_path)) as store:
        for char in ["\ufffd", "\ufffe", "\uffff"]:
            store.state_meta.set(leading + char + "x", char)
        store.state_meta.set("unrelated", "other")
        holder = raw_holder(store)
        prefix = leading + replacement
        expected = holder._read_all(
            "SELECT key, value FROM state_meta WHERE key LIKE ? ESCAPE '\\'",
            (escape_like(prefix) + "%",),
        )
        assert holder.list_meta_prefix(prefix) == [(row[0], row[1]) for row in expected]
