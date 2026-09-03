"""Outbound sent-text store unit tests.

Covers the generic outbound sent-text index (``gateway/sent_text_store.py``)
ported into #95687 from @DanBennettUK's PR #96149, so the threaded-reply PR
closes its own residual case: background/cron-sent messages whose reply text
can't be hydrated get anchored to what we actually said.

The store records ``(chat_id, message_id) -> text`` at send time and looks it
up by reply target on inbound. Verified properties: same-chat scoping (no
cross-chat leakage), bounded entry count + per-entry text cap, best-effort
no-ops on empty inputs, and persistence to a JSON file under HERMES_HOME.
"""

from __future__ import annotations

import json

import pytest

import gateway.sent_text_store as sent_text_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    yield tmp_path


def test_record_and_lookup_roundtrip(store):
    sent_text_store.record("+1555", "m-1", "Good morning.")
    assert sent_text_store.lookup("+1555", "m-1") == "Good morning."


def test_lookup_missing_returns_none(store):
    assert sent_text_store.lookup("+1555", "nope") is None


def test_record_truncates_bounded_length(store):
    long = "x" * 5000
    sent_text_store.record("+1555", "m-2", long)
    got = sent_text_store.lookup("+1555", "m-2")
    assert got is not None and len(got) == sent_text_store._MAX_TEXT_CHARS


def test_capacity_trimmed_to_max_entries(store):
    for i in range(sent_text_store._MAX_ENTRIES + 50):
        sent_text_store.record("chat", f"m-{i}", f"text {i}")
    data = json.load(open(store / "state" / "sent_text_index.json"))
    assert len(data) == sent_text_store._MAX_ENTRIES
    # Oldest entries were evicted, newest retained.
    assert sent_text_store.lookup("chat", "m-0") is None
    last = f"m-{sent_text_store._MAX_ENTRIES + 49}"
    assert sent_text_store.lookup("chat", last) is not None


def test_per_chat_scoping(store):
    """Text recorded in one chat must not leak into another."""
    sent_text_store.record("chatA", "shared-id", "secret A")
    assert sent_text_store.lookup("chatB", "shared-id") is None


def test_noop_on_empty_inputs(store):
    sent_text_store.record("", "m", "text")
    sent_text_store.record(None, "m", "text")
    sent_text_store.record("c", "", "text")
    sent_text_store.record("c", None, "text")
    sent_text_store.record("c", "m", "")
    sent_text_store.record("c", "m", None)
    assert sent_text_store.lookup("", "m") is None
    assert sent_text_store.lookup("c", "") is None
