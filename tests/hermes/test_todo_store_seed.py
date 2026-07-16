"""Tests for plugins.life_ops.todo_store_seed — the generic one-off seed script.

Covers derive_key's determinism/shape and seed_from_contract's insert/skip
counting, idempotency, error handling, and the "closing a seeded key then
re-seeding must not resurrect it" guarantee.

All fixture "todos" below are 100% synthetic/fake placeholder text (this
repo is public) — never real personal data.

HERMES_HOME is redirected to a per-test tmp_path (same explicit-monkeypatch
pattern used elsewhere in tests/hermes/*.py).
"""
from __future__ import annotations

import json
import re

import pytest

from plugins.life_ops import todo_store
from plugins.life_ops import todo_store_seed as seed


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _write_contract(tmp_path, todos):
    path = tmp_path / "journal_brief.latest.json"
    path.write_text(json.dumps({"todos": todos}), encoding="utf-8")
    return str(path)


def _row_status(tmp_path, key):
    conn = todo_store.connect(tmp_path / "todos.db")
    try:
        row = conn.execute("SELECT status FROM todos WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    return row["status"] if row is not None else None


# ---------------------------------------------------------------------------
# derive_key
# ---------------------------------------------------------------------------


class TestDeriveKey:
    @pytest.mark.parametrize(
        "text",
        [
            "Buy synthetic-widget-42 for the demo",
            "Review fake project Zeta before the fake-standup",
            "Schedule a fake-meeting with the imaginary team",
            "Order more fake-gadget-7 samples",
            "",
        ],
    )
    def test_deterministic_same_input_same_output(self, text):
        assert seed.derive_key(text) == seed.derive_key(text)

    def test_reasonable_shape_hyphenated_lowercase_no_punctuation(self):
        for text in [
            "Buy synthetic-widget-42 for the demo",
            "Review fake project Zeta before the fake-standup!",
            "Order more fake-gadget-7 samples, please.",
        ]:
            key = seed.derive_key(text)
            assert key, f"expected non-empty key for {text!r}"
            assert key == key.lower()
            assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", key), key

    def test_different_texts_generally_produce_different_keys(self):
        a = seed.derive_key("Buy synthetic-widget-42 for the demo")
        b = seed.derive_key("Schedule a fake-meeting with the imaginary team")
        assert a != b

    def test_empty_text_does_not_raise(self):
        assert seed.derive_key("") == ""
        assert seed.derive_key(None) == ""  # defensive: falsy input coerced to ""


# ---------------------------------------------------------------------------
# seed_from_contract
# ---------------------------------------------------------------------------


class TestSeedFromContract:
    def _fixture_todos(self):
        return [
            {"key": "fake-explicit-key-1", "text": "Follow up on fake-project Zeta", "source_dates": ["2026-07-10"]},
            {"text": "Buy synthetic-widget-42 for testing", "source_dates": ["2026-07-11"]},
            {"content": "Order more fake-gadget-7 samples", "source_dates": ["2026-07-12"]},
            {"text": "", "source_dates": []},  # malformed: empty content -> empty derived key
        ]

    def test_insert_and_skip_counts(self, tmp_path):
        path = _write_contract(tmp_path, self._fixture_todos())

        result = seed.seed_from_contract(path, "2026-07-14")

        assert "error" not in result
        assert "fake-explicit-key-1" in result["inserted"]
        assert len(result["inserted"]) == 3  # explicit key + 2 derived-key todos
        assert result["skipped_existing"] == []
        assert result["skipped_empty_key"] == 1  # the empty-content item

    def test_explicit_key_used_verbatim(self, tmp_path):
        path = _write_contract(tmp_path, self._fixture_todos())
        seed.seed_from_contract(path, "2026-07-14")
        assert _row_status(tmp_path, "fake-explicit-key-1") == "open"

    def test_derived_key_used_when_key_absent(self, tmp_path):
        path = _write_contract(tmp_path, self._fixture_todos())
        result = seed.seed_from_contract(path, "2026-07-14")

        derived = seed.derive_key("Buy synthetic-widget-42 for testing")
        assert derived in result["inserted"]
        assert _row_status(tmp_path, derived) == "open"

    def test_inserted_row_fields(self, tmp_path):
        todos = [
            {
                "key": "fake-full-key",
                "text": "Review fake project Zeta docs",
                "category": "fake-cat",
                "priority": "high",
                "source_dates": ["2026-07-08", "2026-07-09"],
                "recurring": True,
                "confidence": 0.9,
            }
        ]
        path = _write_contract(tmp_path, todos)
        seed.seed_from_contract(path, "2026-07-14")

        conn = todo_store.connect(tmp_path / "todos.db")
        try:
            row = conn.execute("SELECT * FROM todos WHERE key='fake-full-key'").fetchone()
        finally:
            conn.close()
        assert row["text"] == "Review fake project Zeta docs"
        assert row["category"] == "fake-cat"
        assert row["priority"] == "high"
        assert json.loads(row["source_dates"]) == ["2026-07-08", "2026-07-09"]
        assert row["recurring"] == 1
        assert row["confidence"] == pytest.approx(0.9)
        assert row["first_seen"] == "2026-07-08"  # min(source_dates)
        assert row["last_seen"] == "2026-07-14"  # for_date
        assert row["status"] == "open"
        assert row["origin"] == "journal"

    def test_defaults_applied_when_fields_missing(self, tmp_path):
        todos = [{"key": "fake-bare-key", "text": "Bare fake todo with no extras"}]
        path = _write_contract(tmp_path, todos)
        seed.seed_from_contract(path, "2026-07-14")

        conn = todo_store.connect(tmp_path / "todos.db")
        try:
            row = conn.execute("SELECT * FROM todos WHERE key='fake-bare-key'").fetchone()
        finally:
            conn.close()
        assert row["priority"] == "medium"
        assert row["category"] is None
        assert row["recurring"] == 0
        assert row["confidence"] == pytest.approx(0.5)
        assert row["first_seen"] == "2026-07-14"  # no source_dates -> for_date fallback

    def test_second_run_is_fully_idempotent(self, tmp_path):
        path = _write_contract(tmp_path, self._fixture_todos())
        first = seed.seed_from_contract(path, "2026-07-14")
        assert len(first["inserted"]) == 3

        second = seed.seed_from_contract(path, "2026-07-15")

        assert second["inserted"] == []
        assert sorted(second["skipped_existing"]) == sorted(first["inserted"])
        assert second["skipped_empty_key"] == 1

    def test_missing_file_returns_error_dict_does_not_raise(self, tmp_path):
        result = seed.seed_from_contract(str(tmp_path / "does-not-exist.json"), "2026-07-14")
        assert result["inserted"] == []
        assert result["skipped_existing"] == []
        assert result["skipped_empty_key"] == 0
        assert "error" in result and result["error"]

    def test_invalid_json_returns_error_dict_does_not_raise(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{not valid json at all", encoding="utf-8")
        result = seed.seed_from_contract(str(path), "2026-07-14")
        assert result["inserted"] == []
        assert result["skipped_existing"] == []
        assert result["skipped_empty_key"] == 0
        assert "error" in result and result["error"]

    def test_missing_todos_list_returns_error_dict_does_not_raise(self, tmp_path):
        path = tmp_path / "no-todos-key.json"
        path.write_text(json.dumps({"something_else": []}), encoding="utf-8")
        result = seed.seed_from_contract(str(path), "2026-07-14")
        assert result["inserted"] == []
        assert "error" in result and result["error"]

    def test_non_dict_item_counts_as_skipped_empty_key(self, tmp_path):
        path = _write_contract(tmp_path, ["not-a-dict-item", 42, None])
        result = seed.seed_from_contract(path, "2026-07-14")
        assert result["inserted"] == []
        assert result["skipped_empty_key"] == 3

    def test_closing_a_seeded_key_then_reseeding_does_not_resurrect_it(self, tmp_path):
        path = _write_contract(tmp_path, self._fixture_todos())
        first = seed.seed_from_contract(path, "2026-07-14")
        key = "fake-explicit-key-1"
        assert key in first["inserted"]

        close_result = todo_store.close_todo(key, "done", "test")
        assert close_result["ok"] is True
        assert _row_status(tmp_path, key) == "done"

        second = seed.seed_from_contract(path, "2026-07-16")

        assert key in second["skipped_existing"]
        assert key not in second["inserted"]
        assert _row_status(tmp_path, key) == "done"  # still closed, not reopened
