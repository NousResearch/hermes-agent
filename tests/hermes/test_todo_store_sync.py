"""Tests for services.hermes.todo_store_sync — the morning-chain glue that
wires services.hermes.todo_store into the scheduled path (steps 1 and 3 of
deploy/bin/morning-chain.sh).

Covers export_open_closed_keys (snooze-expiry side effect + JSON export +
error handling), ingest_contract (todos[] reconciliation + resolved_keys[]
closure + malformed-row handling), and the main() CLI (argv parsing,
--dry-run no-mutation guarantee, exit codes).

All fixture data below is 100% synthetic/fake placeholder text (this repo is
public) — never real personal data.

HERMES_HOME is redirected to a per-test tmp_path (same explicit-monkeypatch
pattern used elsewhere in tests/hermes/*.py, e.g. test_todo_store.py /
test_todo_store_seed.py).
"""
from __future__ import annotations

import datetime as _dt
import json

import pytest

from services.hermes import todo_store
from services.hermes import todo_store_sync as sync


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _item(key, text="do the fake-widget-42 review", priority="medium", **kw):
    item = {"key": key, "text": text, "priority": priority}
    item.update(kw)
    return item


def _write_contract(path, contract: dict) -> str:
    p = path / "journal_brief.latest.json"
    p.write_text(json.dumps(contract), encoding="utf-8")
    return str(p)


def _all_rows(tmp_path) -> list[dict]:
    conn = todo_store.connect(tmp_path / "todos.db")
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM todos ORDER BY key").fetchall()]
    finally:
        conn.close()


def _row(tmp_path, key) -> dict | None:
    conn = todo_store.connect(tmp_path / "todos.db")
    try:
        row = conn.execute("SELECT * FROM todos WHERE key=?", (key,)).fetchone()
    finally:
        conn.close()
    return dict(row) if row is not None else None


def _audit_lines(tmp_path) -> list[dict]:
    log_path = tmp_path / "todo-audit.log"
    if not log_path.exists():
        return []
    return [json.loads(l) for l in log_path.read_text(encoding="utf-8").strip().split("\n") if l]


# ---------------------------------------------------------------------------
# export_open_closed_keys
# ---------------------------------------------------------------------------


class TestExportOpenClosedKeys:
    def test_writes_correct_json_for_mix_of_open_snoozed_done(self, tmp_path):
        todo_store.upsert_from_contract(
            [
                _item("fake-open-1", text="fake open task one"),
                _item("fake-open-2", text="fake open task two"),
            ],
            "2026-07-14",
        )
        # snoozed, NOT expired -> must not show up as open, and not closed either
        todo_store.upsert_from_contract([_item("fake-snoozed-future", text="snoozed future")], "2026-07-14")
        todo_store.close_todo("fake-snoozed-future", "snooze", "test", snooze_until="2099-01-01")
        # done -> closed bucket
        todo_store.upsert_from_contract([_item("fake-done-1", text="fake done task")], "2026-07-14")
        todo_store.close_todo("fake-done-1", "done", "test")

        open_path = tmp_path / "out" / "todo-open-keys.json"
        closed_path = tmp_path / "out" / "todo-closed-keys.json"
        result = sync.export_open_closed_keys(str(open_path), str(closed_path))

        assert result == {"open_count": 2, "closed_count": 1}

        open_written = json.loads(open_path.read_text(encoding="utf-8"))
        closed_written = json.loads(closed_path.read_text(encoding="utf-8"))

        assert {d["key"] for d in open_written} == {"fake-open-1", "fake-open-2"}
        assert all(set(d.keys()) == {"key", "text"} for d in open_written)
        assert closed_written == ["fake-done-1"]

        # matches what the store's own read helpers would produce
        assert open_written == todo_store.get_open_keys()
        assert closed_written == todo_store.get_closed_keys()

    def test_reopens_expired_snoozes_before_export(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-expired-snooze", text="past due snooze")], "2026-07-01")
        todo_store.close_todo("fake-expired-snooze", "snooze", "test", snooze_until="2026-07-10")
        before = _row(tmp_path, "fake-expired-snooze")
        assert before["status"] == "snoozed"

        open_path = tmp_path / "todo-open-keys.json"
        closed_path = tmp_path / "todo-closed-keys.json"

        # sanity: past-due relative to "today" as the module computes it
        assert "2026-07-10" <= sync._today()

        result = sync.export_open_closed_keys(str(open_path), str(closed_path))

        after = _row(tmp_path, "fake-expired-snooze")
        assert after["status"] == "open"
        assert after["snoozed_until"] is None

        open_written = json.loads(open_path.read_text(encoding="utf-8"))
        assert "fake-expired-snooze" in {d["key"] for d in open_written}
        assert result["open_count"] == len(open_written)

    def test_returns_correct_counts(self, tmp_path):
        todo_store.upsert_from_contract(
            [_item(f"fake-key-{i}", text=f"fake task {i}") for i in range(3)], "2026-07-14"
        )
        todo_store.close_todo("fake-key-0", "done", "test")
        todo_store.close_todo("fake-key-1", "dismiss", "test")

        open_path = tmp_path / "open.json"
        closed_path = tmp_path / "closed.json"
        result = sync.export_open_closed_keys(str(open_path), str(closed_path))

        assert result["open_count"] == 1  # only fake-key-2 remains open
        assert result["closed_count"] == 2

    def test_unwritable_path_returns_error_dict_not_raise(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-key-1")], "2026-07-14")

        # Pass an existing directory as the target "file" path: the atomic
        # writer's os.replace(tmp, path) onto a directory raises OSError,
        # which export_open_closed_keys must catch rather than propagate.
        bad_open_path = tmp_path  # a directory, not a file
        closed_path = tmp_path / "closed.json"

        result = sync.export_open_closed_keys(str(bad_open_path), str(closed_path))

        assert "error" in result
        assert isinstance(result["error"], str) and result["error"]

    def test_missing_parent_path_created_or_errors_gracefully(self, tmp_path):
        """A path under a location that cannot be created must error, not raise."""
        # Simulate "unwritable" by nesting under a file (not a dir) — mkdir
        # on a path component that is itself a regular file always fails.
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a directory", encoding="utf-8")
        bad_path = blocker / "nested" / "todo-open-keys.json"
        closed_path = tmp_path / "closed.json"

        result = sync.export_open_closed_keys(str(bad_path), str(closed_path))

        assert "error" in result


# ---------------------------------------------------------------------------
# ingest_contract
# ---------------------------------------------------------------------------


class TestIngestContract:
    def test_new_todos_inserted(self, tmp_path):
        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [_item("fake-new-1", text="brand new fake todo")],
            },
        )
        summary = sync.ingest_contract(contract_path)

        assert summary["inserted"] == ["fake-new-1"]
        row = _row(tmp_path, "fake-new-1")
        assert row is not None
        assert row["status"] == "open"
        assert row["text"] == "brand new fake todo"

    def test_todo_matching_existing_open_key_is_refreshed(self, tmp_path):
        todo_store.upsert_from_contract(
            [_item("fake-existing-1", text="old fake text", priority="low")], "2026-07-14"
        )
        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [_item("fake-existing-1", text="refreshed fake text", priority="high")],
            },
        )

        summary = sync.ingest_contract(contract_path)

        assert summary["refreshed"] == ["fake-existing-1"]
        row = _row(tmp_path, "fake-existing-1")
        assert row["text"] == "refreshed fake text"
        assert row["priority"] == "high"
        assert row["last_seen"] == "2026-07-15"

    def test_resolved_keys_entry_closes_matching_open_key(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-resolve-me", text="to be resolved")], "2026-07-14")
        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [],
                "resolved_keys": [{"key": "fake-resolve-me"}],
            },
        )

        summary = sync.ingest_contract(contract_path)

        assert summary["resolved_count"] == 1
        assert summary["resolved_errors"] == []

        row = _row(tmp_path, "fake-resolve-me")
        assert row["status"] == "done"

        audit = _audit_lines(tmp_path)
        matching = [a for a in audit if a["key"] == "fake-resolve-me"]
        assert len(matching) == 1
        assert matching[0]["action"] == "done"
        assert matching[0]["source"] == "journal:resolution"

    def test_resolved_keys_entry_missing_key_is_skipped_not_crash(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-untouched", text="stays open")], "2026-07-14")
        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [],
                "resolved_keys": [{"note": "oops, no 'key' field here"}],
            },
        )

        summary = sync.ingest_contract(contract_path)  # must not raise

        assert summary["resolved_count"] == 0
        assert summary["resolved_errors"] == []
        # the unrelated open todo must be untouched
        row = _row(tmp_path, "fake-untouched")
        assert row["status"] == "open"

    def test_resolved_keys_entry_for_unknown_key_logged_as_resolved_error(self, tmp_path):
        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [],
                "resolved_keys": [{"key": "fake-does-not-exist"}],
            },
        )

        summary = sync.ingest_contract(contract_path)  # must not raise/crash

        assert summary["resolved_count"] == 0
        assert len(summary["resolved_errors"]) == 1
        assert summary["resolved_errors"][0]["key"] == "fake-does-not-exist"
        assert "unknown key" in summary["resolved_errors"][0]["error"]

    def test_missing_contract_file_returns_error_dict(self, tmp_path):
        missing_path = str(tmp_path / "does-not-exist.json")
        summary = sync.ingest_contract(missing_path)
        assert "error" in summary
        assert isinstance(summary["error"], str) and summary["error"]

    def test_malformed_json_returns_error_dict(self, tmp_path):
        bad_path = tmp_path / "journal_brief.latest.json"
        bad_path.write_text("{not valid json !!", encoding="utf-8")
        summary = sync.ingest_contract(str(bad_path))
        assert "error" in summary

    def test_missing_for_date_falls_back_sanely(self, tmp_path):
        contract_path = _write_contract(
            tmp_path,
            {
                # no "for_date" key at all
                "todos": [_item("fake-no-for-date", text="todo with no for_date on contract")],
            },
        )

        summary = sync.ingest_contract(contract_path)  # must not crash

        assert summary["inserted"] == ["fake-no-for-date"]
        row = _row(tmp_path, "fake-no-for-date")
        # falls back to "today" (module's _today()) -- assert it's a
        # real-looking ISO date and matches what the module itself computes.
        assert row["first_seen"] == sync._today()
        assert row["last_seen"] == sync._today()
        # sanity check it's a real parseable date, not a placeholder string
        _dt.date.fromisoformat(row["first_seen"])


# ---------------------------------------------------------------------------
# main() CLI
# ---------------------------------------------------------------------------


class TestMainCli:
    def test_export_subcommand_argv_parsing(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-key-1")], "2026-07-14")
        open_path = tmp_path / "open.json"
        closed_path = tmp_path / "closed.json"

        exit_code = sync.main(
            ["export", "--open-path", str(open_path), "--closed-path", str(closed_path)]
        )

        assert exit_code == 0
        assert open_path.exists()
        assert closed_path.exists()

    def test_ingest_subcommand_argv_parsing_success(self, tmp_path):
        contract_path = _write_contract(
            tmp_path,
            {"for_date": "2026-07-15", "todos": [_item("fake-key-cli")]},
        )

        exit_code = sync.main(["ingest", "--contract", contract_path])

        assert exit_code == 0
        row = _row(tmp_path, "fake-key-cli")
        assert row is not None

    def test_ingest_dry_run_makes_zero_store_mutations(self, tmp_path):
        todo_store.upsert_from_contract([_item("fake-preexisting", text="already here")], "2026-07-14")
        before = _all_rows(tmp_path)
        before_audit = _audit_lines(tmp_path)

        contract_path = _write_contract(
            tmp_path,
            {
                "for_date": "2026-07-15",
                "todos": [
                    _item("fake-preexisting", text="THIS SHOULD NOT BE WRITTEN"),
                    _item("fake-brand-new", text="THIS SHOULD NOT BE INSERTED"),
                ],
                "resolved_keys": [{"key": "fake-preexisting"}],
            },
        )

        exit_code = sync.main(["ingest", "--contract", contract_path, "--dry-run"])

        assert exit_code == 0
        after = _all_rows(tmp_path)
        after_audit = _audit_lines(tmp_path)

        assert after == before, "dry-run must make zero store mutations"
        assert after_audit == before_audit, "dry-run must not touch the audit log"
        # the "new" key must genuinely not exist
        assert _row(tmp_path, "fake-brand-new") is None

    def test_ingest_non_dry_run_unreadable_contract_exits_1(self, tmp_path):
        missing_path = str(tmp_path / "nope.json")
        exit_code = sync.main(["ingest", "--contract", missing_path])
        assert exit_code == 1

    def test_ingest_dry_run_unreadable_contract_also_reports_error_exit(self, tmp_path):
        # Same _load_contract failure path is shared by dry-run and real
        # ingest -- both surface as {"error": ...} -> exit 1, per main()'s
        # "1 if 'error' in result else 0" check.
        missing_path = str(tmp_path / "nope.json")
        exit_code = sync.main(["ingest", "--contract", missing_path, "--dry-run"])
        assert exit_code == 1

    def test_export_subcommand_exits_0_on_success(self, tmp_path):
        open_path = tmp_path / "open.json"
        closed_path = tmp_path / "closed.json"
        assert sync.main(["export", "--open-path", str(open_path), "--closed-path", str(closed_path)]) == 0

    def test_ingest_subcommand_exits_0_on_success_dry_run(self, tmp_path):
        contract_path = _write_contract(tmp_path, {"for_date": "2026-07-15", "todos": []})
        assert sync.main(["ingest", "--contract", contract_path, "--dry-run"]) == 0
