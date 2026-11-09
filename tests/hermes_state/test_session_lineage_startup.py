"""Startup inventory patience and fail-closed contracts for cross-profile writes."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
import hermes_state_lineage as lineage


def test_append_waits_for_sibling_constructor_schema(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    db.create_session("current", "cli")
    sibling = root / "profiles" / "starting" / "state.db"
    sibling.parent.mkdir(parents=True)
    schema_entered, release_schema, initialized = (
        threading.Event(), threading.Event(), threading.Event()
    )
    original_init = SessionDB._init_schema

    def pause_schema(self):
        if self.db_path == sibling:
            schema_entered.set()
            assert release_schema.wait(10), "constructor was not released"
        original_init(self)

    def open_sibling():
        other = SessionDB(sibling)
        initialized.set()
        return other

    polls = []

    def resume_startup(delay):
        polls.append(delay)
        release_schema.set()
        assert initialized.wait(10), "sibling could not finish while inventory waited"

    monkeypatch.setattr(SessionDB, "_init_schema", pause_schema)
    other = None
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(open_sibling)
            try:
                assert schema_entered.wait(10), "sibling did not reach schema initialization"
                assert sibling.exists()
                monkeypatch.setattr(
                    lineage, "time",
                    SimpleNamespace(monotonic=time.monotonic, sleep=resume_startup),
                )
                db.append_message("current", "user", "saved through startup")
                assert polls, "inventory skipped the unfinished sibling"
            finally:
                release_schema.set()
                other = future.result(timeout=10)
        assert [message["content"] for message in db.get_messages("current")] == [
            "saved through startup"
        ]
        assert other.get_session("current") is None
    finally:
        if other is not None:
            other.close()
        db.close()


@pytest.mark.parametrize("incomplete", [False, True])
def test_unfinished_inventory_exhausts_shared_deadline_without_writes(
    tmp_path, monkeypatch, incomplete,
):
    root = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(root))
    db = SessionDB(root / "state.db")
    db.create_session("current", "cli")
    sibling = root / "profiles" / "unfinished" / "state.db"
    sibling.parent.mkdir(parents=True)
    with sqlite3.connect(sibling) as conn:
        conn.execute(
            "CREATE TABLE sessions (id TEXT)" if incomplete
            else "CREATE TABLE unrelated (id TEXT)"
        )
    before = sibling.read_bytes()
    clock, waits = [10.0], []

    def advance(delay):
        waits.append(delay)
        clock[0] += delay

    monkeypatch.setattr(
        lineage, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=advance),
    )
    reason = "incomplete sessions schema" if incomplete else "lacks sessions table"
    try:
        with lineage._lineage_inventory_deadline(10.125):
            with pytest.raises(lineage.SessionLineageInventoryError, match=reason):
                db.append_message("current", "user", "must not be saved")
        assert waits and sum(waits) == pytest.approx(0.125)
        assert clock[0] == pytest.approx(10.125)
        assert db.get_messages("current") == []
        assert sibling.read_bytes() == before
    finally:
        db.close()
