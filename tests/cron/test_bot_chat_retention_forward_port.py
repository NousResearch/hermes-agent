"""Bounded crash-recoverable compatibility receipts."""
import json
import time
from pathlib import Path

import pytest
from cron import bot_chat_legacy as legacy, delivery_queue as queue


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(queue, "DELIVERY_DB", tmp_path / "cron/deliveries.db")
    return tmp_path


def test_sequence_floor_survives_backward_clock_without_scan(store, monkeypatch):
    monkeypatch.setattr(time, "time_ns", lambda: 900)
    a = legacy.defer("a" * 64, {"id": "a"}, "body", store, for_failure=False)
    monkeypatch.setattr(time, "time_ns", lambda: 100)
    monkeypatch.setattr(Path, "glob", lambda *a: pytest.fail("sequence must not scan"))
    b = legacy.defer("b" * 64, {"id": "b"}, "body", store, for_failure=False)
    assert b["sequence"] > a["sequence"]
    assert json.loads((legacy._root() / ".sequence").read_text()) == b["sequence"]


def test_terminal_index_failure_recovers_orphan_without_scan(store, monkeypatch):
    monkeypatch.setattr(legacy, "MAX_TERMINAL_RECEIPTS", 1, raising=False)
    a, b = "a" * 64, "b" * 64
    legacy.defer(a, {"id": "a"}, "body", store, for_failure=False)
    real_write = legacy.atomic_json_write
    failures = []

    def fail_index(path, value, **kw):
        if path.name == ".terminal-index" and not failures:
            failures.append(True)
            raise OSError("injected index commit failure")
        return real_write(path, value, **kw)

    monkeypatch.setattr(legacy, "atomic_json_write", fail_index)
    legacy.drain_legacy_pending()  # missing destination state -> terminal
    assert failures == [True]
    assert "content" not in legacy.read_pending(a)
    # An ordinary producer must discover the orphan by the fixed journal path.
    with monkeypatch.context() as m:
        m.setattr(Path, "glob", lambda *a: pytest.fail("recovery must not scan"))
        legacy.defer(b, {"id": "b"}, "body", store, for_failure=False)
    assert json.loads((legacy._root() / ".terminal-index").read_text()) == [a]
    legacy.drain_legacy_pending()
    assert legacy.read_pending(a) is None
    assert legacy.read_pending(b)["status"] == "ambiguous"


@pytest.mark.parametrize("metadata", [".terminal-index", ".terminal-intent"])
@pytest.mark.parametrize("bad", ["../outside", "nested/outside", "absolute", "A" * 64])
def test_recovery_validates_metadata_before_paths(store, monkeypatch, metadata, bad):
    monkeypatch.setattr(legacy, "MAX_TERMINAL_RECEIPTS", 1, raising=False)
    active, done = "a" * 64, "b" * 64
    legacy.defer(active, {"id": "active"}, "body", store, for_failure=False)
    root = legacy._root()
    if bad == "absolute":
        bad = str(store / "outside")
    outside = root / (bad + ".json")
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text(json.dumps({"status": "settled"}))
    (root / metadata).write_text(json.dumps([bad, active] if metadata.endswith("index") else bad))
    original = Path.read_text

    def guarded(path, *a, **kw):
        assert path.resolve() != outside.resolve(), "unvalidated metadata used as path"
        return original(path, *a, **kw)

    monkeypatch.setattr(Path, "read_text", guarded)
    legacy.defer(done, {"id": "done"}, "body", store, for_failure=False)
    # Terminalize only done; preserve the active evidence even in a corrupt index.
    monkeypatch.setattr(legacy, "_queued", lambda root: [(root / (done + ".json"), legacy.read_pending(done))])
    legacy.drain_legacy_pending()
    assert outside.exists()
    assert legacy.read_pending(active)["status"] == "queued"
    assert json.loads((root / ".terminal-index").read_text()) == [done]
    assert json.loads((root / ".terminal-intent").read_text()) is None
