"""`pm gc --dry-run` (and the doctor's plan) report every class the sweep would reclaim without
touching the disk or rewriting facts.json; the executed sweep still removes exactly those classes."""
from __future__ import annotations

import argparse
import json
import os
import time

from pm import paths
from pm.cli import _gc_store_plan
from pm.download_state import GC_GRACE_SECONDS
from pm.lock import Facts
from pm.store import Store


def _store(tmp_path):
    """A store holding one of every candidate class plus what must survive the sweep."""
    root = tmp_path / "tools"
    root.mkdir()
    (root / "python-1" / "bin").mkdir(parents=True)
    (root / "python-1" / "bin" / "py.exe").write_text("x", encoding="utf-8")
    (root / "fetch-deadbeef").mkdir()
    (root / "fetch-deadbeef" / "pkg.zip").write_text("y" * 11, encoding="utf-8")
    (root / ".staging-abcd").mkdir()
    (root / ".reclaim-1234").mkdir()
    (root / ".reclaim-1234" / "tool.dll").write_text("z", encoding="utf-8")
    (root / ".previous-python-0").mkdir()
    (root / ".install.lock").touch()
    facts_path = root / "facts.json"
    facts_path.write_text(json.dumps(
        {"schema": 1, "packages": {"python": {"entry": "python-1", "version": "1"}}}), encoding="utf-8")
    return Store(root), Facts(facts_path)


def _partials(tmp_path, monkeypatch):
    """One grace-expired partial and one an in-flight download still owns."""
    partials = tmp_path / "partials"
    partials.mkdir()
    old = partials / "old.part"
    old.write_text("p" * 7, encoding="utf-8")
    fresh = partials / "fresh.part"
    fresh.write_text("q", encoding="utf-8")
    past = time.time() - 2 * GC_GRACE_SECONDS
    os.utime(old, (past, past))
    monkeypatch.setattr(paths, "partials_root", lambda: partials)
    return old, fresh


def test_plan_lists_every_reclaimable_class_and_touches_nothing(tmp_path, monkeypatch):
    store, facts = _store(tmp_path)
    _old, fresh = _partials(tmp_path, monkeypatch)
    before = sorted(str(p) for p in store.root.rglob("*"))
    facts_bytes = facts.path.read_bytes()

    candidates, kept = _gc_store_plan(store, facts)

    assert kept == 1
    assert [item["name"] for item in candidates["store_entries"]] == ["fetch-deadbeef"]
    assert [item["name"] for item in candidates["staging"]] == [".staging-abcd"]
    assert [item["name"] for item in candidates["set_aside"]] == [".reclaim-1234"]
    assert [item["name"] for item in candidates["partials"]] == ["old.part"]
    assert candidates["store_entries"][0]["bytes"] == 11
    assert candidates["partials"][0]["bytes"] == 7
    assert sorted(str(p) for p in store.root.rglob("*")) == before
    assert fresh.exists()
    # The projection must not take the sweep's side effects: facts.json keeps its bytes.
    assert facts.path.read_bytes() == facts_bytes


def test_plan_yields_when_the_store_lock_is_held_and_wait_is_false(tmp_path):
    from pm.filesystem import lock_fd

    store, facts = _store(tmp_path)
    fd = os.open(store.root / ".install.lock", os.O_CREAT | os.O_RDWR, 0o600)
    assert lock_fd(fd, wait=False)
    try:
        assert _gc_store_plan(store, facts, lock_wait=False) is None
    finally:
        os.close(fd)
    # The waiting form (pm gc --dry-run) still reports once it owns the lock like the sweep.
    candidates, kept = _gc_store_plan(store, facts)
    assert kept == 1
    assert [item["name"] for item in candidates["store_entries"]] == ["fetch-deadbeef"]


def test_dry_run_reports_without_deleting_then_collect_matches(tmp_path, monkeypatch, capsys):
    import pm.cli

    store, _facts = _store(tmp_path)
    old, fresh = _partials(tmp_path, monkeypatch)
    monkeypatch.setattr(paths, "writable_store_root", lambda: store.root)
    monkeypatch.setattr(pm.cli, "_store", lambda: Store(store.root))
    monkeypatch.setattr(pm.cli, "_facts", lambda: Facts(store.root / "facts.json"))
    monkeypatch.setattr(pm.cli, "repo_root", lambda: tmp_path / "proj")

    assert pm.cli.cmd_gc(argparse.Namespace(dry_run=True, json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry-run"
    assert [item["name"] for item in report["store"]["store_entries"]] == ["fetch-deadbeef"]
    assert report["store"]["kept"] == 1
    assert report["totals"]["count"] == 4  # entry + staging + set-aside + expired partial
    assert report["totals"]["bytes"] == 19  # pkg.zip + tool.dll + old.part
    assert report["dependency_generations"] == []
    assert (store.root / "fetch-deadbeef").is_dir()
    assert (store.root / ".staging-abcd").is_dir()
    assert (store.root / ".reclaim-1234").is_dir()
    assert old.exists() and fresh.exists()

    assert pm.cli.cmd_gc(argparse.Namespace(dry_run=False, json=True)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "collect"
    assert report["store"] == {"removed": 3, "kept": 1}
    assert not (store.root / "fetch-deadbeef").exists()
    assert not (store.root / ".staging-abcd").exists()
    assert not (store.root / ".reclaim-1234").exists()
    assert (store.root / "python-1").is_dir()
    assert (store.root / ".previous-python-0").is_dir()
    assert not old.exists() and fresh.exists()


def test_dry_run_summary_line_names_the_classes_and_the_reclaim_hint(tmp_path, monkeypatch, capsys):
    import pm.cli

    store, _facts = _store(tmp_path)
    _partials(tmp_path, monkeypatch)
    monkeypatch.setattr(paths, "writable_store_root", lambda: store.root)
    monkeypatch.setattr(pm.cli, "_store", lambda: Store(store.root))
    monkeypatch.setattr(pm.cli, "_facts", lambda: Facts(store.root / "facts.json"))
    monkeypatch.setattr(pm.cli, "repo_root", lambda: tmp_path / "proj")

    assert pm.cli.cmd_gc(argparse.Namespace(dry_run=True, json=False)) == 0
    out = capsys.readouterr().out
    assert "gc dry-run: would remove" in out
    assert "1 store entry(s) (11 B)" in out
    assert "1 expired partial(s) (7 B)" in out
    assert "run `hermes pm gc` to reclaim" in out

    assert pm.cli.cmd_gc(argparse.Namespace(dry_run=False, json=False)) == 0
    out = capsys.readouterr().out
    # The sweep prints one `removing <name>` per reclaimed entry before the summary line.
    assert "removing .staging-abcd" in out and "removing fetch-deadbeef" in out
    assert out.rstrip().endswith("gc: removed 3, kept 1; removed 0 dependency generations, "
                                 "0 PM runtime generations, 0 install state dir(s) of deleted checkouts")
