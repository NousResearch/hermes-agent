"""Curator run hygiene: what a pass costs on disk and who gets to run it.

A weekly pass on a 1.9 GB skills tree (97% curator backups + ledger) held the CLI prompt for
six minutes; two CLIs launched 12 s apart both ran it.
"""

import importlib
import threading
from pathlib import Path

import pytest


@pytest.fixture
def env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import tools.skill_usage as usage
    import agent.curator as curator
    import agent.curator_backup as cb
    for m in (usage, cb, curator):
        importlib.reload(m)
    monkeypatch.setattr(curator, "_load_config", lambda: {})
    monkeypatch.setattr(curator, "_run_llm_review", lambda prompt: "llm-stub")
    yield {"home": home, "curator": curator, "cb": cb}
    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=10.0)


def _snapshots(home: Path):
    d = home / "skills" / ".curator_backups"
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.exists() else []


def test_prune_only_pass_takes_no_snapshot_but_still_ages_old_ones_out(env, monkeypatch):
    cb, curator, home = env["cb"], env["curator"], env["home"]
    (home / "skills" / "alpha").mkdir()
    (home / "skills" / "alpha" / "SKILL.md").write_text("---\nname: alpha\n---\n", encoding="utf-8")
    monkeypatch.setattr(cb, "get_keep", lambda: 5)
    for _ in range(3):
        assert cb.snapshot_skills(reason="old") is not None
    assert len(_snapshots(home)) == 3
    monkeypatch.setattr(cb, "get_keep", lambda: 1)

    curator.run_curator_review(synchronous=True, consolidate=False)
    survivors = _snapshots(home)
    assert len(survivors) == 1, "prune-only pass must apply retention without adding a snapshot"

    curator.run_curator_review(synchronous=True, consolidate=True)
    reasons = [r.get("reason") for r in cb.list_backups()]
    assert "pre-curator-run" in reasons, "consolidation still snapshots first"
    assert len(reasons) <= 2, "and retention still applies (the new snapshot never prunes itself)"


def test_only_one_process_claims_a_due_pass(env, monkeypatch):
    curator, home = env["curator"], env["home"]
    monkeypatch.setattr(curator, "should_run_now", lambda now=None: True)
    started, release = threading.Event(), threading.Event()
    runs = []

    def _slow_review(**kw):
        runs.append(kw)
        started.set()
        release.wait(10)
        return {}

    monkeypatch.setattr(curator, "run_curator_review", _slow_review)
    holder = threading.Thread(target=curator.maybe_run_curator, daemon=True)
    holder.start()
    assert started.wait(5)
    try:
        assert curator.maybe_run_curator() is None, "a second launch must not run the pass concurrently"
        assert len(runs) == 1
    finally:
        release.set()
        holder.join(5)
    assert not curator._run_claim_path().exists(), "claim released after the pass"
    assert curator.maybe_run_curator() is not None, "and the next due pass can claim again"
