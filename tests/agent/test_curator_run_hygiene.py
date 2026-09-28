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


def _read_state(home: Path) -> dict:
    import json
    p = home / "skills" / ".curator_state"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def test_async_llm_pass_writes_the_owning_profile_not_the_launch_one(env, monkeypatch):
    """The curator-review daemon thread starts with an EMPTY contextvars context, so home
    resolution falls back to the process env — the LAUNCH profile. In a multiplexed
    gateway the Curator tick runs under the served profile's scope and returns long before
    the async pass finishes; the pass then wrote .curator_state / logs/curator into the
    launch profile's home (and, with consolidate on, would review/patch the launch
    profile's skills tree). The gate reproduces that ordering: the scope is gone before
    the thread's first home resolution."""
    import time as _time

    import pytest as _pytest

    curator = env["curator"]
    launch_home = env["home"]
    served_home = launch_home.parent / ".hermes-b"
    (served_home / "skills").mkdir(parents=True)

    gate = threading.Event()
    real_report = curator._safe_curated_report

    def _gated_report():
        gate.wait(10.0)  # hold until the owning scope has exited, as in production
        return real_report()

    monkeypatch.setattr(curator, "_safe_curated_report", _gated_report)

    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    token = set_hermes_home_override(served_home)
    try:
        curator.run_curator_review(synchronous=False, consolidate=False)
    finally:
        reset_hermes_home_override(token)
        gate.set()  # the thread now runs with the owning scope already gone

    # load_state()'s base dict always carries last_run_duration_seconds=None, and the
    # synchronous pre-pass save (run_curator_review persists BEFORE forking the thread)
    # already wrote it — only the async pass can turn it non-None.
    deadline = _time.monotonic() + 10.0
    while _time.monotonic() < deadline:
        if _read_state(served_home).get("last_run_duration_seconds") is not None:
            break
        _time.sleep(0.02)
    else:
        _pytest.fail(
            "async pass never recorded the served profile's state; "
            f"launch={_read_state(launch_home)!r}")

    for t in threading.enumerate():
        if t.name == "curator-review" and t.is_alive():
            t.join(timeout=10.0)

    leaked = _read_state(launch_home)
    assert leaked.get("last_run_duration_seconds") is None, (
        "async LLM pass leaked the launch profile's .curator_state: "
        f"{leaked!r}")


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
