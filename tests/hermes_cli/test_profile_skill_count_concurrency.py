"""Profile scans share work across dashboard and RPC callers."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import pytest

from hermes_cli import profiles

SKILL_NAME = "SKILL.md"
SKILL_TEXT = "---\nname: example\ndescription: Example skill\n---\n"
WAIT_SECONDS = 5
CONTENTION_SECONDS = 2
CALLERS = 4


@pytest.mark.parametrize("distinct_profiles", [False, True])
def test_skill_scans_are_serialized_and_reuse_completed_counts(tmp_path, monkeypatch, distinct_profiles):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(profiles, "_SKILL_COUNT_CACHE", {})
    roots = [tmp_path / str(i if distinct_profiles else 0) for i in range(CALLERS)]
    cached_root = tmp_path / "cached"
    for root in set(roots) | {cached_root}:
        skill = root / "skills" / "category" / "example" / SKILL_NAME
        skill.parent.mkdir(parents=True)
        skill.write_text(SKILL_TEXT)
    assert profiles._count_skills(cached_root) == 1
    scan_entered = threading.Event()
    overlapping_scan = threading.Event()
    release_scan = threading.Event()
    counter_lock = threading.Lock()
    active = peak = scans = 0
    original_rglob = Path.rglob

    def held_scan(path, pattern):
        nonlocal active, peak, scans
        with counter_lock:
            active += 1
            scans += 1
            peak = max(peak, active)
            if active > 1:
                overlapping_scan.set()
        scan_entered.set()
        try:
            assert release_scan.wait(WAIT_SECONDS)
            yield from original_rglob(path, pattern)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(Path, "rglob", held_scan)
    with ThreadPoolExecutor(max_workers=CALLERS + 1) as executor:
        futures = [executor.submit(profiles._count_skills, roots[0])]
        try:
            assert scan_entered.wait(WAIT_SECONDS)
            futures.extend(executor.submit(profiles._count_skills, root) for root in roots[1:])
            cached = executor.submit(profiles._count_skills, cached_root)
            assert cached.result(timeout=CONTENTION_SECONDS) == 1
            overlapping_scan.wait(CONTENTION_SECONDS)
        finally:
            release_scan.set()
        counts = [future.result(timeout=WAIT_SECONDS) for future in futures]
    assert counts == [1] * CALLERS
    assert peak == 1
    assert scans == len(set(roots))


def test_slow_scan_result_is_fresh_when_published(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(profiles, "_SKILL_COUNT_CACHE", {})
    root = tmp_path / "profile"
    skill = root / "skills" / "example" / SKILL_NAME
    skill.parent.mkdir(parents=True)
    skill.write_text(SKILL_TEXT)
    clock = [100.0]
    scans = 0
    original_rglob = Path.rglob

    def slow_scan(path, pattern):
        nonlocal scans
        scans += 1
        clock[0] += profiles._SKILL_COUNT_TTL_SECONDS + 1
        yield from original_rglob(path, pattern)

    monkeypatch.setattr(profiles.time, "time", lambda: clock[0])
    monkeypatch.setattr(Path, "rglob", slow_scan)
    assert profiles._count_skills(root) == 1
    assert profiles._count_skills(root) == 1
    assert scans == 1
    clock[0] += profiles._SKILL_COUNT_TTL_SECONDS + 1
    assert profiles._count_skills(root) == 1
    assert scans == 2
