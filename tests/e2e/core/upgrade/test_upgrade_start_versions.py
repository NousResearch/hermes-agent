"""The upgrade suites start from a real shipped release, of each updater generation on request.

Releases changed tag format from ``v<calver>`` to ``v0.<minor>.<patch>``; a start version chosen by
the old format stayed on the last calver release while users moved to the new line. These checks
run against this checkout's own tags, in the full-history e2e-upgrade job.
"""
from __future__ import annotations

import subprocess

import pytest

from tests.e2e.core.upgrade import _install_helpers as I
from tests.e2e.core.upgrade import _helpers as H


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=H.WORKTREE, capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture(autouse=True)
def _fresh_resolver(monkeypatch):
    monkeypatch.delenv("HERMES_E2E_UPGRADE_BASE", raising=False)
    monkeypatch.delenv("HERMES_E2E_N1_REF", raising=False)
    I.upgrade_base.cache_clear()
    if not I.release_tags(before=True):
        pytest.skip("no release tag reachable before HEAD (fetch tags)")
    yield
    I.upgrade_base.cache_clear()


def test_default_start_is_the_newest_release_before_head():
    base = I.upgrade_base()
    newest = max(I.release_tags(before=True), key=lambda t: int(_git("log", "-1", "--format=%ct", t)))
    assert base is not None and base.ref == newest
    assert "+" not in base.ref and not base.ref.startswith(("rc.", "abandoned-"))


@pytest.mark.parametrize("generation", [name for name, _ in I.UPDATER_GENERATIONS])
def test_a_generation_name_starts_from_the_newest_release_that_ships_it(generation, monkeypatch):
    monkeypatch.setenv("HERMES_E2E_UPGRADE_BASE", generation)
    base = I.upgrade_base()
    assert I.updater_generation(base.ref) == generation
    newer = I.release_tags(before=True)[: I.release_tags(before=True).index(base.ref)]
    assert all(I.updater_generation(t) != generation for t in newer)


def test_an_explicit_ref_and_the_older_variable_both_pin_the_start(monkeypatch):
    oldest = I.release_tags(before=True)[-1]
    monkeypatch.setenv("HERMES_E2E_N1_REF", oldest)
    assert I.upgrade_base().ref == oldest
    I.upgrade_base.cache_clear()
    monkeypatch.setenv("HERMES_E2E_UPGRADE_BASE", "HEAD~1")
    assert I.upgrade_base().commit == _git("rev-parse", "HEAD~1")


def test_a_published_move_renames_the_file_in_the_new_release(tmp_path):
    origin = I.make_origin(tmp_path, _git("rev-parse", "HEAD"))
    target = I.publish_commit(origin, tmp_path, "release: move", {"docs/marker.txt": "x\n"},
                              moves={"gateway/run_background.py": "gateway/run_turn_background.py"})
    status = I.git("diff", "--name-status", "-M", f"{target}~1", target, cwd=origin)
    assert "R100\tgateway/run_background.py\tgateway/run_turn_background.py" in status.splitlines()


def test_a_scenario_generation_wins_over_the_suite_override(monkeypatch):
    monkeypatch.setenv("HERMES_E2E_UPGRADE_BASE", "by-path")
    assert I.updater_generation(I.upgrade_base("handoff").ref) == "handoff"
