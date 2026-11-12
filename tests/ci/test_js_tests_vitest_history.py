"""The JS job orders each vitest run from scratch, never from a restored cache.

Vitest writes per-file durations to ``node_modules/.vite/vitest`` and runs the
longest files first next time. js-tests.yml caches the installed node_modules
trees, so the run that misses the cache saves its history and every later hit
replays that order: the slowest jsdom suites start cold together and push light
files past their 15s timeout (main went red from run 41717106556 on).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/js-tests.yml"
HISTORY = ".vite/vitest/da39a3ee5e6b4b0d3255bfef95601890afd80709/results.json"


def _steps():
    return YAML(typ="base").load(WORKFLOW.read_text(encoding="utf-8"))["jobs"]["check"]["steps"]


def _cached_trees(steps) -> list[str]:
    restore = next(step for step in steps if step.get("id") == "node-modules-cache")
    return [line.strip() for line in restore["with"]["path"].splitlines() if line.strip()]


def _concrete(pattern: str) -> str:
    return pattern.replace("*", "pkg")


def test_restored_trees_lose_their_vitest_history_before_any_check(tmp_path):
    steps = _steps()
    names = [step.get("name") for step in steps]
    drop = names.index("Drop vitest run history from the restored tree")
    assert names.index("Restore node_modules") < drop < names.index("Run all workspace checks")

    trees = [_concrete(pattern) for pattern in _cached_trees(steps)]
    for tree in trees:
        for rel in (HISTORY, "left-pad/package.json", ".vite/deps/_metadata.json"):
            path = tmp_path / tree / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    subprocess.run(["bash", "-e", "-o", "pipefail", "-c", steps[drop]["run"]], cwd=tmp_path, check=True)

    for tree in trees:
        assert not (tmp_path / tree / ".vite/vitest").exists(), tree
        # Installed packages and other Vite caches are what the cache is for.
        assert (tmp_path / tree / "left-pad/package.json").is_file(), tree
        assert (tmp_path / tree / ".vite/deps/_metadata.json").is_file(), tree
