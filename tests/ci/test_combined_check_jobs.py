"""Checks merged into one job keep reporting independently.

Cheap checks that shared a runner, a trigger and a checkout were merged into single jobs
(repo-guards.yml, lint.yml) so each stops paying a full billed minute for seconds of work. What the
separate jobs gave for free is that one red check never stopped another from running. In one job
the default step condition is `success()`, so a step added without a status function silently
stops running after any earlier failure and hides its own result.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
COMBINED = (("repo-guards.yml", "guards"), ("lint.yml", "lint"))
# Steps that must keep the default success(): the clean-worktree check skips after a failed step
# on purpose (that step may have left its own temp files; tests/ci/test_clean_worktree_check.py).
SUCCESS_ONLY = "./.github/actions/check-clean-worktree"


def _steps(filename: str, job: str) -> list[dict]:
    workflow = YAML(typ="base").load((ROOT / ".github/workflows" / filename).read_text(encoding="utf-8"))
    return workflow["jobs"][job]["steps"]


@pytest.mark.parametrize("filename,job", COMBINED)
def test_every_check_runs_whatever_an_earlier_check_did(filename, job):
    steps = _steps(filename, job)
    first_check = next(i for i, step in enumerate(steps) if "run" in step)
    offenders = []
    for step in steps[first_check:]:
        if step.get("uses") == SUCCESS_ONLY:
            continue
        condition = str(step.get("if", ""))
        if "!cancelled()" not in condition and "always()" not in condition:
            offenders.append(step.get("name") or step.get("id") or step.get("uses"))
    assert not offenders, f"{filename}:{job} steps that stop after any earlier failure: {offenders}"
