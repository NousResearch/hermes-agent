"""Which ci.yaml runs a newer run on the same group may cancel.

The real ``concurrency`` block is evaluated with the repo's Actions-expression
evaluator for each way the workflow starts. A re-dispatch on a fix/debug branch
replaces the stale run instead of queuing behind it; main and stable-release
runs are never cancelled.
"""
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from tests.ci import _gha_expr as gha

CI = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yaml"


def _cancels(event: str, ref: str, release: bool = False) -> bool:
    block = YAML(typ="base").load(CI.read_text(encoding="utf-8"))["concurrency"]
    ctx = {"github": {"event_name": event, "ref": ref, "run_id": "1",
                      "event": {"pull_request": {"number": "7"}}},
           "inputs": {"release": release}}
    return gha.truthy(gha.render(block["cancel-in-progress"], ctx))


@pytest.mark.parametrize(("event", "ref", "release", "cancels"), [
    ("pull_request", "refs/pull/7/merge", False, True),
    ("workflow_dispatch", "refs/heads/fix/some-branch", False, True),
    ("workflow_dispatch", "refs/heads/main", False, False),
    ("push", "refs/heads/main", False, False),
    ("workflow_call", "refs/heads/main", True, False),
    ("workflow_dispatch", "refs/heads/fix/some-branch", True, False),
    ("workflow_dispatch", "refs/tags/v0.23.1", False, False),
])
def test_only_pr_and_side_branch_dispatch_runs_are_superseded(event, ref, release, cancels):
    assert _cancels(event, ref, release) is cancels
