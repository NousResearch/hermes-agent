"""Guards that scan every lane's files run once on every pull request.

The /tmp-literal, Bash-shebang and AGENTS.md-size guards read catalog YAML, docs, TypeScript and
frontend AGENTS.md files as well as Python. Gated on detect's `python` lane, a PR that changed only
such files skipped them and main's push was the first run to report the finding (#152338 put a
literal /tmp in plugin-catalog/memtomem.yaml that way). Here the real classifier's lanes for a
one-file diff are replayed through ci.yaml and the workflows it calls.
"""

from __future__ import annotations

import pytest

from hermes_agent.compat.paths import current_path
from tests.ci import _gha_expr as gha
from tests.ci.test_update_ci_routing import _REPO, _ci_run, _real_classifier

# scripts/check name -> the script it runs; a step may call either.
_TREE_GUARDS = {
    "no-tmp": "scripts/check_no_tmp_literals.py",
    "shebangs": "scripts/check_bash_shebangs.py",
    "agents-md": "scripts/ci/check_agents_md_size.py",
}
_DIFFS = (
    "plugin-catalog/memtomem.yaml",
    "apps/desktop/AGENTS.md",
    "apps/desktop/src/lib/agent-transfer.ts",
    "website/src/components/RouteLink.tsx",
    "run_agent.py",
)
_STEP_STATUS = {"always": True, "success": True, "failure": False, "cancelled": False}


def _runs(step: dict, guard: str) -> bool:
    text = str(step.get("run", ""))
    return _TREE_GUARDS[guard] in text or f"check --only {guard}" in text


def _times_run(run: dict, guard: str) -> int:
    """Steps that run ``guard`` and whose ``if:`` passes, in every job the replay reached."""
    count, pending = 0, list(run.values())
    while pending:
        node = pending.pop()
        pending.extend((node.get("jobs") or {}).values())
        steps = (node.get("body") or {}).get("steps") or []
        if not steps:
            continue
        ctx = {**node["ctx"], "__status__": _STEP_STATUS,
               "steps": {s["id"]: {"outcome": "success", "conclusion": "success", "outputs": {}}
                         for s in steps if "id" in s}}
        count += sum(1 for s in steps if _runs(s, guard) and gha.condition(s.get("if"), ctx))
    return count


@pytest.mark.parametrize("path", _DIFFS)
def test_a_one_file_pr_runs_every_tree_guard_once(path):
    assert (_REPO / current_path(path)).is_file(), f"{path} is gone: update this table"
    run = _ci_run(_real_classifier([current_path(path)]))
    times = {guard: _times_run(run, guard) for guard in _TREE_GUARDS}
    assert times == dict.fromkeys(_TREE_GUARDS, 1), f"{path}: guard -> steps that ran it: {times}"
