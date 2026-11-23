"""Download presentation follows publication, not the unconditional green verdict."""
import json
import shlex

import pytest

from scripts.releases import lanes, stable
from tests.ci.desktop_release_roles import gate
from tests.ci.test_stable_release_graph import workflow


def reachable_renderers(jobs, needs, selected, policy):
    rendered = []
    for name in ("publish-canary", "complete"):
        job = jobs[name]
        if not gate(job["if"], {}, needs):
            continue
        failed = False
        for step in job["steps"]:
            condition = step.get("if", "success()")
            # The evaluator uses bracket indexing for JSON properties.
            condition = condition.replace(".bundles", "['bundles']")
            if not gate(condition, {}, needs, job_if=False, failed=failed):
                continue
            command = shlex.split(step.get("run", ""))
            if command[:5] == ["python", "-m", "scripts.releases.stable", "gate", "admit"]:
                try:
                    stable.require_gate(needs, command[4:], lanes=selected, policy=policy)
                except ValueError:
                    failed = True
            if "render-builds-table.py" in step.get("run", ""):
                rendered.append((name, step))
    return rendered


@pytest.mark.parametrize("policy,red", [("canary", True), ("canary", False), ("stable", False)])
def test_published_candidate_gets_nonblocking_downloads_even_when_suite_is_red(policy, red):
    jobs = workflow("stable-release.yml")["jobs"]
    selected = lanes.CANARY if policy == "canary" else lanes.DEFAULT
    runs = lanes.jobs_to_run(selected, policy)
    needs = {name: {"result": "success" if runs.get(name, True) else "skipped"}
             for name in jobs}
    needs["admit"]["outputs"] = {
        "lanes": json.dumps(lanes.flags(selected)), "jobs": json.dumps(runs),
        "policy": policy, "require-green": "false" if red else "true",
    }
    if red:
        for name in ("ci", "acceptance", "publication"):
            needs[name]["result"] = "failure"
    rendered = reachable_renderers(jobs, needs, selected, policy)
    assert len(rendered) == 1
    job, step = rendered[0]
    assert job == ("publish-canary" if policy == "canary" else "complete")
    assert step["continue-on-error"] == "true"
    assert "--candidate-manifest-sha256" in step["run"]
    assert "--no-channel-page" in step["run"]
    # Failed/cancelled publication never runs presentation under implicit success().
    assert not gate(step.get("if", "success()"), {}, needs, job_if=False, failed=True)
