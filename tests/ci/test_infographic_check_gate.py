"""Guard: every always-on sub-workflow of ci.yaml must be a gate dependency.

The ``all-checks-pass`` job is the only check branch protection requires.
A sub-workflow job that runs unconditionally (no ``if:``) but is missing
from its ``needs`` list is advisory-only -- its failure can never block a
merge, even when the workflow's own docstring says it must enforce a
policy (regression: ``infographic-check`` was silently advisory).
"""
from pathlib import Path

import yaml

CI = Path(__file__).resolve().parents[2] / ".github" / "workflows" / "ci.yaml"


def _ci():
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def test_unconditional_sub_workflows_are_gate_dependencies():
    ci = _ci()
    jobs = ci["jobs"]
    gate_needs = set(jobs["all-checks-pass"]["needs"])
    unconditional = [
        name
        for name, job in jobs.items()
        if name not in {"detect", "all-checks-pass", "ci-timings"}
        and "uses" in job  # sub-workflow calls only
        and "if" not in job
    ]
    assert unconditional, "expected at least one unconditional sub-workflow"
    missing = sorted(set(unconditional) - gate_needs)
    assert not missing, (
        "unconditional sub-workflows missing from all-checks-pass needs: %s" % missing
    )
