"""A job reads another job's result only through a `needs` entry it declares.

GitHub rejects a job graph that references an unknown job or needs itself, so
cycles are the platform's own validation. What it does not fail closed on is a
result read from a job that was never declared: `needs.build-win32-release.result`
resolves to nothing when the result job's `needs` list only names itself, and the
dispatch burns its whole fan-out before anything reports a problem.
"""
import re
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github/workflows"

EXPRESSION = re.compile(r"\$\{\{(.*?)\}\}", re.S)
JOB_REFERENCE = re.compile(r"\bneeds\.([A-Za-z0-9_-]+)")


def _loaded():
    yaml = YAML(typ="base")
    return {
        path.name: yaml.load(path.read_text(encoding="utf-8"))
        for path in sorted(WORKFLOWS.glob("*.y*ml"))
    }


def _referenced_jobs(node, key=None):
    """Job names this job reads through `needs.` — the `if:` and every expression."""
    if isinstance(node, dict):
        for sub_key, value in node.items():
            yield from _referenced_jobs(value, str(sub_key))
    elif isinstance(node, list):
        for value in node:
            yield from _referenced_jobs(value, key)
    elif isinstance(node, str):
        # `run:` scripts may contain their own `needs.items()` — expressions only.
        yield from JOB_REFERENCE.findall(node) if key == "if" else ()
        for expression in EXPRESSION.finditer(node):
            yield from JOB_REFERENCE.findall(expression.group(1))


def test_jobs_only_read_declared_dependencies():
    violations = []
    for filename, workflow in _loaded().items():
        for name, job in ((workflow or {}).get("jobs") or {}).items():
            if not isinstance(job, dict):
                continue
            declared = job.get("needs") or []
            if isinstance(declared, str):
                declared = [declared]
            for read in sorted(set(_referenced_jobs(job))):
                if read not in declared:
                    violations.append(f"{filename}: {name} reads '{read}' without needing it")
    assert not violations, "workflow job graph:\n" + "\n".join(violations)


def test_no_workflow_runs_from_a_tag_push():
    violations = []
    for filename, workflow in _loaded().items():
        triggers = (workflow or {}).get("on") or {}
        if not isinstance(triggers, dict) or "push" not in triggers:
            continue
        push = triggers["push"]
        # This nightly canary also gates releases with dedicated spend-capped keys.
        if filename == "live-providers.yml" and push == {"tags": ["v*"]}:
            continue
        if not isinstance(push, dict) or not ({"branches", "branches-ignore"} & set(push)):
            violations.append(filename)
        elif {"tags", "tags-ignore"} & set(push):
            violations.append(filename)
    assert not violations, "tag-triggered workflows: " + ", ".join(violations)
