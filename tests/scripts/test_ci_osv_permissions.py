"""The fork's reusable OSV caller must not invalidate the whole CI graph."""

from pathlib import Path

import yaml


WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"


def _workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))


def test_osv_caller_grants_the_reusable_workflows_permissions():
    caller = _workflow("ci.yaml")["jobs"]["osv-scanner"]
    callee = _workflow("osv-scanner.yml")

    assert caller["uses"] == "./.github/workflows/osv-scanner.yml"
    assert caller["permissions"] == callee["permissions"]


def test_osv_remains_required_with_sarif_permissions_scoped_to_its_job():
    ci = _workflow("ci.yaml")

    assert "osv-scanner" in ci["jobs"]["all-checks-pass"]["needs"]
    assert "security-events" not in ci["permissions"]
    assert ci["jobs"]["osv-scanner"]["permissions"] == {
        "actions": "read",
        "contents": "read",
        "security-events": "write",
    }
