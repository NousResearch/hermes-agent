"""Only stable failures may enter the protected-head publication reconciler."""

import pytest

from tests.ci.desktop_release_roles import gate
from tests.ci.test_stable_release_graph import workflow


@pytest.mark.parametrize("kind", ["Channel", "Canary", "Commit", "Stable"])
@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "success"])
def test_reconciler_evaluates_actual_run_title(kind, conclusion):
    claim = "rc.1-v1.2.3" if kind == "Stable" else f"{kind.lower()}.20261028T120000Z-v1.2.3"
    github = {
        "event_name": "workflow_run", "repository": "o/r",
        "event": {"workflow_run": {
            "display_title": f"{kind} release {claim}", "conclusion": conclusion,
            "event": "workflow_dispatch", "head_repository": {"full_name": "o/r"},
        }},
    }
    expression = workflow("stable-release-publication.yml")["jobs"]["reconcile"]["if"]
    assert gate(expression, {}, {}, github=github) == (kind == "Stable" and conclusion != "success")


@pytest.mark.parametrize("overrides", [
    {"event": "push"}, {"head_repository": {"full_name": "fork/r"}},
])
def test_stable_reconciliation_retains_event_and_repository_custody(overrides):
    github = {
        "event_name": "workflow_run", "repository": "o/r",
        "event": {"workflow_run": {
            "display_title": "Stable release rc.1-v1.2.3", "conclusion": "failure",
            "event": "workflow_dispatch", "head_repository": {"full_name": "o/r"}, **overrides,
        }},
    }
    expression = workflow("stable-release-publication.yml")["jobs"]["reconcile"]["if"]
    assert not gate(expression, {}, {}, github=github)


def test_manual_stable_reconciliation_remains_available():
    expression = workflow("stable-release-publication.yml")["jobs"]["reconcile"]["if"]
    assert gate(expression, {}, {}, github={"event_name": "workflow_dispatch"})
