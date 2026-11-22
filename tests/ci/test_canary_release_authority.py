"""Exercise the publisher and its dependent cleanup job on forks and upstream."""
from pathlib import Path

import hermes_yaml
import pytest

from tests.ci._gha_expr import condition


@pytest.mark.parametrize(("repository", "branch", "event", "eligible"), [
    ("calidreamconstruction/hermes-agent", "main", "schedule", False),
    ("calidreamconstruction/hermes-agent", "main", "workflow_dispatch", False),
    ("NousResearch/hermes-agent", "feature", "workflow_dispatch", False),
    ("NousResearch/hermes-agent", "main", "schedule", True),
    ("NousResearch/hermes-agent", "main", "workflow_dispatch", True),
])
def test_canary_publisher_and_cleanup_admission(repository, branch, event, eligible):
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/canary-release.yml"
    jobs = hermes_yaml.safe_load(workflow.read_text(encoding="utf-8"))["jobs"]
    context = {"github": {
        "repository": repository, "ref_name": branch, "event_name": event,
        "event": {"repository": {"default_branch": "main"}},
    }}
    tag_runs = condition(jobs["tag"].get("if"), context)
    assert tag_runs is eligible

    assert jobs["prune"]["needs"] == "tag"
    context["needs"] = {"tag": {"result": "success" if tag_runs else "skipped"}}
    context["__status__"] = {"always": True, "success": tag_runs}
    assert condition(jobs["prune"].get("if"), context) is eligible
