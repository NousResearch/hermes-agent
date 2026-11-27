"""Reusable release Docker calls must not evict another claim's pending work."""
import re

import pytest

from tests.ci.desktop_release_roles import evaluate
from tests.ci.test_stable_release_graph import workflow


def concurrency(*, run, phase="", event="workflow_dispatch", ref="refs/heads/main", pr=""):
    config = workflow("docker.yml")["concurrency"]
    github = {"run_id": run, "ref": ref, "event_name": event,
              "event": {"pull_request": {"number": pr}}}
    inputs = {"release-phase": phase}
    group = re.sub(r"\$\{\{(.*?)\}\}",
                   lambda match: str(evaluate(match[1], inputs, {}, github=github)),
                   config["group"])
    return group, evaluate(config["cancel-in-progress"], inputs, {}, github=github)


def test_release_calls_are_isolated_by_caller_run_and_phase():
    calls = [concurrency(run=run, phase=phase) for run in (101, 102)
             for phase in ("test", "publish")]
    assert len({group for group, _ in calls}) == len(calls)
    assert not any(cancel for _, cancel in calls)


@pytest.mark.parametrize("event,pr,cancel", [("push", "", False), ("pull_request", 42, True)])
def test_standalone_calls_keep_ref_or_pr_scoped_concurrency(event, pr, cancel):
    first = concurrency(run=101, event=event, pr=pr)
    second = concurrency(run=102, event=event, pr=pr)
    assert first == second
    assert first == (f"docker-{pr or 'refs/heads/main'}-standalone", cancel)
