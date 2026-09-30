"""The inherited daily canary is upstream-only; a manual dispatch still works.

A fork inherits ``canary-release.yml`` complete with its schedule, and nothing in
the job stopped it there: the run tagged its own main, then failed on a body
GitHub refuses and on a dispatch its repository has no credentials to finish, so
every fork with Actions enabled failed this job daily for reasons it could not
fix. The scheduled run is now upstream-only.
"""
from pathlib import Path

from ruamel.yaml import YAML

from tests.ci.desktop_release_roles import gate

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "NousResearch/hermes-agent"


def tag_job():
    workflow = YAML(typ="base").load(
        (ROOT / ".github/workflows/canary-release.yml").read_text(encoding="utf-8"))
    return workflow["jobs"]["tag"]


def admitted(*, repository, event_name, ref_name="main"):
    github = {
        "repository": repository,
        "event_name": event_name,
        "ref_name": ref_name,
        "event": {"repository": {"default_branch": "main"}},
    }
    return gate(tag_job()["if"], {}, {}, github=github)


def test_the_scheduled_canary_runs_upstream_only():
    assert admitted(repository=UPSTREAM, event_name="schedule")
    assert not admitted(repository="jiangkoumo/hermes-agent", event_name="schedule")


def test_a_manual_dispatch_still_runs_off_upstream():
    assert admitted(repository="jiangkoumo/hermes-agent", event_name="workflow_dispatch")


def test_a_feature_branch_dispatch_never_tags():
    assert not admitted(repository=UPSTREAM, event_name="workflow_dispatch", ref_name="fix/thing")
    assert not admitted(
        repository="jiangkoumo/hermes-agent", event_name="workflow_dispatch", ref_name="fix/thing")
