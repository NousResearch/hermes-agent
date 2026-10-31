"""The pre-build log classifies a direct bundle dispatch; no release.py command recreates one."""
import json

import pytest

from scripts.releases import dispatch_log
from scripts.releases.bundle_env import parse_assignments
from scripts.releases.job_groups import ALL_JOBS
from tests.ci.test_desktop_release_tag_admission import _workflow

SHA = "a" * 40
REPOSITORY = "fixture-owner/fixture-repo"
BRANCH = "main"


def env(**values):
    base = {
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_SHA": SHA,
        "GITHUB_RUN_ID": "35629258153",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_ACTOR": "ethernet8023",
        "DEFAULT_BRANCH": BRANCH,
        "TAG": "",
        "BUILD_COMMIT": "",
        "RELEASE_PHASE": "",
        "UPLOAD_RELEASE": "false",
        "JOBS": ALL_JOBS,
        "TERMUX_UPGRADE_FROM_TAG": "",
        "BUNDLE_ENV_JSON": "{}",
    }
    base.update(values)
    return base


def test_pre_build_setup_prints_the_dispatch_before_any_other_work():
    steps = _workflow()["jobs"]["validate"]["steps"]
    printed = steps[1]
    assert printed["name"] == "Print the release.py command for this run"
    assert printed["run"] == "python3 -m scripts.releases.dispatch_log"
    assert "if" not in printed
    forwarded = {key: value for key, value in printed["env"].items()}
    assert forwarded["BUILD_COMMIT"] == "${{ inputs.build_commit }}"
    assert not {"CHANNEL", "BRANDING", "DISPOSABLE_CHANNEL", "DISPOSABLE_RECEIVERS"} & set(forwarded)
    assert forwarded["BUNDLE_ENV_JSON"] == "${{ inputs.bundle_env }}"
    assert forwarded["JOBS"] == "${{ inputs.jobs }}"
    assert forwarded["DEFAULT_BRANCH"] == "${{ github.event.repository.default_branch }}"
    assert "GITHUB_TOKEN" not in forwarded and "GH_TOKEN" not in forwarded


@pytest.mark.parametrize("override", [
    {"TAG": "v0.1.0-canary.20260921120000", "UPLOAD_RELEASE": "true"},
    {"TAG": "v0.1.0", "RELEASE_PHASE": "candidate"},
    {"TAG": "v0.1.0", "RELEASE_PHASE": "publish"},
    {"TAG": "canary.20260921T120000Z-v0.1.0", "RELEASE_PHASE": "candidate"},
])
def test_unrecreatable_dispatches_name_the_kind_and_no_command(override):
    text = dispatch_log.report(env(**override))
    assert "release.py: this dispatch is not a release.py --build-commit run" in text
    assert "workflow: " not in text
    assert "receipt: this dispatch writes no post-build receipt tag" in text
    assert "kind: unknown" not in text


@pytest.mark.parametrize("extra", [{}, {"JOBS": "termux"},
                                   {"BUNDLE_ENV_JSON": json.dumps({"HERMES_HOME": None})}])
def test_a_tagless_commit_dispatch_is_legacy_with_no_replay_and_no_receipt(extra):
    # release.py --build-commit runs Stable Release on a commit claim now; a direct tagless commit
    # dispatch is the legacy mode, which no release.py command recreates and which mints no receipt.
    text = dispatch_log.report(env(BUILD_COMMIT=SHA, **extra))
    assert "kind: commit" in text
    assert "a legacy tagless commit build" in text and "workflow: " not in text
    assert "receipt: this dispatch writes no post-build receipt tag" in text


def test_an_unreadable_jobs_value_is_reported_not_raised():
    # The log must not fail the build; admission refuses the input next.
    text = dispatch_log.report(env(BUILD_COMMIT=SHA, JOBS="mac,mac"))
    assert "a legacy tagless commit build" in text and "workflow: " not in text
    assert '"jobs": "mac,mac"' in text


def test_report_omits_values_the_workflow_did_not_forward(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "secret-value")
    text = dispatch_log.report(env(BUILD_COMMIT=SHA))
    assert "secret-value" not in text


def test_module_prints_the_report_from_the_process_environment(monkeypatch, capsys):
    for key, value in env(BUILD_COMMIT=SHA).items():
        monkeypatch.setenv(key, value)
    dispatch_log.main()
    text = capsys.readouterr().out
    assert "kind: commit" in text and "gh workflow run" not in text


def test_the_facts_carry_the_dispatched_bundle_env():
    baked = parse_assignments(["HERMES_SHARED_AUTH_DIR=/shared"], ["HERMES_HOME"])
    facts = dispatch_log.describe(env(BUILD_COMMIT=SHA, BUNDLE_ENV_JSON=json.dumps(baked)))
    assert facts["bundle_env"] == baked
    assert not {"channel", "disposable_channel", "disposable_run", "branding"} & set(facts)
