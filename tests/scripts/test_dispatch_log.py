"""The pre-build log must name the release.py command that started the run."""
import json
import shlex

import pytest

from scripts.releases import dispatch_log
from scripts.releases.bundle_env import parse_assignments
from scripts.releases.channel_build import dispatch_command as channel_dispatch
from scripts.releases.commit_build import dispatch_command as commit_dispatch
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
        "CHANNEL": "",
        "DISPOSABLE_CHANNEL": "",
        "DISPOSABLE_RECEIVERS": "false",
        "RELEASE_PHASE": "",
        "UPLOAD_RELEASE": "false",
        "JOBS": ALL_JOBS,
        "TERMUX_UPGRADE_FROM_TAG": "",
        "BUNDLE_ENV_JSON": "{}",
        "R2_DISPOSABLE_RUN": "",
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
    assert forwarded["CHANNEL"] == "${{ inputs.channel }}"
    assert forwarded["BUNDLE_ENV_JSON"] == "${{ inputs.bundle_env }}"
    assert forwarded["JOBS"] == "${{ inputs.jobs }}"
    assert forwarded["DEFAULT_BRANCH"] == "${{ github.event.repository.default_branch }}"
    assert "GITHUB_TOKEN" not in forwarded and "GH_TOKEN" not in forwarded


@pytest.mark.parametrize("kind,extra", [
    ("commit", {}),
    ("channel", {"CHANNEL": "magic-test"}),
    ("commit", {"BUNDLE_ENV_JSON": json.dumps({"HERMES_SKIP_INTRO": "1", "HERMES_HOME": None})}),
])
def test_printed_command_is_the_dispatcher_command(kind, extra):
    values = env(BUILD_COMMIT=SHA, **extra)
    baked = json.loads(values["BUNDLE_ENV_JSON"])
    if kind == "channel":
        expected = channel_dispatch(values["CHANNEL"], SHA, REPOSITORY, BRANCH, baked or None)
    else:
        expected = commit_dispatch(SHA, REPOSITORY, BRANCH, baked or None)
    text = dispatch_log.report(values)
    assert "kind: " + kind in text
    assert "workflow: " + shlex.join(expected) in text
    flags = dispatch_log.command_flags(dispatch_log.describe(values))
    assert "release.py: " + shlex.join(["python", "scripts/release.py", "--publish", "--remote", "<remote>", *flags]) in text
    assert "receipt: v<version>+" + kind + ".<run-created-utc>.35629258153" in text
    facts = json.loads(text.split("facts: ", 1)[1].splitlines()[0])
    assert facts["run_id"] == "35629258153"
    assert facts["bundle_env"] == baked


@pytest.mark.parametrize("override", [
    {"TAG": "v0.1.0-canary.20260921120000", "UPLOAD_RELEASE": "true"},
    {"TAG": "v0.1.0", "RELEASE_PHASE": "candidate"},
    {"TAG": "v0.1.0", "RELEASE_PHASE": "publish"},
    {"DISPOSABLE_CHANNEL": "probe", "BUILD_COMMIT": SHA, "R2_DISPOSABLE_RUN": "99"},
])
def test_unrecreatable_dispatches_name_the_kind_and_no_command(override):
    text = dispatch_log.report(env(**override))
    assert "release.py: this dispatch is not a release.py --build-commit run" in text
    assert "workflow: " not in text
    assert "receipt: this dispatch writes no post-build receipt tag" in text
    assert "kind: unknown" not in text


@pytest.mark.parametrize("jobs", ["termux", "darwin-arm64,win32-x64"])
def test_a_partial_jobs_commit_has_a_receipt_but_no_release_command(jobs):
    text = dispatch_log.report(env(BUILD_COMMIT=SHA, JOBS=jobs))
    assert "release.py: this dispatch is not a release.py --build-commit run" in text
    assert "receipt: v<version>+commit.<run-created-utc>.35629258153" in text


def test_an_unreadable_jobs_value_is_reported_not_raised():
    # The log must not fail the build; admission refuses the input next.
    text = dispatch_log.report(env(BUILD_COMMIT=SHA, JOBS="mac,mac"))
    assert "release.py: this dispatch is not a release.py --build-commit run" in text
    assert '"jobs": "mac,mac"' in text


def test_report_omits_values_the_workflow_did_not_forward(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_R2_SECRET_ACCESS_KEY", "secret-value")
    text = dispatch_log.report(env(BUILD_COMMIT=SHA))
    assert "secret-value" not in text


def test_module_prints_the_report_from_the_process_environment(monkeypatch, capsys):
    for key, value in env(BUILD_COMMIT=SHA, CHANNEL="magic-test").items():
        monkeypatch.setenv(key, value)
    dispatch_log.main()
    text = capsys.readouterr().out
    assert "kind: channel" in text
    assert "gh workflow run desktop-bundled-release.yml" in text


def test_bundle_env_round_trips_the_cli_flags():
    baked = parse_assignments(["HERMES_SKIP_INTRO=1"], ["HERMES_HOME"])
    facts = dispatch_log.describe(env(BUILD_COMMIT=SHA, BUNDLE_ENV_JSON=json.dumps(baked)))
    flags = dispatch_log.command_flags(facts)
    assert flags == ["--build-commit", SHA, "--bundle-unset", "HERMES_HOME", "--bundle-env", "HERMES_SKIP_INTRO=1"]
