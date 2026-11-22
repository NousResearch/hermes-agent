"""Trust-boundary invariants for commit and channel claims on Stable Release.

Commit and channel claims execute unreviewed commit code on release runners:
- Every native build leg (Windows, macOS, Termux) must run under ``cache-mode: read``
  and skip every cache-save step, while diagnostic Linux legs (``cache-mode: write``)
  must not run at all.
- Neither the Termux APT GPG signing key nor its passphrase may ever be referenced or
  reachable in a commit/channel Termux build.
- ``win32-assembly-v1`` must restore only by exact lockfile key (no prefix ``restore-keys``).
- Commit claims in ``stable-release.yml`` may run bundles + smoke only and must never
  trigger shared-cache suites, Docker/Nix/PM builds, Store submission, or stable/sandbox
  reconciliation workflows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.releases import lanes as lane_set
from tests.ci import workflow_steps
from tests.ci.desktop_release_roles import (
    CANARY_TAG,
    DOWNLOADABLE_DISPATCHES,
    LINUX_TARGETS,
    NATIVE_TARGETS,
    SHA,
    admitted,
    evaluate,
    gate,
    linux_builds,
    load,
    native_builds,
    selection_gates,
    stage_step,
    termux_builder,
    termux_builds,
    universal_assembler,
)

ROOT = Path(__file__).resolve().parents[2]
COMMIT_TAG = "commit.20261026T180000Z-v0.33.0"
CHANNEL_TAG = "channel.20261026T180000Z-v0.33.0"


def _workflow(name: str) -> dict:
    return load(ROOT / ".github/workflows" / name)


def _candidate_dispatch(tag: str) -> dict:
    return {**DOWNLOADABLE_DISPATCHES["candidate"], "tag": tag}


@pytest.mark.parametrize("untrusted_inputs", [
    pytest.param(_candidate_dispatch(COMMIT_TAG), id="commit-claim"),
    pytest.param(_candidate_dispatch(CHANNEL_TAG), id="channel-claim"),
    pytest.param(DOWNLOADABLE_DISPATCHES["commit"], id="legacy-build-commit"),
    pytest.param(DOWNLOADABLE_DISPATCHES["channel"], id="legacy-channel"),
])
def test_untrusted_claims_route_only_to_read_only_native_legs_and_skip_linux(untrusted_inputs):
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    legs = native_builds(jobs)
    gates = selection_gates(jobs)
    needs = admitted(["validate"], channel=bool(untrusted_inputs.get("channel")))

    for target in NATIVE_TARGETS:
        release_job = jobs[legs[(target, "release")]]
        commit_job = jobs[legs[(target, "commit")]]
        assert release_job["cache-mode"] == "write"
        assert commit_job["cache-mode"] == "read"
        assert not gate(release_job["if"], untrusted_inputs, needs), (target, "release")
        assert gate(commit_job["if"], untrusted_inputs, needs), (target, "commit")

        gate_needs = {
            **needs,
            legs[(target, "release")]: {"result": "skipped"},
            legs[(target, "commit")]: {"result": "success"},
        }
        assert evaluate(jobs[gates[target]]["env"]["SELECTED_BUILD_SUCCEEDED"],
                        untrusted_inputs, gate_needs, job_if=False) is True, target

    for target, name in linux_builds(jobs).items():
        assert target in LINUX_TARGETS
        assert not gate(jobs[name]["if"], untrusted_inputs, needs), name


@pytest.mark.parametrize("trusted_tag", ["v0.33.0", CANARY_TAG])
def test_release_and_canary_tags_keep_write_mode_native_and_termux_legs(trusted_tag):
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    inputs = _candidate_dispatch(trusted_tag)
    needs = admitted(["validate"])

    legs = native_builds(jobs)
    gates = selection_gates(jobs)
    for target in NATIVE_TARGETS:
        assert gate(jobs[legs[(target, "release")]]["if"], inputs, needs), target
        assert not gate(jobs[legs[(target, "commit")]]["if"], inputs, needs), target
        gate_needs = {
            **needs,
            legs[(target, "release")]: {"result": "success"},
            legs[(target, "commit")]: {"result": "skipped"},
        }
        assert evaluate(jobs[gates[target]]["env"]["SELECTED_BUILD_SUCCEEDED"],
                        inputs, gate_needs, job_if=False) is True

    t_legs = termux_builds(jobs)
    assert gate(jobs[t_legs["release"]]["if"], inputs, needs)
    assert not gate(jobs[t_legs["commit"]]["if"], inputs, needs)
    t_gate = jobs[termux_builder(jobs)]
    t_needs = {**needs, t_legs["release"]: {"result": "success"}, t_legs["commit"]: {"result": "skipped"}}
    assert evaluate(t_gate["env"]["SELECTED_BUILD_SUCCEEDED"], inputs, t_needs, job_if=False) is True


def test_termux_commit_branch_is_read_only_and_has_no_apt_signing_key_access():
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    t_legs = termux_builds(jobs)
    release_job = jobs[t_legs["release"]]
    commit_job = jobs[t_legs["commit"]]
    gate_job = jobs[termux_builder(jobs)]

    assert release_job["cache-mode"] == "write"
    assert commit_job["cache-mode"] == "read"
    assert commit_job.get("permissions") == {"contents": "read", "packages": "read"}

    # The commit Termux job must never expose GH_TOKEN or R2 secrets in job-level env,
    # never push to GHCR, never reference the APT GPG key or passphrase, and never save caches.
    job_env = commit_job.get("env") or {}
    assert "GH_TOKEN" not in job_env
    assert "GITHUB_TOKEN" not in job_env
    assert "CLOUDFLARE_R2_SECRET_ACCESS_KEY" not in job_env
    assert "CLOUDFLARE_R2_ACCESS_KEY_ID" not in job_env
    serialized_commit = json.dumps(commit_job)
    assert "TERMUX_APT_GPG_KEY" not in serialized_commit
    assert "TERMUX_APT_GPG_PASSPHRASE" not in serialized_commit
    assert "stage_apt_repo.py" not in serialized_commit
    assert not any(str(step.get("uses", "")).startswith("actions/cache/save") for step in commit_job["steps"])
    builder_step = next(step for step in commit_job["steps"] if step.get("id") == "builder_image")
    assert "GITHUB_TOKEN" not in (builder_step.get("env") or {})
    assert (builder_step.get("env") or {}).get("TERMUX_BUILDER_NO_PUSH") == "1"
    setup_pm = next(step for step in commit_job["steps"] if step.get("uses") == "./.github/actions/setup-pm")
    assert setup_pm["with"]["save-tools-cache"] is False
    assert setup_pm["with"]["save-node-cache"] is False
    assert setup_pm["with"]["cache-python"] is False
    assert setup_pm["with"]["archive-inputs"] is False

    # Untrusted dispatches route only to termux-deb-commit (never termux-deb-release).
    needs = admitted(["validate"])
    for inputs in (_candidate_dispatch(COMMIT_TAG), _candidate_dispatch(CHANNEL_TAG),
                   DOWNLOADABLE_DISPATCHES["commit"]):
        assert not gate(release_job["if"], inputs, needs)
        assert gate(commit_job["if"], inputs, needs)
        gate_needs = {**needs, t_legs["release"]: {"result": "skipped"}, t_legs["commit"]: {"result": "success"}}
        assert evaluate(gate_job["env"]["SELECTED_BUILD_SUCCEEDED"], inputs, gate_needs, job_if=False) is True
        # Defense in depth: even the APT signing steps inside termux-deb-release refuse commit/channel tags.
        for step in release_job["steps"]:
            if step.get("name") in ("Write the APT signing key", "Stage the APT repo handoff to R2"):
                assert not gate(step["if"], inputs, needs, job_if=False), step["name"]


_STAGING_STUB = '''import json, os, sys
with open(os.environ["STAGING_RECEIPTS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({{"copy": {copy!r}, "module": __spec__.name, "cwd": os.getcwd(),
                             "argv": sys.argv[1:]}}) + "\\n")
'''


def _claim_staging_legs():
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    legs = native_builds(jobs)
    rows = [(legs[(target, "commit")], target, "apps/desktop/release") for target in NATIVE_TARGETS]
    return [*rows, (termux_builds(jobs)["commit"], None, "termux-build")]


@pytest.mark.parametrize("legacy", ["true", "false"])
@pytest.mark.parametrize("job_name,target,artifacts", _claim_staging_legs())
def test_claim_staging_runs_the_controller_recorder_on_the_built_artifacts(
        tmp_path, job_name, target, artifacts, legacy):
    """A claim's checkout is unreviewed and may predate claims, so it never stages itself."""
    job = _workflow("desktop-bundled-release.yml")["jobs"][job_name]
    checkout = tmp_path / "checkout"
    for copy, base in (("claimed", checkout), ("controller", checkout / ".build/commit-controller")):
        for module in ("scripts/bundles/release_artifacts.py", "scripts/releases/handoff.py"):
            (base / module).parent.mkdir(parents=True, exist_ok=True)
            (base / module).write_text(_STAGING_STUB.format(copy=copy), encoding="utf-8")
    (checkout / artifacts).mkdir(parents=True)
    receipts = tmp_path / "receipts.jsonl"
    step = stage_step(job)
    replay = {**step, "env": {**job.get("env", {}), **step.get("env", {}),
                              "GITHUB_WORKSPACE": str(checkout), "STAGING_RECEIPTS": str(receipts)}}
    context = {
        "inputs": _candidate_dispatch(COMMIT_TAG), "matrix": {"target": {"label": target}},
        "needs": {"validate": {"result": "success", "outputs": {
            "sha": SHA, "channel-build": "", "archive-tag": COMMIT_TAG, "legacy-commit-build": legacy}}},
        "github": {"workspace": str(checkout)},
    }
    workflow_steps.outputs(replay, context, cwd=checkout)

    calls = [json.loads(line) for line in receipts.read_text(encoding="utf-8").splitlines()]
    assert {call["module"] for call in calls} == {"scripts.bundles.release_artifacts", "scripts.releases.handoff"}
    for call in calls:
        assert call["copy"] == "controller", call
        argv = call["argv"]
        for flag in ("--root", "--out"):
            if flag in argv:
                path = Path(call["cwd"], argv[argv.index(flag) + 1]).resolve()
                assert path.is_relative_to((checkout / artifacts).resolve()), (flag, call)
    (record,) = [call for call in calls if call["module"] == "scripts.bundles.release_artifacts"]
    assert ("--legacy-commit-build" in record["argv"]) == (legacy == "true")


def test_every_cache_save_step_skips_commit_and_channel_claims():
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    untrusted_cases = (
        _candidate_dispatch(COMMIT_TAG),
        _candidate_dispatch(CHANNEL_TAG),
        DOWNLOADABLE_DISPATCHES["commit"],
        DOWNLOADABLE_DISPATCHES["channel"],
    )
    steps_ctx = {
        "prepare": {"outcome": "success"},
        "mac-signatures": {"outputs": {"cache-matched-key": "different"}},
    }
    save_steps = 0
    for job_name, job in jobs.items():
        for step in job.get("steps", []):
            uses = str(step.get("uses", ""))
            is_save = uses.startswith("actions/cache/save") or (
                uses == "./.github/actions/desktop-build-cache"
                and (step.get("with") or {}).get("phase") == "save"
            )
            if not is_save:
                continue
            save_steps += 1
            expr = step.get("if")
            assert expr, f"{job_name}/{step.get('name')} saves cache unconditionally"
            for inputs in untrusted_cases:
                if "hashFiles(" in str(expr) or "matrix.target.label" in str(expr):
                    # Verify the static guard explicitly rejects commit./channel. tags as well as build_commit/channel.
                    assert "!startsWith(inputs.tag, 'commit.')" in str(expr), (job_name, step.get("name"))
                    assert "!startsWith(inputs.tag, 'channel.')" in str(expr), (job_name, step.get("name"))
                    assert "inputs.build_commit == ''" in str(expr), (job_name, step.get("name"))
                else:
                    assert not gate(expr, inputs, {}, job_if=False, steps=steps_ctx), (job_name, step.get("name"))
    assert save_steps >= 6


def test_win32_assembly_cache_uses_exact_key_without_prefix_restore_keys():
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    assembler = jobs[universal_assembler(jobs)]
    assert assembler["cache-mode"] == "read"
    restore = next(step for step in assembler["steps"]
                   if step.get("name") == "Restore the bundle-assembly caches")
    assert "win32-assembly-v1-" in restore["with"]["key"]
    assert "restore-keys" not in restore["with"]


def test_commit_policy_runs_bundles_and_smoke_only_and_never_touches_release_or_sandbox():
    assert lane_set.COMMIT == frozenset({"windows", "macos-arm64", "smoke"})
    refused = lane_set.POLICY_REFUSED["commit"]
    for forbidden in ("docker", "docker-tests", "ci", "nix", "pm-bundle", "install-e2e",
                      "termux-checks", "windows-live", "upgrade", "windows-store", "linux",
                      "bootstrap-version"):
        assert forbidden in refused, forbidden
        with pytest.raises(ValueError, match="commit"):
            lane_set.select(with_=[forbidden], policy="commit")

    # --with termux is allowed (runs read-only termux-deb-commit, never termux-checks).
    with_termux = lane_set.select(with_=["termux"], policy="commit")
    assert with_termux == frozenset({"windows", "macos-arm64", "smoke", "termux"})
    runs = lane_set.jobs_to_run(with_termux, "commit")
    assert {job for job, active in runs.items() if active} == {
        "candidates-darwin-arm64", "candidates-win32-arm64", "candidates-win32-x64",
        "candidates-win32-bundle", "candidates-termux", "candidate-manifest", "publish-commit",
    }

    # Commit concurrency isolates per claim tag; stable/sandbox/reconciler ignore Commit release runs.
    stable_wf = _workflow("stable-release.yml")
    assert evaluate(stable_wf["concurrency"]["group"], {"tag": COMMIT_TAG}, {}) == COMMIT_TAG
    assert evaluate(stable_wf["concurrency"]["group"], {"tag": "v0.33.0"}, {}) == "stable-release"

    pub_wf = _workflow("stable-release-publication.yml")
    commit_wf_run = {"event_name": "workflow_run", "repository": "NousResearch/hermes-agent",
                     "event": {"workflow_run": {
                         "conclusion": "success",
                         "display_title": f"Commit release {COMMIT_TAG}",
                         "head_repository": {"full_name": "NousResearch/hermes-agent"},
                     }}}
    assert not gate(pub_wf["jobs"]["reconcile"]["if"], {}, {}, github=commit_wf_run)
    sandbox_wf = _workflow("sandbox-image.yml")
    assert not gate(sandbox_wf["jobs"]["build"]["if"], {}, {}, github=commit_wf_run)
    # Stable release still passes sandbox-image.yml.
    stable_wf_run = {**commit_wf_run, "event": {"workflow_run": {
        **commit_wf_run["event"]["workflow_run"], "display_title": "Stable release rc.1-v0.33.0",
    }}}
    assert gate(sandbox_wf["jobs"]["build"]["if"], {}, {}, github=stable_wf_run)

    # Desktop store submission never runs on commit claims, and post-build receipt tag step is retired.
    desktop = _workflow("desktop-bundled-release.yml")["jobs"]
    store_if = desktop["stable-store"]["if"]
    assert not gate(store_if, {"release-phase": "publish", "skip-store": False, "tag": COMMIT_TAG},
                    {"validate": {"result": "success"}})
    assert not any("commit_build receipt" in step.get("run", "")
                   for step in desktop["commit-builds-summary"]["steps"])
    assert desktop["stage-receipt-darwin-arm64"]["cache-mode"] == "read"
