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
import os
import shutil
import sys
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
    commit_claim_stagers,
    evaluate,
    gate,
    linux_builds,
    load,
    native_builds,
    needs_of,
    selection_gates,
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
    stagers = commit_claim_stagers(jobs)
    needs = admitted(["validate"], channel=bool(untrusted_inputs.get("channel")))
    is_commit_claim = (
        untrusted_inputs.get("release-phase") == "candidate"
        and str(untrusted_inputs.get("tag", "")).startswith("commit.")
    )

    for target in NATIVE_TARGETS:
        release_job = jobs[legs[(target, "release")]]
        commit_job = jobs[legs[(target, "commit")]]
        stager_name = stagers[target]
        stager_job = jobs[stager_name]
        assert release_job["cache-mode"] == "write"
        assert commit_job["cache-mode"] == "read"
        assert not gate(release_job["if"], untrusted_inputs, needs), (target, "release")
        assert gate(commit_job["if"], untrusted_inputs, needs), (target, "commit")
        stager_needs = {**needs, legs[(target, "commit")]: {"result": "success"}}
        assert gate(stager_job["if"], untrusted_inputs, stager_needs) is is_commit_claim, (target, stager_name)

        gate_needs = {
            **needs,
            legs[(target, "release")]: {"result": "skipped"},
            legs[(target, "commit")]: {"result": "success"},
            stager_name: {"result": "success" if is_commit_claim else "skipped"},
        }
        assert evaluate(jobs[gates[target]]["env"]["SELECTED_BUILD_SUCCEEDED"],
                        untrusted_inputs, gate_needs, job_if=False) is True, target
        if is_commit_claim:
            # A commit claim must fail its selection gate if the fresh controller staging job failed or skipped.
            for bad in ("failure", "skipped"):
                bad_needs = {**gate_needs, stager_name: {"result": bad}}
                assert evaluate(jobs[gates[target]]["env"]["SELECTED_BUILD_SUCCEEDED"],
                                untrusted_inputs, bad_needs, job_if=False) is False, (target, bad)

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
    stagers = commit_claim_stagers(jobs)
    for target in NATIVE_TARGETS:
        assert gate(jobs[legs[(target, "release")]]["if"], inputs, needs), target
        assert not gate(jobs[legs[(target, "commit")]]["if"], inputs, needs), target
        assert not gate(jobs[stagers[target]]["if"], inputs, {**needs, legs[(target, "commit")]: {"result": "success"}}), target
        gate_needs = {
            **needs,
            legs[(target, "release")]: {"result": "success"},
            legs[(target, "commit")]: {"result": "skipped"},
            stagers[target]: {"result": "skipped"},
        }
        assert evaluate(jobs[gates[target]]["env"]["SELECTED_BUILD_SUCCEEDED"],
                        inputs, gate_needs, job_if=False) is True

    t_legs = termux_builds(jobs)
    assert gate(jobs[t_legs["release"]]["if"], inputs, needs)
    assert not gate(jobs[t_legs["commit"]]["if"], inputs, needs)
    assert not gate(jobs[stagers["termux"]]["if"], inputs, {**needs, t_legs["commit"]: {"result": "success"}})
    t_gate = jobs[termux_builder(jobs)]
    t_needs = {
        **needs,
        t_legs["release"]: {"result": "success"},
        t_legs["commit"]: {"result": "skipped"},
        stagers["termux"]: {"result": "skipped"},
    }
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
    termux_action_use = "./.build/termux-action/.github/actions/build-termux-deb"
    commit_build = next(step for step in commit_job["steps"] if step.get("uses") == termux_action_use)
    assert commit_build["with"]["build-mode"] == "commit"
    assert commit_build["with"]["save-caches"] in (False, "false")
    assert commit_build["with"]["push-builder-image"] in (False, "false")
    assert "github-token" not in commit_build["with"]
    release_build = next(step for step in release_job["steps"] if step.get("uses") == termux_action_use)
    assert release_build["with"]["build-mode"] == "release"
    assert release_build["with"]["push-builder-image"] in (True, "true")
    for untrusted in (_candidate_dispatch(COMMIT_TAG), _candidate_dispatch(CHANNEL_TAG),
                      DOWNLOADABLE_DISPATCHES["commit"]):
        assert not gate(release_build["with"]["save-caches"], untrusted, {}, job_if=False)
    setup_pm = next(step for step in commit_job["steps"] if step.get("uses") == "./.github/actions/setup-pm")
    assert setup_pm["with"]["save-tools-cache"] is False
    assert setup_pm["with"]["save-node-cache"] is False
    assert setup_pm["with"]["cache-python"] is False
    assert setup_pm["with"]["archive-inputs"] is False

    # Untrusted dispatches route only to termux-deb-commit (never termux-deb-release).
    stager_name = commit_claim_stagers(jobs)["termux"]
    needs = admitted(["validate"])
    for inputs in (_candidate_dispatch(COMMIT_TAG), _candidate_dispatch(CHANNEL_TAG),
                   DOWNLOADABLE_DISPATCHES["commit"]):
        is_claim = inputs.get("release-phase") == "candidate" and str(inputs.get("tag", "")).startswith("commit.")
        assert not gate(release_job["if"], inputs, needs)
        assert gate(commit_job["if"], inputs, needs)
        assert gate(jobs[stager_name]["if"], inputs, {**needs, t_legs["commit"]: {"result": "success"}}) is is_claim
        gate_needs = {
            **needs,
            t_legs["release"]: {"result": "skipped"},
            t_legs["commit"]: {"result": "success"},
            stager_name: {"result": "success" if is_claim else "skipped"},
        }
        assert evaluate(gate_job["env"]["SELECTED_BUILD_SUCCEEDED"], inputs, gate_needs, job_if=False) is True
        if is_claim:
            for bad in ("failure", "skipped"):
                assert evaluate(gate_job["env"]["SELECTED_BUILD_SUCCEEDED"], inputs,
                                {**gate_needs, stager_name: {"result": bad}}, job_if=False) is False
        # Defense in depth: even the APT signing steps inside termux-deb-release refuse commit/channel tags.
        for step in release_job["steps"]:
            if step.get("name") in ("Write the APT signing key", "Stage the APT repo handoff to R2"):
                assert not gate(step["if"], inputs, needs, job_if=False), step["name"]


_STAGING_STUB = '''import json, os, sys
with open(os.environ["STAGING_RECEIPTS"], "a", encoding="utf-8") as stream:
    stream.write(json.dumps({{"copy": {copy!r}, "module": __spec__.name, "cwd": os.getcwd(),
                             "argv": sys.argv[1:]}}) + "\\n")
'''


def test_commit_claim_staging_runs_on_fresh_controller_only_jobs_without_builder_r2_fallback():
    """Unreviewed build jobs never check out .build/commit-controller or stage commit claims to R2."""
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    legs = native_builds(jobs)
    t_legs = termux_builds(jobs)
    stagers = commit_claim_stagers(jobs)
    claim = _candidate_dispatch(COMMIT_TAG)
    needs = admitted(["validate"])
    commit_legs = {**{target: legs[(target, "commit")] for target in NATIVE_TARGETS}, "termux": t_legs["commit"]}

    for target, build_name in commit_legs.items():
        build_job = jobs[build_name]
        serialized = json.dumps(build_job)
        assert ".build/commit-controller" not in serialized, build_name
        # Builder-side R2 staging steps must skip commit claims completely.
        for step in build_job["steps"]:
            if "scripts.releases.handoff stage" in step.get("run", ""):
                assert not gate(step.get("if", "true"), claim, needs, job_if=False), (build_name, step["name"])
        # Builder uploads only built package artifacts for commit claims (never sidecar metadata JSON).
        uploads = [step for step in build_job["steps"]
                   if str(step.get("uses", "")).startswith("actions/upload-artifact@")]
        assert len(uploads) == 1, build_name
        upload = uploads[0]
        assert gate(upload["if"], claim, needs, job_if=False)
        assert not gate(upload["if"], _candidate_dispatch("v0.33.0"), needs, job_if=False)
        assert not gate(upload["if"], DOWNLOADABLE_DISPATCHES["commit"], needs, job_if=False)
        assert "version-info" not in str(upload["with"]["path"])
        assert upload["with"]["if-no-files-found"] == "error"

        # Fresh controller-only staging job depends on validate + build leg, checks out only github.sha.
        stager_name = stagers[target]
        stager = jobs[stager_name]
        assert needs_of(stager) == ["validate", build_name], (stager_name, stager.get("needs"))
        assert stager["cache-mode"] == "read", stager_name
        assert stager.get("permissions") == {"contents": "read"}, stager_name
        checkouts = [step for step in stager["steps"] if "checkout@" in str(step.get("uses", ""))]
        assert len(checkouts) == 1, stager_name
        assert checkouts[0]["with"]["ref"] == "${{ github.sha }}", stager_name
        assert checkouts[0]["with"]["persist-credentials"] in (False, "false"), stager_name
        assert "validate.outputs.sha" not in json.dumps(checkouts[0]), stager_name
        (download,) = [step for step in stager["steps"]
                       if str(step.get("uses", "")).startswith("actions/download-artifact@")]
        expected_name = (
            "commit-claim-termux-${{ github.run_id }}-${{ github.run_attempt }}"
            if target == "termux"
            else "commit-claim-${{ env.TARGET }}-${{ github.run_id }}-${{ github.run_attempt }}"
        )
        assert download["with"]["name"] == expected_name, (stager_name, download["with"])


@pytest.mark.parametrize("legacy", ["true", "false"])
@pytest.mark.parametrize("target", [*NATIVE_TARGETS, "termux"])
def test_claim_staging_runs_the_controller_recorder_on_the_built_artifacts(
        tmp_path, target, legacy):
    """Fresh controller staging jobs run only controller modules on the downloaded artifacts."""
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    stager = jobs[commit_claim_stagers(jobs)[target]]
    controller = tmp_path / "controller"
    for module in ("scripts/bundles/release_artifacts.py", "scripts/releases/handoff.py"):
        (controller / module).parent.mkdir(parents=True, exist_ok=True)
        (controller / module).write_text(_STAGING_STUB.format(copy="controller"), encoding="utf-8")
    staged = controller / "staged-artifacts"
    (staged / "deb" if target == "termux" else staged).mkdir(parents=True)
    extra_bin = tmp_path / "bin"
    extra_bin.mkdir()
    if target == "darwin-arm64":
        (staged / "HermesBundled-0.33.0-mac-arm64.zip").write_bytes(b"zip-fixture")
        ditto = extra_bin / "ditto"
        ditto.write_text("#!/usr/bin/env bash\nset -e\n[ \"$1\" = -x ] && [ \"$2\" = -k ] && mkdir -p \"$4/Hermes.app\"\n", encoding="utf-8")
        ditto.chmod(0o755)
    receipts = tmp_path / "receipts.jsonl"
    (step,) = [s for s in stager["steps"] if "scripts.releases.handoff stage" in s.get("run", "")]
    replay = {**step, "env": {**stager.get("env", {}), **step.get("env", {}),
                              "GITHUB_WORKSPACE": str(controller), "STAGING_RECEIPTS": str(receipts),
                              "PATH": f"{extra_bin}{os.pathsep}{os.environ['PATH']}"}}
    context = {
        "inputs": _candidate_dispatch(COMMIT_TAG),
        "needs": {"validate": {"result": "success", "outputs": {
            "sha": SHA, "channel-build": "", "archive-tag": COMMIT_TAG, "legacy-commit-build": legacy}}},
        "github": {"workspace": str(controller)},
    }
    workflow_steps.outputs(replay, context, cwd=controller)

    calls = [json.loads(line) for line in receipts.read_text(encoding="utf-8").splitlines()]
    assert {call["module"] for call in calls} == {"scripts.bundles.release_artifacts", "scripts.releases.handoff"}
    for call in calls:
        assert call["copy"] == "controller", call
        argv = call["argv"]
        for flag in ("--root", "--out"):
            if flag in argv:
                path = Path(call["cwd"], argv[argv.index(flag) + 1]).resolve()
                assert path.is_relative_to(staged.resolve()), (flag, call)
    (record,) = [call for call in calls if call["module"] == "scripts.bundles.release_artifacts"]
    assert ("--legacy-commit-build" in record["argv"]) == (legacy == "true")


def test_untrusted_commit_legs_disable_persisted_credentials_and_strip_write_scopes():
    """Every build leg executing unreviewed commit code strips git credentials and write scopes."""
    jobs = _workflow("desktop-bundled-release.yml")["jobs"]
    legs = native_builds(jobs)
    t_legs = termux_builds(jobs)
    expected_perms = {
        legs[("win32-x64", "commit")]: {"contents": "read", "id-token": "write"},
        legs[("win32-arm64", "commit")]: {"contents": "read", "id-token": "write"},
        legs[("darwin-arm64", "commit")]: {"contents": "read"},
        t_legs["commit"]: {"contents": "read", "packages": "read"},
    }
    for job_name, expected in expected_perms.items():
        job = jobs[job_name]
        assert job.get("permissions") == expected, (job_name, job.get("permissions"))
        checkouts = [step for step in job["steps"] if "checkout@" in str(step.get("uses", ""))]
        assert checkouts, job_name
        for step in checkouts:
            assert (step.get("with") or {}).get("persist-credentials") in (False, "false"), (job_name, step)


@pytest.mark.parametrize("legacy_script", [False, True])
def test_build_termux_deb_action_never_saves_caches_or_pushes_ghcr_in_commit_mode(tmp_path, legacy_script):
    """The shared Termux composite action respects save-caches=false and push-builder-image=false."""
    action = load(ROOT / ".github/actions/build-termux-deb/action.yml")
    steps = action["runs"]["steps"]
    save_steps = [s for s in steps if str(s.get("uses", "")).startswith("actions/cache/save")]
    assert len(save_steps) == 2
    for step in save_steps:
        assert not gate(step["if"], {"save-caches": "false"}, {}, job_if=False), step["name"]
        assert gate(step["if"], {"save-caches": "true"}, {}, job_if=False), step["name"]

    push_step = next(s for s in steps if s.get("id") == "builder_image_push")
    local_step = next(s for s in steps if s.get("id") == "builder_image_local")
    assert not gate(push_step["if"], {"push-builder-image": "false"}, {}, job_if=False)
    assert gate(push_step["if"], {"push-builder-image": "true"}, {}, job_if=False)
    assert gate(local_step["if"], {"push-builder-image": "false"}, {}, job_if=False)
    assert not gate(local_step["if"], {"push-builder-image": "true"}, {}, job_if=False)
    assert "GITHUB_TOKEN" not in (local_step.get("env") or {})

    # Replay builder_image_local on both current and pre-PR build_builder_image.sh trees.
    repo = tmp_path / "repo"
    scripts_dir = repo / "scripts/termux"
    scripts_dir.mkdir(parents=True)
    shutil.copy2(ROOT / "scripts/termux/termux-builder.Dockerfile", scripts_dir / "termux-builder.Dockerfile")
    if legacy_script:
        (scripts_dir / "build_builder_image.sh").write_text("#!/usr/bin/env bash\necho 'legacy script should not run' >&2; exit 88\n", encoding="utf-8")
    else:
        shutil.copy2(ROOT / "scripts/termux/build_builder_image.sh", scripts_dir / "build_builder_image.sh")
    shutil.copytree(ROOT / "pm", repo / "pm", ignore=shutil.ignore_patterns("__pycache__"))
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "python3").symlink_to(sys.executable)
    docker_log = tmp_path / "docker.log"
    docker_stub = tools / "docker"
    docker_stub.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$DOCKER_LOG\"\n"
        "case \"$1 $2\" in\n"
        "  'manifest inspect') exit 1 ;;\n"
        "  'build '*) exit 0 ;;\n"
        "  'run '*) echo 'builder image smoke OK'; exit 0 ;;\n"
        "  'login '*|'push '*) echo \"FORBIDDEN docker $1\" >&2; exit 99 ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    docker_stub.chmod(0o755)
    replay = {**local_step, "env": {**local_step.get("env", {}),
                                    "DOCKER_LOG": str(docker_log),
                                    "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}"}}
    out = workflow_steps.outputs(replay, {"github": {"repository_owner": "NousResearch"}}, cwd=repo)
    assert out["image"].startswith("hermes-termux-builder-local:"), out
    calls = docker_log.read_text(encoding="utf-8").splitlines()
    assert not any(c.startswith(("login ", "push ")) for c in calls), calls


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
