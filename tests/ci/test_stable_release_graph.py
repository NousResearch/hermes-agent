"""The release workflow's dependency graph enforces publication ordering."""
import os
import subprocess
from pathlib import Path

from ruamel.yaml import YAML

from tests.ci.desktop_release_roles import gate

ROOT = Path(__file__).resolve().parents[2]


def workflow(name):
    return YAML(typ="base").load((ROOT / ".github/workflows" / name).read_text(encoding="utf-8"))


def direct_needs(jobs, name):
    needs = jobs[name].get("needs", [])
    return [needs] if isinstance(needs, str) else needs


def ancestors(jobs, name):
    seen = set()
    pending = [name]
    while pending:
        needs = jobs[pending.pop()].get("needs", [])
        for item in [needs] if isinstance(needs, str) else needs:
            if item not in seen:
                seen.add(item)
                pending.append(item)
    return seen


def test_admit_can_read_the_claim_draft():
    jobs = workflow("stable-release.yml")["jobs"]
    assert jobs["admit"]["permissions"]["contents"] == "write"


def test_release_reuses_whole_ci_and_docker_before_publication():
    jobs = workflow("stable-release.yml")["jobs"]
    assert jobs["ci"]["uses"] == "./.github/workflows/ci.yaml"
    assert jobs["ci"]["with"]["release"] == "true"
    assert "secrets" not in jobs["ci"]
    assert jobs["docker"]["uses"] == jobs["publish-docker"]["uses"]
    assert jobs["docker"]["with"]["release-phase"] == "test"
    # B7: every gate and every signed candidate starts straight after admit.
    # The acceptance join requires CI, so a slow gate cannot delay a build.
    for parallel in ("docker", "nix", "pm-bundle", "termux-checks", "windows-live", "install-e2e"):
        assert direct_needs(jobs, parallel) == ["admit"], parallel
    candidate_calls = ["candidates-darwin-arm64", "candidates-darwin-x64", "candidates-win32-arm64",
                       "candidates-win32-x64", "candidates-win32-bundle", "candidates-termux"]
    # The Windows bundle group is the only candidate that waits on a peer.
    for call in candidate_calls:
        assert set(direct_needs(jobs, call)) <= {
            "admit", "candidates-win32-arm64", "candidates-win32-x64"}, call
    required = {"ci", "docker", "nix", "pm-bundle", "install-e2e", "windows-packaged",
                "macos-packaged-arm64", "macos-packaged-x64", "termux-checks", "windows-live",
                "candidate-manifest", "transitions-darwin-arm64", "transitions-darwin-x64",
                "transitions-win32", "bootstrap-version", *candidate_calls}
    assert required <= ancestors(jobs, "acceptance")
    # B3: stable calls one build group at a time; the bundle group waits for
    # both Windows arches, and the Linux groups are not called at all.
    assert "candidates" not in jobs
    for name, group in zip(candidate_calls, ("darwin-arm64", "darwin-x64", "win32-arm64",
                                            "win32-x64", "win32-bundle", "termux")):
        call = jobs[name]
        assert call["uses"] == "./.github/workflows/desktop-bundled-release.yml"
        assert call["with"]["release-phase"] == "candidate"
        assert call["with"]["jobs"] == group
        assert {"tag", "claim-tag", "claim-object"} <= set(call["with"])
    assert {"candidates-win32-arm64", "candidates-win32-x64"} <= set(jobs["candidates-win32-bundle"]["needs"])
    assert not any("linux" in name for name in jobs)
    # B4: each install arm starts from its own receipt, and the manifest is
    # written after every candidate call (so after every smoke).
    for name in ("transitions", "macos-packaged"):
        assert name not in jobs
    assert set(jobs["candidate-manifest"]["needs"]) == \
        {"admit", *candidate_calls}
    for receipt, call, packaged, matrix in (
            ("transitions-darwin-arm64", "candidates-darwin-arm64", "macos-packaged-arm64", "macos"),
            ("transitions-darwin-x64", "candidates-darwin-x64", "macos-packaged-x64", "macos"),
            ("transitions-win32", "candidates-win32-bundle", "windows-packaged", "windows")):
        assert jobs[receipt]["needs"] == ["admit", call]
        assert jobs[packaged]["needs"] == receipt
        assert jobs[packaged]["strategy"]["matrix"] == \
            "${{ fromJSON(needs." + receipt + ".outputs." + matrix + ") }}"
    # B5: publish-docker starts when the docker tests pass; it does not wait
    # for the acceptance join. publish-bundles still does.
    assert jobs["publish-docker"]["needs"] == ["admit", "docker"]
    assert {"admit", "docker"} <= ancestors(jobs, "publish-docker")
    assert "acceptance" not in ancestors(jobs, "publish-docker")
    assert required <= ancestors(jobs, "publish-bundles")
    assert {"publish-docker", "publish-bundles", "publication"} <= ancestors(jobs, "complete")
    assert "promote-docker" not in jobs and "promote-bundles" not in jobs
    # The gates must run to judge a failed or skipped prerequisite, not skip with it.
    for name in ("acceptance", "publication", "complete"):
        failed = {need: {"result": "failure"} for need in ancestors(jobs, name)}
        assert gate(jobs[name]["if"], {}, failed), name


def test_claim_flags_remove_exactly_the_jobs_the_gate_expects_skipped():
    """The gate's SKIPPED_BY table and the workflow's conditions describe the same graph."""
    from scripts.releases.stable import SKIPPED_BY

    jobs = workflow("stable-release.yml")["jobs"]
    outputs = {"skipTests": "needs.admit.outputs.skip-tests",
               "skipBundles": "needs.admit.outputs.skip-bundles"}

    def needs_of(name):
        needs = jobs[name].get("needs", [])
        return [needs] if isinstance(needs, str) else needs

    def removed_by(name, flag):
        condition = str(jobs[name].get("if", ""))
        if f"{outputs[flag]} != 'true'" in condition:
            return True
        # Without a status function a job skips when any job it needs skipped.
        return ("always()" not in condition and "!cancelled()" not in condition
                and any(flag in SKIPPED_BY.get(need, ()) and removed_by(need, flag)
                        for need in needs_of(name)))

    for name, flags in SKIPPED_BY.items():
        assert name in jobs
        for flag in flags:
            assert removed_by(name, flag), f"{name} does not skip under {flag}"
    # The image publish-docker pushes and the candidates are built under
    # skip-tests; they are told to run without their own tests.
    for name in ("docker", "candidates-darwin-arm64", "candidates-darwin-x64", "candidates-win32-arm64",
                 "candidates-win32-x64", "candidates-win32-bundle", "candidates-termux"):
        assert jobs[name]["with"]["skip-tests"] == "${{ needs.admit.outputs.skip-tests == 'true' }}"
    assert {"skip-bundles", "skip-tests"} <= set(jobs["admit"]["outputs"])
    for name in ("acceptance", "publication", "complete"):
        gate = next(step for step in jobs[name]["steps"]
                    if "scripts.releases.stable gate" in step.get("run", ""))
        assert gate["env"]["SKIP_BUNDLES"] == "${{ needs.admit.outputs.skip-bundles }}"
        assert gate["env"]["SKIP_TESTS"] == "${{ needs.admit.outputs.skip-tests }}"
        gated = gate["run"].split(" gate ", 1)[1].split()
        assert set(gated) <= set(needs_of(name)), name


def test_all_applicable_ci_jobs_are_aggregated_and_desktop_e2e_stays_deferred():
    jobs = workflow("ci.yaml")["jobs"]
    checks = {name for name, job in jobs.items() if "uses" in job}
    assert checks <= set(jobs["all-checks-pass"]["needs"])
    assert not gate(jobs["e2e-desktop"]["if"], {}, {})
    assert "workflow_call" in workflow("ci.yaml")["on"]


def test_claim_custody_and_final_payload_identity_reach_every_privileged_phase():
    release = workflow("stable-release.yml")
    jobs = release["jobs"]
    # The claim is the one record of the attempt's policy; a dispatch cannot override it.
    inputs = set(release["on"]["workflow_dispatch"]["inputs"])
    assert not {"autopublish", "skip-bundles", "skip-tests"}.intersection(inputs)
    assert {"claim-tag", "claim-object", "tag", "commit", "version", "release-id", "release-epoch"} <= \
        set(jobs["admit"]["outputs"])
    for name in ("publish-bundles", *("candidates-darwin-arm64", "candidates-darwin-x64",
                                      "candidates-win32-arm64", "candidates-win32-x64",
                                      "candidates-win32-bundle", "candidates-termux")):
        call = jobs[name]["with"]
        assert call["tag"] == "${{ needs.admit.outputs.tag }}"
        assert call["claim-tag"] == "${{ needs.admit.outputs.claim-tag }}"
        assert call["claim-object"] == "${{ needs.admit.outputs.claim-object }}"
    desktop = workflow("desktop-bundled-release.yml")
    assert "release-epoch" in desktop["jobs"]["validate"]["outputs"]
    assert "HERMES_RELEASE_EPOCH" not in desktop["jobs"]["termux-deb"]["env"]
    # B3: each build group stages its own receipt before its smoke, and the
    # receipt URL and digest cross the call boundary as workflow outputs.
    assert "manifest-url" not in desktop["on"]["workflow_call"]["outputs"]
    assert "manifest-sha256" not in desktop["on"]["workflow_call"]["outputs"]
    receipts = {"darwin-arm64": "stage-receipt-darwin-arm64", "darwin-x64": "stage-receipt-darwin-x64",
                "win32-bundle": "assemble-win32-bundle"}
    for group, job in receipts.items():
        producer = desktop["jobs"][job]
        assert producer["outputs"]["receipt-url"] == "${{ steps.receipt.outputs.receipt-url }}"
        assert producer["outputs"]["receipt-sha256"] == "${{ steps.receipt.outputs.receipt-sha256 }}"
        for suffix, output in (("url", "receipt-url"), ("sha256", "receipt-sha256")):
            expected = "${{ jobs." + job + ".outputs." + output + " }}"
            assert desktop["on"]["workflow_call"]["outputs"][f"{group}-receipt-{suffix}"]["value"] == expected
    for name in ("smoke-darwin-arm64", "smoke-darwin-x64"):
        assert f"stage-receipt-{name.removeprefix('smoke-')}" in desktop["jobs"][name]["needs"]
    for call, key, output in (("candidates-darwin-arm64", "RECEIPT_URL", "darwin-arm64-receipt-url"),
                              ("candidates-darwin-arm64", "RECEIPT_SHA256", "darwin-arm64-receipt-sha256"),
                              ("candidates-darwin-x64", "RECEIPT_URL", "darwin-x64-receipt-url"),
                              ("candidates-darwin-x64", "RECEIPT_SHA256", "darwin-x64-receipt-sha256"),
                              ("candidates-win32-bundle", "RECEIPT_URL", "win32-bundle-receipt-url"),
                              ("candidates-win32-bundle", "RECEIPT_SHA256", "win32-bundle-receipt-sha256")):
        receipt_job = {"candidates-darwin-arm64": "transitions-darwin-arm64",
                       "candidates-darwin-x64": "transitions-darwin-x64",
                       "candidates-win32-bundle": "transitions-win32"}[call]
        expected = "${{ needs." + call + ".outputs." + output + " }}"
        assert jobs[receipt_job]["steps"][-1]["env"][key] == expected
    for name in ("docker", "nix"):
        assert jobs[name]["with"]["version"] == "${{ needs.admit.outputs.version }}"
    assert "version" not in jobs["pm-bundle"]["with"]
    assert "release-epoch" not in jobs["docker"]["with"]
    assert "release-epoch" not in jobs["nix"]["with"]
    assert "release-epoch" not in jobs["pm-bundle"]["with"]
    assert "release-epoch" not in jobs["publish-docker"]["with"]
    complete = jobs["complete"]["steps"]
    # A6: the green workflow validates the accepted candidate archive; the
    # final tag and the retarget move to the publication pass.
    validation = next(i for i, step in enumerate(complete)
                      if step.get("name", "").startswith("Validate the accepted candidate archive"))
    assert "DOCKER_MANIFEST_DIGEST" not in complete[validation]["env"]
    assert "RELEASE_ID" not in complete[validation]["env"]
    # B4: complete and the render read the manifest that stable-release.yml's
    # candidate-manifest job wrote.
    assert complete[validation]["env"]["CANDIDATE_MANIFEST_SHA256"] == \
        "${{ needs.candidate-manifest.outputs.manifest-sha256 }}"
    assert jobs["publish-bundles"]["with"]["manifest-sha256"] == \
        "${{ needs.candidate-manifest.outputs.manifest-sha256 }}"
    render = next(i for i, step in enumerate(complete) if step.get("name", "").startswith("Render the admitted"))
    reconcile = next(i for i, step in enumerate(complete) if step.get("name", "").startswith("Reconcile ordered"))
    assert validation < render < reconcile
    assert not any("Create the final tag" in step.get("name", "") for step in complete)


def test_docker_dev_stamp_checkout_has_release_history():
    build = workflow("docker.yml")["jobs"]["build"]
    checkout = next(step for step in build["steps"] if "actions/checkout@" in step.get("uses", ""))

    assert checkout["with"]["fetch-depth"] == "0"


def test_release_gates_extract_consumer_facing_versions():
    release_jobs = workflow("stable-release.yml")["jobs"]
    bootstrap = next(
        step["run"] for step in release_jobs["bootstrap-version"]["steps"]
        if step.get("name") == "Stamp and verify the Cargo and Tauri release identity"
    )
    assert "cargo metadata" in bootstrap
    assert "tauri.conf.json" in bootstrap
    assert "uv build --wheel --sdist" not in bootstrap

    docker = workflow("docker.yml")["jobs"]
    image_check = next(
        step["run"] for step in docker["build"]["steps"]
        if step.get("name") == "Verify release image identity"
    )
    assert '["baseVersion"]' in image_check and "$RELEASE_VERSION" in image_check

    nix = workflow("nix.yml")["jobs"]
    nix_check = next(
        step["run"] for step in nix["flake-check"]["steps"]
        if step.get("name") == "Verify release package runtime identity"
    )
    assert '"$package/bin/hermes" --version' in nix_check
    assert "actual != expected" in nix_check


def test_packaged_stamp_writers_receive_versions_without_rewriting_python_metadata():
    docker_steps = workflow("docker.yml")["jobs"]["build"]["steps"]
    assert not any(step.get("name") == "Stamp release build context" for step in docker_steps)
    write = next(step["run"] for step in docker_steps if step.get("name") == "Write install stamp")
    assert "--base-version" in write and "--display-version" in write

    nix_steps = workflow("nix.yml")["jobs"]["flake-check"]["steps"]
    prepare = next(step["run"] for step in nix_steps if step.get("name") == "Prepare isolated release source")
    assert "scripts/releases/stamping.py" in prepare


def test_publication_reconciler_has_every_recovery_trigger_and_shared_lock():
    stable = workflow("stable-release.yml")
    publication = workflow("stable-release-publication.yml")

    assert stable["concurrency"] == publication["concurrency"] == {
        "group": "stable-release", "cancel-in-progress": "false",
    }
    assert {"workflow_dispatch", "workflow_run"} <= set(publication["on"])
    assert publication["on"]["workflow_run"] == {
        "workflows": ["Stable Release"], "types": ["completed"],
    }
    reconcile = publication["jobs"]["reconcile"]
    assert reconcile["environment"] == "release-signing"
    assert publication["permissions"] == {"contents": "write", "actions": "write"}
    # Reconcile after a dispatched run of this repository that did not succeed;
    # a manual dispatch of the reconciler always runs.
    def run_of(**overrides):
        workflow_run = {"conclusion": "failure", "event": "workflow_dispatch",
                        "head_repository": {"full_name": "o/r"}, **overrides}
        return {"event_name": "workflow_run", "repository": "o/r", "event": {"workflow_run": workflow_run}}

    assert gate(reconcile["if"], {}, {}, github=run_of())
    assert gate(reconcile["if"], {}, {}, github={"event_name": "workflow_dispatch"})
    for refused in ({"conclusion": "success"}, {"event": "push"}, {"head_repository": {"full_name": "fork/r"}}):
        assert not gate(reconcile["if"], {}, {}, github=run_of(**refused)), refused
    checkout = reconcile["steps"][0]
    assert checkout["with"]["ref"] == "${{ github.event.repository.default_branch }}"
    assert checkout["with"]["persist-credentials"] == "false"
    assert not any(step.get("run", "").startswith("sleep ") for step in reconcile["steps"])


def test_docker_recovery_refuses_to_replace_a_divergent_version_tag(tmp_path):
    publish = workflow("docker.yml")["jobs"]["release-publish-manifest"]
    step = next(item for item in publish["steps"] if item.get("name") == "Create both immutable versioned manifest lists")
    digest_dir = tmp_path / "digests"
    digest_dir.mkdir()
    for variant in ("slim", "desktop"):
        for arch, digest in (("amd64", "a" * 64), ("arm64", "b" * 64)):
            artifact = digest_dir / f"docker-publish-digest-{variant}-{arch}-0.21.5"
            artifact.mkdir()
            (artifact / f"{arch}.digest").write_text(f"sha256:{digest}\n", encoding="utf-8")

    marker = tmp_path / "create-called"
    bindir = tmp_path / "bin"
    bindir.mkdir()
    docker = bindir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "if 'create' in sys.argv:\n"
        "    pathlib.Path(os.environ['CREATE_MARKER']).write_text('called')\n"
        "    raise SystemExit(0)\n"
        "print(json.dumps({'manifests': [{'digest': 'sha256:' + 'c' * 64}]}))\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c",
         step["run"].replace("/tmp/digests", str(digest_dir))], cwd=tmp_path,
        env={**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}",
             "IMAGE_NAME": "owner/repo", "RELEASE_TAG": "0.21.5",
             "CREATE_MARKER": str(marker)},
        capture_output=True, text=True, encoding="utf-8",
    )
    assert result.returncode != 0
    assert "versioned Docker manifest differs" in result.stderr
    assert not marker.exists()
