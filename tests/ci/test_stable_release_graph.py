"""The release workflow's dependency graph enforces publication ordering."""
import json
import os
import subprocess
from pathlib import Path

import pytest
from ruamel.yaml import YAML

from scripts.releases import lanes as lane_set

from tests.ci.desktop_release_roles import evaluate, gate
from tests.ci.workflow_steps import is_checkout

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
        # The packaged leg names the claim commit, so it reads admit directly.
        assert jobs[packaged]["needs"] == ["admit", receipt]
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


LANE_SETS = {
    "default": lane_set.DEFAULT,
    "everything": frozenset(lane_set.NAMES),
    "windows and docker, untested": lane_set.select(only=["windows", "docker"]),
    "docker only": lane_set.select(only=["docker"]),
    "no mac": lane_set.select(without=["macos-arm64"]),
    "no tests": lane_set.select(without=lane_set.NAMES[5:]),
    "nothing": frozenset(),
    # Each lane alone and each lane missing, so no job's condition can borrow another lane's flag.
    **{f"only {name}": frozenset({name}) for name in lane_set.NAMES},
    **{f"everything but {name}": frozenset(lane_set.NAMES) - {name} for name in lane_set.NAMES},
}


@pytest.mark.parametrize("policy", lane_set.POLICIES)
@pytest.mark.parametrize("lanes", LANE_SETS.values(), ids=LANE_SETS)
def test_every_lane_controlled_job_runs_exactly_when_the_lane_table_says(lanes, policy):
    """The workflow's `if` conditions and the gate's expectations describe the same graph."""
    jobs = workflow("stable-release.yml")["jobs"]
    runs = lane_set.jobs_to_run(lanes, policy)
    admit = {"result": "success", "outputs": {"lanes": json.dumps(lane_set.flags(lanes)),
                                              "jobs": json.dumps(runs), "policy": policy}}

    def result(name):
        if name == "admit":
            return admit
        # What the job does under these lanes is what the table says (a job outside it never skips).
        return {"result": "success" if runs.get(name, True) else "skipped",
                "outputs": {"baseline": "published"}}

    for name, wanted in runs.items():
        assert name in jobs, name
        needs = {need: result(need) for need in direct_needs(jobs, name)}
        assert gate(jobs[name].get("if", "true"), {}, needs) == wanted, (name, sorted(lanes))
    # A job that is not lane-controlled never skips with the lanes.
    for name in set(jobs) - set(runs) - {"admit", "acceptance", "publication", "complete",
                                         "candidate-manifest", "publish-bundles"}:
        assert "outputs.jobs" not in str(jobs[name].get("if", "")), name


def test_a_canary_runs_every_stable_job_with_every_lane_and_differs_only_in_publication():
    """canary-release.yml runs this same workflow: nothing to keep in sync. With the canary's lanes
    (every lane but the signed-package upgrade arms) the canary runs exactly the jobs a stable
    release with every lane runs, except those arms and the other policy's publication job."""
    jobs = workflow("stable-release.yml")["jobs"]
    # The canary app cannot upgrade a stable install; install-e2e tests old stable -> this commit.
    assert lane_set.CANARY == frozenset(lane_set.NAMES) - {"upgrade"}
    canary = lane_set.jobs_to_run(lane_set.CANARY, "canary")
    assert set(canary) <= set(jobs)
    upgrade = {job for job, needs in lane_set.JOB_LANES.items() if "upgrade" in needs}
    assert upgrade == {"transitions-darwin-arm64", "transitions-darwin-x64", "transitions-win32",
                       "macos-packaged-arm64", "macos-packaged-x64", "windows-packaged"}
    stable_all = lane_set.jobs_to_run(lane_set.NAMES, "stable")
    # The publication jobs that run under exactly one of the two policies.
    either = {job for job, policies in lane_set.POLICY_JOBS.items() if ("stable" in policies) != ("canary" in policies)}
    assert either == {"publish-bundles", "publish-canary"}
    assert {job for job in canary if canary[job] != stable_all[job]} == either | upgrade
    # Every other build and test job runs: macOS Intel and Termux included, which a default claim
    # leaves out, and install-e2e, which updates released stable tags to the canary commit.
    not_canary = {job for job, policies in lane_set.POLICY_JOBS.items() if "canary" not in policies}
    assert not_canary == {"publish-bundles", "publish-commit"}
    assert {job for job, runs in canary.items() if not runs} == not_canary | upgrade
    for opted_out in ("candidates-darwin-x64", "candidates-termux", "termux-checks"):
        assert canary[opted_out] and not lane_set.jobs_to_run(lane_set.DEFAULT)[opted_out], opted_out
    with pytest.raises(ValueError, match="policy"):
        lane_set.jobs_to_run(lane_set.CANARY, "preview")
    # The run is titled by policy; stable keeps the title the sequencer matches runs by.
    from scripts.releases.stable import claim_run_title
    title = workflow("stable-release.yml")["run-name"]
    prefix, rest = title.split(" release ", 1)
    assert rest == "${{ inputs.tag }}"
    assert evaluate(prefix, {"tag": "rc.1-v1.2.3"}, {}) == "Stable"
    assert evaluate(prefix, {"tag": "canary.20261021T064100Z-v1.2.3"}, {}) == "Canary"
    assert claim_run_title("rc.1-v1.2.3") == "Stable release rc.1-v1.2.3"


def test_a_commit_build_runs_the_default_jobs_but_upgrade_tests_and_publishes_nowhere():
    """release.py --build-commit dispatches this same workflow on commit.<stamp>-vX.Y.Z. A commit
    build runs the stable default lanes minus what it can never run (no upgrade test of any kind;
    commit builds don't support updating), and its only publication job renders its builds page."""
    jobs = workflow("stable-release.yml")["jobs"]
    assert lane_set.COMMIT == lane_set.DEFAULT - lane_set.COMMIT_REFUSED
    assert lane_set.COMMIT_REFUSED == {"upgrade", "install-e2e", "windows-store"}
    commit = lane_set.jobs_to_run(lane_set.COMMIT, "commit")
    assert set(commit) <= set(jobs)
    stable = lane_set.jobs_to_run(lane_set.DEFAULT, "stable")
    # Relational: every job that is neither a publication job nor needs a refused lane runs exactly
    # as it does in a default stable release, so a lane added later runs in commit builds too.
    refused = {job for job, needs in lane_set.JOB_LANES.items() if needs & lane_set.COMMIT_REFUSED}
    for job in set(commit) - set(lane_set.POLICY_JOBS) - refused:
        assert commit[job] == stable[job], job
    assert not any(commit[job] for job in refused)
    assert {job for job, runs in commit.items() if runs and job in lane_set.POLICY_JOBS} == {"publish-commit"}
    assert {job for job in commit if commit[job] != stable[job]} == {
        "publish-bundles", "publish-docker", "publish-commit", "install-e2e",
        "transitions-darwin-arm64", "transitions-win32", "macos-packaged-arm64", "windows-packaged"}
    # Every refused lane is refused outright, not just off by default.
    for lane in lane_set.COMMIT_REFUSED:
        with pytest.raises(ValueError, match="commit claim cannot run"):
            lane_set.select(with_=[lane], policy="commit")
    # Neither a stable release nor a canary runs it.
    for policy, lanes in (("stable", lane_set.NAMES), ("canary", lane_set.CANARY)):
        assert not lane_set.jobs_to_run(lanes, policy)["publish-commit"], policy

    publish = jobs["publish-commit"]
    assert publish["needs"] == ["admit", "candidate-manifest", "acceptance"]
    manifest = {"result": "success"}

    def admit_with(green: str, runs: dict = commit) -> dict:
        return {"result": "success", "outputs": {"jobs": json.dumps(runs), "policy": "commit",
                                                 "require-green": green}}

    for green, red_suite_renders in (("false", True), ("true", False)):
        needs = {"admit": admit_with(green), "candidate-manifest": manifest}
        assert gate(publish["if"], {}, {**needs, "acceptance": {"result": "failure"}}) is red_suite_renders
        assert gate(publish["if"], {}, {**needs, "acceptance": {"result": "success"}})
        for result in ("failure", "skipped", "cancelled"):
            assert not gate(publish["if"], {}, {**needs, "candidate-manifest": {"result": result},
                                                "acceptance": {"result": "success"}}), result
    # A commit build never reaches stable or canary publication, the Docker push or the Store.
    accepted = {"admit": admit_with("false"), "acceptance": {"result": "success"}, "candidate-manifest": manifest,
                "docker": {"result": "success"}}
    for name in ("publish-bundles", "publish-canary", "publish-docker"):
        assert not gate(jobs[name]["if"], {}, accepted), name
    store = workflow("desktop-bundled-release.yml")["jobs"]["stable-store"]["if"]
    assert not gate(store, {"release-phase": "publish", "skip-store": False,
                            "tag": "commit.20261026T180000Z-v1.2.3"}, {"validate": {"result": "success"}})
    # Its page is rendered from the pinned accepted candidate into its own claim archive only.
    render = next(step for step in publish["steps"] if "render-builds-table.py" in step.get("run", ""))
    for flag in ('--archive "$RELEASE_CLAIM_TAG"', '--candidate-manifest-sha256 "$CANDIDATE_MANIFEST_SHA256"',
                 '--candidate-commit "$RELEASE_COMMIT"', '--bundle-env-json "$CLAIM_BUNDLE_ENV"',
                 '--summary-out "$GITHUB_STEP_SUMMARY"'):
        assert flag in render["run"], flag
    assert publish["env"]["CLAIM_BUNDLE_ENV"] == "${{ needs.admit.outputs.bundle-env }}"
    complete_render = next(step for step in jobs["complete"]["steps"]
                           if step.get("name") == "Render the admitted candidate smoke results")
    assert not gate(complete_render["if"], {}, {"admit": {"result": "success", "outputs": {
        "lanes": json.dumps(lane_set.flags(lane_set.COMMIT)), "policy": "commit"}}}, job_if=False)
    # Publication judges it like every other publication job.
    assert "publish-commit" in jobs["publication"]["needs"]
    gate_step = next(step for step in jobs["publication"]["steps"] if "stable gate" in step.get("run", ""))
    assert "publish-commit" in gate_step["run"].split()
    # Its bundle defaults reach every candidate call from the admitted claim only.
    assert jobs["admit"]["outputs"]["bundle-env"] == "${{ steps.admit.outputs.bundle-env }}"
    for name in lane_set.CANDIDATE_CALLS:
        assert jobs[name]["with"]["bundle_env"] == "${{ needs.admit.outputs.bundle-env }}", name
    # Titled Commit, and each commit build runs on its own (never queued behind a release).
    title = workflow("stable-release.yml")["run-name"]
    prefix = title.split(" release ", 1)[0]
    assert evaluate(prefix, {"tag": "commit.20261026T180000Z-v1.2.3"}, {}) == "Commit"
    group = workflow("stable-release.yml")["concurrency"]["group"]
    assert evaluate(group, {"tag": "commit.20261026T180000Z-v1.2.3"}, {}) == \
        "commit.20261026T180000Z-v1.2.3"
    assert evaluate(group, {"tag": "canary.20261026T180000Z-v1.2.3"}, {}) == "canary-release-pipeline"
    assert evaluate(group, {"tag": "rc.1-v1.2.3"}, {}) == "stable-release"


def test_canary_publication_follows_its_claims_green_policy_and_never_touches_stable():
    jobs = workflow("stable-release.yml")["jobs"]
    canary = lane_set.jobs_to_run(lane_set.CANARY, "canary")
    publish = jobs["publish-canary"]
    assert publish["needs"] == ["admit", "candidate-manifest", "acceptance"]
    # Custody re-checks the same flag the job's `if` reads.
    assert publish["env"]["REQUIRE_GREEN"] == "${{ needs.admit.outputs.require-green }}"
    assert jobs["admit"]["outputs"]["require-green"] == "${{ steps.admit.outputs.require-green }}"
    manifest = {"result": "success"}

    def admit_with(green: str) -> dict:
        return {"result": "success", "outputs": {"jobs": json.dumps(canary), "policy": "canary",
                                                 "require-green": green}}

    for green, red_suite_publishes in (("false", True), ("true", False)):
        needs = {"admit": admit_with(green), "candidate-manifest": manifest}
        # Without require-green a red suite reports in the same run and does not hold the canary
        # back; with it, publication waits for acceptance exactly as a stable release does.
        assert gate(publish["if"], {}, {**needs, "acceptance": {"result": "failure"}}) is red_suite_publishes
        assert gate(publish["if"], {}, {**needs, "acceptance": {"result": "success"}})
        # A failed or skipped candidate (a build or smoke went red) publishes nothing either way.
        for result in ("failure", "skipped", "cancelled"):
            assert not gate(publish["if"], {}, {**needs, "candidate-manifest": {"result": result},
                                                "acceptance": {"result": "success"}}), result
        assert not gate(publish["if"], {}, {**needs, "acceptance": {"result": "success"}}, cancelled=True)
    admit = admit_with("false")
    # A stable release never runs it, and a canary never runs stable publication (Store included).
    stable_admit = {"result": "success", "outputs": {"jobs": json.dumps(lane_set.jobs_to_run(lane_set.NAMES))}}
    assert not gate(publish["if"], {}, {"admit": stable_admit, "candidate-manifest": manifest,
                                        "acceptance": {"result": "success"}})
    bundles = jobs["publish-bundles"]
    accepted = {"admit": admit, "acceptance": {"result": "success"}, "candidate-manifest": manifest}
    assert not gate(bundles["if"], {}, accepted)
    store = workflow("desktop-bundled-release.yml")["jobs"]["stable-store"]["if"]
    publish_phase = {"release-phase": "publish", "skip-store": False}
    assert gate(store, {**publish_phase, "tag": "rc.1-v1.2.3"}, {"validate": {"result": "success"}})
    assert not gate(store, {**publish_phase, "tag": "canary.20261021T064100Z-v1.2.3"},
                    {"validate": {"result": "success"}})
    # Publication runs the canary controller under the job's own needs, with the pinned candidate.
    steps = publish["steps"]
    assert any("release_artifacts publish" in step.get("run", "") for step in steps)
    advance = next(step for step in steps if "channel_releases canary" in step.get("run", ""))
    assert publish["env"]["RELEASE_NEEDS"] == "${{ toJSON(needs) }}"
    assert publish["env"]["CANDIDATE_MANIFEST_SHA256"] == "${{ needs.candidate-manifest.outputs.manifest-sha256 }}"
    assert publish["env"]["RELEASE_TAG_OBJECT"] == "${{ needs.admit.outputs.claim-object }}"
    assert advance["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert publish["permissions"] == {"contents": "write"}
    # The stable publication pass (stable heads, stable/latest aliases, Store commit) is stable-only;
    # the canary aliases move only for a canary whose image was pushed.
    complete = jobs["complete"]["steps"]
    canary_needs = {"admit": admit, "publish-docker": {"result": "success"},
                    "publish-canary": {"result": "success"}}
    for step in complete:
        name = step.get("name", "")
        if name.startswith(("Reconcile ordered", "Set up Docker Buildx for ordered", "Log in to Docker Hub for ordered")):
            assert not gate(step["if"], {}, canary_needs, job_if=False), name
            assert gate(step["if"], {}, {"admit": {"result": "success", "outputs": {"policy": "stable"}}},
                        job_if=False), name
        if "canary aliases" in name or "canary Docker aliases" in name:
            # They follow the published canary: a red suite the canary's claim does not wait for
            # (the gate step failing) still moves them, ...
            assert gate(step["if"], {}, canary_needs, job_if=False, failed=True), name
            # ... but never ahead of the canary itself. publish-canary is skipped when a
            # require-green claim's acceptance is red, a package failed, or the lanes are narrowed.
            for held in ("skipped", "failure", "cancelled"):
                assert not gate(step["if"], {}, {**canary_needs, "publish-canary": {"result": held}},
                                job_if=False, failed=True), (name, held)
            assert not gate(step["if"], {}, {**canary_needs, "publish-docker": {"result": "failure"}},
                            job_if=False), name


def test_lane_inputs_reach_the_workflows_they_configure():
    jobs = workflow("stable-release.yml")["jobs"]
    # Tests the image and the candidates run for themselves follow their own lane.
    assert jobs["docker"]["with"]["skip-tests"] == "${{ !fromJSON(needs.admit.outputs.lanes)['docker-tests'] }}"
    for name in ("candidates-darwin-arm64", "candidates-darwin-x64", "candidates-win32-arm64",
                 "candidates-win32-x64", "candidates-win32-bundle", "candidates-termux"):
        assert jobs[name]["with"]["skip-tests"] == "${{ !fromJSON(needs.admit.outputs.lanes).smoke }}", name
    # Which Windows packages the shared builds make follows the two Windows lanes, and the Store
    # submission follows the Store lane.
    store_off = "${{ !fromJSON(needs.admit.outputs.lanes)['windows-store'] }}"
    for name in ("candidates-win32-arm64", "candidates-win32-x64", "candidates-win32-bundle"):
        assert jobs[name]["with"]["skip-signed"] == "${{ !fromJSON(needs.admit.outputs.lanes).windows }}", name
        assert jobs[name]["with"]["skip-store"] == store_off, name
        # The MSIX smokes run in the bundle call, so the bundle never waits for them.
        assert jobs[name]["with"]["windows-smokes-with-bundle"] == "true", name
    assert jobs["publish-bundles"]["with"]["skip-store"] == store_off
    desktop = workflow("desktop-bundled-release.yml")["jobs"]
    assert "inputs.skip-store" in desktop["stable-store"]["if"]
    assert {"lanes", "jobs"} <= set(jobs["admit"]["outputs"])
    for name in ("acceptance", "publication", "complete"):
        gate_step = next(step for step in jobs[name]["steps"]
                         if "scripts.releases.stable gate" in step.get("run", ""))
        assert gate_step["env"] == {"RELEASE_NEEDS": "${{ toJSON(needs) }}",
                                    "LANES": "${{ needs.admit.outputs.lanes }}",
                                    "POLICY": "${{ needs.admit.outputs.policy }}"}
        gated = gate_step["run"].split(" gate ", 1)[1].split()
        assert set(gated) <= set(direct_needs(jobs, name)), name


def test_a_store_only_build_is_handed_no_signing_variables():
    """after-pack and electron-builder sign whenever AZURE_SIGN_* are set; the Azure login is skipped
    with the signed lane, so the build step must see them empty."""
    jobs = workflow("desktop-bundled-release.yml")["jobs"]
    step = next(step for step in jobs["build-win32-x64-release"]["steps"] if step.get("name") == "Build and package")
    azure = {key: value for key, value in step["env"].items() if key.startswith("AZURE_") and key != "AZURE_TOKEN_CREDENTIALS"}
    assert set(azure) == {"AZURE_SIGN_ENDPOINT", "AZURE_SIGN_ACCOUNT", "AZURE_SIGN_PROFILE", "AZURE_SIGN_PUBLISHER",
                          "AZURE_CLIENT_ID", "AZURE_TENANT_ID"}
    for key, value in azure.items():
        assert value == f"${{{{ inputs.skip-signed != true && vars.{key} || '' }}}}", key


def test_admit_emits_the_same_lanes_and_jobs_the_gate_reads(tmp_path):
    from scripts.releases import stable

    output = tmp_path / "output"
    summary = tmp_path / "summary"
    admitted = {"claim_tag": "rc.1-v1.2.3", "claim_object": "a" * 40, "tag": "v1.2.3", "commit": "b" * 40,
                "version": "1.2.3", "attempt": 1, "claim_epoch": 1, "autopublish": False,
                "lanes": lane_set.select(only=["windows", "docker"]), "require_green": True,
                "policy": "stable"}
    env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_REPOSITORY": "o/r"}
    from unittest import mock
    with mock.patch.object(stable, "check_claim", return_value=admitted), \
            mock.patch.object(stable, "output", return_value=json.dumps(
                {"tagName": "rc.1-v1.2.3", "isDraft": True, "isPrerelease": False, "databaseId": 7})):
        stable.admit(env)
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert lane_set.from_flags(json.loads(emitted["lanes"])) == admitted["lanes"]
    assert json.loads(emitted["jobs"]) == lane_set.jobs_to_run(admitted["lanes"])
    assert emitted["policy"] == "stable" and emitted["docker-tag"] == "rc.1-v1.2.3"
    assert "on: docker, windows" in summary.read_text(encoding="utf-8")


def test_a_skipped_upgrade_arm_never_skips_a_gated_publish_job():
    """The signed-package upgrade arms skip when there is no published baseline (the first bundled
    release). A gated job downstream of one without a status function inherits that skip, and the
    publication gate then refuses it: rc.11-v0.21.7 blocked on publish-bundles=skipped."""
    from scripts.releases.stable import PACKAGED_BY

    jobs = workflow("stable-release.yml")["jobs"]

    for name, job in jobs.items():
        if set(PACKAGED_BY) & ancestors(jobs, name):
            condition = str(job.get("if", ""))
            assert "always()" in condition or "!cancelled()" in condition, (
                f"{name} skips whenever an upgrade arm skips")


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
    assert not {"autopublish", "lanes", "skip-bundles", "skip-tests"}.intersection(inputs)
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
    checkout = next(step for step in build["steps"] if is_checkout(step))

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
    # baseVersion is the stable core: a canary's +canary.<stamp> lives only in displayVersion.
    assert '["baseVersion"]' in image_check and '"${RELEASE_VERSION%%+*}"' in image_check

    nix = workflow("nix.yml")["jobs"]
    nix_check = next(
        step["run"] for step in nix["flake-check"]["steps"]
        if step.get("name") == "Verify release package runtime identity"
    )
    assert '"$package/bin/hermes" --version' in nix_check
    assert "actual != expected" in nix_check
    # A path: flake has no git metadata; without ?rev= the stamp has no commit
    # and the packaged CLI reports "unknown".
    assert 'release-source?rev=$CANDIDATE_COMMIT' in nix_check
    flake_check = next(step["run"] for step in nix["flake-check"]["steps"] if step.get("name") == "nix flake check")
    assert "release-source?rev={1}" in flake_check and "env.CANDIDATE_COMMIT" in flake_check


def test_versioned_docker_images_belong_to_one_attempt():
    # The final vX.Y.Z does not exist until publish; a second attempt pushing
    # under it collides with the first attempt's immutable manifest list.
    jobs = workflow("stable-release.yml")["jobs"]
    # docker-tag is the claim tag for stable (stable.docker_tag); a canary's `+` becomes `-`.
    for name in ("docker", "publish-docker"):
        assert jobs[name]["with"]["tag"] == "${{ needs.admit.outputs.docker-tag }}"


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

    assert publication["concurrency"] == {"group": "stable-release", "cancel-in-progress": "false"}
    # A stable run shares the reconciler's lock; a canary run of the same workflow never does.
    group = stable["concurrency"]["group"]
    assert stable["concurrency"]["cancel-in-progress"] == "false"
    assert evaluate(group, {"tag": "rc.1-v1.2.3"}, {}) == "stable-release"
    assert evaluate(group, {"tag": "canary.20261021T064100Z-v1.2.3"}, {}) == "canary-release-pipeline"
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
    for refused in ({"conclusion": "success"}, {"event": "push"}, {"head_repository": {"full_name": "fork/r"}},
                    # A red canary is a Stable Release run too; it must never be reconciled or rerun.
                    {"display_title": "Canary release canary.20261021T064100Z-v1.2.3"},
                    # So is a commit build: it has no stable draft or head either.
                    {"display_title": "Commit release commit.20261026T180000Z-v1.2.3"}):
        assert not gate(reconcile["if"], {}, {}, github=run_of(**refused)), refused
    assert gate(reconcile["if"], {}, {}, github=run_of(display_title="Stable release rc.1-v1.2.3"))
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
        check=False,
    )
    assert result.returncode != 0
    assert "versioned Docker manifest differs" in result.stderr
    assert not marker.exists()


def test_every_run_step_receives_the_variables_it_reads():
    """A run step reading $VAR it was never given gets an empty string: rc.14-v0.21.7 passed
    every build and gate, then 'complete' died rendering the builds table without
    CLOUDFLARE_R2_PUBLIC_URL or RELEASE_CLAIM_TAG."""
    import re

    runner = {"GITHUB_REPOSITORY", "GITHUB_OUTPUT", "GITHUB_STEP_SUMMARY", "GITHUB_ENV", "GITHUB_SHA",
              "GITHUB_RUN_ID", "GITHUB_WORKSPACE", "RUNNER_TEMP", "HOME", "PATH"}
    for name, job in workflow("stable-release.yml")["jobs"].items():
        given = set(job.get("env") or {})
        for step in job.get("steps", []):
            used = set(re.findall(r"\$\{?([A-Z][A-Z0-9_]+)", step.get("run", ""))) - runner
            missing = used - given - set(step.get("env") or {})
            assert not missing, f"{name} / {step.get('name') or step['run'][:40]}: {sorted(missing)}"


def test_the_builds_table_render_cannot_block_the_release_and_has_its_r2_inputs():
    """The table is presentation: it never fails 'Stable release is green', and it gets the
    R2 credentials its listing and page writes need."""
    step = next(step for step in workflow("stable-release.yml")["jobs"]["complete"]["steps"]
                if step.get("name") == "Render the admitted candidate smoke results")
    assert str(step.get("continue-on-error")).lower() == "true"  # base YAML loader: a string
    assert {"CLOUDFLARE_R2_ACCOUNT_ID", "CLOUDFLARE_R2_ACCESS_KEY_ID", "CLOUDFLARE_R2_SECRET_ACCESS_KEY",
            "CLOUDFLARE_R2_BUCKET", "CLOUDFLARE_R2_PUBLIC_URL", "RELEASE_CLAIM_TAG"} <= set(step["env"])


@pytest.mark.parametrize("arch", ["arm64", "x64"])
def test_macos_smoke_runs_without_a_receipt_outside_the_candidate_phase(arch):
    """Only a candidate stages a receipt; a canary's skipped stage-receipt must not skip the smoke,
    or its publish gate (which needs every job to succeed) fails on every canary."""
    smoke = workflow("desktop-bundled-release.yml")["jobs"][f"smoke-darwin-{arch}"]["if"]
    base = {"validate": {"result": "success", "outputs": {"sha": "a" * 40, f"darwin-{arch}": "true"}},
            f"build-darwin-{arch}": {"result": "success"}}
    inputs = {"release-phase": "", "upload_release": True, "skip-tests": False}
    skipped = {**base, f"stage-receipt-darwin-{arch}": {"result": "skipped"}}
    staged = {**base, f"stage-receipt-darwin-{arch}": {"result": "success"}}
    assert gate(smoke, inputs, skipped)
    assert not gate(smoke, {**inputs, "release-phase": "candidate"}, skipped)
    assert gate(smoke, {**inputs, "release-phase": "candidate"}, staged)
