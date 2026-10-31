"""A canary is a Stable Release run on its own tag: every lane, canary destinations.

canary-release.yml dispatches stable-release.yml with ``canary.<stamp>-vX.Y.Z``. These tests pin
the parts of the stable controller that branch on that tag: admission (the claim message's lanes,
starting from every lane but the signed-package upgrade arms, and its green policy; the tag stamp
as the epoch; main only), candidate validation (canary package versions), the Docker tag, and
publication to the canary channels only.
"""
from __future__ import annotations

import copy
import json
from datetime import UTC, datetime

import pytest

from scripts.releases import lanes as lane_set
from scripts.releases import stable

BASE = "https://releases.example"
CANARY = "canary.20261021T064100Z-v1.2.3"
VERSION = "1.2.3+canary.20261021T064100Z"
COMMIT = "a" * 40
TAG_OBJECT = "b" * 40
CONTROLLER_ENV = {
    "GITHUB_REPOSITORY": "example/hermes-agent",
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_WORKFLOW_REF": "example/hermes-agent/.github/workflows/stable-release.yml@refs/heads/main",
    "GITHUB_SHA": "e" * 40,
}


def claim_message(**changes) -> str:
    """The claim ``scripts/release.py --canary`` writes into the canary tag."""
    claim = {"schema": 1, "kind": "canary", "version": "1.2.3", "commit": COMMIT,
             "lanes": sorted(lane_set.CANARY), "lanesRevision": lane_set.LANES_REVISION,
             "requireGreen": True, **changes}
    return json.dumps({key: value for key, value in claim.items() if value is not None})


def _git(*, branches="refs/remotes/origin/main", kind="tag", message=None):
    ref = f"refs/tags/{CANARY}"
    message = claim_message() if message is None else message

    def git(argv):
        if argv[1] == "ls-remote":
            return f"{TAG_OBJECT}\t{ref}\n{COMMIT}\t{ref}^{{}}"
        if argv[1:3] == ["cat-file", "-t"]:
            return kind
        if argv[1] == "rev-parse":
            return TAG_OBJECT if argv[-1] == ref else COMMIT
        if argv[1] == "for-each-ref":
            return branches if argv[-1] == "refs/remotes/origin/main" else ""
        if argv[1] == "tag":
            assert argv[2:] == ["-l", CANARY, "--format=%(contents)"]
            return message
        return ""

    return git


def test_a_canary_claim_runs_its_claimed_lanes_from_its_tag_stamp_on_main_only():
    env = {**CONTROLLER_ENV, "RELEASE_CLAIM_TAG": CANARY, "RELEASE_CLAIM_OBJECT": TAG_OBJECT}
    epoch = int(datetime(2026, 10, 21, 6, 41, tzinfo=UTC).timestamp())
    assert stable.check_claim(env, _git()) == {
        "claim_tag": CANARY, "tag": CANARY, "version": VERSION, "attempt": None, "commit": COMMIT,
        "claim_object": TAG_OBJECT, "autopublish": True, "lanes": lane_set.CANARY,
        "require_green": True, "bundle_env": {}, "claim_epoch": epoch, "policy": "canary",
    }
    # The canary's tag is its payload tag and archive alike.
    assert stable.stable_context({**env, "RELEASE_TAG": CANARY}, _git())[:2] == (CANARY, COMMIT)
    # A release branch may cut a stable release, never a canary.
    with pytest.raises(ValueError, match="not on main"):
        stable.check_claim(env, _git(branches=""))
    with pytest.raises(ValueError, match="annotated"):
        stable.check_claim(env, _git(kind="commit"))
    with pytest.raises(ValueError, match="tag moved"):
        stable.check_claim({**env, "RELEASE_CLAIM_OBJECT": "c" * 40}, _git())
    # Only the default-branch controller may admit one.
    with pytest.raises(ValueError, match="controller"):
        stable.check_claim({**env, "GITHUB_WORKFLOW_REF": env["GITHUB_WORKFLOW_REF"].replace(
            "stable-release", "desktop-bundled-release")}, _git())
    with pytest.raises(ValueError, match="calendar"):
        stable.canary_epoch("canary.20260231T000000Z-v1.2.3")


def test_the_claim_message_carries_the_canary_lanes_and_green_policy():
    env = {**CONTROLLER_ENV, "RELEASE_CLAIM_TAG": CANARY, "RELEASE_CLAIM_OBJECT": TAG_OBJECT}
    narrow = sorted(lane_set.CANARY - {"macos-intel", "termux", "termux-checks"})
    claim = stable.check_claim(env, _git(message=claim_message(lanes=narrow, requireGreen=False)))
    assert claim["lanes"] == frozenset(narrow) and claim["require_green"] is False
    # A canary is its own app identity: no claim may ask for the signed-package upgrade arms.
    with pytest.raises(ValueError, match="cannot run upgrade"):
        stable.check_claim(env, _git(message=claim_message(lanes=sorted(lane_set.NAMES))))
    for bad in (claim_message(requireGreen=None), claim_message(requireGreen="no"),
                claim_message(version="1.2.4"), claim_message(commit="c" * 40),
                claim_message(kind="stable"), claim_message(lanesRevision=1),
                claim_message(extra=True), claim_message(lanes=["windows", "docker"]), "not json"):
        with pytest.raises(ValueError, match="Canary claim metadata is invalid"):
            stable.check_claim(env, _git(message=bad))


def test_lanes_select_starts_each_policy_from_its_own_base():
    assert lane_set.select(policy="canary") == lane_set.CANARY
    assert lane_set.select() == lane_set.DEFAULT
    assert lane_set.select(policy="canary", without=["termux"]) == lane_set.CANARY - {"termux"}
    assert lane_set.select(policy="canary", only=["windows", "smoke"]) == {"windows", "smoke"}
    with pytest.raises(ValueError, match="cannot run upgrade"):
        lane_set.select(policy="canary", with_=["upgrade"])
    with pytest.raises(ValueError, match="unknown release policy"):
        lane_set.select(policy="nightly")


def test_a_canary_without_the_full_desktop_set_builds_and_tests_but_publishes_nothing():
    # A canary only advances channels, so a claim without a package the channel serves still
    # builds, smokes and runs its suites, and leaves every canary head where it is.
    for lanes in (lane_set.select(policy="canary", without=["windows"]),
                  lane_set.select(policy="canary", only=["macos-arm64", "smoke", "ci"])):
        jobs = lane_set.jobs_to_run(lanes, "canary")
        assert jobs["candidate-manifest"] and not jobs["publish-canary"]
        assert stable.gate_expectations(["publish-canary"], lanes, "canary") == {"publish-canary": "skipped"}
    assert lane_set.jobs_to_run(lane_set.select(policy="canary", without=["termux"]), "canary")["publish-canary"]


def _admitted(require_green=True):
    return {"claim_tag": CANARY, "tag": CANARY, "version": VERSION, "attempt": None, "commit": COMMIT,
            "claim_object": TAG_OBJECT, "autopublish": True, "lanes": lane_set.CANARY,
            "require_green": require_green, "claim_epoch": stable.canary_epoch(CANARY),
            "policy": "canary"}


@pytest.mark.parametrize("draft", [True, False])
@pytest.mark.parametrize("require_green", [True, False])
def test_canary_admission_needs_its_prerelease_and_emits_the_canary_policy(tmp_path, monkeypatch, draft,
                                                                           require_green):
    output, summary = tmp_path / "output", tmp_path / "summary"
    env = {"GITHUB_OUTPUT": str(output), "GITHUB_STEP_SUMMARY": str(summary), "GITHUB_REPOSITORY": "o/r"}
    monkeypatch.setattr(stable, "check_claim", lambda _env: _admitted(require_green))
    release = {"tagName": CANARY, "isDraft": draft, "isPrerelease": True, "databaseId": 7}
    monkeypatch.setattr(stable, "output", lambda _argv: json.dumps(release))
    stable.admit(env)
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert emitted["policy"] == "canary"
    assert emitted["require-green"] == ("true" if require_green else "false")
    assert emitted["docker-tag"] == CANARY
    assert lane_set.from_flags(json.loads(emitted["lanes"])) == lane_set.CANARY
    jobs = json.loads(emitted["jobs"])
    assert jobs == lane_set.jobs_to_run(lane_set.CANARY, "canary")
    assert jobs["publish-canary"] and not jobs["publish-bundles"]
    assert jobs["candidates-darwin-x64"] and jobs["candidates-termux"] and jobs["termux-checks"]
    # Old stable -> this commit runs unbundled (install-e2e); no signed-package upgrade arm runs.
    assert jobs["install-e2e"]
    assert not any(jobs[job] for job in ("transitions-win32", "windows-packaged", "macos-packaged-arm64"))
    assert "## Canary candidate" in summary.read_text(encoding="utf-8")
    # A canary owns a prerelease; a non-prerelease would be stable's draft.
    monkeypatch.setattr(stable, "output", lambda _argv: json.dumps({**release, "isPrerelease": False}))
    with pytest.raises(ValueError, match="prerelease"):
        stable.admit(env)


def test_privileged_phases_see_the_canary_channel(tmp_path, monkeypatch):
    output = tmp_path / "output"
    monkeypatch.setattr(stable, "stable_context", lambda _env: (CANARY, COMMIT, _admitted()))
    stable.verify({"GITHUB_OUTPUT": str(output)})
    emitted = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    assert emitted == {"tag": CANARY, "sha": COMMIT, "channel": "canary", "payload-version": VERSION,
                       "release-epoch": str(stable.canary_epoch(CANARY))}


def canary_candidates(tag=CANARY, *, digest="2" * 64):
    from scripts.termux.deb_version import deb_version_for_tag

    quad = stable.canary_windows_version(tag)
    packages = []
    for platform in ("windows", "macos"):
        for arch in ("x64", "arm64"):
            suffix = ".msixbundle" if platform == "windows" else ".zip"
            packages.append({
                "platform": platform, "arch": arch, "tag": tag, "commit": COMMIT,
                "identity": "NousResearch.HermesBundledCanary",
                "version": quad if platform == "windows" else stable.payload_version(tag),
                **({"executableVersion": quad, "publisher": "CN=Test", "applicationId": "HermesBundledCanary"}
                   if platform == "windows" else {"teamId": "ABCDEFGHIJ"}),
                "artifact": {"sha256": digest, "url": f"{BASE}/releases/tag/{tag}/{arch}{suffix}"},
            })
    packages.append({"platform": "termux", "arch": "aarch64", "tag": tag, "commit": COMMIT,
                     "identity": "hermes-agent", "version": deb_version_for_tag(tag),
                     "artifact": {"sha256": digest, "url": f"{BASE}/releases/tag/{tag}/deb/hermes.deb"}})
    return {"schema": 2, "tag": tag, "commit": COMMIT, "releaseEpoch": stable.canary_epoch(tag),
            "archive": tag, "packages": packages,
            "smoke_results": {job: {"result": "success"} for job in stable.SMOKE_JOBS}}


def test_canary_candidates_carry_the_canary_package_versions():
    from scripts.releases import channel_releases

    # yy.mmdd.hh.mmss, the quad scripts/msix-shared.mjs canaryPackageVersionAt stamps.
    assert stable.canary_windows_version(CANARY) == "26.1021.6.4100"
    assert channel_releases.canary_windows_version(CANARY) == "26.1021.6.4100"
    manifest = canary_candidates()
    rows = stable.validate_candidates(manifest, CANARY, COMMIT, BASE, stable.canary_epoch(CANARY), archive=CANARY)
    assert set(rows) == set(lane_set.TARGET_LANE)
    stable.require_targets_match_claim(manifest, lane_set.CANARY)
    stable.require_smokes_match_claim(manifest, lane_set.CANARY)
    # A canary stamped with stable's quad, or a Termux package without its ~canary version, is refused.
    wrong = copy.deepcopy(manifest)
    for row in wrong["packages"]:
        if row["platform"] == "windows":
            row["version"] = row["executableVersion"] = stable.stable_windows_version(wrong["releaseEpoch"])
    with pytest.raises(ValueError, match="Windows candidate version"):
        stable.validate_candidates(wrong, CANARY, COMMIT, BASE, archive=CANARY)
    plain = copy.deepcopy(manifest)
    plain["packages"][-1]["version"] = f"{VERSION}-1"
    with pytest.raises(ValueError, match="Termux"):
        stable.validate_candidates(plain, CANARY, COMMIT, BASE, archive=CANARY)
    # Only the stable shape may ever take a final receipt tag.
    stable.require_release_identity(CANARY, COMMIT)
    with pytest.raises(ValueError, match="stable payload"):
        stable.require_stable_identity(CANARY, COMMIT)


def test_the_canary_gate_expects_its_own_publication_job():
    required = ["acceptance", "publish-docker", "publish-bundles", "publish-canary"]
    assert stable.gate_expectations(required, lane_set.CANARY, "canary") == {
        "acceptance": "success", "publish-docker": "success",
        "publish-bundles": "skipped", "publish-canary": "success"}
    assert stable.gate_expectations(required, lane_set.NAMES, "stable") == {
        "acceptance": "success", "publish-docker": "success",
        "publish-bundles": "success", "publish-canary": "skipped"}
    canary_needs = {"acceptance": {"result": "success"}, "publish-docker": {"result": "success"},
                    "publish-bundles": {"result": "skipped"}, "publish-canary": {"result": "success"}}
    stable.require_gate(canary_needs, required, lanes=lane_set.CANARY, policy="canary")
    # Read as stable (a missing POLICY), the same canary run fails closed.
    with pytest.raises(ValueError, match="publish-bundles=skipped"):
        stable.require_gate(canary_needs, required, lanes=lane_set.CANARY)


def test_canary_images_get_their_own_tag_and_aliases_and_never_stable_ones():
    from scripts.releases import docker

    tag = stable.docker_tag(CANARY)
    # The canary tag is already a valid Docker tag, so it names its image unchanged.
    assert tag == CANARY and stable.docker_tag("rc.2-v1.2.3") == "rc.2-v1.2.3"
    assert docker.is_canary_image_tag(tag) and not docker.is_canary_image_tag("v1.2.3-canary.20261021T064100Z")
    assert docker.require_release_image_tag(tag) == tag
    assert docker.build_manifest(tag, COMMIT, {"amd64": "1" * 64, "arm64": "2" * 64})["tag"] == tag
    with pytest.raises(docker.DockerReleaseError):
        docker.require_stable_tag(tag)
    with pytest.raises(docker.DockerReleaseError):
        docker.promote_stable(tag, "sha256:" + "1" * 64, "v1.2.3", run=lambda _argv: "")
    with pytest.raises(docker.DockerReleaseError):
        docker.promote_canary("rc.2-v1.2.3", run=lambda _argv: "")

    digests = {"": "sha256:" + "1" * 64, "-desktop": "sha256:" + "2" * 64}
    aliases = dict(digests)
    created = []

    def run(argv):
        if argv[3] == "inspect":
            reference = argv[4].split(":", 1)[1]
            for suffix in ("-desktop", ""):
                if reference.endswith(suffix):
                    name = reference[: len(reference) - len(suffix)] if suffix else reference
                    break
            return digests[suffix] if name == tag else aliases.get(suffix) if name == "canary" else "missing"
        created.append(argv)
        return ""

    docker.promote_canary(tag, run=run, sleep=lambda _seconds: None)
    targets = [argv[argv.index("-t") + 1] for argv in created]
    assert targets == [f"{docker.IMAGE}:canary", f"{docker.IMAGE}:canary-desktop"]
    assert not any("stable" in target or "latest" in target for target in targets)


def test_canary_promotion_writes_only_canary_feeds(tmp_path, monkeypatch):
    from scripts.bundles import release_artifacts
    from scripts.releases import r2

    manifest = canary_candidates()
    manifest["files"] = [{"path": row["artifact"]["url"].rsplit(f"/{CANARY}/", 1)[1],
                          "url": row["artifact"]["url"], "sha256": row["artifact"]["sha256"]}
                         for row in manifest["packages"]]
    materialized, appinstallers, pointers, finalized = [], [], [], []
    monkeypatch.setattr(release_artifacts, "materialize", lambda _m, root, **kw: materialized.append(kw))
    monkeypatch.setattr(r2, "finalize", lambda **kw: finalized.append(kw))

    def appinstaller(root, **kw):
        appinstallers.append(kw)
        (root / "canary.appinstaller").write_text("<AppInstaller/>", encoding="utf-8")

    monkeypatch.setattr(release_artifacts, "publish_canary_appinstaller", appinstaller)
    monkeypatch.setattr(release_artifacts, "_pointer_publisher", lambda tag, base: (
        lambda key, file, upload=True: pointers.append((key, upload))))
    apt = tmp_path / "apt" / "dists" / "hermes-canary"
    apt.mkdir(parents=True)
    (apt / "InRelease").write_text("signed", encoding="utf-8")
    release_artifacts.promote(manifest, tmp_path, BASE)

    assert release_artifacts.release_channel(CANARY) == "canary"
    assert release_artifacts.release_channel("v1.2.3") == "stable"
    assert finalized == [{"tag": CANARY, "dir": tmp_path, "archive": CANARY, "arches": ("arm64", "x64"),
                          "verify_bytes": False}]
    # The canary feed keeps its accepted bundle beside it; the descriptor is read back, not re-uploaded.
    assert materialized[1]["only"] == {"x64.msixbundle"}
    assert appinstallers[0]["tag"] == CANARY and appinstallers[0]["version"] == "26.1021.6.4100"
    assert pointers == [("releases/win32/canary/canary.appinstaller", False),
                        ("releases/termux/canary/dists/hermes-canary/InRelease", True)]
    assert not any("stable" in key for key, _upload in pointers)


def _custody_run(*, default="main"):
    ref = f"refs/tags/{CANARY}"

    def run(argv):
        if argv[:2] == ["gh", "api"]:
            return default
        if argv[:3] == ["gh", "release", "view"]:
            return json.dumps({"tagName": CANARY, "isDraft": True, "isPrerelease": True})
        if argv[1] == "rev-parse":
            return TAG_OBJECT if argv[-1].endswith("^{tag}") else COMMIT
        if argv[1] == "ls-remote":
            return f"{TAG_OBJECT}\t{ref}\n{COMMIT}\t{ref}^{{}}"
        return ""

    return run


def test_the_canary_controller_custody_is_the_stable_pipeline_after_its_accepted_candidate():
    from hermes_cli.release_channels import ChannelError
    from scripts.releases import channel_releases

    needs = {"admit": {"result": "success"}, "candidate-manifest": {"result": "success"},
             "acceptance": {"result": "failure"}}
    # Without require-green, a red suite never blocks the canary's publication.
    env = {**CONTROLLER_ENV, "RELEASE_TAG": CANARY, "RELEASE_TAG_OBJECT": TAG_OBJECT, "REQUIRE_GREEN": "false",
           "RELEASE_COMMIT": COMMIT, "RELEASE_NEEDS": json.dumps(needs), "GITHUB_ACTIONS": "true"}
    assert channel_releases.admit_transaction("canary-release", env, run=_custody_run()) == (CANARY, COMMIT)
    # With it (or without the flag at all: custody fails closed), acceptance must have passed.
    for green in ({"REQUIRE_GREEN": "true"}, {"REQUIRE_GREEN": ""}):
        with pytest.raises(ValueError, match="acceptance=failure"):
            channel_releases.admit_transaction("canary-release", {**env, **green}, run=_custody_run())
    green = {**env, "REQUIRE_GREEN": "true",
             "RELEASE_NEEDS": json.dumps({**needs, "acceptance": {"result": "success"}})}
    assert channel_releases.admit_transaction("canary-release", green, run=_custody_run()) == (CANARY, COMMIT)
    failed = {**needs, "candidate-manifest": {"result": "failure"}}
    with pytest.raises(ValueError, match="candidate-manifest=failure"):
        channel_releases.admit_transaction("canary-release", {**env, "RELEASE_NEEDS": json.dumps(failed)},
                                           run=_custody_run())
    # The controller ref of another branch is not the controller.
    with pytest.raises(ChannelError, match="existing release workflow"):
        channel_releases.admit_transaction("canary-release", env, run=_custody_run(default="develop"))
    # Nor is any other workflow: the direct desktop-bundled-release dispatch once published canaries
    # from its native legs alone, and must not even with every one of those legs green.
    legacy_needs = {name: {"result": "success"} for name in (
        "validate", "build-win32-x64", "build-win32-arm64", "build-darwin-arm64", "build-darwin-x64",
        "build-linux-x64", "build-linux-arm64", "builds-table", "assemble-win32-bundle",
        "smoke-darwin-arm64", "smoke-darwin-x64", "smoke-win32-arm64", "smoke-win32-x64",
        "publish-win32-updater", "publish-darwin-updater")}
    legacy = {**env, "RELEASE_NEEDS": json.dumps(legacy_needs),
              "GITHUB_WORKFLOW_REF": env["GITHUB_WORKFLOW_REF"].replace("stable-release", "desktop-bundled-release")}
    with pytest.raises(ChannelError, match="existing release workflow"):
        channel_releases.admit_transaction("canary-release", legacy, run=_custody_run())
    # A stable tag can never ride the canary policy.
    with pytest.raises(ChannelError, match="policy/tag mismatch"):
        channel_releases.admit_transaction("canary-release", {**env, "RELEASE_TAG": "v1.2.3"},
                                           run=_custody_run())


def test_the_pruner_keeps_the_live_canary_however_old(monkeypatch, capsys):
    from scripts.releases import r2

    live, stale = "canary.20260101T000000Z-v1.2.3", "canary.20251201T000000Z-v1.2.2"
    keys = [f"releases/tag/{live}/x64.msixbundle", f"releases/termux/canary/pool/{live}/hermes.deb",
            f"releases/tag/{stale}/x64.msixbundle", "releases/canary/release-candidates.json"]
    monkeypatch.setattr(r2, "credentials", lambda: (None, "https://r2.example", "bucket"))
    monkeypatch.setattr(r2, "list_objects", lambda **_kw: {"keys": keys, "lastModified": {}})
    pointer = json.dumps({"archive": live})
    monkeypatch.setattr(r2, "get_object", lambda *_args, **_kw: pointer)
    r2.prune(14, dry_run=True, now_epoch=datetime(2026, 10, 21, tzinfo=UTC).timestamp())
    doomed = capsys.readouterr().out
    assert stale in doomed and live not in doomed
    # An unreadable pointer could hide the live canary: refuse to prune at all.
    monkeypatch.setattr(r2, "get_object", lambda *_args, **_kw: None)
    with pytest.raises(RuntimeError, match="live canary"):
        r2.prune(14, dry_run=True, now_epoch=datetime(2026, 10, 21, tzinfo=UTC).timestamp())
