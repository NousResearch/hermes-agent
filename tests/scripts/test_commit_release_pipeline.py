"""A commit build is a Stable Release run on its own claim tag: default lanes minus every upgrade
test, the commit identity, and no publication beyond its own builds page.

release.py --build-commit claims ``commit.<stamp>-vX.Y.Z`` on any pushed commit and dispatches
stable-release.yml. These tests pin what branches on that tag: the claim message (lanes, green
policy, bundle defaults), admission (any pushed commit, maintainers only, no GitHub release),
the tag-mode package identity (commit-build stamps, stamped Windows quad and epoch), the writers
that must refuse a commit tag (channels, feeds, Docker publication), retention and the CLI.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from scripts.releases import lanes as lane_set
from scripts.releases import stable
from tests.scripts.test_release_tags import release_repo  # (a fixture)

STAMP = "20261026T180000Z"
TAG = f"commit.{STAMP}-v1.2.3"
VERSION = f"1.2.3+commit.{STAMP}"
COMMIT = "a" * 40
TAG_OBJECT = "b" * 40
CONTROLLER_ENV = {
    "GITHUB_REPOSITORY": "example/hermes-agent",
    "GITHUB_EVENT_NAME": "workflow_dispatch",
    "GITHUB_REF": "refs/heads/main",
    "GITHUB_WORKFLOW_REF": "example/hermes-agent/.github/workflows/stable-release.yml@refs/heads/main",
    "GITHUB_SHA": "e" * 40,
    "GITHUB_ACTOR": "maintainer",
}


def claim(**changes) -> dict:
    """The claim ``release.py --build-commit`` writes into the commit tag."""
    metadata = {"schema": 1, "kind": "commit", "version": "1.2.3", "commit": COMMIT,
                "lanes": sorted(lane_set.COMMIT), "lanesRevision": lane_set.LANES_REVISION,
                "requireGreen": True, "bundleEnv": {}, **changes}
    return {key: value for key, value in metadata.items() if value is not None}


def _git(*, message=None, permission="write", on_main=False):
    """git/gh for one remote annotated commit claim; calls are recorded."""
    ref = f"refs/tags/{TAG}"
    message = json.dumps(claim()) if message is None else message
    calls = []

    def run(argv):
        calls.append(argv)
        if argv[:2] == ["gh", "api"]:
            assert argv[2].startswith("repos/example/hermes-agent/collaborators/")
            return permission
        if argv[1] == "rev-parse":
            return TAG_OBJECT if argv[-1] == ref else COMMIT
        if argv[1] == "cat-file":
            return "tag"
        if argv[1] == "ls-remote":
            return f"{TAG_OBJECT}\t{ref}\n{COMMIT}\t{ref}^{{}}"
        if argv[1] == "for-each-ref":
            return "refs/remotes/origin/main" if on_main else ""
        if argv[1] == "tag":
            return message
        return ""

    run.calls = calls
    return run


def test_the_version_helpers_name_a_commit_claim_in_every_language():
    from hermes_cli.update_channel import (commit_tag_for_date, commit_timestamp, is_canary_tag,
                                           is_commit_tag, release_version, tag_core, tag_kind)
    from scripts.termux.deb_version import channel_for_tag, deb_version_for_tag, release_version_for_tag

    assert commit_tag_for_date("1.2.3", STAMP) == TAG
    assert (release_version(TAG), tag_core(TAG), tag_kind(TAG), commit_timestamp(TAG)) == \
        (VERSION, "1.2.3", "commit", STAMP)
    assert is_commit_tag(TAG) and not is_canary_tag(TAG)
    assert (deb_version_for_tag(TAG), release_version_for_tag(TAG), channel_for_tag(TAG)) == \
        (f"1.2.3~commit.{STAMP}-1", VERSION, "commit")
    # A near miss is no commit claim: the shapes stay exact.
    for near in (f"commit.{STAMP}-1.2.3", f"commit.{STAMP[:-1]}-v1.2.3", f"v1.2.3+commit.{STAMP}",
                 f"commit-{STAMP}-v1.2.3", f"commit.{STAMP}-v1.2.3/x"):
        assert not is_commit_tag(near) and release_version(near) is None, near


def test_a_commit_claim_starts_from_the_default_lanes_minus_every_upgrade_test():
    assert lane_set.select(policy="commit") == lane_set.DEFAULT - {"upgrade", "install-e2e", "windows-store"}
    assert lane_set.select(policy="commit", with_=["termux"]) == lane_set.COMMIT | {"termux"}
    assert lane_set.select(policy="commit", only=["docker", "ci"]) == {"docker", "ci"}
    for refused in ("upgrade", "install-e2e", "windows-store"):
        with pytest.raises(ValueError, match="commit claim cannot run"):
            lane_set.select(policy="commit", only=["docker", refused])


def test_the_claim_message_carries_lanes_green_policy_and_bundle_defaults():
    parsed = stable.stamped_claim(json.dumps(claim(bundleEnv={"HERMES_PREVIEW_FULL_CONNECTORS": "1"})),
                                  tag=TAG, commit=COMMIT)
    assert parsed["lanes"] == lane_set.COMMIT and parsed["requireGreen"] is True
    assert parsed["bundleEnv"] == {"HERMES_PREVIEW_FULL_CONNECTORS": "1"}
    for bad in (claim(bundleEnv=None),                        # a commit claim always records its defaults
                claim(bundleEnv={"NOT_ALLOWED": "x"}),        # only the permitted names
                claim(kind="canary"), claim(version="1.2.4"), claim(commit="c" * 40),
                claim(requireGreen="yes"), claim(extra=True),
                claim(lanes=sorted(lane_set.COMMIT | {"install-e2e"})),
                claim(lanes=sorted(lane_set.COMMIT | {"upgrade"}))):
        with pytest.raises(ValueError, match="Commit claim metadata is invalid"):
            stable.stamped_claim(json.dumps(bad), tag=TAG, commit=COMMIT)
    # The canary validator never accepts a commit tag, nor a canary's claim a bundle default.
    with pytest.raises(ValueError, match="Canary claim"):
        stable.canary_claim(json.dumps(claim()), tag=TAG, commit=COMMIT)
    canary = f"canary.{STAMP}-v1.2.3"
    with pytest.raises(ValueError, match="Canary claim metadata is invalid"):
        stable.stamped_claim(json.dumps({**claim(kind="canary"), "lanes": sorted(lane_set.CANARY)}),
                             tag=canary, commit=COMMIT)


def test_any_pushed_commit_is_admitted_for_maintainers_only():
    env = {**CONTROLLER_ENV, "RELEASE_CLAIM_TAG": TAG, "RELEASE_CLAIM_OBJECT": TAG_OBJECT}
    epoch = int(datetime(2026, 10, 26, 18, 0, tzinfo=UTC).timestamp())
    run = _git(message=json.dumps(claim(bundleEnv={"HERMES_PREVIEW_FULL_CONNECTORS": "1"},
                                        requireGreen=False)))
    # Not on main: a commit build builds any commit its claim tag pushed.
    assert stable.check_claim(env, run) == {
        "claim_tag": TAG, "tag": TAG, "version": VERSION, "attempt": None, "commit": COMMIT,
        "claim_object": TAG_OBJECT, "autopublish": True, "lanes": lane_set.COMMIT, "require_green": False,
        "bundle_env": {"HERMES_PREVIEW_FULL_CONNECTORS": "1"}, "claim_epoch": epoch, "policy": "commit",
    }
    assert ["gh", "api", "repos/example/hermes-agent/collaborators/maintainer/permission",
            "--jq", ".permission"] in run.calls
    # The dispatching and the triggering actor both need write.
    for permission in ("read", "triage", "none"):
        with pytest.raises(ValueError, match="write, maintain or admin"):
            stable.check_claim(env, _git(permission=permission))
    with pytest.raises(ValueError, match="maintainer"):
        stable.check_claim({**env, "GITHUB_ACTOR": ""}, _git())
    triggering = []

    def gh_by_actor(argv):
        if argv[:2] == ["gh", "api"]:
            triggering.append(argv[2])
            return "read" if "/outsider/" in argv[2] else "write"
        return _git()(argv)

    with pytest.raises(ValueError, match="write, maintain or admin"):
        stable.check_claim({**env, "GITHUB_TRIGGERING_ACTOR": "outsider"}, gh_by_actor)
    # Custody is the canary's: an annotated tag on origin, unmoved, from the main controller.
    with pytest.raises(ValueError, match="tag moved"):
        stable.check_claim({**env, "RELEASE_CLAIM_OBJECT": "c" * 40}, _git())
    with pytest.raises(ValueError, match="controller"):
        stable.check_claim({**env, "GITHUB_REF": "refs/heads/feature"}, _git())


def test_admission_needs_no_github_release_and_emits_the_commit_policy(tmp_path, monkeypatch):
    out, summary = tmp_path / "out", tmp_path / "summary"
    env = {**CONTROLLER_ENV, "RELEASE_CLAIM_TAG": TAG, "GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary)}
    admitted = {"claim_tag": TAG, "tag": TAG, "version": VERSION, "attempt": None, "commit": COMMIT,
                "claim_object": TAG_OBJECT, "autopublish": True, "lanes": lane_set.COMMIT,
                "require_green": True, "bundle_env": {"HERMES_PREVIEW_FULL_CONNECTORS": "1"},
                "claim_epoch": 1, "policy": "commit"}
    monkeypatch.setattr(stable, "check_claim", lambda _env: admitted)
    monkeypatch.setattr(stable, "output", lambda argv: pytest.fail(f"a commit build asked {argv}"))
    stable.admit(env)
    outputs = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    assert outputs["policy"] == "commit" and outputs["release-id"] == "" and outputs["docker-tag"] == TAG
    assert json.loads(outputs["bundle-env"]) == {"HERMES_PREVIEW_FULL_CONNECTORS": "1"}
    jobs = json.loads(outputs["jobs"])
    assert jobs["publish-commit"] and not any(jobs[name] for name in (
        "publish-bundles", "publish-canary", "publish-docker", "install-e2e", "windows-packaged"))
    assert "Commit candidate" in summary.read_text(encoding="utf-8")


def test_candidate_legs_get_bundle_defaults_from_the_claim_only(tmp_path, monkeypatch):
    out = tmp_path / "out"
    claimed = {"policy": "commit", "claim_epoch": 1, "bundle_env": {"HERMES_PREVIEW_FULL_CONNECTORS": "1"}}
    monkeypatch.setattr(stable, "stable_context", lambda env: (TAG, COMMIT, claimed))
    env = {"GITHUB_OUTPUT": str(out), "BUNDLE_ENV_JSON": '{"HERMES_PREVIEW_FULL_CONNECTORS": "1"}'}
    stable.verify(env)
    outputs = dict(line.split("=", 1) for line in out.read_text(encoding="utf-8").splitlines())
    assert outputs["channel"] == "commit" and outputs["payload-version"] == VERSION
    for injected in ('{}', '{"HERMES_PREVIEW_FULL_CONNECTORS": "other"}'):
        with pytest.raises(ValueError, match="Bundle environment differs"):
            stable.verify({**env, "BUNDLE_ENV_JSON": injected})
    # A release never carries bundle defaults at all.
    monkeypatch.setattr(stable, "stable_context", lambda env: ("v1.2.3", COMMIT, {"claim_epoch": 1}))
    with pytest.raises(ValueError, match="Bundle environment differs"):
        stable.verify(env)


def test_a_commit_claim_is_versioned_from_its_stamp_like_a_canary():
    epoch = int(datetime(2026, 10, 26, 18, 0, tzinfo=UTC).timestamp())
    assert stable.stamped_epoch(TAG) == epoch
    assert stable.release_windows_version(TAG, None) == stable.stamped_windows_version(TAG) == "26.1026.18.0"
    assert stable.payload_version(TAG) == VERSION
    stable.require_release_identity(TAG, COMMIT)
    # The canary-only helpers stay canary-only.
    for helper in (stable.canary_epoch, stable.canary_windows_version):
        with pytest.raises(ValueError, match="canary"):
            helper(TAG)


def test_package_provenance_binds_a_commit_claim_to_a_commit_build_stamp():
    from scripts.bundles.release_artifacts import stamp_matches

    stamp = {"commit": COMMIT, "tag": TAG, "baseVersion": VERSION, "source": "commit-build"}
    stamp_matches(stamp, TAG, COMMIT)
    with pytest.raises(ValueError, match="update ownership"):
        stamp_matches({**stamp, "source": "build"}, TAG, COMMIT)
    release = {"commit": COMMIT, "tag": "v1.2.3", "baseVersion": "1.2.3"}
    stamp_matches({**release, "source": "build"}, "v1.2.3", COMMIT)
    with pytest.raises(ValueError, match="update ownership"):
        stamp_matches({**release, "source": "commit-build"}, "v1.2.3", COMMIT)


def test_the_build_bridge_hands_packagers_the_tag_and_the_commit_identity(tmp_path):
    from scripts.bundles.desktop_inputs import identity_environment

    request = SimpleNamespace(validate_channel=lambda: None, channel_request=None, commit=COMMIT, tag=TAG,
                              version=VERSION, bundle_env={}, release_epoch=1)
    env = identity_environment(request, "bundled", {"HERMES_BUILD_COMMIT": "stale"})
    assert env["HERMES_PAYLOAD_TAG"] == TAG and env["HERMES_BUILD_COMMIT"] == COMMIT
    release = SimpleNamespace(**{**vars(request), "tag": "v1.2.3", "version": "1.2.3"})
    env = identity_environment(release, "bundled", {"HERMES_BUILD_COMMIT": "stale"})
    assert env["HERMES_PAYLOAD_TAG"] == "v1.2.3" and "HERMES_BUILD_COMMIT" not in env


def test_writers_that_publish_refuse_a_commit_tag():
    from scripts.bundles import release_artifacts
    from scripts.releases import darwin, docker, r2

    for channel_of in (r2.channel_for_tag, release_artifacts.release_channel):
        with pytest.raises(ValueError, match="no release channel"):
            channel_of(TAG)
    with pytest.raises(ValueError, match="macOS"):
        darwin.merge_mac_feeds({"arm64-commit-mac.yml": ""}, TAG)
    # Its image is built and tested, never pushed under a public tag.
    assert docker.require_tested_image_tag(TAG) == TAG
    docker.build_manifest(TAG, COMMIT, {"amd64": "1" * 64, "arm64": "2" * 64})
    with pytest.raises(docker.DockerReleaseError):
        docker.require_release_image_tag(TAG)
    manifest = json.dumps({"schema": docker.MANIFEST_SCHEMA, "tag": TAG, "commit": COMMIT}).encode()
    with pytest.raises(docker.DockerReleaseError):
        docker.parse_manifest(manifest)


def test_commit_builds_keep_ninety_days_and_legacy_commit_pages_age_out():
    from scripts.releases import r2

    day = 86400
    now = datetime(2026, 10, 26, tzinfo=UTC).timestamp()
    cutoff = datetime.fromtimestamp(now - r2.COMMIT_KEEP_DAYS * day, tz=UTC).strftime("%Y%m%d")
    old, fresh = "releases/tag/commit.20260701T000000Z-v1.2.3/", "releases/tag/commit.20261001T000000Z-v1.2.3/"
    keys = [old + "index.html", old + "HermesBundledCommitaaaaaaa-1.2.3+commit.20260701T000000Z-win-x64.msix",
            fresh + "index.html", "releases/commit/" + "a" * 40 + "/index.html",
            "releases/commit/" + "b" * 40 + "/index.html", "releases/tag/v1.2.3/index.html",
            "releases/tag/canary.20260701T000000Z-v1.2.3/index.html"]
    last_modified = {keys[3]: now - 100 * day, keys[4]: now - 10 * day}
    assert r2.commit_doomed_keys(keys, cutoff, last_modified, now - r2.COMMIT_KEEP_DAYS * day) == \
        [keys[0], keys[1], keys[3]]


def test_the_builds_page_of_a_commit_build_touches_no_release_or_channel(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "render_builds_table_commit", Path(__file__).resolve().parents[2] / "scripts/render-builds-table.py")
    rbt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rbt)
    written = {}
    candidate = {"files": [{"path": f"HermesBundledCommitaaaaaaa-{VERSION}-win-x64.msix"}], "smoke_results": {}}
    monkeypatch.setattr(rbt.stable, "read_admitted_candidate",
                        lambda tag, commit, base, digest, archive: candidate if (tag, archive) == (TAG, TAG) else None)
    monkeypatch.setattr(rbt, "r2_object_names",
                        lambda archive: [f"releases/tag/{archive}/{row['path']}" for row in candidate["files"]])
    monkeypatch.setattr(rbt, "write_page", lambda key, html, base: written.setdefault(key, html))
    monkeypatch.setattr(rbt.subprocess, "run", lambda *a, **k: pytest.fail("a commit build has no GitHub release"))
    monkeypatch.setattr(rbt, "write_channel_page", lambda *a, **k: pytest.fail("a commit build has no channel page"))
    summary = tmp_path / "summary.md"
    monkeypatch.setattr(sys, "argv", ["render-builds-table.py", "--tag", TAG, "--archive", TAG, "--repo", "o/r",
                                      "--r2-base-url", "https://releases.example", "--candidate-manifest-sha256",
                                      "f" * 64, "--candidate-commit", COMMIT, "--summary-out", str(summary),
                                      "--bundle-env-json", '{"HERMES_HOME": null}'])
    assert rbt.main() == 0
    assert list(written) == [f"releases/tag/{TAG}/index.html"]
    page = written[f"releases/tag/{TAG}/index.html"]
    # It links its source commit, not a GitHub release it does not have, and lists its defaults.
    assert f"https://github.com/o/r/commit/{COMMIT}" in page and "/releases/tag/" not in page.split("<h2>")[0]
    assert "<code>HERMES_HOME</code>" in page and "Unset" in page
    with pytest.raises(ValueError, match="exact source commit"):
        rbt.render_page(TAG, {}, "https://releases.example")
    with pytest.raises(ValueError, match="Only a commit build page"):
        rbt.render_page("v1.2.3", {}, "https://releases.example", bundle_env={"HERMES_HOME": None})
    text = summary.read_text(encoding="utf-8")
    assert f"Commit build {TAG}" in text and f"releases/tag/{TAG}/index.html" in text



def build_args(**options):
    """``release.py --build-commit HEAD --remote origin`` with every option at its default."""
    defaults = {"build_commit": "HEAD", "remote": "origin", "publish": True, "only": None, "with_lanes": None,
                "without_lanes": None, "require_green": True, "bundle_env": [], "bundle_unset": []}
    return SimpleNamespace(**{**defaults, **options})


@pytest.fixture
def commit_repo(tmp_path, monkeypatch, request):
    """A real repository and bare remote; GitHub (gh) is an inert recorder."""
    import subprocess

    from scripts import release
    from scripts.releases import commit_build

    git = request.getfixturevalue("release_repo")
    # tests.scripts.test_release_tags patches its own import of release.py; commit_build uses this one.
    monkeypatch.setattr(release, "REPO_ROOT", tmp_path)
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "hermes-agent"\nversion = "1.2.3"\n',
                                             encoding="utf-8")
    git("add", "pyproject.toml")
    git("commit", "-qm", "version")
    remote = tmp_path.parent / f"{tmp_path.name}-remote.git"
    git("init", "--bare", "-q", str(remote))
    git("remote", "add", "origin", "https://github.com/fixture/release")
    rewrite = ["-c", f"url.{remote.as_posix()}.insteadOf=https://github.com/fixture/release"]
    calls = []
    actual_run = subprocess.run

    def run(argv, *args, **kwargs):
        if argv[0] != "gh":
            if argv[:2] in (["git", "push"], ["git", "ls-remote"]):
                argv = ["git", *rewrite, *argv[1:]]
            return actual_run(argv, *args, **kwargs)
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="main\n", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    which = release.shutil.which
    monkeypatch.setattr(release.shutil, "which", lambda name: "gh" if name == "gh" else which(name))
    # Reachability from a pushed ref is commit_build.require_pushed's own concern (tested with it);
    # here the revision resolves to the checkout's HEAD.
    monkeypatch.setattr(commit_build, "resolve_revision", lambda rev, remote_name, repo: git("rev-parse", rev))
    monkeypatch.setenv("CLOUDFLARE_R2_PUBLIC_URL", "https://releases.example")
    remote_git = lambda *args: actual_run(["git", "--git-dir", str(remote), *args], capture_output=True,
                                           text=True, encoding="utf-8", check=True).stdout.strip()
    return git, remote_git, calls


def test_build_commit_dry_run_creates_and_dispatches_nothing(commit_repo, capsys):
    from scripts.releases.commit_build import cmd_build_commit

    git, remote_git, calls = commit_repo
    cmd_build_commit(build_args(publish=False))
    out = capsys.readouterr().out
    assert "Dry run: no tag was created" in out and "stable-release.yml" in out
    assert git("tag", "--list", "commit.*") == "" and remote_git("tag", "--list") == ""
    assert not any(call[1:3] == ["workflow", "run"] for call in calls)


def test_build_commit_claims_the_commit_and_runs_stable_release_on_it(commit_repo, capsys):
    from hermes_cli.update_channel import is_commit_tag
    from scripts.releases.commit_build import cmd_build_commit

    git, remote_git, calls = commit_repo
    head = git("rev-parse", "HEAD")
    cmd_build_commit(build_args(with_lanes=["termux"], without_lanes=["nix"], require_green=False,
                                bundle_env=["HERMES_PREVIEW_FULL_CONNECTORS=1"], bundle_unset=["HERMES_HOME"]))
    (tag,) = git("tag", "--list", "commit.*").split()
    assert is_commit_tag(tag) and tag.endswith("-v1.2.3")
    assert git("cat-file", "-t", tag) == "tag" and git("rev-parse", f"{tag}^{{commit}}") == head
    assert remote_git("rev-parse", f"{tag}^{{commit}}") == head
    # Admission reads exactly what the claim recorded.
    parsed = stable.stamped_claim(git("tag", "-l", tag, "--format=%(contents)"), tag=tag, commit=head)
    assert parsed["lanes"] == (lane_set.COMMIT | {"termux"}) - {"nix"}
    assert parsed["requireGreen"] is False
    assert parsed["bundleEnv"] == {"HERMES_PREVIEW_FULL_CONNECTORS": "1", "HERMES_HOME": None}
    dispatches = [call for call in calls if call[1:3] == ["workflow", "run"]]
    assert dispatches == [["gh", "workflow", "run", "stable-release.yml", "--ref", "main",
                           "--repo", "fixture/release", "-f", f"tag={tag}"]]
    assert f"https://releases.example/releases/tag/{tag}/index.html" in capsys.readouterr().out


def test_build_commit_refuses_a_lane_a_commit_build_cannot_run(commit_repo):
    from scripts.releases.commit_build import cmd_build_commit

    git, _remote_git, calls = commit_repo
    for lane in ("upgrade", "install-e2e", "windows-store"):
        with pytest.raises(SystemExit, match="commit claim cannot run"):
            cmd_build_commit(build_args(with_lanes=[lane]))
    assert git("tag", "--list", "commit.*") == "" and not calls


def test_commit_claim_tags_are_pruned_after_ninety_days(commit_repo, capsys):
    from datetime import timedelta

    from scripts import release

    git, remote_git, _calls = commit_repo
    old = f"commit.{(datetime.now(UTC) - timedelta(days=120)).strftime('%Y%m%dT%H%M%SZ')}-v1.2.3"
    fresh = f"commit.{(datetime.now(UTC) - timedelta(days=30)).strftime('%Y%m%dT%H%M%SZ')}-v1.2.3"
    for tag in (old, fresh, "commit.not-a-stamp-v1.2.3"):
        git("tag", "-a", tag, "-m", "{}")
        git("push", "-q", "origin", f"refs/tags/{tag}")
    release._prune_old_commit_claims(SimpleNamespace(publish=False), "origin")
    assert f"Would delete {old}" in capsys.readouterr().out
    assert old in remote_git("tag", "--list")
    release._prune_old_commit_claims(SimpleNamespace(publish=True), "origin")
    assert set(remote_git("tag", "--list").split()) == {fresh, "commit.not-a-stamp-v1.2.3"}
    assert set(git("tag", "--list", "commit.*").split()) == {fresh, "commit.not-a-stamp-v1.2.3"}
