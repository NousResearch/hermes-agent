"""The release entrypoint claims an attempt, cuts a draft, and dispatches the gate.

The claim is the push of one attempt ref, ``rc.<N>-vX.Y.Z``, and it pushes
exactly that ref. A version is spent only by publication; an abandoned attempt
frees its version for attempt N+1. At most one attempt, of any version, is
outstanding. A release that never starts is an error, not a warning the
operator has to notice.
"""
import json
import subprocess
import threading

import pytest


def git(repo, *args):
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


@pytest.fixture
def source(tmp_path):
    origin = tmp_path / "origin.git"
    repo = tmp_path / "source"
    origin.mkdir()
    repo.mkdir()
    git(origin, "init", "--bare", "--quiet")
    git(repo, "init", "--initial-branch=main", "--quiet")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.test")
    (repo / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="0.0.0"\n', encoding="utf-8")
    git(repo, "add", "pyproject.toml")
    git(repo, "commit", "--quiet", "-m", "Initial")
    git(repo, "remote", "add", "origin", str(origin))
    git(repo, "push", "--quiet", "-u", "origin", "main")
    return repo


def _advance(repo, message):
    git(repo, "commit", "--allow-empty", "--quiet", "-m", message)
    git(repo, "push", "--quiet", "origin", "main")
    return git(repo, "rev-parse", "HEAD")


def _claim(repo, version, commit, attempt=1):
    metadata = json.dumps({
        "schema": 1, "version": version, "attempt": attempt, "commit": commit,
        "autopublish": False, "claimEpoch": 1_790_000_000,
    }, sort_keys=True, separators=(",", ":"))
    ref = f"rc.{attempt}-v{version}"
    git(repo, "tag", "-a", ref, commit, "-m", metadata)
    git(repo, "push", "--quiet", "origin", f"refs/tags/{ref}")


def _mark(repo, version, attempt=1):
    ref = f"abandoned-rc.{attempt}-v{version}"
    git(repo, "tag", "-a", ref, f"rc.{attempt}-v{version}^{{commit}}", "-m", "abandoned")
    git(repo, "push", "--quiet", "origin", f"refs/tags/{ref}")


def _publish_tag(repo, version, commit):
    git(repo, "tag", "-a", f"v{version}", commit, "-m", "published")
    git(repo, "push", "--quiet", "origin", f"refs/tags/v{version}")


def _release(repo, commit, **overrides):
    from scripts.releases.entrypoint import release

    arguments = {"bump": "patch", "repo": repo, "remote": "origin",
                 "repository": "example/hermes-agent", "execute": lambda _command: None}
    return release(commit, **{**arguments, **overrides})


def _must_not_execute(command):
    if command[:3] != ["gh", "run", "list"]:
        pytest.fail(f"a refused release must not run {command}")
    return "[]"


def test_release_claims_the_first_attempt_creates_a_draft_and_dispatches(source):
    commit = git(source, "rev-parse", "HEAD")
    calls = []

    def execute(command):
        calls.append(command)
        if command[:3] == ["gh", "run", "list"]:
            return json.dumps([{"databaseId": 7, "url": "https://github.com/example/hermes-agent/actions/runs/7",
                                "headBranch": "rc.1-v0.21.5", "status": "queued"}])
        if command[:3] == ["gh", "release", "create"]:
            return "https://github.com/example/hermes-agent/releases/tag/untagged-0123abcd\n"
        return ""

    result = _release(source, commit, execute=execute, autopublish=True, skip_bundles=True)

    assert result["version"] == "0.21.5"
    assert result["tag"] == "rc.1-v0.21.5"
    assert result["commit"] == commit
    # A draft lives at the page gh reports, never at releases/tag/<claim>.
    assert result["url"] == "https://github.com/example/hermes-agent/releases/tag/untagged-0123abcd"
    assert result["final_url"] == "https://github.com/example/hermes-agent/releases/tag/v0.21.5"
    assert git(source, "rev-parse", "rc.1-v0.21.5^{commit}") == commit
    claim = json.loads(git(source, "tag", "-l", "rc.1-v0.21.5", "--format=%(contents)"))
    assert isinstance(claim.pop("claimEpoch"), int)
    # The claim is the one record of the attempt's policy, flags included.
    assert claim == {
        "attempt": 1,
        "autopublish": True,
        "commit": commit,
        "schema": 1,
        "skipBundles": True,
        "skipTests": False,
        "version": "0.21.5",
    }
    create = calls[0]
    assert create[:4] == ["gh", "release", "create", "rc.1-v0.21.5"]
    assert "--verify-tag" in create and "--draft" in create
    assert "--generate-notes" not in create
    assert "--notes-file" in create
    assert create[-2:] == ["--title", "Hermes Agent v0.21.5"]
    assert calls[1:] == [
        ["gh", "workflow", "run", "stable-release.yml", "--ref", "rc.1-v0.21.5",
         "--repo", "example/hermes-agent", "--raw-field", "tag=rc.1-v0.21.5"],
        ["gh", "run", "list", "--repo", "example/hermes-agent", "--workflow", "stable-release.yml",
         "--branch", "rc.1-v0.21.5", "--json", "databaseId,url,headBranch,status"],
    ]
    assert result["run_url"] == "https://github.com/example/hermes-agent/actions/runs/7"
    # The claim push names the claim ref and nothing else.
    pushed = git(source, "ls-remote", "origin", "refs/tags/*")
    assert pushed.splitlines() == [
        f"{git(source, 'rev-parse', 'rc.1-v0.21.5')}\trefs/tags/rc.1-v0.21.5",
        f"{commit}\trefs/tags/rc.1-v0.21.5^{{}}",
    ]


def test_release_output_names_the_wait_and_the_publish_step():
    from scripts.releases.entrypoint import next_steps

    result = {"version": "0.21.5", "tag": "rc.1-v0.21.5", "autopublish": False,
              "skip_bundles": False, "skip_tests": False,
              "run_url": "https://github.com/example/hermes-agent/actions/runs/7",
              "url": "https://github.com/example/hermes-agent/releases/tag/untagged-0123abcd",
              "final_url": "https://github.com/example/hermes-agent/releases/tag/v0.21.5"}
    text = next_steps(result)
    assert "Workflow: " + result["run_url"] in text
    assert "The release workflow started on rc.1-v0.21.5." in text
    # The draft is reachable before the workflow is green, not only after it.
    assert text.index(result["url"]) < text.index("Wait for that workflow to finish.")
    assert result["final_url"] in text
    assert "python scripts/release.py publish --version 0.21.5 --remote origin" in text

    automatic = next_steps({**result, "autopublish": True})
    assert "Autopublish is on." in automatic
    assert "publish --version" not in automatic
    assert "skipped" not in text
    skipped = next_steps({**result, "skip_bundles": True, "skip_tests": True})
    assert "Bundles are skipped." in skipped and "Tests are skipped." in skipped


def test_a_final_tag_for_the_next_version_refuses_the_cut(source):
    """A started publication owns its version even before its release is public."""
    from scripts.releases.entrypoint import ReleaseRefused

    commit = git(source, "rev-parse", "HEAD")
    git(source, "tag", "v0.21.5", commit)
    git(source, "push", "-q", "origin", "refs/tags/v0.21.5")

    with pytest.raises(ReleaseRefused, match="v0.21.5 already has a final tag"):
        _release(source, commit, execute=_must_not_execute)
    assert "rc." not in git(source, "ls-remote", "origin", "refs/tags/*")


def test_second_cut_after_abandon_is_attempt_two_of_the_same_version(source):
    first = git(source, "rev-parse", "HEAD")
    _claim(source, "0.21.5", first)
    _mark(source, "0.21.5")

    result = _release(source, _advance(source, "fix"))

    assert result["tag"] == "rc.2-v0.21.5"
    assert result["version"] == "0.21.5"


@pytest.mark.parametrize("bump", ["patch", "minor"])
def test_an_outstanding_attempt_of_any_version_blocks_the_next_cut(source, bump):
    from scripts.releases.entrypoint import ReleaseRefused

    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))

    with pytest.raises(ReleaseRefused, match="publish or abandon it first"):
        _release(source, _advance(source, "later"), bump=bump, execute=_must_not_execute)
    assert git(source, "tag", "--list", "rc.*") == "rc.1-v0.21.5"


def test_a_published_attempt_no_longer_blocks(source):
    commit = git(source, "rev-parse", "HEAD")
    _claim(source, "0.21.5", commit)
    _publish_tag(source, "0.21.5", commit)

    result = _release(source, _advance(source, "later"), published=("0.21.5", commit))

    assert result["tag"] == "rc.1-v0.21.6"


def test_new_attempt_must_descend_from_the_published_head(source):
    from scripts.releases.entrypoint import ReleaseRefused

    earlier = git(source, "rev-parse", "HEAD")
    published = _advance(source, "published")

    with pytest.raises(ReleaseRefused, match="does not descend from the published stable head"):
        _release(source, earlier, published=("0.21.4", published), execute=_must_not_execute)
    assert git(source, "tag", "--list", "rc.*") == ""


def test_an_abandoned_attempt_does_not_constrain_the_next_cut(source):
    root = git(source, "rev-parse", "HEAD")
    good = _advance(source, "good")
    bad = _advance(source, "bad")
    _claim(source, "0.21.5", bad)
    _mark(source, "0.21.5")

    result = _release(source, good, published=("0.21.4", root))

    assert result["tag"] == "rc.2-v0.21.5"
    assert result["commit"] == good


def test_refusal_names_the_run_and_both_ways_out(source, capsys):
    from scripts.releases.entrypoint import ReleaseRefused

    blocked = git(source, "rev-parse", "HEAD")
    _claim(source, "0.21.5", blocked)

    def execute(command):
        assert command == ["gh", "run", "list", "--repo", "example/hermes-agent", "--workflow",
                           "stable-release.yml", "--branch", "rc.1-v0.21.5",
                           "--json", "databaseId,url,headBranch,headSha,status"]
        return json.dumps([
            {"databaseId": 98, "url": "https://github.com/example/hermes-agent/actions/runs/98",
             "headBranch": "rc.1-v0.21.5", "headSha": "0" * 40, "status": "completed"},
            {"databaseId": 99, "url": "https://github.com/example/hermes-agent/actions/runs/99",
             "headBranch": "rc.1-v0.21.5", "headSha": blocked, "status": "completed"},
        ])

    with pytest.raises(ReleaseRefused):
        _release(source, _advance(source, "later"), bump="minor", execute=execute)
    text = capsys.readouterr().err
    assert "https://github.com/example/hermes-agent/actions/runs/99" in text
    assert "runs/98" not in text
    # The abandon command names the outstanding attempt's version, not the derived 0.22.0.
    assert "python scripts/release.py abandon --version 0.21.5 --remote origin" in text
    assert "gh run rerun 99 --failed --repo example/hermes-agent" in text


def test_refusal_for_an_attempt_that_never_started_offers_only_abandon(source, capsys):
    from scripts.releases.entrypoint import ReleaseRefused

    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))

    with pytest.raises(ReleaseRefused):
        _release(source, _advance(source, "later"), execute=_must_not_execute)
    text = capsys.readouterr().err
    assert "No workflow run is listed for rc.1-v0.21.5." in text
    assert "python scripts/release.py abandon --version 0.21.5 --remote origin" in text
    assert "gh run rerun" not in text


def test_successive_attempts_reserve_increasing_native_epochs(source):
    _release(source, git(source, "rev-parse", "HEAD"))
    _mark(source, "0.21.5")
    _release(source, _advance(source, "next"))

    first_epoch = int(git(source, "for-each-ref", "refs/tags/rc.1-v0.21.5",
                          "--format=%(taggerdate:unix)"))
    second_epoch = int(git(source, "for-each-ref", "refs/tags/rc.2-v0.21.5",
                           "--format=%(taggerdate:unix)"))
    assert second_epoch > first_epoch


def test_draft_body_is_fenced_at_top_and_bottom():
    from scripts.releases.draft_warning import (
        WARNING_CLOSE, WARNING_OPEN, draft_body,
    )

    body = draft_body(version="0.21.5", attempt_ref="rc.2-v0.21.5",
                      notes="## What's changed\n- x")
    assert body.startswith(WARNING_OPEN) and body.rstrip().endswith(WARNING_CLOSE)
    assert body.count(WARNING_OPEN) == 2 and body.count(WARNING_CLOSE) == 2
    assert "python scripts/release.py publish --version 0.21.5" in body
    assert "python scripts/release.py abandon --version 0.21.5" in body
    assert "## What's changed\n- x" in body
    # The warning names the attempt and the version it would burn.
    assert "rc.2-v0.21.5" in body


@pytest.mark.parametrize("published_by", ["identity", "seed_tag"])
def test_release_draft_lists_the_commits_since_the_stable_base(source, published_by):
    from scripts.releases.draft_warning import WARNING_CLOSE, WARNING_OPEN

    shipped = _advance(source, "fix: shipped in the published release")
    git(source, "tag", "-a", "v0.21.4", shipped, "-m", "published")
    commit = _advance(source, "feat: new in this release (#123)")
    published = ("0.21.4", shipped) if published_by == "identity" else ("0.21.4", None)
    seen = {}

    def execute(command):
        assert command[:3] != ["gh", "api", "repos/example/hermes-agent/releases/generate-notes"]
        if command[:3] == ["gh", "release", "create"]:
            notes_path = command[command.index("--notes-file") + 1]
            with open(notes_path, encoding="utf-8") as file:
                seen["body"] = file.read()
        if command[:3] == ["gh", "run", "list"]:
            return json.dumps([{"databaseId": 7, "url": "https://github.com/example/hermes-agent/actions/runs/7",
                                "headBranch": "rc.1-v0.21.5", "status": "queued"}])
        return ""

    result = _release(source, commit, execute=execute, published=published)

    assert result["tag"] == "rc.1-v0.21.5"
    body = seen["body"]
    assert body.startswith(WARNING_OPEN) and body.rstrip().endswith(WARNING_CLOSE)
    # The notes are this repository's commits from the stable base to the cut
    # commit, not GitHub's merged-PR list, which a fork leaves empty.
    assert "New in this release" in body
    assert "https://github.com/example/hermes-agent/pull/123" in body
    assert "hipped in the published release" not in body
    assert "compare/v0.21.4...rc.1-v0.21.5" in body
    # The stable workflow renders the download tables into this marker.
    assert "<!-- HERMES_BUILDS_TABLE -->" in body
    assert "python scripts/release.py publish --version 0.21.5" in body
    assert "python scripts/release.py abandon --version 0.21.5" in body


def test_a_dispatch_that_never_starts_is_an_error(source):
    from scripts.releases.entrypoint import ReleaseRefused

    def refuse(command):
        if command[1:3] == ["workflow", "run"]:
            raise RuntimeError("workflow dispatch rejected")

    with pytest.raises(ReleaseRefused, match="never started"):
        _release(source, git(source, "rev-parse", "HEAD"), execute=refuse)
    # The attempt stands outstanding until someone abandons it.
    assert "rc.1-v0.21.5" in git(source, "tag", "--list")


def _oversized_commit(repo, tmp_path):
    """A commit whose subject alone overflows GitHub's release body limit."""
    from scripts.releases.entrypoint import GITHUB_BODY_LIMIT

    message = tmp_path / "message.txt"
    message.write_text("feat: " + "x" * GITHUB_BODY_LIMIT, encoding="utf-8")
    git(repo, "commit", "--allow-empty", "--quiet", "-F", str(message))
    git(repo, "push", "--quiet", "origin", "main")
    return git(repo, "rev-parse", "HEAD")


def test_an_oversized_body_is_refused_before_the_claim(source, tmp_path):
    from scripts.releases.entrypoint import ReleaseRefused

    commit = _oversized_commit(source, tmp_path)

    with pytest.raises(ReleaseRefused, match="--no-changelog"):
        _release(source, commit, execute=lambda command: pytest.fail(f"must not run {command}"))

    # Nothing was claimed, so the attempt is not burned.
    assert git(source, "ls-remote", "origin", "refs/tags/*") == ""
    assert git(source, "tag", "-l", "rc.*") == ""


def test_no_changelog_releases_what_the_full_changelog_could_not(source, tmp_path):
    from scripts.releases.entrypoint import GITHUB_BODY_LIMIT

    commit = _oversized_commit(source, tmp_path)
    seen = {}

    def execute(command):
        if command[:3] == ["gh", "release", "create"]:
            with open(command[command.index("--notes-file") + 1], encoding="utf-8") as file:
                seen["body"] = file.read()
        return ""

    result = _release(source, commit, execute=execute, no_changelog=True)

    assert result["tag"] == "rc.1-v0.21.5"
    assert len(seen["body"]) <= GITHUB_BODY_LIMIT
    assert "<!-- HERMES_BUILDS_TABLE -->" in seen["body"]


@pytest.mark.parametrize("argv", [
    ["release", "--commit", "HEAD", "--no-changelog"],
    ["--no-changelog", "release", "--commit", "HEAD"],
])
def test_release_command_accepts_no_changelog(argv, monkeypatch):
    from scripts import release as release_script
    from scripts.releases import entrypoint

    seen = {}
    monkeypatch.setattr(entrypoint, "cmd_release", lambda args: seen.setdefault("args", args))
    monkeypatch.setattr(release_script.sys, "argv", ["release.py", *argv])

    release_script.main()

    assert seen["args"].no_changelog is True


def test_publish_and_abandon_output_name_the_result():
    from scripts.releases.entrypoint import abandon_steps, publish_steps

    published = publish_steps({"version": "0.21.5", "repository": "example/hermes-agent",
                               "run_url": "https://github.com/example/hermes-agent/actions/runs/9"})
    assert "Requested publication of v0.21.5." in published
    assert "Workflow: https://github.com/example/hermes-agent/actions/runs/9" in published
    assert "moves the stable channel" in published

    abandoned = abandon_steps({"version": "0.21.5", "tag": "rc.1-v0.21.5",
                               "marker": "abandoned-rc.1-v0.21.5"})
    assert "Cleared rc.1-v0.21.5." in abandoned
    assert "abandoned-rc.1-v0.21.5" in abandoned
    assert "The next cut is rc.2-v0.21.5." in abandoned
    assert "cannot be reused" not in abandoned


def test_publish_dispatches_the_sequencer():
    from scripts.releases.entrypoint import ReleaseRefused, publish

    calls = []
    published = publish("0.21.5", repository="example/hermes-agent", dispatch=calls.append)
    assert published["requested"] == "v0.21.5"
    assert published["version"] == "0.21.5"
    assert published["repository"] == "example/hermes-agent"
    assert calls == [
        ["gh", "workflow", "run", "stable-release-publication.yml",
         "--repo", "example/hermes-agent", "--raw-field", "version=0.21.5"],
    ]

    with pytest.raises(ReleaseRefused, match="burned or superseded by 0\\.21\\.6"):
        publish(
            "0.21.5", repository="example/hermes-agent",
            dispatch=lambda _command: pytest.fail("superseded publish must not dispatch"),
            inspect=lambda _command: pytest.fail("superseded publish must not inspect drafts"),
            head_version=lambda: "0.21.6",
        )


def test_publish_preflight_finds_the_draft_on_the_outstanding_attempt(source):
    from scripts.releases.entrypoint import ReleaseRefused, publish

    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))
    calls = []

    def inspect(command):
        tag = command[3]
        if tag != "rc.1-v0.21.5":
            raise ReleaseRefused("release not found")
        return json.dumps({"tagName": tag, "isDraft": True, "isPrerelease": False})

    publish("0.21.5", repository="example/hermes-agent", dispatch=calls.append,
            inspect=inspect, head_version=lambda: None,
            repo=source, remote="origin")
    assert calls[-1][-1] == "version=0.21.5"

    # A draft left on the old v{version}-rc shape is not the publish draft.
    def old_shape(command):
        tag = command[3]
        if tag != "v0.21.5-rc":
            raise ReleaseRefused("release not found")
        return json.dumps({"tagName": tag, "isDraft": True, "isPrerelease": False})

    with pytest.raises(ReleaseRefused, match="burned or has no release draft"):
        publish("0.21.5", repository="example/hermes-agent",
                dispatch=lambda _command: pytest.fail("must not dispatch"),
                inspect=old_shape, head_version=lambda: None,
                repo=source, remote="origin")


def _abandon(repo, version, *, draft=None, calls=None):
    from scripts.releases.entrypoint import ReleaseRefused, abandon

    def inspect(command):
        if draft is None or command[3] != draft["tagName"]:
            raise ReleaseRefused("release not found")
        return json.dumps(draft)

    return abandon(version, repo=repo, remote="origin", repository="example/hermes-agent",
                   delete=(calls if calls is not None else []).append, inspect=inspect)


def test_abandon_of_a_draft_deletes_it_writes_the_marker_and_frees_the_version(source):
    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))
    calls = []

    result = _abandon(source, "0.21.5", calls=calls,
                      draft={"tagName": "rc.1-v0.21.5", "isDraft": True, "isPrerelease": False})

    assert result == {"version": "0.21.5", "tag": "rc.1-v0.21.5",
                      "marker": "abandoned-rc.1-v0.21.5", "repository": "example/hermes-agent"}
    assert calls == [["gh", "release", "delete", "rc.1-v0.21.5", "--repo", "example/hermes-agent", "--yes"]]
    remote = git(source, "ls-remote", "origin", "refs/tags/*")
    assert "refs/tags/abandoned-rc.1-v0.21.5" in remote
    assert "refs/tags/rc.1-v0.21.5" in remote
    marker = json.loads(git(source, "tag", "-l", "abandoned-rc.1-v0.21.5", "--format=%(contents)"))
    assert marker == {"attempt": 1, "attemptRef": "rc.1-v0.21.5", "schema": 1, "version": "0.21.5"}
    assert git(source, "rev-parse", "abandoned-rc.1-v0.21.5^{commit}") == git(
        source, "rev-parse", "rc.1-v0.21.5^{commit}")
    assert _release(source, _advance(source, "fix"))["tag"] == "rc.2-v0.21.5"


def test_abandon_of_a_draftless_burned_attempt_writes_the_marker(source):
    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))
    calls = []

    assert _abandon(source, "0.21.5", calls=calls)["marker"] == "abandoned-rc.1-v0.21.5"
    assert calls == []


@pytest.mark.parametrize("version", ["0.21.5", "0.22.0"])
def test_abandon_refuses_without_an_outstanding_attempt_of_that_version(source, version):
    from scripts.releases.entrypoint import ReleaseRefused

    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))
    if version == "0.21.5":
        _mark(source, "0.21.5")

    with pytest.raises(ReleaseRefused, match="no outstanding attempt to abandon"):
        _abandon(source, version)


def test_abandon_refuses_a_release_that_was_published(source):
    from scripts.releases.entrypoint import ReleaseRefused

    _claim(source, "0.21.5", git(source, "rev-parse", "HEAD"))
    with pytest.raises(ReleaseRefused, match="published and cannot be abandoned"):
        _abandon(source, "0.21.5",
                 draft={"tagName": "rc.1-v0.21.5", "isDraft": False, "isPrerelease": False})
    assert "abandoned-rc" not in git(source, "ls-remote", "origin", "refs/tags/*")


def test_abandon_clears_one_of_two_attempts_a_concurrent_cut_left(source):
    from scripts.releases.entrypoint import ReleaseRefused

    commit = git(source, "rev-parse", "HEAD")
    _claim(source, "0.21.5", commit)
    _claim(source, "0.22.0", commit)

    assert _abandon(source, "0.22.0")["marker"] == "abandoned-rc.1-v0.22.0"
    with pytest.raises(ReleaseRefused, match="rc.1-v0.21.5 is outstanding"):
        _release(source, _advance(source, "later"), execute=_must_not_execute)


def _clones(source, tmp_path):
    origin = git(source, "remote", "get-url", "origin")
    clones = []
    for name in ("left", "right"):
        clone = tmp_path / name
        git(tmp_path, "clone", "--quiet", origin, str(clone))
        git(clone, "config", "user.name", "Test")
        git(clone, "config", "user.email", "test@example.test")
        clones.append(clone)
    return origin, clones


def test_concurrent_claim_loser_reports_the_remote_winner_and_the_version_stays_free(
        source, tmp_path, monkeypatch):
    from scripts.releases import entrypoint
    from scripts.releases.versioning import derive_next_version, next_attempt

    old = git(source, "rev-parse", "HEAD")
    new = _advance(source, "later")
    origin, (left, right) = _clones(source, tmp_path)
    git(left, "checkout", "--quiet", old)

    barrier = threading.Barrier(2)
    winner_pushed = threading.Event()
    original_git = entrypoint._git

    def racing_git(repo, *args):
        if args[:2] == ("push", "origin") and args[-1] == "refs/tags/rc.1-v0.21.5":
            barrier.wait(timeout=10)
            if repo == left:
                if not winner_pushed.wait(timeout=10):
                    raise TimeoutError("winning claim did not finish")
            else:
                try:
                    return original_git(repo, *args)
                finally:
                    winner_pushed.set()
        return original_git(repo, *args)

    monkeypatch.setattr(entrypoint, "_git", racing_git)
    outcomes = {}

    def claim(name, repo, commit):
        try:
            outcomes[name] = _release(repo, commit)
        except Exception as error:
            outcomes[name] = error

    threads = [
        threading.Thread(target=claim, args=("old", left, old)),
        threading.Thread(target=claim, args=("new", right, new)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=15)

    assert outcomes["new"]["commit"] == new
    assert isinstance(outcomes["old"], entrypoint.ReleaseRefused)
    assert "was claimed by Test" in str(outcomes["old"])
    assert new in str(outcomes["old"])
    assert git(left, "rev-parse", "rc.1-v0.21.5^{commit}") == new

    fresh = tmp_path / "fresh"
    git(tmp_path, "clone", "--quiet", origin, str(fresh))
    refs = git(fresh, "tag", "--list", "rc.*").splitlines()
    assert derive_next_version(published=None, bump="patch") == "0.21.5"
    assert next_attempt("0.21.5", refs) == 2


def test_a_concurrent_cut_of_another_version_is_stopped_before_dispatch(
        source, tmp_path, monkeypatch):
    from scripts.releases import entrypoint

    commit = git(source, "rev-parse", "HEAD")
    _origin, (left, right) = _clones(source, tmp_path)
    original_git = entrypoint._git

    def interleaved_git(repo, *args):
        # The other maintainer cuts 0.22.0 after this cut passed its pre-check.
        if repo == left and args[:2] == ("push", "origin") and args[-1] == "refs/tags/rc.1-v0.21.5":
            monkeypatch.setattr(entrypoint, "_git", original_git)
            _release(right, commit, bump="minor")
        return original_git(repo, *args)

    monkeypatch.setattr(entrypoint, "_git", interleaved_git)
    with pytest.raises(entrypoint.ReleaseRefused, match="more than one outstanding attempt"):
        _release(left, commit, execute=lambda command: pytest.fail(f"must not run {command}"))
    assert {"rc.1-v0.21.5", "rc.1-v0.22.0"} <= set(git(left, "tag", "--list", "rc.*").splitlines())
