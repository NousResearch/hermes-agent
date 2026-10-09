"""Stable publication is ordered, held by default, and idempotent.

A green draft publishes when it opted into autopublish, or when a later green
release needs it resolved. A running older claim blocks only the versions above
it; already-resolvable older green claims still make progress.
"""
import hashlib
import json

import pytest


def _claims(*rows):
    return [dict(version=version, state=state, autopublish=autopublish)
            for version, state, autopublish in rows]


def _flips(steps):
    return [step["flip"] for step in steps if "flip" in step]


def test_a_sole_green_draft_waits_without_autopublish():
    from scripts.releases.sequencer import plan

    assert plan(_claims(("0.21.5", "green", False)), head="0.21.4") == []


def test_autopublish_flips_the_current_green_release():
    from scripts.releases.sequencer import plan

    steps = plan(_claims(("0.21.5", "green", True)), head="0.21.4")
    assert _flips(steps) == ["0.21.5"]
    assert steps[-1] == {"advance": "0.21.5"}


def test_a_newer_green_release_flushes_the_older_waiting_draft():
    from scripts.releases.sequencer import plan

    steps = plan(_claims(
        ("0.21.5", "green", False),
        ("0.21.6", "green", False),
    ), head="0.21.4")

    assert _flips(steps) == ["0.21.5", "0.21.6"]
    assert steps == [
        {"flip": "0.21.5"}, {"advance": "0.21.5"},
        {"flip": "0.21.6"}, {"advance": "0.21.6"},
    ]


def test_a_running_newer_claim_does_not_flush_a_held_draft():
    from scripts.releases.sequencer import plan

    steps = plan(_claims(
        ("0.21.5", "green", False),
        ("0.21.6", "running", False),
        ("0.21.7", "green", True),
    ), head="0.21.4")

    assert steps == []
    assert plan(_claims(
        ("0.21.5", "running", False),
        ("0.21.6", "green", True),
    ), head="0.21.4") == []
    assert plan(_claims(
        ("0.21.5", "green", False),
        ("0.21.6", "published", False),
    ), head="0.21.4") == []


def test_unstarted_claim_waits_for_its_grace_period_before_burning():
    from datetime import datetime, timedelta, timezone

    from scripts.releases.sequencer import classify_runs

    claimed = datetime(2026, 9, 22, 1, 0, tzinfo=timezone.utc)
    assert classify_runs([], claimed_at=claimed, now=claimed + timedelta(minutes=59)) == (
        "running", None,
    )
    assert classify_runs([], claimed_at=claimed, now=claimed + timedelta(hours=1)) == (
        "burned", None,
    )


def test_failed_run_retries_twice_before_burning():
    from scripts.releases.sequencer import classify_runs, retry_due

    failed = {"id": 42, "status": "completed", "conclusion": "failure", "run_attempt": 1}
    state, retry = classify_runs([failed])
    assert state == "running"
    assert retry is not None
    assert retry_due([{"version": "0.21.5", "state": state, "retry": retry}]) == [{
        "version": "0.21.5", "run_id": 42, "attempt": 2,
    }]
    newer = dict(retry)
    newer["run_id"] = 43
    assert retry_due([
        {"version": "0.21.5", "state": state, "retry": retry},
        {"version": "0.21.6", "state": state, "retry": newer},
    ]) == [{"version": "0.21.5", "run_id": 42, "attempt": 2}]
    failed["run_attempt"] = 2
    state, retry = classify_runs([failed])
    assert retry_due([{"version": "0.21.5", "state": state, "retry": retry}])[0]["attempt"] == 3
    failed["run_attempt"] = 3
    assert classify_runs([failed]) == ("burned", None)


@pytest.mark.parametrize(("attempt", "reruns"), [(1, True), (3, False)])
def test_the_pass_that_observes_a_failure_reruns_it_unless_burned(attempt, reruns):
    """No cron re-wakes the reconciler, so the failure event's own pass must rerun."""
    from datetime import datetime, timezone

    from scripts.releases.sequencer import reconcile

    commit = "a" * 40
    tags = {"rc.1-v0.21.5": ("1" * 40, commit, _claim_message("0.21.5", 1, commit))}
    failed = {
        "id": 42, "status": "completed", "conclusion": "failure", "run_attempt": attempt,
        # Completed this instant: the pass observing the failure event.
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "head_branch": "rc.1-v0.21.5", "head_sha": commit,
    }
    discover_run = _discover_run(tags, workflow_runs=[failed])
    posted = []

    def run(argv):
        if argv[:4] == ["gh", "api", "--method", "POST"]:
            posted.append(argv[-1])
            return ""
        if argv == ["gh", "api", "repos/example/project/actions/runs/42"]:
            return json.dumps({"id": 42, "run_attempt": attempt + 1, "status": "queued"})
        return discover_run(argv)

    steps = reconcile({"GITHUB_REPOSITORY": "example/project"}, run=run,
                      read_head=lambda: None, advance_head=lambda record: None)
    if reruns:
        assert posted == ["repos/example/project/actions/runs/42/rerun-failed-jobs"]
        assert steps == [{"retry": "0.21.5", "attempt": 2}]
    else:
        assert posted == []
        assert steps == []


def test_a_succeeded_run_is_green_only_with_its_draft():
    from scripts.releases.sequencer import classify_final_release, classify_runs

    succeeded = {"id": 9, "status": "completed", "conclusion": "success", "run_attempt": 1}
    assert classify_runs([succeeded], has_draft=True) == ("green", None)
    # The tool is the only thing that deletes drafts; a missing draft after
    # success is an error, not an inferred abandonment.
    with pytest.raises(ValueError, match="without a draft"):
        classify_runs([succeeded], has_draft=False)
    draft = {"id": 9, "tag_name": "rc.1-v0.21.5", "draft": True, "prerelease": False}
    assert classify_final_release("v0.21.5", "rc.1-v0.21.5", draft) == ("green", True)


def test_a_burned_claim_is_spent_and_skipped():
    from scripts.releases.sequencer import classify_final_release, plan

    assert classify_final_release("v0.21.5", "v0.21.5-rc", None) == ("burned", False)

    steps = plan(_claims(
        ("0.21.5", "burned", False),
        ("0.21.6", "green", True),
    ), head="0.21.4")

    assert _flips(steps) == ["0.21.6"]
    assert steps[-1] == {"advance": "0.21.6"}


def test_advances_each_published_version_and_ignores_a_later_burned_claim():
    from scripts.releases.sequencer import plan

    assert plan(_claims(
        ("0.21.5", "published", False),
        ("0.21.6", "published", False),
    ), head="0.21.4") == [
        {"advance": "0.21.5"},
        {"advance": "0.21.6"},
    ]
    assert plan(_claims(
        ("0.21.5", "green", False),
        ("0.21.6", "burned", False),
    ), head="0.21.4") == []


def test_explicit_publish_uses_the_same_oldest_first_plan():
    from scripts.releases.sequencer import plan

    assert plan(_claims(
        ("0.21.5", "green", False),
        ("0.21.6", "green", False),
    ), head="0.21.4", requested_version="0.21.6") == [
        {"flip": "0.21.5"},
        {"advance": "0.21.5"},
        {"flip": "0.21.6"},
        {"advance": "0.21.6"},
    ]


def test_published_history_at_or_below_the_head_is_an_idempotent_noop():
    from scripts.releases.sequencer import plan

    assert plan(_claims(
        ("0.21.4", "published", False),
        ("0.21.5", "green", False),
    ), head="0.21.4") == []


def test_an_unpublished_claim_below_the_head_is_refused():
    from scripts.releases.sequencer import plan

    with pytest.raises(ValueError, match="backwards"):
        plan(_claims(("0.21.4", "green", True)), head="0.21.5")


def _claim_message(version, attempt, commit, *, autopublish=False,
                   epoch=1_790_000_000):
    return {"schema": 1, "version": version, "attempt": attempt, "commit": commit,
            "autopublish": autopublish, "skipBundles": False, "skipTests": False,
            "claimEpoch": epoch}


def _final_message(version, attempt, commit, *, release_id, epoch=1_790_000_000):
    claim_tag = f"rc.{attempt}-v{version}"
    return {
        "schema": 1, "version": version, "commit": commit,
        "autopublish": False, "claimEpoch": epoch, "claimTag": claim_tag,
        "claimTagObject": "1" * 40, "releaseId": release_id,
        "candidateManifestSha256": "a" * 64, "dockerManifestDigest": "sha256:" + "b" * 64,
        "archive": f"releases/tag/{claim_tag}/",
    }


def _discover_run(tags, releases=(), workflow_runs=()):
    def run(argv):
        if argv[:2] == ["git", "fetch"]:
            return ""
        if argv[:3] == ["git", "ls-remote", "--tags"]:
            return "\n".join(
                f"{sha}\trefs/tags/{tag}\n{commit}\trefs/tags/{tag}^{{}}"
                for tag, (sha, commit, _message) in tags.items()
            )
        if argv[:2] == ["git", "rev-parse"]:
            return tags[argv[-1].removeprefix("refs/tags/")][0]
        if argv[:3] == ["git", "cat-file", "-t"]:
            return "tag"
        if argv[:3] == ["git", "cat-file", "-p"]:
            message = next(m for s, _c, m in tags.values() if s == argv[3])
            return f"tagger Fixture <fixture@example.test> {message['claimEpoch']} +0000\n"
        if argv[:3] == ["git", "tag", "-l"]:
            return json.dumps(tags[argv[3]][2])
        if argv[:4] == ["gh", "api", "--paginate", "--slurp"]:
            if "/releases?" in argv[-1]:
                return json.dumps([list(releases)])
            return json.dumps([{"workflow_runs": list(workflow_runs)}])
        raise AssertionError(argv)

    return run


def test_discover_reads_an_attempt_ref_and_keeps_the_version():
    from scripts.releases.sequencer import discover

    commit = "a" * 40
    tags = {
        "rc.2-v0.21.5": ("1" * 40, commit, _claim_message("0.21.5", 2, commit)),
        "v0.21.5": ("2" * 40, commit,
                    _final_message("0.21.5", 2, commit, release_id=7)),
    }
    releases = [{
        "id": 7, "tag_name": "v0.21.5", "draft": False, "prerelease": False,
        "published_at": "2026-09-22T01:00:00Z",
    }]
    records = discover("example/project", _discover_run(tags, releases=releases))
    assert records[0]["version"] == "0.21.5"
    assert records[0]["claim_tag"] == "rc.2-v0.21.5"
    assert records[0]["state"] == "published"
    assert records[0]["claim_epoch"] == 1_790_000_000


def test_a_marker_ref_clears_the_attempt():
    from scripts.releases.sequencer import discover

    commit = "a" * 40
    tags = {
        "rc.1-v0.21.5": ("1" * 40, commit, _claim_message("0.21.5", 1, commit)),
        "abandoned-rc.1-v0.21.5": ("2" * 40, commit, {}),
    }
    failed = {
        "id": 42, "status": "completed", "conclusion": "failure",
        "run_attempt": 1, "updated_at": "2026-09-22T01:14:59Z",
        "head_branch": "rc.1-v0.21.5", "head_sha": commit,
    }
    records = discover("example/project",
                       _discover_run(tags, workflow_runs=[failed]))
    assert records[0]["state"] == "burned"


def test_two_outstanding_attempts_are_refused_across_versions():
    from scripts.releases.sequencer import discover

    commit = "a" * 40
    tags = {
        "rc.1-v0.21.5": ("1" * 40, commit, _claim_message("0.21.5", 1, commit)),
        "rc.1-v0.22.0": ("2" * 40, commit, _claim_message("0.22.0", 1, commit)),
    }
    with pytest.raises(ValueError, match="more than one outstanding attempt"):
        discover("example/project", _discover_run(tags))


def _sequencer_fixture(*versions, manifest_digest, docker_digest="sha256:" + "b" * 64,
                       drafts_on_claim_tag=False, skip_bundles=False):
    """Two green claims with their final tags; releases start as drafts."""
    commit = "a" * 40
    tags = {}
    releases = []
    for index, version in enumerate(versions, start=1):
        claim_tag, tag = f"rc.1-v{version}", f"v{version}"
        claim_object, final_object = str(index) * 40, str(index + 2) * 40
        claim = {
            "schema": 1, "version": version, "attempt": 1, "commit": commit,
            "autopublish": False, "skipBundles": skip_bundles, "skipTests": False,
            "claimEpoch": 1_790_000_000 + index,
        }
        final = {
            "schema": 1, "version": version, "commit": commit,
            "autopublish": False, "claimEpoch": claim["claimEpoch"],
            "claimTag": claim_tag, "claimTagObject": claim_object,
            "releaseId": index,
            "candidateManifestSha256": manifest_digest,
            "dockerManifestDigest": docker_digest,
            "archive": f"releases/tag/{claim_tag}/",
        }
        tags[claim_tag] = (claim_object, commit, claim)
        tags[tag] = (final_object, commit, final)
        releases.append({
            "id": index, "tag_name": claim_tag if drafts_on_claim_tag else tag,
            "draft": True, "prerelease": False, "published_at": None,
            "body": "notes",
        })

    def run(argv):
        if argv[:2] == ["git", "fetch"]:
            return ""
        if argv[:3] == ["git", "ls-remote", "--tags"]:
            return "\n".join(
                f"{sha}\trefs/tags/{tag}\n{target}\trefs/tags/{tag}^{{}}"
                for tag, (sha, target, _message) in tags.items()
            )
        if argv[:2] == ["git", "ls-remote"]:
            lines = []
            for ref in argv[2:]:
                name = ref.removeprefix("refs/tags/").removesuffix("^{}")
                entry = tags.get(name)
                sha, target = (entry[0], entry[1]) if entry else ("", "")
                if sha:
                    lines.append(f"{sha}\t{ref if ref.endswith('^{}') else ref}")
                    if ref.endswith("^{}"):
                        lines[-1] = f"{target}\t{ref}"
            return "\n".join(lines)
        if argv[:2] == ["git", "rev-parse"]:
            return tags[argv[-1].removeprefix("refs/tags/")][0]
        if argv[:3] == ["git", "cat-file", "-t"]:
            return "tag"
        if argv[:3] == ["git", "cat-file", "-p"]:
            epoch = next(message["claimEpoch"] for object_id, _target, message in tags.values()
                         if object_id == argv[3])
            return f"tagger Fixture <fixture@example.test> {epoch} +0000\n"
        if argv[:3] == ["git", "tag", "-l"]:
            return json.dumps(tags[argv[3]][2])
        if "tag" in argv and "-a" in argv:
            tag = argv[argv.index("-a") + 1]
            tags[tag] = (str(len(tags)) * 40, tags[claim_tag][1], tags[tag][2])
            return ""
        if argv[:2] == ["git", "push"]:
            return ""
        if argv[:3] == ["docker", "buildx", "imagetools"]:
            return json.dumps(docker_digest)
        if argv[:4] == ["gh", "api", "--paginate", "--slurp"]:
            if "/releases?" in argv[-1]:
                return json.dumps([releases])
            return json.dumps([{"workflow_runs": []}])
        if argv[:3] == ["gh", "api", "--method"]:
            release = next(row for row in releases if row["id"] == int(argv[4].rsplit("/", 1)[1]))
            for _flag, value in zip(argv[5::2], argv[6::2]):
                name, _, raw = value.partition("=")
                if name == "tag_name":
                    release["tag_name"] = raw
                elif name == "draft":
                    release["draft"] = raw == "true"
                    if not release["draft"]:
                        release["published_at"] = "2026-09-22T01:00:00Z"
                elif name == "body":
                    release["body"] = raw
                elif name == "prerelease":
                    release["prerelease"] = raw == "true"
            return "{}"
        if argv[:2] == ["gh", "api"] and "/releases/" in argv[2]:
            release = next(row for row in releases if row["id"] == int(argv[2].rsplit("/", 1)[1]))
            return json.dumps(release)
        raise AssertionError(argv)

    return tags, releases, run


def test_reconcile_discovers_custody_flips_then_advances_oldest_first():
    from scripts.releases.sequencer import reconcile

    manifest_digest = hashlib.sha256(b"m").hexdigest()
    _tags, releases, run = _sequencer_fixture("0.21.5", "0.21.6", manifest_digest=manifest_digest)
    events = []
    archive_keys = []
    head = ["0.21.4"]

    def read_archive(key):
        archive_keys.append(key)
        return b"m"

    def advance(record):
        events.append(("advance", record["tag"]))
        head[0] = record["version"]

    steps = reconcile(
        {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.6"},
        run=run, read_head=lambda: head[0], advance_head=advance,
        read_archive=read_archive,
    )

    assert steps == [
        {"flip": "0.21.5"}, {"advance": "0.21.5"},
        {"flip": "0.21.6"}, {"advance": "0.21.6"},
    ]
    assert events == [
        ("advance", "v0.21.5"), ("advance", "v0.21.6"),
    ]
    # Each pass hashes the manifest from its own attempt archive path.
    assert archive_keys == [
        "releases/tag/rc.1-v0.21.5/release-candidates.json",
        "releases/tag/rc.1-v0.21.6/release-candidates.json",
    ]
    for release, version in zip(releases, ("0.21.5", "0.21.6")):
        assert release["tag_name"] == f"v{version}" and release["draft"] is False


def test_the_store_check_joins_the_pass_after_the_aliases_move(monkeypatch):
    from scripts.releases import channel_releases, docker, sequencer, store

    manifest_digest = hashlib.sha256(b"m").hexdigest()
    _tags, releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest_digest, drafts_on_claim_tag=True)
    events = []
    head = ["0.21.4"]
    monkeypatch.setattr(
        channel_releases, "advance_stable",
        lambda env, record, root: (head.append(record["version"]),
                                   events.append(("feed", record["version"]))))
    monkeypatch.setattr(
        docker, "promote_stable",
        lambda claim_tag, digest: events.append(("aliases", claim_tag)))
    seen_env = []
    monkeypatch.setattr(
        store, "check_from_env",
        lambda env: seen_env.append(env) or events.append(("store",)))

    steps = sequencer.reconcile(
        {"GITHUB_REPOSITORY": "example/project", "MS_STORE_PRODUCT_ID": "9NTEST"},
        run=run, read_head=lambda: head[-1], advance_head=None,
        read_archive=lambda _key: b"m",
    )

    assert steps == [{"advance": "0.21.5"}]
    assert events == [("feed", "0.21.5"), ("aliases", "rc.1-v0.21.5"), ("store",)]
    assert seen_env[0]["MS_STORE_PRODUCT_ID"] == "9NTEST"


def test_a_publish_that_died_before_the_retarget_is_repaired():
    from scripts.releases.sequencer import reconcile

    manifest_digest = hashlib.sha256(b"m").hexdigest()
    _tags, releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest_digest, drafts_on_claim_tag=True)
    events = []
    head = ["0.21.4"]

    def advance(record):
        events.append(("advance", record["tag"]))
        head[0] = record["version"]

    steps = reconcile(
        {"GITHUB_REPOSITORY": "example/project"},
        run=run, read_head=lambda: head[0], advance_head=advance,
        read_archive=lambda _key: b"m",
    )

    # The final tag existed and the draft was still on the attempt ref: the
    # retarget and the publication rerun, then the head advances.
    assert steps == [{"advance": "0.21.5"}]
    assert events == [("advance", "v0.21.5")]
    assert releases[0]["tag_name"] == "v0.21.5" and releases[0]["draft"] is False


def test_a_release_that_skipped_bundles_ships_only_its_tag_release_and_docker_aliases(monkeypatch):
    """Its receipt binds no manifest; the R2 head and Store stay put; one pass finishes it."""
    from scripts.releases import channel_releases, docker, sequencer, store

    docker_digest = "sha256:" + "b" * 64
    _tags, releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=None, docker_digest=docker_digest, skip_bundles=True)
    alias = [None]
    events = []

    def promote(claim_tag, digest):
        events.append(("aliases", claim_tag))
        alias[0] = digest

    monkeypatch.setattr(channel_releases, "advance_stable",
                        lambda *_args: pytest.fail("the protected R2 head moved"))
    monkeypatch.setattr(store, "check_from_env", lambda _env: pytest.fail("the Store was checked"))
    monkeypatch.setattr(channel_releases, "stable_head_version", lambda _env: "0.21.4")
    monkeypatch.setattr(docker, "promote_stable", promote)
    monkeypatch.setattr(docker, "stable_alias_digest", lambda: alias[0])
    env = {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.5"}

    def no_archive(_key):
        pytest.fail("a release that skipped bundles has no archive to read")

    steps = sequencer.reconcile(env, run=run, read_archive=no_archive)

    assert steps == [{"flip": "0.21.5"}, {"advance": "0.21.5"}]
    assert events == [("aliases", "rc.1-v0.21.5")]
    assert releases[0]["tag_name"] == "v0.21.5" and releases[0]["draft"] is False
    # The R2 head still names 0.21.4, but the stable alias carries the 0.21.5
    # receipt digest, so the next pass has nothing left to advance.
    assert sequencer.reconcile(env, run=run, read_archive=no_archive) == []
    assert events == [("aliases", "rc.1-v0.21.5")]
