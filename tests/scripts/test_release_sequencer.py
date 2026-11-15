"""Stable publication is ordered, held by default, and idempotent.

A green draft publishes when it opted into autopublish, or when a later green
release needs it resolved. A running older claim blocks only the versions above
it; already-resolvable older green claims still make progress.
"""
import hashlib
import json
import subprocess

import pytest
from datetime import UTC


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

    claimed = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)
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
        "updated_at": datetime.now(UTC).isoformat(),
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
            # git lists refs sorted by name (rc.10 sorts before rc.2).
            return "\n".join(
                f"{sha}\trefs/tags/{tag}\n{commit}\trefs/tags/{tag}^{{}}"
                for tag, (sha, commit, _message) in sorted(tags.items())
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


def test_abandoned_attempts_before_a_later_winner_are_burned_not_compared_to_the_final_tag():
    """v0.21.6's shape: rc.1-3 abandoned, rc.4 won and became v0.21.6. The final tag names
    rc.4; the abandoned predecessors sit at other commits and must reconcile as burned
    instead of crashing every later pass with "v0.21.6 points at a different commit"."""
    from scripts.releases.sequencer import discover

    commits = {n: str(n) * 40 for n in (1, 2, 3, 4)}
    tags = {}
    for n, commit in commits.items():
        tags[f"rc.{n}-v0.21.6"] = (f"{n}a".ljust(40, "0"), commit,
                                   _claim_message("0.21.6", n, commit, epoch=1_790_000_000 + n))
        if n < 4:
            tags[f"abandoned-rc.{n}-v0.21.6"] = (f"{n}b".ljust(40, "0"), commit, {})
    final = _final_message("0.21.6", 4, commits[4], release_id=9, epoch=1_790_000_004)
    final["claimTagObject"] = tags["rc.4-v0.21.6"][0]  # the final tag binds the winning claim's object
    tags["v0.21.6"] = ("f" * 40, commits[4], final)
    releases = [{"id": 9, "tag_name": "v0.21.6", "draft": False, "prerelease": False,
                 "published_at": "2026-10-08T11:51:57Z"}]
    records = {r["claim_tag"]: r for r in discover("example/project", _discover_run(tags, releases=releases))}
    assert {tag: r["state"] for tag, r in records.items()} == {
        "rc.1-v0.21.6": "burned", "rc.2-v0.21.6": "burned", "rc.3-v0.21.6": "burned",
        "rc.4-v0.21.6": "published",
    }


def test_a_tenth_attempt_is_the_version_s_record_not_the_ninth():
    """git sorts rc.10 between rc.1 and rc.2; discover orders attempts numerically so the
    newest attempt is last and reconcile hands IT (not rc.9) to publish."""
    from scripts.releases.sequencer import discover

    tags = {}
    for n in range(1, 11):
        commit = f"{n:x}".rjust(40, "c")
        tags[f"rc.{n}-v0.21.7"] = (f"{n:x}".rjust(40, "a"), commit,
                                   _claim_message("0.21.7", n, commit, epoch=1_790_000_000 + n))
        if n < 10:
            tags[f"abandoned-rc.{n}-v0.21.7"] = (f"{n:x}".rjust(40, "b"), commit, {})
    green = {"id": 77, "status": "completed", "conclusion": "success", "run_attempt": 1,
             "updated_at": "2026-10-11T01:00:00Z", "head_branch": "rc.10-v0.21.7",
             "head_sha": tags["rc.10-v0.21.7"][1]}
    draft = {"id": 11, "tag_name": "rc.10-v0.21.7", "draft": True, "prerelease": False}
    records = discover("example/project", _discover_run(tags, releases=[draft], workflow_runs=[green]))
    assert [r["attempt"] for r in records] == list(range(1, 11))
    newest = {r["version"]: r for r in records}["0.21.7"]
    assert (newest["claim_tag"], newest["state"], newest["release_id"]) == ("rc.10-v0.21.7", "green", 11)


def test_a_same_commit_retry_after_an_abandoned_attempt_publishes():
    """rc.1 abandoned, rc.2 re-cut at the SAME commit wins: rc.1 is burned (its marker), not
    compared against v0.21.6's metadata, which names rc.2."""
    from scripts.releases.sequencer import discover

    commit = "a" * 40
    tags = {
        "rc.1-v0.21.6": ("1" * 40, commit, _claim_message("0.21.6", 1, commit, epoch=1_790_000_001)),
        "abandoned-rc.1-v0.21.6": ("3" * 40, commit, {}),
        "rc.2-v0.21.6": ("2" * 40, commit, _claim_message("0.21.6", 2, commit, epoch=1_790_000_002)),
    }
    final = _final_message("0.21.6", 2, commit, release_id=9, epoch=1_790_000_002)
    final["claimTagObject"] = "2" * 40
    tags["v0.21.6"] = ("f" * 40, commit, final)
    releases = [{"id": 9, "tag_name": "v0.21.6", "draft": False, "prerelease": False,
                 "published_at": "2026-10-08T11:51:57Z"}]
    records = discover("example/project", _discover_run(tags, releases=releases))
    assert [(r["claim_tag"], r["state"]) for r in records] == [
        ("rc.1-v0.21.6", "burned"), ("rc.2-v0.21.6", "published")]


def test_a_final_tag_at_another_commit_than_a_live_attempt_is_still_refused():
    """Only an abandoned attempt is exempt: a NON-abandoned attempt whose version's final tag
    points elsewhere is a real inconsistency and keeps failing loudly."""
    from scripts.releases.sequencer import discover

    tags = {
        "rc.1-v0.21.6": ("1a".ljust(40, "0"), "1" * 40, _claim_message("0.21.6", 1, "1" * 40)),
        "v0.21.6": ("f" * 40, "2" * 40, _final_message("0.21.6", 2, "2" * 40, release_id=9)),
    }
    with pytest.raises(ValueError, match="points at a different commit than rc.1-v0.21.6"):
        discover("example/project", _discover_run(tags))


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
                       drafts_on_claim_tag=False, skip_bundles=False, final_tags=True, lanes=None):
    """Two green claims with their final tags; releases start as drafts.

    final_tags=False is a fresh green claim: no receipt tag yet, the draft on
    the attempt ref, and one successful Stable Release run.
    """
    commit = "a" * 40
    tags = {}
    releases = []
    workflow_runs = []
    for index, version in enumerate(versions, start=1):
        claim_tag, tag = f"rc.1-v{version}", f"v{version}"
        claim_object, final_object = str(index) * 40, str(index + 2) * 40
        claim = {
            "schema": 1, "version": version, "attempt": 1, "commit": commit,
            "autopublish": False, "skipBundles": skip_bundles, "skipTests": False,
            "claimEpoch": 1_790_000_000 + index,
        }
        if lanes is not None:
            claim = {**{k: v for k, v in claim.items() if not k.startswith("skip")}, "lanes": sorted(lanes)}
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
        if final_tags:
            tags[tag] = (final_object, commit, final)
        else:
            workflow_runs.append({"id": 100 + index, "run_attempt": 1, "status": "completed",
                                  "conclusion": "success", "head_branch": claim_tag,
                                  "head_sha": commit})
        releases.append({
            "id": index, "tag_name": claim_tag if drafts_on_claim_tag or not final_tags else tag,
            "draft": True, "prerelease": False, "published_at": None,
            "body": "notes",
        })

    def rev_parse(argv):
        # git fails for a tag that does not exist (a fresh claim has no receipt yet).
        name = argv[-1].removeprefix("refs/tags/")
        if name not in tags:
            raise subprocess.CalledProcessError(128, argv)
        return tags[name][0]

    def create_tag(argv):
        # A new receipt records the message the code wrote; a repeated one keeps its own.
        tag = argv[argv.index("-a") + 1]
        message = tags[tag][2] if tag in tags else json.loads(argv[argv.index("-m") + 1])
        tags[tag] = (str(len(tags)) * 40, argv[argv.index("-a") + 2], message)
        return ""

    def run(argv):
        if argv[:2] == ["git", "fetch"]:
            return ""
        if argv[:3] == ["git", "ls-remote", "--tags"]:
            # git lists refs sorted by name (rc.10 sorts before rc.2).
            return "\n".join(
                f"{sha}\trefs/tags/{tag}\n{target}\trefs/tags/{tag}^{{}}"
                for tag, (sha, target, _message) in sorted(tags.items())
            )
        if argv[:2] == ["git", "ls-remote"]:
            lines = []
            for ref in argv[2:]:
                name = ref.removeprefix("refs/tags/").removesuffix("^{}")
                entry = tags.get(name)
                sha, target = (entry[0], entry[1]) if entry else ("", "")
                if sha:
                    lines.append(f"{sha}\t{ref}")
                    if ref.endswith("^{}"):
                        lines[-1] = f"{target}\t{ref}"
            return "\n".join(lines)
        if argv[:2] == ["git", "rev-parse"]:
            return rev_parse(argv)
        if argv[:3] == ["git", "cat-file", "-t"]:
            return "tag"
        if argv[:3] == ["git", "cat-file", "-p"]:
            epoch = next(message["claimEpoch"] for object_id, _target, message in tags.values()
                         if object_id == argv[3])
            return f"tagger Fixture <fixture@example.test> {epoch} +0000\n"
        if argv[:3] == ["git", "tag", "-l"]:
            return json.dumps(tags[argv[3]][2])
        if "tag" in argv and "-a" in argv:
            return create_tag(argv)
        if argv[0] == "git" and "push" in argv:
            return ""
        if argv[:3] == ["docker", "buildx", "imagetools"]:
            return json.dumps(docker_digest)
        if argv[:4] == ["gh", "api", "--paginate", "--slurp"]:
            if "/releases?" in argv[-1]:
                return json.dumps([releases])
            return json.dumps([{"workflow_runs": workflow_runs}])
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


def test_a_fresh_flip_advances_the_head_with_the_receipt_it_just_wrote():
    """The normal publish: no receipt tag when discover() runs, flip and advance in one pass.

    discover() records no receipt digests for a claim without its final tag, so
    the advance must see the ones publish_attempt() just wrote. Otherwise the
    channel advance compares the archive digest against None and every
    first-pass publish fails after making the release public (0.21.7).
    """
    from scripts.releases.sequencer import reconcile

    manifest_digest = hashlib.sha256(b"m").hexdigest()
    docker_digest = "sha256:" + "c" * 64
    tags, releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest_digest, docker_digest=docker_digest, final_tags=False)
    assert "v0.21.5" not in tags
    head = ["0.21.4"]
    advanced = []

    def advance(record):
        advanced.append({key: record[key] for key in (
            "tag", "candidate_manifest_sha256", "docker_manifest_digest")})
        head[0] = record["version"]

    steps = reconcile(
        {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.5"},
        run=run, read_head=lambda: head[0], advance_head=advance,
        read_archive=lambda _key: b"m",
    )

    assert steps == [{"flip": "0.21.5"}, {"advance": "0.21.5"}]
    assert advanced == [{"tag": "v0.21.5", "candidate_manifest_sha256": manifest_digest,
                         "docker_manifest_digest": docker_digest}]
    assert tags["v0.21.5"][2]["candidateManifestSha256"] == manifest_digest
    assert releases[0]["tag_name"] == "v0.21.5" and releases[0]["draft"] is False


def test_the_store_release_joins_the_pass_after_the_aliases_move(monkeypatch):
    from scripts.releases import channel_releases, docker, sequencer, store

    manifest = json.dumps({"files": [{"path": "Store-Product-win.msixbundle"}]}).encode()
    manifest_digest = hashlib.sha256(manifest).hexdigest()
    _tags, _releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest_digest, drafts_on_claim_tag=True)
    events = []
    head = ["0.21.4"]
    monkeypatch.setattr(
        channel_releases, "advance_stable",
        lambda env, record, root: (head.append(record["version"]),
                                   events.append(("feed", record["version"]))))
    monkeypatch.setattr(
        docker, "promote_stable",
        lambda claim_tag, digest, release_tag: events.append(("aliases", claim_tag, release_tag)))
    seen = []
    monkeypatch.setattr(
        store, "release_from_env",
        lambda env, manifest: seen.append((env, manifest)) or events.append(("store",)))

    steps = sequencer.reconcile(
        {"GITHUB_REPOSITORY": "example/project", "MS_STORE_PRODUCT_ID": "9NTEST"},
        run=run, read_head=lambda: head[-1], advance_head=None,
        read_archive=lambda _key: manifest,
    )

    assert steps == [{"advance": "0.21.5"}]
    assert events == [("feed", "0.21.5"), ("aliases", "rc.1-v0.21.5", "v0.21.5"), ("store",)]
    # The Store releases the draft named by the receipt-bound archived manifest.
    assert seen[0][0]["MS_STORE_PRODUCT_ID"] == "9NTEST"
    assert seen[0][1] == json.loads(manifest)


@pytest.mark.parametrize("recovery_head", ["0.21.5", "0.21.6"])
def test_store_release_recovers_after_alias_movement_without_rewinding_channels(monkeypatch, recovery_head):
    from scripts.releases import channel_releases, docker, sequencer, store
    from tests.scripts.test_release_store import STORE_FILE, _Api

    manifest = json.dumps({"files": [{"path": STORE_FILE}]}).encode()
    _tags, _releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=hashlib.sha256(manifest).hexdigest(), drafts_on_claim_tag=True)
    head = ["0.21.4"]
    events = []
    api = _Api()
    release_from_env = store.release_from_env
    env = {"GITHUB_REPOSITORY": "example/project", "MS_STORE_PRODUCT_ID": "9NTEST",
           "MS_STORE_TENANT_ID": "T", "MS_STORE_CLIENT_ID": "C", "MS_STORE_CLIENT_SECRET": "S"}

    def advance(_env, record, _root):
        head[0] = record["version"]
        events.append("feed")

    def interrupt_after_aliases(*_args):
        events.append("aliases")
        raise KeyboardInterrupt("cancelled after aliases moved")

    def release_store(environment, candidate):
        events.append("store")
        return release_from_env(environment, candidate, run=api)

    monkeypatch.setattr(channel_releases, "advance_stable", advance)
    monkeypatch.setattr(docker, "promote_stable", interrupt_after_aliases)
    monkeypatch.setattr(store, "release_from_env", release_store)
    with pytest.raises(KeyboardInterrupt, match="after aliases moved"):
        sequencer.reconcile(env, run=run, read_head=lambda: head[0], read_archive=lambda _key: manifest)
    assert head[0] == "0.21.5" and events == ["feed", "aliases"]
    assert api.mode == "Manual" and api.commits == 0

    # Channel movement is already durable; retry only the receipt-bound Store work,
    # even if another publication has since advanced the channels further.
    head[0] = recovery_head
    monkeypatch.setattr(channel_releases, "advance_stable",
                        lambda *_args: pytest.fail("Store recovery moved the desktop head"))
    monkeypatch.setattr(docker, "promote_stable",
                        lambda *_args: pytest.fail("Store recovery moved the Docker aliases"))
    assert sequencer.reconcile(
        env, run=run, read_head=lambda: head[0], read_archive=lambda _key: manifest) == []
    assert head[0] == recovery_head and events == ["feed", "aliases", "store"]
    assert api.mode == "Immediate" and api.commits == 1
    sequencer.reconcile(env, run=run, read_head=lambda: head[0], read_archive=lambda _key: manifest)
    assert api.commits == 1


@pytest.mark.parametrize("pending_is_latest", [True, False])
def test_store_reconciliation_finds_the_newest_matching_published_submission(monkeypatch, pending_is_latest):
    from scripts.releases import channel_releases, docker, sequencer, store
    from tests.scripts.test_release_store import STORE_FILE, _Api

    newest_file = "Store-HermesBundled-2026.7000.1.0-win.msixbundle"
    manifests = {
        f"releases/tag/rc.1-v{version}/release-candidates.json":
            json.dumps({"files": [{"path": filename}]}).encode()
        for version, filename in [("0.21.5", STORE_FILE), ("0.21.6", newest_file)]
    }
    tags, releases, run = _sequencer_fixture("0.21.5", "0.21.6", manifest_digest=None)
    for version, raw in zip(("0.21.5", "0.21.6"), manifests.values()):
        tags[f"v{version}"][2]["candidateManifestSha256"] = hashlib.sha256(raw).hexdigest()
    for row in releases:
        row.update(draft=False, published_at="2026-09-22T01:00:00Z")
    api = _Api(files=[newest_file if pending_is_latest else STORE_FILE])
    seen = []
    release_from_env = store.release_from_env

    def release_store(env, manifest):
        result = release_from_env(env, manifest, run=api)
        seen.append((store.store_bundle(manifest), result))
        return result

    monkeypatch.setattr(store, "release_from_env", release_store)
    monkeypatch.setattr(channel_releases, "advance_stable",
                        lambda *_args: pytest.fail("Store reconciliation moved the desktop head"))
    monkeypatch.setattr(docker, "promote_stable",
                        lambda *_args: pytest.fail("Store reconciliation moved the Docker aliases"))
    env = {"GITHUB_REPOSITORY": "example/project", "MS_STORE_PRODUCT_ID": "9NTEST",
           "MS_STORE_TENANT_ID": "T", "MS_STORE_CLIENT_ID": "C", "MS_STORE_CLIENT_SECRET": "S"}
    assert sequencer.reconcile(
        env, run=run, read_head=lambda: "0.21.6", read_archive=manifests.__getitem__) == []
    # A newer public release may have failed its non-blocking Store upload;
    # only a package mismatch permits trying an older public release.
    assert seen == ([(newest_file, "immediate")] if pending_is_latest else
                    [(newest_file, "superseded"), (STORE_FILE, "immediate")])
    assert api.mode == "Immediate" and api.commits == 1


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

    def promote(claim_tag, digest, release_tag):
        events.append(("aliases", claim_tag, release_tag))
        alias[0] = digest

    monkeypatch.setattr(channel_releases, "advance_stable",
                        lambda *_args: pytest.fail("the protected R2 head moved"))
    monkeypatch.setattr(store, "release_from_env", lambda *_a: pytest.fail("the Store was released"))
    monkeypatch.setattr(channel_releases, "stable_head_version", lambda _env: "0.21.4")
    monkeypatch.setattr(docker, "promote_stable", promote)
    monkeypatch.setattr(docker, "stable_alias_digest", lambda: alias[0])
    env = {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.5"}

    def no_archive(_key):
        pytest.fail("a release that skipped bundles has no archive to read")

    steps = sequencer.reconcile(env, run=run, read_archive=no_archive)

    assert steps == [{"flip": "0.21.5"}, {"advance": "0.21.5"}]
    assert events == [("aliases", "rc.1-v0.21.5", "v0.21.5")]
    assert releases[0]["tag_name"] == "v0.21.5" and releases[0]["draft"] is False
    # The R2 head still names 0.21.4, but the stable alias carries the 0.21.5
    # receipt digest, so the next pass has nothing left to advance.
    assert sequencer.reconcile(env, run=run, read_archive=no_archive) == []
    assert events == [("aliases", "rc.1-v0.21.5", "v0.21.5")]


@pytest.mark.parametrize("lanes,docker,store_checked", [
    # --only windows,docker: a partial desktop set. Only the aliases move; the channel stays.
    (["windows", "docker"], True, True),
    # --only windows: no channel moves at all, so the release is finished once public.
    (["windows"], False, True),
    # --only macos-arm64,docker: no Store, no channel.
    (["macos-arm64", "docker"], True, False),
])
def test_a_partial_desktop_release_moves_only_what_it_built(monkeypatch, lanes, docker, store_checked):
    from scripts.releases import channel_releases, docker as docker_module, sequencer, store

    docker_digest = "sha256:" + "b" * 64 if docker else None
    archived = json.dumps({"files": [{"path": "Store-Product-win.msixbundle"}]}).encode()
    manifest = hashlib.sha256(archived).hexdigest()
    _tags, releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest, docker_digest=docker_digest, lanes=lanes)
    alias = [None]
    events = []

    def promote(claim_tag, digest, release_tag):
        events.append(("aliases", claim_tag, release_tag))
        alias[0] = digest

    monkeypatch.setattr(channel_releases, "advance_stable",
                        lambda *_args: pytest.fail("a partial desktop set moved the protected head"))
    monkeypatch.setattr(store, "release_from_env", lambda *_a: events.append(("store",)))
    monkeypatch.setattr(channel_releases, "stable_head_version", lambda _env: "0.21.4")
    monkeypatch.setattr(docker_module, "promote_stable", promote)
    monkeypatch.setattr(docker_module, "stable_alias_digest", lambda: alias[0])
    env = {"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.5"}

    steps = sequencer.reconcile(env, run=run, read_archive=lambda _key: archived)

    assert steps == [{"flip": "0.21.5"}, {"advance": "0.21.5"}]
    assert events == ([("aliases", "rc.1-v0.21.5", "v0.21.5")] if docker else []) + \
        ([("store",)] if store_checked else [])
    assert releases[0]["tag_name"] == "v0.21.5" and releases[0]["draft"] is False
    # Finished: the alias carries the receipt's digest, or the release moves no channel at all.
    assert sequencer.reconcile(env, run=run, read_archive=lambda _key: archived) == []


def test_the_sequencer_narrates_every_phase_on_stderr_and_keeps_stdout_json(capsys):
    """Discovery, each claim's state and why, the plan, every step and every command."""
    from scripts.releases.sequencer import reconcile

    manifest_digest = hashlib.sha256(b"m").hexdigest()
    _tags, _releases, run = _sequencer_fixture(
        "0.21.5", manifest_digest=manifest_digest, final_tags=False)
    head = ["0.21.4"]

    def advance(record):
        head[0] = record["version"]

    reconcile({"GITHUB_REPOSITORY": "example/project", "REQUESTED_VERSION": "0.21.5"},
              run=run, read_head=lambda: head[0], advance_head=advance,
              read_archive=lambda _key: b"m")
    captured = capsys.readouterr()
    assert captured.out == ""
    err = captured.err
    for expected in (
        "Stable sequencer: repository=example/project requested=0.21.5",
        "::group::Discover stable claims",
        "rc.1-v0.21.5 @ aaaaaaaaaa: GREEN — run 101 attempt 1 completed/success; draft on rc.1-v0.21.5",
        "Channel head: 0.21.4",
        "plan 0.21.5: green — publish then advance (requested)",
        "Steps: flip 0.21.5, advance 0.21.5",
        "[1/2] Publish 0.21.5",
        f"Receipt v0.21.5: manifest {manifest_digest}",
        "::notice title=Stable sequencer::v0.21.5 is public",
        "[2/2] Advance channels to 0.21.5",
        "Channel head after the pass: 0.21.5 (planned 0.21.5)",
        "Stable sequencer finished: flip 0.21.5, advance 0.21.5",
        "  $ git push origin refs/tags/v0.21.5",
    ):
        assert expected in err, expected
    assert err.count("::group::") == err.count("::endgroup::")


def test_a_held_claim_says_why_it_was_held(capsys):
    from scripts.releases.sequencer import plan

    plan([{"version": "0.21.5", "state": "green", "autopublish": False}], head="0.21.4")
    assert "plan 0.21.5: green but held" in capsys.readouterr().err


def test_main_prints_only_the_steps_on_stdout_and_annotates_a_failure(monkeypatch, capsys):
    from scripts.releases import sequencer

    monkeypatch.setattr(sequencer, "reconcile", lambda env: [{"advance": "0.21.5"}])
    sequencer.main([], {})
    assert json.loads(capsys.readouterr().out) == [{"advance": "0.21.5"}]

    def boom(env):
        raise ValueError("Stable protected head did not reach the planned version")

    monkeypatch.setattr(sequencer, "reconcile", boom)
    with pytest.raises(ValueError):
        sequencer.main([], {})
    captured = capsys.readouterr()
    assert captured.out == ""
    assert ("::error title=Stable sequencer failed::ValueError: "
            "Stable protected head did not reach the planned version") in captured.err
