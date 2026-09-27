"""The thin release entrypoint: claim an attempt, cut the draft, dispatch the gate.

Nothing here builds. The claim is an annotated attempt ref ``rc.<N>-vX.Y.Z``
pushed as exactly that ref. A version is spent only by publication: an attempt
that fails is abandoned with a marker ref, and the next cut is attempt N+1 of
the same version. At most one attempt, of any version, is outstanding.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scripts.releases.draft_warning import draft_body
from scripts.releases.versioning import (
    SEED, attempt_ref, derive_next_version, marker_ref, next_attempt,
    outstanding_attempts, parse_attempt_ref, parse_marker_ref,
)

WORKFLOW = "stable-release.yml"
# A fetch refspec may hold one ``*``; the parsers filter what the globs over-match.
_ATTEMPT_GLOBS = ("rc.*", "abandoned-rc.*")


class ReleaseRefused(RuntimeError):
    """The release cannot proceed, and nothing was silently skipped."""


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True, encoding="utf-8").strip()


def _claims(repo: Path) -> list[str]:
    """Every local attempt ref and abandon marker ref."""
    listed = _git(repo, "tag", "--list", *_ATTEMPT_GLOBS)
    return [ref for ref in listed.splitlines()
            if parse_attempt_ref(ref) or parse_marker_ref(ref)]


def _claim_commit(repo: Path, tag: str) -> str:
    return _git(repo, "rev-parse", f"{tag}^{{commit}}")


def _refresh_claims(repo: Path, remote: str) -> None:
    _git(
        repo, "fetch", remote,
        "+refs/heads/main:refs/remotes/hermes-release/main",
        *(f"+refs/tags/{glob}:refs/tags/{glob}" for glob in _ATTEMPT_GLOBS),
    )


def _require_remote_main(repo: Path, commit: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "refs/remotes/hermes-release/main"],
        cwd=repo, capture_output=True,
    )
    if result.returncode != 0:
        raise ReleaseRefused(f"{commit} is not on origin/main")


def _claim_collision(repo: Path, remote: str, tag: str, error: Exception) -> ReleaseRefused:
    subprocess.run(["git", "tag", "--delete", tag], cwd=repo, capture_output=True)
    try:
        _git(repo, "fetch", remote, f"+refs/tags/{tag}:refs/tags/{tag}")
        details = _git(
            repo, "for-each-ref", f"refs/tags/{tag}",
            "--format=%(taggername)|%(taggerdate:iso-strict)|%(*objectname)",
        )
    except subprocess.CalledProcessError:
        return ReleaseRefused(f"claim {tag} could not be pushed: {error}")
    actor, when, commit = details.split("|", 2)
    return ReleaseRefused(f"{tag} was claimed by {actor} at {when} for {commit}")


def _outstanding_attempts(repo: Path, remote: str) -> list[tuple[str, int, str]]:
    """Attempts with no abandon marker whose version has no final tag on ``remote``.

    The predicate is the shared one in ``versioning``; this wrapper only owns
    the remote lookup of the final tags.
    """
    published: dict[str, bool] = {}

    def is_published(version: str) -> bool:
        if version not in published:
            published[version] = bool(_git(repo, "ls-remote", remote, f"refs/tags/v{version}"))
        return published[version]

    return outstanding_attempts(_claims(repo), is_published)


def _outstanding_attempt(repo: Path, remote: str) -> tuple[str, int, str] | None:
    """The one outstanding attempt as ``(version, attempt, ref)``, or None."""
    outstanding = _outstanding_attempts(repo, remote)
    if len(outstanding) > 1:
        refs = ", ".join(ref for *_rest, ref in outstanding)
        raise ReleaseRefused(
            f"more than one outstanding attempt ({refs}) — abandon all but one before releasing")
    return outstanding[0] if outstanding else None


def _attempt_run(execute, repository: str, ref: str, commit: str) -> dict | None:
    """The workflow run for ``ref`` at ``commit``, or None when none is listed."""
    raw = execute([
        "gh", "run", "list", "--repo", repository, "--workflow", WORKFLOW,
        "--branch", ref, "--json", "databaseId,url,headBranch,headSha,status",
    ])
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return None
    return next((row for row in rows if isinstance(row, dict) and row.get("headSha") == commit
                 and row.get("url") and row.get("databaseId") is not None), None)


def _refuse_outstanding(outstanding: tuple[str, int, str], *, repo: Path, remote: str,
                        repository: str, execute) -> ReleaseRefused:
    """Print the blocking run and both ways out, then return the refusal to raise."""
    version, _attempt, ref = outstanding
    run = _attempt_run(execute, repository, ref, _claim_commit(repo, ref))
    lines = [f"{ref} is outstanding: it has no abandon marker and v{version} is not published."]
    lines.append(f"Workflow: {run['url']}" if run else f"No workflow run is listed for {ref}.")
    lines += ["If the attempt is unfixable, abandon it:",
              f"    python scripts/release.py abandon --version {version} --remote {remote}"]
    if run:
        lines += ["If the attempt is fixable, rerun its failed jobs:",
                  f"    gh run rerun {run['databaseId']} --failed --repo {repository}"]
    print("\n".join(lines), file=sys.stderr)
    return ReleaseRefused(f"{ref} is outstanding — publish or abandon it first")


def _require_ancestry(repo: Path, commit: str, published_commit: str | None) -> None:
    """A new attempt descends from the published stable head; abandoned attempts do not bind it."""
    if published_commit is None:
        return
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", published_commit, commit], cwd=repo, capture_output=True)
    if ancestor.returncode != 0:
        raise ReleaseRefused(
            f"{commit} does not descend from the published stable head {published_commit}")


def _next_claim_epoch(repo: Path) -> int:
    epochs = _git(repo, "for-each-ref", "refs/tags/rc.*", "--format=%(refname:strip=2) %(taggerdate:unix)")
    previous = [int(stamp) for ref, _, stamp in (line.partition(" ") for line in epochs.splitlines())
                if parse_attempt_ref(ref) and stamp.isdigit()]
    return max(int(time.time()), max(previous, default=0) + 1)


# GitHub refuses a release body longer than this with HTTP 422.
GITHUB_BODY_LIMIT = 125_000


def _changelog(repo: Path, repository: str, *, commit: str, tag: str, version: str,
               published: tuple[str, str | None], no_changelog: bool) -> str:
    """The commit changelog from the stable base to the cut commit.

    GitHub's generate-notes lists merged pull requests since the last published
    release, so it is empty on a repository that merges none (a fork).
    """
    from scripts import release as release_script

    published_version, base = published
    receipt = f"v{published_version}"
    if base is None and subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"refs/tags/{receipt}^{{commit}}"],
            cwd=repo, capture_output=True).returncode == 0:
        # Before the first publication the seed version's tag is the base.
        base = receipt
    commits = release_script.get_commits(since_tag=base, until=commit, cwd=repo)
    return release_script.generate_changelog(
        commits, tag, version, repo_url=f"https://github.com/{repository}",
        prev_tag=receipt if base else None, no_changelog=no_changelog,
    )


def release(commit: str, *, bump: str, repo: Path, remote: str, repository: str,
            execute, autopublish: bool = False, no_changelog: bool = False,
            skip_bundles: bool = False, skip_tests: bool = False,
            published: tuple[str, str | None] = (SEED, None)) -> dict:
    """Claim the next attempt of the derived version, cut its draft, and start the gate.

    ``published`` is the stable channel's ``(version, commit)``; the commit is
    None before the first publication. ``skip_bundles`` and ``skip_tests`` are
    written into the claim, which is the one record every later job reads.
    """
    _refresh_claims(repo, remote)
    _require_remote_main(repo, commit)
    outstanding = _outstanding_attempt(repo, remote)
    if outstanding is not None:
        raise _refuse_outstanding(outstanding, repo=repo, remote=remote,
                                  repository=repository, execute=execute)
    published_version, published_commit = published
    _require_ancestry(repo, commit, published_commit)
    version = derive_next_version(published=published_version, bump=bump)
    if _git(repo, "ls-remote", remote, f"refs/tags/v{version}"):
        # A final tag records a started publication. Until its GitHub release is
        # public the published identity still names the previous version.
        raise ReleaseRefused(
            f"v{version} already has a final tag. Its publication is still finishing. "
            "Wait for Stable Release Publication, then cut again.")
    attempt = next_attempt(version, _claims(repo))
    tag = attempt_ref(version, attempt)
    # Built before the claim: a body GitHub refuses would otherwise burn the attempt.
    body = draft_body(version=version, attempt_ref=tag, notes=_changelog(
        repo, repository, commit=commit, tag=tag, version=version, published=published,
        no_changelog=no_changelog,
    ))
    if len(body) > GITHUB_BODY_LIMIT:
        raise ReleaseRefused(
            f"the {tag} draft body is {len(body)} characters; GitHub accepts at most "
            f"{GITHUB_BODY_LIMIT}. Nothing was claimed. Re-run with --no-changelog.")
    claim_epoch = _next_claim_epoch(repo)
    claim = json.dumps({
        "schema": 1,
        "version": version,
        "attempt": attempt,
        "commit": commit,
        "autopublish": autopublish,
        "skipBundles": skip_bundles,
        "skipTests": skip_tests,
        "claimEpoch": claim_epoch,
    }, sort_keys=True, separators=(",", ":"))
    subprocess.check_output(
        ["git", "tag", "-a", tag, commit, "-m", claim], cwd=repo,
        text=True, encoding="utf-8",
        env={**os.environ, "GIT_COMMITTER_DATE": f"@{claim_epoch} +0000"},
    )
    try:
        _git(repo, "push", remote, f"refs/tags/{tag}")
    except subprocess.CalledProcessError as error:
        raise _claim_collision(repo, remote, tag, error) from error
    ref = f"refs/tags/{tag}"
    remote_ref = dict(line.split()[::-1] for line in _git(
        repo, "ls-remote", remote, ref, f"{ref}^{{}}",
    ).splitlines())
    if (remote_ref.get(ref) != _git(repo, "rev-parse", ref)
            or remote_ref.get(f"{ref}^{{}}") != commit):
        raise ReleaseRefused(f"claim {tag} did not persist with exact remote custody")
    # The pre-check and the push are not one atomic step: a concurrent cut of a
    # different version passes the same check. Re-read before anything starts.
    _refresh_claims(repo, remote)
    _outstanding_attempt(repo, remote)
    try:
        # The body carries a warning at both ends, so it is handed over as a file.
        file = tempfile.NamedTemporaryFile(
            "w", suffix=".md", delete=False, encoding="utf-8", newline="\n")
        try:
            file.write(body)
        finally:
            file.close()
        try:
            # gh prints the new release's page. A draft is not reachable
            # through releases/tag/<tag> (GitHub serves it at an untagged-*
            # URL), so this printed page is the only link to it.
            url = (execute([
                "gh", "release", "create", tag, "--repo", repository,
                "--verify-tag", "--draft", "--notes-file", file.name,
                "--title", f"Hermes Agent v{version}",
            ]) or "").strip()
        finally:
            os.unlink(file.name)
        execute([
            "gh", "workflow", "run", WORKFLOW, "--ref", tag, "--repo", repository,
            "--raw-field", f"tag={tag}",
        ])
        found = execute([
            "gh", "run", "list", "--repo", repository, "--workflow", WORKFLOW,
            "--branch", tag, "--json", "databaseId,url,headBranch,status",
        ])
    except Exception as exc:
        raise ReleaseRefused(f"release {tag} never started: {exc}") from exc
    run_url = _dispatched_run(found, tag)
    return {"version": version, "tag": tag, "commit": commit, "url": url,
            "final_url": f"https://github.com/{repository}/releases/tag/v{version}",
            "run_url": run_url, "autopublish": autopublish,
            "skip_bundles": skip_bundles, "skip_tests": skip_tests}


def _dispatched_run(raw: str, tag: str) -> str:
    """The run URL for the claim ref. Empty when the list has no such run yet."""
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return ""
    matched = [row for row in rows if isinstance(row, dict) and row.get("headBranch") == tag and row.get("url")]
    if len(matched) != 1:
        return ""
    return str(matched[0]["url"])


def _latest_run(raw: str) -> str:
    """The URL of the newest listed run. Empty when the list names none."""
    try:
        rows = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return ""
    for row in rows:
        if isinstance(row, dict) and row.get("url"):
            return str(row["url"])
    return ""


def _release_view(tag: str, repository: str, inspect) -> dict | None:
    try:
        return json.loads(inspect([
            "gh", "release", "view", tag, "--repo", repository,
            "--json", "tagName,isDraft,isPrerelease",
        ]))
    except ReleaseRefused as exc:
        if "not found" in str(exc).lower():
            return None
        raise


def _outstanding_ref(repo: Path, version: str) -> str:
    """The attempt ref whose draft the publication pass must find."""
    attempt = next_attempt(version, _claims(repo)) - 1
    if attempt < 1:
        raise ReleaseRefused(f"stable {version} has no claimed attempt to publish")
    return attempt_ref(version, attempt)


def _preflight_publish(version: str, repository: str, inspect, head_version,
                       repo: Path | None, remote: str | None) -> None:
    requested = tuple(map(int, version.split(".")))
    head = head_version()
    if head and tuple(map(int, head.split("."))) >= requested:
        raise ReleaseRefused(f"stable {version} is burned or superseded by {head}")
    if repo is None or remote is None:
        raise ValueError("preflight needs the release repository to find the attempt ref")
    # The draft lives on the attempt ref, never on the final tag and never on
    # the old v{version}-rc shape.
    _refresh_claims(repo, remote)
    attempt_ref = _outstanding_ref(repo, version)
    rows = [(tag, row) for tag in (f"v{version}", attempt_ref)
            if (row := _release_view(tag, repository, inspect)) is not None]
    if len(rows) != 1 or rows[0][0] != attempt_ref or rows[0][1].get("isDraft") is not True:
        raise ReleaseRefused(f"stable {version} is burned or has no release draft")


def publish(version: str, *, repository: str, dispatch, inspect=None, head_version=None,
            repo: Path | None = None, remote: str | None = None) -> dict:
    """Request ordered publication through the one production sequencer."""
    tag = f"v{version}"
    from hermes_cli.update_channel import STABLE_TAG_RE

    if not STABLE_TAG_RE.fullmatch(tag):
        raise ReleaseRefused(f"{version} is not a stable version")
    if inspect is not None and head_version is not None:
        _preflight_publish(version, repository, inspect, head_version, repo, remote)
    dispatch([
        "gh", "workflow", "run", "stable-release-publication.yml",
        "--repo", repository, "--raw-field", f"version={version}",
    ])
    return {"requested": tag, "version": version, "repository": repository}


def abandon(version: str, *, repo: Path, remote: str, repository: str, delete, inspect=None) -> dict:
    """Clear the outstanding attempt of ``version``. The attempt ref stays; the marker is the record.

    The draft goes first: a cleared attempt with a live draft could still be
    published by hand, while a draftless outstanding attempt is just abandoned
    again. This reads every outstanding attempt, not the one-attempt view, so
    it still clears one when a concurrent cut left two.
    """
    _refresh_claims(repo, remote)
    matching = [found for found in _outstanding_attempts(repo, remote) if found[0] == version]
    if len(matching) != 1:
        raise ReleaseRefused(f"stable {version} has no outstanding attempt to abandon")
    _version, attempt, tag = matching[0]
    if inspect is not None:
        draft = _release_view(tag, repository, inspect)
        if draft is not None:
            if draft.get("isDraft") is not True:
                raise ReleaseRefused(f"{tag} is published and cannot be abandoned")
            delete(["gh", "release", "delete", tag, "--repo", repository, "--yes"])
    marker = marker_ref(version, attempt)
    message = json.dumps({"schema": 1, "version": version, "attempt": attempt, "attemptRef": tag},
                         sort_keys=True, separators=(",", ":"))
    _git(repo, "tag", "-a", marker, f"{tag}^{{commit}}", "-m", message)
    try:
        _git(repo, "push", remote, f"refs/tags/{marker}")
    except subprocess.CalledProcessError as error:
        raise _claim_collision(repo, remote, marker, error) from error
    return {"version": version, "tag": tag, "marker": marker, "repository": repository}


def next_steps(result: dict) -> str:
    """Say what started, what the operator waits for, and the next action."""
    version = result["version"]
    lines = [
        f"Claimed {result['tag']} for v{version}. The release workflow started on {result['tag']}.",
        f"Workflow: {result['run_url']}" if result.get("run_url") else "Workflow: the run is not listed yet. Open the Actions tab for this claim.",
        f"Draft release: {result['url']}",
        "The draft exists now. Edit its notes while the workflow runs; the edits carry through to publication.",
        "Wait for that workflow to finish. It builds and tests this commit.",
    ]
    if result["skip_bundles"]:
        lines.append("Bundles are skipped. Only the tag, the GitHub release and the Docker image "
                     "ship; the desktop, Termux and Store channels stay on the previous release.")
    if result["skip_tests"]:
        lines.append("Tests are skipped. No CI, E2E, native smoke or upgrade acceptance job runs. "
                     "The build is published untested.")
    if result["autopublish"]:
        lines.append("Autopublish is on. A green workflow publishes the release. You do not run publish.")
    else:
        lines.append("Autopublish is off. The release stays a draft after the workflow is green.")
        lines.append("When it is green, publish the release to push this build live:")
        lines.append(f"    python scripts/release.py publish --version {version} --remote origin")
    lines.append(f"After publication the release is at {result['final_url']}.")
    return "\n".join(lines)


def cmd_release(args) -> None:
    """The ``release`` subcommand: claim, draft, dispatch."""
    from scripts import release as release_script

    repo = release_script.REPO_ROOT
    remote = release_script.resolve_push_remote(args.remote)
    repository = release_script.remote_github_repo(remote)
    if not repository:
        raise SystemExit(f"release: remote {remote!r} does not point at a GitHub repository")
    commit = _git(repo, "rev-parse", "--verify", f"{args.commit}^{{commit}}")

    def execute(command: list[str]) -> None:
        completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "release command failed")
        return completed.stdout

    from scripts.releases.versioning import published_stable_identity
    result = release(
        commit, bump=args.bump, repo=repo, remote=remote, repository=repository,
        execute=execute, autopublish=args.autopublish, no_changelog=args.no_changelog,
        skip_bundles=args.skip_bundles, skip_tests=args.skip_tests,
        published=published_stable_identity(repository),
    )
    print(next_steps(result))


def _command_repository(args) -> tuple[Path, str, str]:
    from scripts import release as release_script

    repo = release_script.REPO_ROOT
    remote = release_script.resolve_push_remote(args.remote)
    repository = release_script.remote_github_repo(remote)
    if not repository:
        raise SystemExit(f"release: remote {remote!r} does not point at a GitHub repository")
    return repo, remote, repository


def _execute(repo: Path, command: list[str]) -> None:
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise ReleaseRefused(completed.stderr.strip() or "release command failed")


def _inspect(repo: Path, command: list[str]) -> str:
    completed = subprocess.run(command, cwd=repo, capture_output=True, text=True, encoding="utf-8")
    if completed.returncode != 0:
        raise ReleaseRefused(completed.stderr.strip() or "release inspection failed")
    return completed.stdout


def publish_steps(result: dict) -> str:
    """Say that publication was requested, and where to watch it."""
    version = result["version"]
    lines = [
        f"Requested publication of v{version}.",
        f"Workflow: {result['run_url']}" if result.get("run_url") else "Workflow: the run is not listed yet. Open the Actions tab.",
        "Wait for that workflow to finish. It publishes the draft and moves the stable channel.",
        f"The release page is https://github.com/{result['repository']}/releases/tag/v{version}.",
    ]
    return "\n".join(lines)


def abandon_steps(result: dict) -> str:
    """Say which attempt is cleared and which attempt the next cut takes."""
    version, tag = result["version"], result["tag"]
    _version, attempt = parse_attempt_ref(tag)
    return "\n".join([
        f"Cleared {tag}. The marker {result['marker']} records it.",
        f"v{version} is not spent. The next cut is rc.{attempt + 1}-v{version}.",
    ])


def cmd_publish(args) -> None:
    repo, remote, repository = _command_repository(args)
    from scripts.releases.versioning import published_stable_version
    result = publish(args.version, repository=repository,
                     dispatch=lambda command: _execute(repo, command),
                     inspect=lambda command: _inspect(repo, command),
                     head_version=lambda: published_stable_version(repository),
                     repo=repo, remote=remote)
    listed = _inspect(repo, [
        "gh", "run", "list", "--repo", repository, "--workflow", "stable-release-publication.yml",
        "--json", "databaseId,url,headBranch,status", "--limit", "1",
    ])
    print(publish_steps({**result, "run_url": _latest_run(listed)}))


def cmd_abandon(args) -> None:
    repo, remote, repository = _command_repository(args)
    result = abandon(args.version, repo=repo, remote=remote, repository=repository,
                     delete=lambda command: _execute(repo, command),
                     inspect=lambda command: _inspect(repo, command))
    print(abandon_steps(result))
