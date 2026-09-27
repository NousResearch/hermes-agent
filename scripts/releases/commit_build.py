"""Admit and dispatch tagless builds without changing release channels."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path

WORKFLOW = "desktop-bundled-release.yml"


def require_commit(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError("Commit builds require an exact full 40-character SHA")
    return value


def output(argv: list[str], repo: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=repo, text=True, encoding="utf-8", timeout=60).strip()


def require_pushed(commit: str, remote: str, repo: Path | None = None, *, run=output) -> None:
    """Require ancestry from a branch or tag currently advertised by this remote."""
    require_commit(commit)
    advertised = {line.split()[0] for line in run(
        ["git", "ls-remote", remote, "refs/heads/*", "refs/tags/*"], repo).splitlines()}
    containing = set(run(["git", "for-each-ref", f"--contains={commit}", "--format=%(objectname)",
                          f"refs/remotes/{remote}/", "refs/tags/"], repo).splitlines())
    if not advertised.intersection(containing):
        raise ValueError(f"Commit {commit} is not reachable from a pushed branch or tag on {remote}")


def version_at(repo: Path | None, commit: str, *, run=output) -> str:
    require_commit(commit)
    document = tomllib.loads(run(["git", "show", f"{commit}:pyproject.toml"], repo))
    version = document["project"]["version"]
    if not isinstance(version, str) or not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError("Commit packaging requires project.version=X.Y.Z")
    return version


def _controller(env: dict[str, str], commit: str, *, run=output,
                repo: Path | None = None) -> dict:
    default = env.get("DEFAULT_BRANCH", "")
    ref = f"refs/heads/{default}"
    repository = env.get("GITHUB_REPOSITORY", "")
    expected_workflow = f"{repository}/.github/workflows/{WORKFLOW}@{ref}"
    if (not default or not repository or env.get("GITHUB_EVENT_NAME") != "workflow_dispatch"
            or env.get("GITHUB_REF") != ref or env.get("GITHUB_WORKFLOW_REF") != expected_workflow):
        raise ValueError("Commit builds require workflow_dispatch from the repository default-branch workflow")
    actors = {env.get("GITHUB_ACTOR", ""), env.get("GITHUB_TRIGGERING_ACTOR") or env.get("GITHUB_ACTOR", "")}
    for actor in actors:
        if not actor:
            raise ValueError("Commit builds require a repository maintainer")
        permission = run(["gh", "api", f"repos/{repository}/collaborators/{actor}/permission",
                          "--jq", ".permission"], repo)
        if permission not in {"write", "maintain", "admin"}:
            raise ValueError("Commit builds require repository write, maintain or admin permission")
    require_pushed(commit, "origin", repo, run=run)
    return {"repository": repository, "default": default}


def admit(env: dict[str, str], *, run=output, repo: Path | None = None) -> dict[str, str]:
    from scripts.releases.bundle_env import decode

    commit = require_commit(env.get("BUILD_COMMIT", ""))
    if env.get("TAG") or env.get("RELEASE_PHASE") or env.get("UPLOAD_RELEASE", "false") != "false":
        raise ValueError("Commit builds cannot use tag, release-phase or upload_release")
    if env.get("TERMUX_UPGRADE_FROM_TAG"):
        raise ValueError("Commit builds do not run release-channel upgrade acceptance")
    _controller(env, commit, run=run, repo=repo)
    decode(env.get("BUNDLE_ENV_JSON", ""))
    return {"sha": commit, "channel": "commit",
            "payload-version": version_at(repo, commit, run=run)}


def receipt_tag(kind: str, version: str, created_at: str, run_id: str) -> str:
    """Return the canonical post-build receipt identity."""
    from hermes_cli.update_channel import STABLE_TAG_RE

    if kind not in {"channel", "commit"} or not STABLE_TAG_RE.fullmatch("v" + version):
        raise ValueError("Build receipt kind or version is invalid")
    if not re.fullmatch(r"[1-9][0-9]{0,19}", run_id):
        raise ValueError("Build receipt run ID is invalid")
    try:
        instant = datetime.fromisoformat(created_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    except (AttributeError, ValueError) as error:
        raise ValueError("Build receipt creation time is invalid") from error
    if instant.microsecond or created_at != instant.strftime("%Y-%m-%dT%H:%M:%SZ"):
        raise ValueError("Build receipt creation time must be UTC whole seconds")
    return f"v{version}+{kind}.{instant.strftime('%Y%m%dT%H%M%SZ')}.{run_id}"


def _verify_receipt(tag: str, commit: str, record: dict, *, run, repo: Path | None) -> None:
    tag_object = run(["git", "rev-parse", f"refs/tags/{tag}"], repo)
    if (run(["git", "cat-file", "-t", tag_object], repo) != "tag"
            or run(["git", "rev-parse", f"refs/tags/{tag}^{{commit}}"], repo) != commit
            or json.loads(run(["git", "tag", "-l", tag, "--format=%(contents)"], repo)) != record):
        raise ValueError("Build receipt tag differs from this run")


def publish_receipt(kind: str, env: dict[str, str], *, version: str, commit: str,
                    details: dict, run=output, repo: Path | None = None) -> dict:
    """Create one annotated post-build receipt, or verify its exact replay."""
    require_commit(commit)
    controller = _controller(env, commit, run=run, repo=repo)
    run_id = env.get("GITHUB_RUN_ID", "")
    if env.get("GITHUB_ACTIONS") != "true" or not re.fullmatch(r"[1-9][0-9]{0,19}", run_id):
        raise ValueError("Build receipts require a GitHub Actions run ID")
    info = json.loads(run([
        "gh", "api", f"repos/{controller['repository']}/actions/runs/{run_id}",
    ], repo))
    created_at = info.get("created_at")
    if (info.get("id") != int(run_id) or info.get("event") != "workflow_dispatch"
            or info.get("status") != "in_progress"
            or info.get("head_branch") != controller["default"]
            or info.get("head_sha") != env.get("GITHUB_SHA")
            or not isinstance(created_at, str)):
        raise ValueError("Build receipt run differs from the trusted controller")
    tag = receipt_tag(kind, version, created_at, run_id)
    record = {
        "schema": 1,
        "kind": kind,
        "tag": tag,
        "version": version,
        "commit": commit,
        "runId": run_id,
        "runCreatedAt": created_at,
        "details": details,
    }
    ref = f"refs/tags/{tag}"
    remote = {}
    for line in run(["git", "ls-remote", "origin", ref, f"{ref}^{{}}"], repo).splitlines():
        sha, name = line.split()
        remote[name] = sha
    if remote:
        if set(remote) != {ref, f"{ref}^{{}}"} or remote[f"{ref}^{{}}"] != commit:
            raise ValueError("Remote build receipt tag custody changed")
        run(["git", "fetch", "--force", "origin", f"+{ref}:{ref}"], repo)
        _verify_receipt(tag, commit, record, run=run, repo=repo)
        return record

    try:
        local = run(["git", "rev-parse", "--verify", ref], repo)
    except subprocess.CalledProcessError:
        local = ""
    if local:
        _verify_receipt(tag, commit, record, run=run, repo=repo)
    else:
        message = json.dumps(record, sort_keys=True, separators=(",", ":"))
        run([
            "git", "-c", "user.name=Hermes Build Receipt",
            "-c", "user.email=actions@users.noreply.github.com",
            "tag", "-a", tag, commit, "-m", message,
        ], repo)
    try:
        run(["git", "push", "origin", ref], repo)
    except subprocess.CalledProcessError:
        pass
    remote = {}
    for line in run(["git", "ls-remote", "origin", ref, f"{ref}^{{}}"], repo).splitlines():
        sha, name = line.split()
        remote[name] = sha
    if set(remote) != {ref, f"{ref}^{{}}"} or remote[f"{ref}^{{}}"] != commit:
        raise ValueError("Build receipt tag was not published exactly")
    run(["git", "fetch", "--force", "origin", f"+{ref}:{ref}"], repo)
    _verify_receipt(tag, commit, record, run=run, repo=repo)
    return record


def resolve_revision(rev: str, remote: str, repo: Path) -> str:
    if not isinstance(rev, str) or not rev or rev.startswith("-"):
        raise ValueError("Commit builds require a Git revision")
    output(["git", "fetch", "--quiet", remote], repo)
    commit = require_commit(output(["git", "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}"], repo))
    require_pushed(commit, remote, repo)
    return commit


def dispatch_command(commit: str, repository: str, branch: str,
                     bundle_env: dict[str, str | None] | None = None) -> list[str]:
    from scripts.releases.bundle_env import validate

    require_commit(commit)
    command = ["gh", "workflow", "run", WORKFLOW, "--ref", branch, "--repo", repository,
            "-f", f"build_commit={commit}", "-f", "tag=", "-f", "upload_release=false",
            "-f", "termux_upgrade_from_tag="]
    if bundle_env:
        command += ["-f", "bundle_env=" + json.dumps(validate(bundle_env), sort_keys=True)]
    return command


def cmd_build_commit(args) -> None:
    from scripts import release
    from scripts.releases import r2
    from scripts.releases.bundle_env import parse_assignments

    try:
        bundle_env = parse_assignments(args.bundle_env, args.bundle_unset)
        remote = release.resolve_push_remote(args.remote)
        repository = release.remote_github_repo(remote)
        if not repository:
            raise ValueError("commit builds require an explicit GitHub remote")
        commit = resolve_revision(args.build_commit, remote, release.REPO_ROOT)
        branch = release._default_branch(repository)
        if not branch:
            raise ValueError("could not resolve the repository default branch")
        command = dispatch_command(commit, repository, branch, bundle_env)
        page = r2.public_url_for(r2.public_base_url(), r2.commit_page_key_for(commit))
        print(f"Building one-off bundle for commit {commit}")
        print(f"Builds will be available at: {page}.")
        print(f"Workflow command, running from {repository}@{branch}")
        print(f"    {shlex.join(command)}")
        if not args.publish:
            print("Dry run. Add --publish to dispatch.")
            return
        print("Starting workflow!")
        result = subprocess.run(command, cwd=release.REPO_ROOT, capture_output=True, text=True,  # windows-footgun: ok — encoding and replacement policy are on the next line.
                                encoding="utf-8", errors="replace", check=True, timeout=60)
        print((result.stdout or "").strip() or f"Dispatched commit build {commit}. No release was created.")
        print("Wait for that workflow to finish. It builds this commit and uploads the bundles.")
        print(f"The builds page is {page}.")
        print("This build does not publish a release and does not move a channel.")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError):
            # check_output failures carry no captured stderr; don't mask the
            # original error with a TypeError while reporting it.
            stderr = "\n" + (exc.stderr or "")
        raise SystemExit(f"release: commit build refused: {exc}{stderr}") from exc


def main() -> None:
    import sys

    if sys.argv[1:] not in (["admit"], ["receipt"]):
        raise SystemExit("usage: python -m scripts.releases.commit_build {admit|receipt}")
    env = dict(os.environ)
    values = admit(env)
    if sys.argv[1:] == ["admit"]:
        with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in values.items()))
        print(json.dumps(values, sort_keys=True))
        return

    from scripts.releases.bundle_env import decode
    from scripts.releases.stable import require_success

    needs = json.loads(env.get("RELEASE_NEEDS", "{}"))
    require_success(needs, list(needs))
    result = publish_receipt(
        "commit", env, version=values["payload-version"], commit=values["sha"],
        details={"bundleEnv": decode(env.get("BUNDLE_ENV_JSON", ""))},
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
