"""Admit and dispatch tagless builds without changing release channels."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tomllib
from pathlib import Path

WORKFLOW = "desktop-bundled-release.yml"


def require_commit(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{40}", value):
        raise ValueError("Commit builds require an exact full 40-character SHA")
    return value


def output(argv: list[str], repo: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=repo, text=True, encoding="utf-8", timeout=60).strip()


def require_pushed(commit: str, remote: str, repo: Path | None = None) -> None:
    """Require ancestry from a branch or tag currently advertised by this remote."""
    require_commit(commit)
    advertised = {line.split()[0] for line in output(
        ["git", "ls-remote", remote, "refs/heads/*", "refs/tags/*"], repo).splitlines()}
    containing = set(output(["git", "for-each-ref", f"--contains={commit}", "--format=%(objectname)",
                             f"refs/remotes/{remote}/", "refs/tags/"], repo).splitlines())
    if not advertised.intersection(containing):
        raise ValueError(f"Commit {commit} is not reachable from a pushed branch or tag on {remote}")


def version_at(repo: Path | None, commit: str) -> str:
    require_commit(commit)
    document = tomllib.loads(output(["git", "show", f"{commit}:pyproject.toml"], repo))
    version = document["project"]["version"]
    if not isinstance(version, str) or not re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)", version):
        raise ValueError("Commit packaging requires project.version=X.Y.Z")
    return version


def admit(env: dict[str, str]) -> dict[str, str]:
    from scripts.releases.bundle_env import decode

    commit = require_commit(env.get("BUILD_COMMIT", ""))
    if env.get("TAG") or env.get("RELEASE_PHASE") or env.get("UPLOAD_RELEASE", "false") != "false":
        raise ValueError("Commit builds cannot use tag, release-phase or upload_release")
    if env.get("TERMUX_UPGRADE_FROM_TAG"):
        raise ValueError("Commit builds do not run release-channel upgrade acceptance")
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
        permission = output(["gh", "api", f"repos/{repository}/collaborators/{actor}/permission", "--jq", ".permission"])
        if permission not in {"write", "maintain", "admin"}:
            raise ValueError("Commit builds require repository write, maintain or admin permission")
    require_pushed(commit, "origin")
    decode(env.get("BUNDLE_ENV_JSON", ""))
    return {"sha": commit, "channel": "commit", "payload-version": version_at(None, commit)}


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
            "-f", "termux_only=false", "-f", "termux_upgrade_from_tag="]
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
        print(f"Commit: {commit}\nR2: {r2.commit_prefix_for(commit)}\nPage: {page}\nWorkflow: {repository}@{branch}")
        print(shlex.join(command))
        if not args.publish:
            print("Dry run. Add --publish to dispatch. No tag, release or channel is changed.")
            return
        result = subprocess.run(command, cwd=release.REPO_ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", check=True, timeout=60)
        print((result.stdout or "").strip() or f"Dispatched commit build {commit}. No release was created.")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise SystemExit(f"release: commit build refused: {exc}") from exc


def main() -> None:
    import sys

    if sys.argv[1:] != ["admit"]:
        raise SystemExit("usage: python -m scripts.releases.commit_build admit")
    values = admit(dict(os.environ))
    with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write("".join(f"{key}={value}\n" for key, value in values.items()))
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
