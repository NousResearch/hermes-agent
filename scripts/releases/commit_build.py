"""Claim and dispatch commit builds; admit the legacy tagless ones. No release channel moves."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tomllib
from datetime import datetime, timezone, UTC
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


def resolve_revision(rev: str, remote: str, repo: Path) -> str:
    if not isinstance(rev, str) or not rev or rev.startswith("-"):
        raise ValueError("Commit builds require a Git revision")
    output(["git", "fetch", "--quiet", remote], repo)
    commit = require_commit(output(["git", "rev-parse", "--verify", "--end-of-options", f"{rev}^{{commit}}"], repo))
    require_pushed(commit, remote, repo)
    return commit


def claim_message(*, version: str, commit: str, lanes, require_green: bool,
                  bundle_env: dict[str, str | None], channel: str | None = None,
                  branding: str = "preview") -> str:
    """The annotated commit or channel claim's message: what stable.stamped_claim validates at
    admission. A channel claim also names its channel and that channel's branding."""
    from hermes_cli.release_channels import validate_name
    from scripts.releases import lanes as lane_set
    from scripts.releases.bundle_env import validate
    from scripts.releases.channels import BRANDINGS

    require_commit(commit)
    claim = {
        "schema": 1, "kind": "commit", "version": version, "commit": commit,
        "lanes": sorted(lanes), "lanesRevision": lane_set.LANES_REVISION,
        "requireGreen": bool(require_green), "bundleEnv": validate(dict(bundle_env)),
    }
    if channel is not None:
        if branding not in BRANDINGS:
            raise ValueError(f"Unknown channel branding: {branding}")
        claim |= {"kind": "channel", "channel": validate_name(channel), "branding": branding}
    return json.dumps(claim, sort_keys=True, separators=(",", ":"))


def dispatch_command(tag: str, repository: str, branch: str) -> list[str]:
    """Run Stable Release on the commit or channel claim, from the default branch like every release."""
    from hermes_cli.update_channel import is_preview_tag
    from scripts.releases.stable import CONTROLLER_WORKFLOW

    if not is_preview_tag(tag):
        raise ValueError(f"Not a commit or channel build claim tag: {tag!r}")
    return ["gh", "workflow", "run", CONTROLLER_WORKFLOW, "--ref", branch, "--repo", repository,
            "-f", f"tag={tag}"]


def _unused_claim_tag(release, version: str, kind: str = "commit") -> str:
    """<kind>.<now>-v<version>, moved forward a second at a time past any claim already tagged here
    (two builds started within one second). A claim another clone pushed meanwhile still fails the
    push rather than being replaced."""
    from datetime import timedelta

    from hermes_cli.update_channel import channel_tag_for_date, commit_tag_for_date

    for_date = {"commit": commit_tag_for_date, "channel": channel_tag_for_date}[kind]
    moment = datetime.now(UTC).replace(microsecond=0)
    for _ in range(60):
        tag = for_date(version, moment.strftime("%Y%m%dT%H%M%SZ"))
        if release.git_result("rev-parse", "-q", "--verify", f"refs/tags/{tag}").returncode != 0:
            return tag
        moment += timedelta(seconds=1)
    raise ValueError(f"every {kind} claim stamp for the next minute is already tagged here")


def channel_bundle_env(args) -> dict[str, str | None]:
    """--bundle-env/--bundle-unset, plus --channel-specific-data-dirs' per-channel data dirs."""
    from scripts.releases.bundle_env import parse_assignments

    bundle_env = parse_assignments(args.bundle_env, args.bundle_unset)
    if getattr(args, "channel_specific_data_dirs", False):
        bundle_env |= {"HERMES_DATA_DIR_SUFFIX": f"-channel-build-{args.channel}", "HERMES_HOME": None,
                       "HERMES_DESKTOP_USER_DATA_DIR": None, "HERMES_SHARED_AUTH_DIR": None}
    return bundle_env


def cmd_build_commit(args) -> None:
    """--build-commit REV [--channel NAME]: claim a commit or channel build and run the Stable
    Release pipeline on it.

    The claim is an annotated ``commit.<UTC stamp>-v<pyproject version at REV>`` tag on REV (or
    ``channel.…`` with --channel) whose message records the lanes (--only/--with/--without,
    starting from the stable default minus what the kind can never run), whether a red suite holds
    its publication (--[no-]require-green), the bundle defaults (--bundle-env/--bundle-unset) and,
    for a channel, its channel and branding. Without --publish nothing is created.
    """
    from scripts import release
    from scripts.releases import lanes as lane_set, r2

    channel = getattr(args, "channel", None)
    kind = "channel" if channel else "commit"
    try:
        lanes = lane_set.select(only=lane_set.parse_list(args.only),
                                with_=lane_set.parse_list(args.with_lanes) or (),
                                without=lane_set.parse_list(args.without_lanes) or (), policy=kind)
        bundle_env = channel_bundle_env(args)
        remote = release.resolve_push_remote(args.remote)
        repository = release.remote_github_repo(remote)
        if not repository:
            raise ValueError(f"{kind} builds require an explicit GitHub remote")
        commit = resolve_revision(args.build_commit, remote, release.REPO_ROOT)
        version = version_at(release.REPO_ROOT, commit)
        branch = release._default_branch(repository)
        if not branch:
            raise ValueError("could not resolve the repository default branch")
        tag = _unused_claim_tag(release, version, kind)
        message = claim_message(version=version, commit=commit, lanes=lanes,
                                require_green=args.require_green, bundle_env=bundle_env,
                                channel=channel, branding=getattr(args, "branding", "preview"))
        command = dispatch_command(tag, repository, branch)
        page = r2.public_url_for(r2.public_base_url(), r2.staging_key_for(tag, "index.html"))
        target = "the channel" if channel else "its builds page"
        print(f"{kind.capitalize()} build {tag} of {commit}" + (f" for channel {channel}" if channel else ""))
        print(f"Lanes {lane_set.describe(lanes)}")
        print(f"Publishing to {target} waits for every suite." if args.require_green
              else f"Publishing to {target} waits only for the built, smoked candidate; suites report.")
        print(f"Builds will be available at: {page}")
        print(f"Workflow command, running from {repository}@{branch}")
        print(f"    {shlex.join(command)}")
        if not args.publish:
            print("Dry run: no tag was created. Add --publish to claim and dispatch.")
            return
        tagged = release.git_result("tag", "-a", tag, commit, "-m", message)
        if tagged.returncode != 0:
            raise ValueError(f"could not create {tag}: {tagged.stderr.strip()}")
        pushed = release.git_result("push", remote, f"refs/tags/{tag}")
        if pushed.returncode != 0:
            raise ValueError(f"could not push {tag} to {remote}: {pushed.stderr.strip()}")
        print(f"✓ Pushed {tag} to {remote}")
        result = subprocess.run(command, cwd=release.REPO_ROOT, capture_output=True, text=True,  # windows-footgun: ok — encoding and replacement policy are on the next line.
                                encoding="utf-8", errors="replace", check=True, timeout=60)
        print((result.stdout or "").strip() or f"Dispatched Stable Release for {tag}.")
        print(f"The builds page is {page}.")
        print(f"Channel {channel} advances to this build once it is accepted; nothing else is published."
              if channel else "A commit build publishes nowhere: no release, channel, feed, Docker tag or Store.")
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        stderr = ""
        if isinstance(exc, subprocess.CalledProcessError):
            # check_output failures carry no captured stderr; don't mask the
            # original error with a TypeError while reporting it.
            stderr = "\n" + (exc.stderr or "")
        raise SystemExit(f"release: {kind} build refused: {exc}{stderr}") from exc


def main() -> None:
    import sys

    # Commit builds no longer mint a post-build receipt: their claim tag, made before the build,
    # is their identity (release.py --build-commit).
    if sys.argv[1:] != ["admit"]:
        raise SystemExit("usage: python -m scripts.releases.commit_build admit")
    env = dict(os.environ)
    values = admit(env)
    with Path(env["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
        stream.write("".join(f"{key}={value}\n" for key, value in values.items()))
    print(json.dumps(values, sort_keys=True))


if __name__ == "__main__":
    main()
