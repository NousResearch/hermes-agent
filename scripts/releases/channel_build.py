"""Exact-commit channel requests and release-tool dispatch (one-offs stay separate)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

from hermes_cli.release_channels import (
    ChannelError, canonical_json,
    require_commit, require_sha256, validate_name, validate_repository,
    validate_request,
)
from scripts.releases import commit_build, r2
from scripts.releases.bundle_env import parse_assignments, validate
from scripts.releases.channels import ChannelPublisher, R2ChannelStore


def dispatch_command(request: dict, repository: str, branch: str, *, disposable_run: str = "") -> list[str]:
    validate_request(request, repository=repository)
    if not branch or branch.startswith("-"):
        raise ChannelError("Missing repository default branch")
    digest = hashlib.sha256(canonical_json(request)).hexdigest()
    command = ["gh", "workflow", "run", commit_build.WORKFLOW, "--repo", repository, "--ref", branch,
            "-f", "channel_build=" + request["buildId"], "-f", "channel_request_sha256=" + digest,
            "-f", "tag=", "-f", "upload_release=false", "-f", "termux_only=false",
            "-f", "termux_upgrade_from_tag="]
    if disposable_run:
        from scripts.releases.r2_scope import require_run
        command += ["-f", "disposable_run=" + require_run(disposable_run)]
    return command


def prepare_build(*, name: str, revision: str, remote: str, repo: Path, repository: str,
                  default_branch: str, publisher: ChannelPublisher, dispatch,
                  bundle_env: dict | None = None, controller_commit: str | None = None,
                  publish: bool = False) -> dict:
    validate_name(name)
    validate_repository(repository)
    if publisher.repository.casefold() != repository.casefold():
        raise ChannelError("Publisher repository authority mismatch")
    bundle_env = validate({} if bundle_env is None else bundle_env)
    commit = commit_build.resolve_revision(revision, remote, repo)
    version = commit_build.version_at(repo, commit)
    if not default_branch or default_branch.startswith("-"):
        raise ChannelError("Missing repository default branch")
    if controller_commit is None:
        advertised = commit_build.output(["git", "ls-remote", remote, f"refs/heads/{default_branch}"], repo).split()
        if len(advertised) != 2 or advertised[1] != f"refs/heads/{default_branch}":
            raise ChannelError("Could not resolve repository default branch")
        controller_commit = require_commit(advertised[0])
    require_commit(controller_commit)
    current = publisher._read(name)
    if current is not None:
        publisher._preview("allocate", current[0])
    result = {"channel": name, "repository": repository, "commit": commit,
              "sourceVersion": version, "controllerCommit": controller_commit, "created": current is None}
    if not publish:
        return result
    publisher.create(name)
    request = publisher.allocate(name, commit, version, bundle_env, controller_commit)
    command = dispatch_command(request, repository, default_branch,
                               disposable_run=os.environ.get("R2_DISPOSABLE_RUN", ""))
    print(f"Admitted build {request['buildId']}; request SHA256 {hashlib.sha256(canonical_json(request)).hexdigest()}")
    dispatch(command)
    return {**result, "request": request, "command": command}


def resume_build(build_id: str, digest: str, *, publisher: ChannelPublisher,
                 default_branch: str, dispatch, publish: bool = True) -> dict:
    request = publisher.request(build_id, digest)
    current = publisher._read(request["channel"])
    if current is None:
        raise ChannelError("Request channel no longer exists")
    publisher._preview("allocate", current[0])
    if current[0]["identity"] != request["identity"]:
        raise ChannelError("Request channel identity mismatch")
    command = dispatch_command(request, publisher.repository, default_branch,
                               disposable_run=os.environ.get("R2_DISPOSABLE_RUN", ""))
    if publish:
        dispatch(command)
    return {"request": request, "command": command}


def maintainer_authorizer(repository: str):
    validate_repository(repository)
    def authorize(action: str, record: dict) -> None:
        actor = commit_build.output(["gh", "api", "user", "--jq", ".login"])
        permission = commit_build.output(["gh", "api", f"repos/{repository}/collaborators/{actor}/permission", "--jq", ".permission"])
        if permission not in {"write", "maintain", "admin"}:
            raise ChannelError("Channel administration requires a repository maintainer")
        if action == "bootstrap" and permission not in {"maintain", "admin"}:
            raise ChannelError("Protected bootstrap requires maintain or admin permission")
    return authorize


def configured_publisher(repository: str) -> ChannelPublisher:
    from scripts.releases.r2_scope import R2Scope, channel_public_base
    validate_repository(repository)
    scope = R2Scope.configured(repository)
    if scope.prefix:
        actual_id = commit_build.output(["gh", "api", f"repos/{repository}", "--jq", ".id"])
        if actual_id != os.environ.get("GITHUB_REPOSITORY_ID"):
            raise ChannelError("Disposable namespace belongs to another repository")
    base = channel_public_base()
    store = R2ChannelStore(*r2.credentials(), scope=scope)
    def verify_build(request: dict, manifest: dict) -> bool:
        from scripts.releases.channel_releases import verify_bootstrap
        return verify_bootstrap(request, manifest, base, repository)

    return ChannelPublisher(store, repository, base, authorize=maintainer_authorizer(repository),
                            verify_build=verify_build)


def cmd_channel(args) -> None:
    from scripts import release
    try:
        remote = release.resolve_push_remote(args.remote)
        repository = release.remote_github_repo(remote)
        if not repository:
            raise ChannelError("Channel commands require an explicit GitHub remote")
        publisher = configured_publisher(repository)
        operations = {
            "channels": lambda: publisher.list(),
            "resume_channel_build": lambda: resume_build(
                args.resume_channel_build, args.request_sha256, publisher=publisher,
                default_branch=release._default_branch(repository), publish=args.publish,
                dispatch=lambda command: subprocess.run(command, cwd=release.REPO_ROOT, check=True, timeout=60)),
            "bootstrap_channels": lambda: _bootstrap_file(publisher, Path(args.bootstrap_channels), args.publish),
            "retire_channel": lambda: publisher.retire(args.retire_channel, args.to, args.minimum_version,
                                                        publish=args.publish),
            "channel": lambda: prepare_build(
                name=args.channel, revision=args.build_commit, remote=remote, repo=release.REPO_ROOT,
                repository=repository, default_branch=release._default_branch(repository), publisher=publisher,
                dispatch=lambda command: subprocess.run(command, cwd=release.REPO_ROOT, check=True, timeout=60),
                bundle_env=parse_assignments(args.bundle_env, args.bundle_unset), publish=args.publish),
        }
        selected = next(key for key in operations if getattr(args, key, None))
        result = operations[selected]()
        print(json.dumps(result, sort_keys=True, indent=2))
        if not args.publish and selected != "channels":
            print("Dry run: no R2 object or workflow was written. Add --publish to execute.")
    except (OSError, ValueError, subprocess.SubprocessError, r2.R2RequestError) as exc:
        raise SystemExit(f"release: channel operation refused: {exc}") from exc


def _bootstrap_file(publisher: ChannelPublisher, path: Path, publish: bool) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ChannelError("Bootstrap input must be a list of {record, manifest} accepted release facts")
    # Validate every row before the first write; multi-object publication is not atomic.
    for row in value:
        publisher.bootstrap(row["record"], row.get("manifest"), publish=False)
    return [publisher.bootstrap(row["record"], row.get("manifest"), publish=publish) for row in value]


def add_arguments(parser) -> None:
    parser.add_argument("--resume-channel-build", metavar="BUILD_ID", help="Retry an immutable admitted request without allocating")
    parser.add_argument("--request-sha256", type=require_sha256, metavar="SHA256", help="Pinned request digest for --resume-channel-build")
    parser.add_argument("--channel", type=validate_name, metavar="NAME", help="R2 channel for --build-commit")
    parser.add_argument("--channels", action="store_true", help="List authenticated R2 channel records")
    parser.add_argument("--retire-channel", type=validate_name, metavar="NAME", help="Permanently retire a preview channel")
    parser.add_argument("--to", type=validate_name, metavar="NAME", help="Retirement destination channel")
    parser.add_argument("--minimum-version", metavar="X.Y.Z", help="Destination version floor")
    parser.add_argument("--bootstrap-channels", metavar="JSON", help="Preview protected channel bootstrap from accepted release facts")


def validate_arguments(parser, args) -> bool:
    selected = [key for key in ("channel", "channels", "retire_channel", "bootstrap_channels", "resume_channel_build") if getattr(args, key)]
    if bool(args.resume_channel_build) != bool(args.request_sha256):
        parser.error("--resume-channel-build and --request-sha256 are required together")
    retirement = (args.to, args.minimum_version)
    if any(retirement) and not args.retire_channel:
        parser.error("Retirement options require --retire-channel")
    if len(selected) > 1:
        parser.error("Channel administration operations are mutually exclusive")
    if not selected:
        return False
    if (args.bundle_env or args.bundle_unset) and not args.channel:
        parser.error("--bundle-env and --bundle-unset require --build-commit")
    if args.channel and args.build_commit is None:
        parser.error("--channel requires --build-commit")
    if args.retire_channel and not all(retirement):
        parser.error("--retire-channel requires --to and --minimum-version")
    if ((not args.channel and args.build_commit) or any((args.bump, args.canary, args.prune_canaries,
            args.first_release, args.date, args.output, args.no_changelog))):
        parser.error("Channel operations cannot be combined with release/tag operations")
    return True
