"""Channel dispatch: the local command names intent; CI allocates and builds.

A channel build is a preview of an exact pushed commit. The local command does
not touch R2: it resolves the commit, validates the request and dispatches the
default-branch workflow. The workflow's privileged allocation step creates the
channel and mints the immutable build request (see channel_disposable.py).
"""  # noqa: E501

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

from hermes_cli.release_channels import (
    ChannelError,
    validate_name,
    validate_repository,
)
from scripts.releases import commit_build, r2
from scripts.releases.bundle_env import parse_assignments, validate
from scripts.releases.channels import ChannelPublisher, R2ChannelStore


def dispatch_command(
    name: str, commit: str, repository: str, branch: str, bundle_env: dict | None = None
) -> list[str]:
    validate_name(name)
    validate_repository(repository)
    commit_build.require_commit(commit)
    if not branch or branch.startswith("-"):
        raise ChannelError("Missing repository default branch")
    command = [
        "gh",
        "workflow",
        "run",
        commit_build.WORKFLOW,
        "--repo",
        repository,
        "--ref",
        branch,
        "-f",
        "channel=" + name,
        "-f",
        "build_commit=" + commit,
        "-f",
        "tag=",
        "-f",
        "upload_release=false",
        "-f",
        "termux_only=false",
        "-f",
        "termux_upgrade_from_tag=",
    ]
    if bundle_env:
        command += ["-f", "bundle_env=" + json.dumps(bundle_env, sort_keys=True)]
    return command


def prepare_build(
    *,
    name: str,
    revision: str,
    remote: str,
    repo: Path,
    repository: str,
    default_branch: str,
    dispatch,
    bundle_env: dict | None = None,
    publish: bool = False,
) -> dict:
    validate_name(name)
    validate_repository(repository)
    bundle_env = validate({} if bundle_env is None else bundle_env)
    commit = commit_build.resolve_revision(revision, remote, repo)
    version = commit_build.version_at(repo, commit)
    if not default_branch or default_branch.startswith("-"):
        raise ChannelError("Missing repository default branch")
    command = dispatch_command(name, commit, repository, default_branch, bundle_env)
    result = {
        "channel": name,
        "repository": repository,
        "commit": commit,
        "sourceVersion": version,
    }
    if not publish:
        return {**result, "command": command}
    dispatch(command)
    return {**result, "command": command}


def maintainer_authorizer(repository: str):
    validate_repository(repository)

    def authorize(action: str, record: dict) -> None:
        actor = commit_build.output(["gh", "api", "user", "--jq", ".login"])
        permission = commit_build.output([
            "gh",
            "api",
            f"repos/{repository}/collaborators/{actor}/permission",
            "--jq",
            ".permission",
        ])
        if permission not in {"write", "maintain", "admin"}:
            raise ChannelError(
                "Channel administration requires a repository maintainer"
            )
        if action == "bootstrap" and permission not in {"maintain", "admin"}:
            raise ChannelError(
                "Protected bootstrap requires maintain or admin permission"
            )

    return authorize


def configured_publisher(repository: str) -> ChannelPublisher:
    from scripts.releases.r2_scope import R2Scope, channel_public_base

    validate_repository(repository)
    scope = R2Scope.configured(repository)
    if scope.prefix:
        actual_id = commit_build.output([
            "gh",
            "api",
            f"repos/{repository}",
            "--jq",
            ".id",
        ])
        if actual_id != os.environ.get("GITHUB_REPOSITORY_ID"):
            raise ChannelError("Disposable namespace belongs to another repository")
    base = channel_public_base()
    store = R2ChannelStore(*r2.credentials(), scope=scope)

    def verify_build(request: dict, manifest: dict) -> bool:
        from scripts.releases.channel_releases import verify_bootstrap

        return verify_bootstrap(request, manifest, base, repository)

    return ChannelPublisher(
        store,
        repository,
        base,
        authorize=maintainer_authorizer(repository),
        verify_build=verify_build,
    )


def cmd_channel(args) -> None:
    from scripts import release

    try:
        remote = release.resolve_push_remote(args.remote)
        repository = release.remote_github_repo(remote)
        if not repository:
            raise ChannelError("Channel commands require an explicit GitHub remote")
        default_branch = release._default_branch(repository)
        if args.channel:
            bundle_env = parse_assignments(args.bundle_env, args.bundle_unset)
            if args.channel_specific_data_dirs:
                bundle_env["HERMES_DATA_DIR_SUFFIX"] = f"-channel-build-{args.channel}"
                bundle_env["HERMES_HOME"] = None
                bundle_env["HERMES_DESKTOP_USER_DATA_DIR"] = None
                bundle_env["HERMES_SHARED_AUTH_DIR"] = None

            result = prepare_build(
                name=args.channel,
                revision=args.build_commit,
                remote=remote,
                repo=release.REPO_ROOT,
                repository=repository,
                default_branch=default_branch,
                dispatch=lambda command: subprocess.run(
                    command, cwd=release.REPO_ROOT, check=True, timeout=60
                ),
                bundle_env=bundle_env,
                publish=args.publish,
            )
            print(json.dumps(result, sort_keys=True, indent=2))
            if not args.publish:
                print("Dry run: no workflow was dispatched. Add --publish to dispatch.")
            return
        publisher = configured_publisher(repository)
        operations = {
            "channels": lambda: publisher.list(),
            "retire_channel": lambda: publisher.retire(
                args.retire_channel, args.to, args.minimum_version, publish=args.publish
            ),
            "bootstrap_channels": lambda: _bootstrap_file(
                publisher, Path(args.bootstrap_channels), args.publish
            ),
        }
        selected = next(key for key in operations if getattr(args, key, None))
        result = operations[selected]()
        print(json.dumps(result, sort_keys=True, indent=2))
        if not args.publish and selected != "channels":
            print(
                "Dry run: no R2 object or workflow was written. Add --publish to execute."
            )
    except (OSError, ValueError, subprocess.SubprocessError, r2.R2RequestError) as exc:
        raise SystemExit(f"release: channel operation refused: {exc}") from exc


def _bootstrap_file(
    publisher: ChannelPublisher, path: Path, publish: bool
) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, list):
        raise ChannelError(
            "Bootstrap input must be a list of {record, manifest} accepted release facts"
        )
    # Validate every row before the first write; multi-object publication is not atomic.
    for row in value:
        publisher.bootstrap(row["record"], row.get("manifest"), publish=False)
    return [
        publisher.bootstrap(row["record"], row.get("manifest"), publish=publish)
        for row in value
    ]


def add_arguments(parser) -> None:
    parser.add_argument(
        "--channel",
        type=validate_name,
        metavar="NAME",
        help="R2 channel for --build-commit",
    )
    parser.add_argument(
        "--channel-specific-data-dirs",
        action="store_true",
        help="Set HERMES_HOME et al to a directory specific for this channel",
    )
    parser.add_argument(
        "--channels", action="store_true", help="List authenticated R2 channel records"
    )
    parser.add_argument(
        "--retire-channel",
        type=validate_name,
        metavar="NAME",
        help="Permanently retire a preview channel",
    )
    parser.add_argument(
        "--to",
        type=validate_name,
        metavar="NAME",
        help="Retirement destination channel",
    )
    parser.add_argument(
        "--minimum-version", metavar="X.Y.Z", help="Destination version floor"
    )
    parser.add_argument(
        "--bootstrap-channels",
        metavar="JSON",
        help="Preview protected channel bootstrap from accepted release facts",
    )


def validate_arguments(parser, args) -> bool:
    selected = [
        key
        for key in ("channel", "channels", "retire_channel", "bootstrap_channels")
        if getattr(args, key)
    ]
    retirement = (args.to, args.minimum_version)
    if any(retirement) and not args.retire_channel:
        parser.error("Retirement options require --retire-channel")
    if len(selected) > 1:
        parser.error("Channel administration operations are mutually exclusive")
    if not selected:
        return False
    if (args.bundle_env or args.bundle_unset) and not args.channel:
        parser.error("--bundle-env and --bundle-unset require --build-commit")
    if (args.bundle_env or args.bundle_unset) and args.channel_specific_data_dirs:
        from scripts.releases.bundle_env import parse_assignments

        env = parse_assignments(args.bundle_env, args.bundle_unset)
        banned = (
            "HERMES_DATA_DIR_SUFFIX",
            "HERMES_HOME",
            "HERMES_SHARED_AUTH_DIR",
            "HERMES_DESKTOP_USER_DATA_DIR",
        )
        bad_vars = [e for e in env if e in banned]
        if bad_vars:
            bad_set = " ".join([f"--bundle-env {v}" for v in bad_vars if v])
            bad_unset = " ".join([f"--bundle-unset {v}" for v in bad_vars if not v])
            parser.error(
                f"--channel-specific-data-dirs conflicts with:\n{bad_set} {bad_unset}\nRemove those args to use it."
            )
    if args.channel and args.build_commit is None:
        parser.error("--channel requires --build-commit")
    if args.retire_channel and not all(retirement):
        parser.error("--retire-channel requires --to and --minimum-version")
    if (not args.channel and args.build_commit) or any((
        args.bump,
        args.canary,
        args.prune_canaries,
        args.first_release,
        args.date,
        args.output,
        args.no_changelog,
    )):
        parser.error(
            "Channel operations cannot be combined with release/tag operations"
        )
    return True
