"""Channel administration from release.py: list, retire and bootstrap channel records.

Building into a channel is a channel claim (release.py --channel NAME --build-commit REV,
scripts/releases/commit_build.py): Stable Release admits it, allocates its request
(channel_claims.py), builds and tests it like a release and advances the channel.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

from hermes_cli.release_channels import (
    ChannelError,
    validate_name,
    validate_repository,
)
from scripts.releases import commit_build, r2
from scripts.releases.channels import BRANDINGS, ChannelPublisher, R2ChannelStore


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
    from scripts.releases.r2_scope import channel_public_base

    validate_repository(repository)
    base = channel_public_base()
    store = R2ChannelStore(*r2.credentials())

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
        if selected == "channels":
            print("These are the channel records. This command did not change them.")
        elif not args.publish:
            print("Dry run: no R2 object was written. Add --publish to execute.")
        elif selected == "retire_channel":
            print(f"Retired {args.retire_channel}. New installs move to {args.to} at version {args.minimum_version} or newer.")
        else:
            print("Wrote the channel records. Check the JSON above for each record.")
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
        "--branding",
        choices=BRANDINGS,
        default="preview",
        help="App name, icon and package ID for --channel: 'stable' installs as the "
        "regular Hermes app (fixed when the channel is created), 'preview' as "
        "its own side-by-side app",
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
    if args.branding != "preview" and not args.channel:
        parser.error("--branding requires --channel")
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
        args.canary,
        args.prune_canaries,
        args.date,
        args.no_changelog,
    )):
        parser.error(
            "Channel operations cannot be combined with release/tag operations"
        )
    return True
