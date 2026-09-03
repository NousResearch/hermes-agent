"""``hermes profile`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import sys
from typing import Callable


def handle_bot_transfer_action(args) -> bool:
    """Handle profile share/pull/push actions outside the CLI godfile."""
    action = getattr(args, "profile_action", None)
    if action not in {"share", "pull", "push"}:
        return False

    from hermes_cli.bot_transfer import (
        BotTransferError,
        ensure_profile_bot_id,
        get_profile_bot_id,
        profile_is_cloneable,
        pull_bot_profile,
        push_bot_profile,
        set_profile_cloneable,
    )

    try:
        if action == "share":
            from hermes_cli.profiles import (
                get_profile_dir,
                normalize_profile_name,
                validate_profile_name,
            )

            name = normalize_profile_name(args.profile_name)
            validate_profile_name(name)
            profile_dir = get_profile_dir(name)
            if not profile_dir.is_dir():
                raise FileNotFoundError(f"Profile '{name}' does not exist.")
            if args.allow_pull or args.deny_pull:
                allowed = bool(args.allow_pull)
                bot_id = set_profile_cloneable(name, allowed)
            else:
                allowed = profile_is_cloneable(name)
                bot_id = get_profile_bot_id(profile_dir)
            state = "allowed" if allowed else "denied"
            print(f"Remote pull: {state} for '{name}'")
            if bot_id:
                print(f"Bot ID:      {bot_id}")
            elif args.allow_pull:
                print(f"Bot ID:      {ensure_profile_bot_id(profile_dir)}")
        elif action == "pull":
            profile_dir, bot_id = pull_bot_profile(
                args.profile_name,
                remote=getattr(args, "remote_url", None),
                name=getattr(args, "clone_name", None),
            )
            print(f"✓ Pulled bot '{profile_dir.name}' ({bot_id})")
            print(f"  Path: {profile_dir}")
        else:
            remote_name, bot_id = push_bot_profile(
                args.profile_name,
                remote=getattr(args, "remote_url", None),
                name=getattr(args, "clone_name", None),
            )
            print(f"✓ Pushed bot '{remote_name}' ({bot_id})")
    except (
        BotTransferError,
        ValueError,
        FileExistsError,
        FileNotFoundError,
        OSError,
    ) as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    return True


def build_profile_parser(subparsers, *, cmd_profile: Callable) -> None:
    """Attach the ``profile`` subcommand to ``subparsers``."""
    # =========================================================================
    # profile command
    # =========================================================================
    profile_parser = subparsers.add_parser(
        "profile",
        help="Manage profiles — multiple isolated Hermes instances",
    )
    profile_subparsers = profile_parser.add_subparsers(dest="profile_action")

    profile_subparsers.add_parser("list", help="List all profiles")
    profile_use = profile_subparsers.add_parser(
        "use", help="Set sticky default profile"
    )
    profile_use.add_argument("profile_name", help="Profile name (or 'default')")

    profile_create = profile_subparsers.add_parser(
        "create", help="Create a new profile"
    )
    profile_create.add_argument(
        "profile_name", help="Profile name (lowercase, alphanumeric)"
    )
    profile_create.add_argument(
        "--clone",
        action="store_true",
        help="Copy config.yaml, .env, SOUL.md, and skills from active profile",
    )
    profile_create.add_argument(
        "--clone-all",
        action="store_true",
        help="Full copy of active profile (all state, excluding per-profile history)",
    )
    profile_create.add_argument(
        "--clone-from",
        metavar="SOURCE",
        help="Source profile to clone from; implies --clone unless --clone-all is set",
    )
    profile_create.add_argument(
        "--no-alias", action="store_true", help="Skip wrapper script creation"
    )
    profile_create.add_argument(
        "--no-skills",
        action="store_true",
        help="Create an empty profile with no bundled skills (opts out of `hermes update` skill sync)",
    )
    profile_create.add_argument(
        "--description",
        default=None,
        help="One- or two-sentence description of what this profile is good at. "
             "Used by the kanban decomposer to route tasks based on role instead "
             "of profile name alone. Skip and add later via `hermes profile describe`.",
    )

    profile_delete = profile_subparsers.add_parser("delete", help="Delete a profile")
    profile_delete.add_argument("profile_name", help="Profile to delete")
    profile_delete.add_argument(
        "-y", "--yes", action="store_true", help="Skip confirmation prompt"
    )

    profile_describe = profile_subparsers.add_parser(
        "describe",
        help="Read or set a profile's description (used by the kanban orchestrator)",
    )
    profile_describe.add_argument(
        "profile_name",
        nargs="?",
        default=None,
        help="Profile to describe (omit + use --all --auto to sweep)",
    )
    profile_describe.add_argument(
        "--text",
        default=None,
        help="Set description to this exact text (overwrites any existing description)",
    )
    profile_describe.add_argument(
        "--auto",
        action="store_true",
        help="Auto-generate description via the auxiliary LLM "
             "(uses auxiliary.profile_describer)",
    )
    profile_describe.add_argument(
        "--overwrite",
        action="store_true",
        help="With --auto, replace user-authored descriptions too (default: only "
             "fill in missing or previously-auto descriptions)",
    )
    profile_describe.add_argument(
        "--all",
        dest="all_missing",
        action="store_true",
        help="With --auto, run on every profile missing a description",
    )

    profile_show = profile_subparsers.add_parser("show", help="Show profile details")
    profile_show.add_argument("profile_name", help="Profile to show")

    profile_alias = profile_subparsers.add_parser(
        "alias", help="Manage wrapper scripts"
    )
    profile_alias.add_argument("profile_name", help="Profile name")
    profile_alias.add_argument(
        "--remove", action="store_true", help="Remove the wrapper script"
    )
    profile_alias.add_argument(
        "--name",
        dest="alias_name",
        metavar="NAME",
        help="Custom alias name (default: profile name)",
    )

    profile_rename = profile_subparsers.add_parser(
        "rename",
        help="Rename a profile ('default': sets a display name; id unchanged)",
    )
    profile_rename.add_argument("old_name", help="Current profile name")
    profile_rename.add_argument(
        "new_name",
        help="New profile name (for 'default': a display name — the canonical id stays 'default')",
    )

    profile_export = profile_subparsers.add_parser(
        "export", help="Export a profile to archive"
    )
    profile_export.add_argument("profile_name", help="Profile to export")
    profile_export.add_argument(
        "-o", "--output", default=None,
        help="Output file (default: a managed profile-exports/<name>-<timestamp>.tar.gz "
             "under the default Hermes home)",
    )

    profile_import = profile_subparsers.add_parser(
        "import", help="Import a profile from archive"
    )
    profile_import.add_argument("archive", help="Path to .tar.gz archive")
    profile_import.add_argument(
        "--name",
        dest="import_name",
        metavar="NAME",
        help="Profile name (default: inferred from archive)",
    )

    profile_share = profile_subparsers.add_parser(
        "share", help="Control whether a profile may be pulled from this gateway"
    )
    profile_share.add_argument("profile_name", help="Profile to share")
    share_policy = profile_share.add_mutually_exclusive_group()
    share_policy.add_argument(
        "--allow-pull",
        action="store_true",
        help="Allow authenticated remote clients to clone this bot",
    )
    share_policy.add_argument(
        "--deny-pull",
        action="store_true",
        help="Disable remote cloning for this bot",
    )

    profile_pull = profile_subparsers.add_parser(
        "pull", help="Clone a bot from an authenticated remote gateway"
    )
    profile_pull.add_argument("profile_name", help="Remote profile to clone")
    profile_pull.add_argument(
        "--from",
        dest="remote_url",
        help="Remote gateway URL (default: gateway.proxy_url)",
    )
    profile_pull.add_argument(
        "--name", dest="clone_name", help="Local name for the cloned bot"
    )

    profile_push = profile_subparsers.add_parser(
        "push", help="Clone a local bot to an opt-in remote gateway"
    )
    profile_push.add_argument("profile_name", help="Local profile to clone")
    profile_push.add_argument(
        "--to",
        dest="remote_url",
        help="Remote gateway URL (default: gateway.proxy_url)",
    )
    profile_push.add_argument(
        "--name", dest="clone_name", help="Name to use on the remote gateway"
    )

    # ---------- Distribution subcommands (issue #20456) ----------
    profile_install = profile_subparsers.add_parser(
        "install",
        help="Install a profile distribution from a git URL or local directory",
        description=(
            "Install a Hermes profile distribution. SOURCE can be a git URL "
            "(github.com/user/repo, https://..., git@...) or a local "
            "directory containing distribution.yaml at its root."
        ),
    )
    profile_install.add_argument(
        "source",
        help="Distribution source (git URL or local directory)",
    )
    profile_install.add_argument(
        "--name", dest="install_name", metavar="NAME",
        help="Override profile name (default: read from manifest)",
    )
    profile_install.add_argument(
        "--alias", action="store_true",
        help="Create a shell wrapper alias for the installed profile",
    )
    profile_install.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing profile of the same name (user data preserved)",
    )
    profile_install.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip manifest preview confirmation",
    )

    profile_update = profile_subparsers.add_parser(
        "update",
        help="Re-pull a distribution and apply updates (user data preserved)",
        description=(
            "Fetch the distribution from its recorded source and overwrite "
            "distribution-owned files (SOUL.md, skills/, cron/, mcp.json). "
            "User data (memories, sessions, auth, .env) is never touched. "
            "config.yaml is preserved unless --force-config is passed."
        ),
    )
    profile_update.add_argument("profile_name", help="Profile to update")
    profile_update.add_argument(
        "--force-config", action="store_true",
        help="Also overwrite config.yaml (normally preserved to keep user overrides)",
    )
    profile_update.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation",
    )

    profile_info = profile_subparsers.add_parser(
        "info",
        help="Show a profile's distribution manifest (version, requirements, source)",
    )
    profile_info.add_argument("profile_name", help="Profile to inspect")

    profile_parser.set_defaults(func=cmd_profile)
