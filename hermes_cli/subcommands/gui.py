"""``hermes gui`` / ``hermes desktop`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_gui_parser(subparsers, *, cmd_gui: Callable) -> None:
    """Attach the ``gui`` / ``desktop`` subcommand to ``subparsers``."""
    gui_parser = subparsers.add_parser(
        "desktop",
        aliases=["gui"],
        help="Build and launch the native desktop app",
        description=(
            "Launch the Hermes Electron desktop app. By default this installs "
            "workspace Node dependencies, builds the current OS's unpacked "
            "Electron app, then launches that packaged artifact. "
            "Use `hermes desktop instance` for a separate isolated Desktop "
            "shell (distinct from Settings → Connections)."
        ),
    )
    gui_parser.add_argument(
        "--source",
        action="store_true",
        help="Launch via `electron .` against apps/desktop/dist instead of the packaged app",
    )
    gui_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the desktop app but do not launch it (used by the installer's --update flow)",
    )
    gui_parser.add_argument(
        "--fake-boot",
        action="store_true",
        help="Enable deterministic desktop boot delays for validating startup UI",
    )
    gui_parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Force Desktop to ignore any hermes CLI already on PATH during backend resolution",
    )
    gui_parser.add_argument(
        "--hermes-root",
        help="Override the Hermes source root used by Desktop (sets HERMES_DESKTOP_HERMES_ROOT)",
    )
    gui_parser.add_argument(
        "--cwd",
        help="Initial project directory for Desktop chat sessions (sets HERMES_DESKTOP_CWD)",
    )
    gui_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm install/package and launch the existing unpacked app from apps/desktop/release",
    )
    gui_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force a full rebuild even if the content stamp matches",
    )
    gui_parser.set_defaults(func=cmd_gui, desktop_action=None, instance_action=None)

    desktop_sub = gui_parser.add_subparsers(dest="desktop_action", required=False)
    instance = desktop_sub.add_parser(
        "instance",
        help="Manage isolated Desktop shells (separate userData/home; shared runtime)",
        description=(
            "Create and launch named isolated Desktop applications. Each "
            "instance has its own Electron userData, HERMES_HOME, and "
            "single-instance lock, but shares the canonical Hermes install. "
            "This is not Settings → Connections (which keeps one shared shell)."
        ),
    )
    instance.set_defaults(func=cmd_gui)
    instance_sub = instance.add_subparsers(dest="instance_action", required=False)

    create = instance_sub.add_parser(
        "create",
        help="Register a named isolated Desktop instance and install its shortcut",
    )
    create.add_argument(
        "instance_name", help="Instance slug (for example grace or athena)"
    )
    create.add_argument(
        "--ssh-host",
        required=True,
        help="SSH config alias or hostname (remote state stays on that machine)",
    )
    create.add_argument(
        "--remote-hermes-path",
        required=True,
        help="Absolute path to hermes on the remote machine",
    )
    create.add_argument(
        "--remote-profile",
        required=True,
        help="Remote Hermes profile to attach (for example default)",
    )
    create.add_argument(
        "--display-name",
        help="Window/app name (default: 'Hermes <Name>')",
    )
    create.add_argument(
        "--connection-id",
        default="",
        help="Exact Connections registry id this isolated shell belongs to",
    )
    create.add_argument(
        "--ssh-user",
        default="",
        help="SSH username for the selected Connection (omit to use ssh config)",
    )
    create.add_argument(
        "--ssh-port",
        type=int,
        default=22,
        help="SSH port for the selected Connection (default: 22)",
    )
    create.add_argument(
        "--ssh-key-path",
        default="",
        help="Absolute local path to the SSH private key used by this Connection",
    )
    create.add_argument(
        "--skip-ssh-check",
        action="store_true",
        help="Do not probe ssh before writing the instance",
    )
    create.add_argument(
        "--no-shortcut",
        action="store_true",
        help="Create the instance without writing a Desktop shortcut",
    )
    create.set_defaults(func=cmd_gui)

    listed = instance_sub.add_parser("list", help="List isolated Desktop instances")
    listed.set_defaults(func=cmd_gui)

    show = instance_sub.add_parser(
        "show", help="Print one instance manifest (non-secret)"
    )
    show.add_argument("instance_name")
    show.set_defaults(func=cmd_gui)

    launch = instance_sub.add_parser(
        "launch",
        help="Launch or focus an isolated Desktop instance",
    )
    launch.add_argument("instance_name")
    launch.add_argument(
        "--deep-link",
        help="Forward a hermes:// URL into the isolated shell after launch",
    )
    launch.set_defaults(func=cmd_gui)

    shortcut = instance_sub.add_parser(
        "shortcut",
        help="Recreate the OS shortcut for an isolated Desktop instance",
    )
    shortcut.add_argument("instance_name")
    shortcut.set_defaults(func=cmd_gui)

    repair = instance_sub.add_parser(
        "repair",
        help="Refresh named hardlinks after a Desktop update",
    )
    repair.add_argument("instance_name", nargs="?", default=None)
    repair.add_argument(
        "--all",
        dest="all_instances",
        action="store_true",
        help="Repair every registered isolated instance",
    )
    repair.set_defaults(func=cmd_gui)

    remove = instance_sub.add_parser(
        "remove",
        help="Remove the launcher and shortcut; never deletes remote state",
    )
    remove.add_argument("instance_name")
    remove.add_argument(
        "--purge-local",
        action="store_true",
        help="Also delete this instance's local HERMES_HOME and Electron userData",
    )
    remove.add_argument(
        "--force",
        action="store_true",
        help="Remove the launcher even if the named executable appears locked",
    )
    remove.set_defaults(func=cmd_gui)
