"""``hermes plugins`` subcommand parser.

Extracted from ``hermes_cli/main.py:main()`` (god-file Phase 2 follow-up).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_plugins_parser(subparsers, *, cmd_plugins: Callable) -> None:
    """Attach the ``plugins`` subcommand to ``subparsers``."""
    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage plugins — install, update, remove, list",
        description="Install plugins from Git repositories, update, remove, or list them.",
    )
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_action")

    plugins_install = plugins_subparsers.add_parser(
        "install", help="Install a plugin from a Git URL or owner/repo"
    )
    plugins_install.add_argument(
        "identifier",
        help="Git URL or owner/repo shorthand (e.g. anpicasso/hermes-plugin-chrome-profiles)",
    )
    plugins_install.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Remove existing plugin and reinstall",
    )
    _install_enable_group = plugins_install.add_mutually_exclusive_group()
    _install_enable_group.add_argument(
        "--enable",
        action="store_true",
        help="Auto-enable the plugin after install (skip confirmation prompt)",
    )
    _install_enable_group.add_argument(
        "--no-enable",
        action="store_true",
        help="Install disabled (skip confirmation prompt); enable later with `hermes plugins enable <name>`",
    )

    plugins_update = plugins_subparsers.add_parser(
        "update", help="Pull latest changes for an installed plugin"
    )
    plugins_update.add_argument("name", help="Plugin name to update")

    plugins_remove = plugins_subparsers.add_parser(
        "remove", aliases=["rm", "uninstall"], help="Remove an installed plugin"
    )
    plugins_remove.add_argument("name", help="Plugin directory name to remove")

    plugins_list = plugins_subparsers.add_parser(
        "list", aliases=["ls"], help="List installed plugins"
    )
    plugins_list.add_argument(
        "--enabled",
        action="store_true",
        help="Show only enabled plugins",
    )
    plugins_list.add_argument(
        "--user",
        action="store_true",
        help="Show only user-installed plugins (including git plugins)",
    )
    plugins_list.add_argument(
        "--no-bundled",
        action="store_true",
        help="Hide bundled plugins",
    )
    plugins_list.add_argument(
        "--plain",
        action="store_true",
        help="Print compact plain-text output instead of a Rich table",
    )
    plugins_list.add_argument(
        "--json",
        action="store_true",
        help="Emit backwards-compatible plugin list JSON",
    )
    plugins_list.add_argument(
        "--inventory-json",
        action="store_true",
        help="Emit stable schema-v1 plugin inventory JSON",
    )

    plugins_enable = plugins_subparsers.add_parser(
        "enable", help="Enable a disabled plugin"
    )
    plugins_enable.add_argument("name", help="Plugin name to enable")
    _enable_override_group = plugins_enable.add_mutually_exclusive_group()
    _enable_override_group.add_argument(
        "--allow-tool-override",
        action="store_true",
        help="Grant this plugin permission to replace built-in tools "
        "(e.g. shell_exec, write_file). Skips the confirmation prompt.",
    )
    _enable_override_group.add_argument(
        "--no-allow-tool-override",
        action="store_true",
        help="Enable without granting built-in tool override (skip prompt).",
    )

    plugins_disable = plugins_subparsers.add_parser(
        "disable", help="Disable a plugin without removing it"
    )
    plugins_disable.add_argument("name", help="Plugin name to disable")
    plugins_parser.set_defaults(func=cmd_plugins)
