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
        help="Manage and validate plugins",
        description=(
            "Install, update, remove, list, or validate native Hermes plugins "
            "and portable Agent Plugins v1 packages. Portable packages install disabled."
        ),
    )
    plugins_subparsers = plugins_parser.add_subparsers(dest="plugins_action")

    plugins_install = plugins_subparsers.add_parser(
        "install", help="Install a plugin from a Git URL, owner/repo, or index name"
    )
    plugins_install.add_argument(
        "identifier",
        help=(
            "Git URL, owner/repo shorthand (e.g. anpicasso/hermes-plugin-chrome-profiles), "
            "or a bare plugin name resolved through the community index "
            "(see `hermes plugins search`)"
        ),
    )
    plugins_install.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Remove existing plugin and reinstall",
    )
    plugins_install.add_argument(
        "--ref",
        metavar="COMMIT_SHA",
        help="Install exactly one immutable 40-character Git commit SHA",
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

    plugins_search = plugins_subparsers.add_parser(
        "search", help="Search the community plugin index"
    )
    plugins_search.add_argument(
        "term",
        nargs="?",
        default="",
        help="Search term matched fuzzily against name, description, and tags "
        "(omit to browse the full index)",
    )
    plugins_search.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON",
    )
    plugins_search.add_argument(
        "--capability",
        metavar="CAP",
        help="Filter by declared capability (e.g. tools, platform, commands)",
    )
    plugins_search.add_argument(
        "--refresh",
        action="store_true",
        help="Bypass the local cache and re-fetch the index",
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
        help="Print machine-readable JSON",
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

    plugins_capabilities = plugins_subparsers.add_parser(
        "capabilities",
        help="Show declared vs granted capabilities per plugin",
        description=(
            "Show each plugin's declared capabilities (from plugin.yaml) "
            "against what the user has granted. Capabilities are a consent "
            "and audit layer over host API surfaces — NOT a sandbox."
        ),
    )
    plugins_capabilities.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Plugin id to inspect (omit to list all plugins with capabilities)",
    )

    plugins_doctor = plugins_subparsers.add_parser(
        "doctor", help="Validate a plugin with the real runtime contracts"
    )
    plugins_doctor.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Plugin path or installed plugin id (default: current directory)",
    )
    plugins_doctor.add_argument(
        "--ci",
        action="store_true",
        help="Exit non-zero when validation reports an error",
    )

    plugins_pack = plugins_subparsers.add_parser(
        "pack",
        help="Declarative, shareable plugin sets (hermes-pack.yaml)",
        description=(
            "Install, export, or inspect plugin packs — a single YAML file "
            "pinning a set of plugins to exact commit SHAs, with optional "
            "non-secret config seeds. Installing a pack fans out to ordinary "
            "pinned installs; capability consent stays per-plugin."
        ),
    )
    pack_subparsers = plugins_pack.add_subparsers(dest="pack_action")

    pack_install = pack_subparsers.add_parser(
        "install", help="Review and install a pack from a file path or https URL"
    )
    pack_install.add_argument(
        "source", help="Path to a hermes-pack.yaml file, or an https:// URL"
    )
    pack_install.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Reinstall plugins that already exist",
    )

    pack_export = pack_subparsers.add_parser(
        "export",
        help="Emit a pack YAML for the current install on stdout",
    )
    pack_export.add_argument(
        "--enabled-only",
        action="store_true",
        help="Only include plugins currently in plugins.enabled",
    )
    pack_export.add_argument(
        "--name",
        default="my-hermes-pack",
        help="Pack name to embed in the exported YAML",
    )

    pack_show = pack_subparsers.add_parser(
        "show", help="Dry-run: parse and display a pack without installing"
    )
    pack_show.add_argument(
        "source", help="Path to a hermes-pack.yaml file, or an https:// URL"
    )

    plugins_show = plugins_subparsers.add_parser(
        "show",
        aliases=["info"],
        help="Show details for a single plugin (including emits/listens)",
    )
    plugins_show.add_argument("name", help="Plugin name or key to show")

    plugins_parser.set_defaults(func=cmd_plugins)
