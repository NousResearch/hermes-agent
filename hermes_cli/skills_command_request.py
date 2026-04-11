"""Shared parser tree and request model for `hermes skills` and `/skills`."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass


class SkillsParseError(ValueError):
    """Raised when the shared skills parser rejects input."""


class _SkillsArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise SkillsParseError(message)

    def exit(self, status=0, message=None):
        if message:
            raise SkillsParseError(message.strip())
        raise SkillsParseError(f"parser exited with status {status}")


@dataclass(frozen=True)
class SkillsCommandRequest:
    action: str
    command_source: str
    page: int = 1
    size: int = 20
    source_filter: str = "all"
    query: str = ""
    limit: int = 10
    identifier: str = ""
    category: str = ""
    force: bool = False
    skip_confirm: bool = False
    invalidate_cache: bool = False
    name: str = ""
    skill_path: str = ""
    target: str = "github"
    repo: str = ""
    snapshot_action: str = ""
    path: str = ""
    tap_action: str = ""
    config_available: bool = True
    error: str = ""
    help_text: str = ""


def register_skills_subcommands(
    skills_parser: argparse.ArgumentParser,
    *,
    include_config: bool = True,
) -> None:
    skills_subparsers = skills_parser.add_subparsers(dest="skills_action")

    skills_browse = skills_subparsers.add_parser("browse", help="Browse all available skills (paginated)")
    skills_browse.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    skills_browse.add_argument("--size", type=int, default=20, help="Results per page (default: 20)")
    skills_browse.add_argument("--source", default="all",
                               choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"],
                               help="Filter by source (default: all)")

    skills_search = skills_subparsers.add_parser("search", help="Search skill registries")
    skills_search.add_argument("query", nargs="+", help="Search query")
    skills_search.add_argument("--source", default="all", choices=["all", "official", "skills-sh", "well-known", "github", "clawhub", "lobehub"])
    skills_search.add_argument("--limit", type=int, default=10, help="Max results")

    skills_install = skills_subparsers.add_parser("install", help="Install a skill")
    skills_install.add_argument("identifier", help="Skill identifier (e.g. openai/skills/skill-creator)")
    skills_install.add_argument("--category", default="", help="Category folder to install into")
    skills_install.add_argument("--force", action="store_true", help="Install despite blocked scan verdict")
    skills_install.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt (needed in TUI mode)")
    skills_install.add_argument("--now", action="store_true", help="Invalidate prompt cache immediately")

    skills_inspect = skills_subparsers.add_parser("inspect", help="Preview a skill without installing")
    skills_inspect.add_argument("identifier", help="Skill identifier")

    skills_list = skills_subparsers.add_parser("list", help="List installed skills")
    skills_list.add_argument("--source", default="all", choices=["all", "hub", "builtin", "local"])

    skills_check = skills_subparsers.add_parser("check", help="Check installed hub skills for updates")
    skills_check.add_argument("name", nargs="?", help="Specific skill to check (default: all)")

    skills_update = skills_subparsers.add_parser("update", help="Update installed hub skills")
    skills_update.add_argument("name", nargs="?", help="Specific skill to update (default: all outdated skills)")

    skills_audit = skills_subparsers.add_parser("audit", help="Re-scan installed hub skills")
    skills_audit.add_argument("name", nargs="?", help="Specific skill to audit (default: all)")

    skills_uninstall = skills_subparsers.add_parser("uninstall", help="Remove a hub-installed skill")
    skills_uninstall.add_argument("name", help="Skill name to remove")
    skills_uninstall.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    skills_uninstall.add_argument("--now", action="store_true", help="Invalidate prompt cache immediately")

    skills_publish = skills_subparsers.add_parser("publish", help="Publish a skill to a registry")
    skills_publish.add_argument("skill_path", help="Path to skill directory")
    skills_publish.add_argument("--to", default="github", choices=["github", "clawhub"], help="Target registry")
    skills_publish.add_argument("--repo", default="", help="Target GitHub repo (e.g. openai/skills)")

    skills_snapshot = skills_subparsers.add_parser("snapshot", help="Export/import skill configurations")
    snapshot_subparsers = skills_snapshot.add_subparsers(dest="snapshot_action")
    snap_export = snapshot_subparsers.add_parser("export", help="Export installed skills to a file")
    snap_export.add_argument("output", help="Output JSON file path (use - for stdout)")
    snap_import = snapshot_subparsers.add_parser("import", help="Import and install skills from a file")
    snap_import.add_argument("input", help="Input JSON file path")
    snap_import.add_argument("--force", action="store_true", help="Force install despite caution verdict")

    skills_tap = skills_subparsers.add_parser("tap", help="Manage skill sources")
    tap_subparsers = skills_tap.add_subparsers(dest="tap_action")
    tap_subparsers.add_parser("list", help="List configured taps")
    tap_add = tap_subparsers.add_parser("add", help="Add a GitHub repo as skill source")
    tap_add.add_argument("repo", help="GitHub repo (e.g. owner/repo)")
    tap_rm = tap_subparsers.add_parser("remove", help="Remove a tap")
    tap_rm.add_argument("name", help="Tap name to remove")

    if include_config:
        skills_subparsers.add_parser("config", help="Interactive skill configuration — enable/disable individual skills")


def skills_subcommand_names(*, command_source: str = "cli") -> list[str]:
    parser = _SkillsArgumentParser(prog="skills", add_help=False)
    register_skills_subcommands(parser, include_config=(command_source != "slash"))
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    return sorted(action.choices.keys())


def skills_subcommand_tree(*, command_source: str = "cli") -> dict[str, list[str]]:
    parser = _SkillsArgumentParser(prog="skills", add_help=False)
    register_skills_subcommands(parser, include_config=(command_source != "slash"))
    action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    tree: dict[str, list[str]] = {}
    for name, subparser in action.choices.items():
        nested: list[str] = []
        for sub_action in getattr(subparser, "_actions", []):
            if isinstance(sub_action, argparse._SubParsersAction):
                nested = list(sub_action.choices.keys())
                break
        tree[name] = nested
    return tree


def format_skills_help(
    *,
    command_source: str,
    subcommands: list[str] | None = None,
) -> str:
    parser = _SkillsArgumentParser(
        prog="/skills" if command_source == "slash" else "skills",
        add_help=True,
    )
    register_skills_subcommands(parser, include_config=(command_source != "slash"))

    if not subcommands:
        return parser.format_help()

    current = parser
    remaining = list(subcommands)
    while remaining:
        token = remaining.pop(0)
        subparser_action = next(
            (
                action
                for action in getattr(current, "_actions", [])
                if isinstance(action, argparse._SubParsersAction)
            ),
            None,
        )
        if subparser_action is None or token not in subparser_action.choices:
            return parser.format_help()
        current = subparser_action.choices[token]

    return current.format_help()


def request_from_namespace(args, *, command_source: str) -> SkillsCommandRequest:
    action = getattr(args, "skills_action", None) or "help"
    skip_confirm = getattr(args, "yes", False)
    if command_source == "slash" and action in {"install", "uninstall"}:
        skip_confirm = True
    invalidate_cache = bool(getattr(args, "now", False))
    if command_source == "cli" and action in {"install", "uninstall"}:
        invalidate_cache = True

    path = ""
    if action == "snapshot":
        if getattr(args, "snapshot_action", None) == "export":
            path = getattr(args, "output", "")
        elif getattr(args, "snapshot_action", None) == "import":
            path = getattr(args, "input", "")

    tap_action = getattr(args, "tap_action", "") or ""
    if command_source == "slash" and action == "tap" and not tap_action:
        tap_action = "list"

    return SkillsCommandRequest(
        action=action,
        command_source=command_source,
        page=getattr(args, "page", 1),
        size=getattr(args, "size", 20),
        source_filter=getattr(args, "source", "all"),
        query=" ".join(getattr(args, "query", []) or []) if isinstance(getattr(args, "query", ""), list) else (getattr(args, "query", "") or ""),
        limit=getattr(args, "limit", 10),
        identifier=getattr(args, "identifier", "") or "",
        category=getattr(args, "category", "") or "",
        force=bool(getattr(args, "force", False)),
        skip_confirm=bool(skip_confirm),
        invalidate_cache=invalidate_cache,
        name=getattr(args, "name", "") or "",
        skill_path=getattr(args, "skill_path", "") or "",
        target=getattr(args, "to", "github") or "github",
        repo=getattr(args, "repo", "") or "",
        snapshot_action=getattr(args, "snapshot_action", "") or "",
        path=path,
        tap_action=tap_action,
        config_available=(command_source != "slash"),
    )


def parse_skills_slash_command(cmd: str) -> SkillsCommandRequest:
    try:
        parts = shlex.split(cmd.strip())
    except ValueError as exc:
        return SkillsCommandRequest(action="error", command_source="slash", error=str(exc))
    if parts and parts[0].lower() == "/skills":
        parts = parts[1:]
    if not parts:
        return SkillsCommandRequest(action="help", command_source="slash")

    if any(part in {"--help", "-h"} for part in parts):
        help_parts = [part for part in parts if part not in {"--help", "-h"}]
        if help_parts and help_parts[0] == "config":
            return SkillsCommandRequest(
                action="error",
                command_source="slash",
                error="`/skills config` is only available in the interactive CLI.",
            )
        return SkillsCommandRequest(
            action="help",
            command_source="slash",
            help_text=format_skills_help(command_source="slash", subcommands=help_parts),
        )

    parser = _SkillsArgumentParser(prog="/skills", add_help=False)
    register_skills_subcommands(parser, include_config=False)
    try:
        args = parser.parse_args(parts)
    except SkillsParseError as exc:
        if parts and parts[0] == "config":
            return SkillsCommandRequest(
                action="error",
                command_source="slash",
                error="`/skills config` is only available in the interactive CLI.",
            )
        return SkillsCommandRequest(action="error", command_source="slash", error=str(exc))
    return request_from_namespace(args, command_source="slash")
