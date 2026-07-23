"""``hermes rules`` subcommand parser and handler.

Usage::

    hermes rules list [--scope profile|project]
    hermes rules create <name> [--body TEXT] [--description TEXT]
                             [--always-apply] [--globs GLOB [GLOB ...]]
                             [--scope profile|project]
    hermes rules show <name> [--scope profile|project]
    hermes rules delete <name> [--scope profile|project]
    hermes rules apply [--scope profile|project]

Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

from agent.auto_config import render_frontmatter, validate_rule_frontmatter
from agent.rules_configure_tool import (
    get_active_profile_dir,
    resolve_rules_dir_for_scope,
)
from agent.rules_loader import (
    discover_project_rules_dirs,
    format_rules_for_prompt,
    load_active_rules,
    load_rules,
    match_glob_rules,
    partition_rules,
    parse_frontmatter,
)


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_rules_parser(subparsers, *, cmd_rules: Callable) -> None:
    """Attach the ``rules`` subcommand to ``subparsers``."""
    rules_parser = subparsers.add_parser(
        "rules",
        help="Manage project and profile rules",
        description=(
            "List, create, inspect, delete, and apply rules. "
            "Rules live in .hermes/rules/ (project) or "
            "~/.hermes/profiles/<profile>/rules/ (profile)."
        ),
    )
    rules_sub = rules_parser.add_subparsers(dest="rules_action")

    # list
    list_parser = rules_sub.add_parser("list", help="List active rules")
    list_parser.add_argument(
        "--scope",
        choices=["profile", "project", "all"],
        default="all",
        help="'profile' lists only profile rules, 'project' lists project rules, "
             "'all' lists both with scope labels (default: all)",
    )
    list_parser.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)",
    )

    # create
    create_parser = rules_sub.add_parser("create", help="Create a rule")
    create_parser.add_argument("name", help="Rule name (e.g. ui/conventions)")
    create_parser.add_argument("--body", default="", help="Rule body text")
    create_parser.add_argument("--description", default="", help="Short label")
    create_parser.add_argument(
        "--always-apply", action="store_true", dest="always_apply",
        help="Always inject this rule into the system prompt",
    )
    create_parser.add_argument(
        "--globs", nargs="+", dest="globs",
        help="File patterns that activate this rule (e.g. *.vue)",
    )
    create_parser.add_argument(
        "--scope", choices=["profile", "project"], default=None,
        help="Where to write the rule. "
             "Default: 'project' if .hermes/rules/ exists, else 'profile'.",
    )
    create_parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite an existing rule with the same name",
    )

    # show
    show_parser = rules_sub.add_parser("show", help="Show a rule's content")
    show_parser.add_argument("name", help="Rule name")
    show_parser.add_argument(
        "--scope", choices=["profile", "project"], default=None,
        help="Scope to search. Default: auto-detect.",
    )
    show_parser.add_argument(
        "--format", choices=["markdown", "json"], default="markdown",
        help="Output format (default: markdown)",
    )

    # delete
    delete_parser = rules_sub.add_parser("delete", help="Delete a rule")
    delete_parser.add_argument("name", help="Rule name")
    delete_parser.add_argument(
        "--scope", choices=["profile", "project"], default=None,
        help="Scope to search. Default: auto-detect.",
    )
    delete_parser.add_argument(
        "--force", "-f", action="store_true",
        help="Skip confirmation prompt",
    )

    # apply
    apply_parser = rules_sub.add_parser(
        "apply", help="Print the rules block that would be injected into the system prompt"
    )
    apply_parser.add_argument(
        "--cwd", default=None,
        help="Working directory for project rule discovery. Default: cwd.",
    )
    apply_parser.add_argument(
        "--touched", nargs="*",
        help="Simulate glob matching against these file paths",
    )

    rules_parser.set_defaults(func=cmd_rules)


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Dispatch to the appropriate action handler."""
    action = getattr(args, "rules_action", None)
    if action == "list":
        return _run_list(args)
    elif action == "create":
        return _run_create(args)
    elif action == "show":
        return _run_show(args)
    elif action == "delete":
        return _run_delete(args)
    elif action == "apply":
        return _run_apply(args)
    else:
        print("hermes rules: no action specified. run 'hermes rules --help'", file=sys.stderr)
        return 1


# ---------------------------------------------------------------------------
# Action handlers
# ---------------------------------------------------------------------------


def _rule_scope_for_cwd(cwd: Optional[Path] = None) -> tuple[Path, Path]:
    """Return (project_rules_dir, profile_rules_dir) for display."""
    from agent.rules_loader import resolve_rules_dir
    profile_dir = get_active_profile_dir()
    project_dirs = discover_project_rules_dirs(cwd)
    project_dir = project_dirs[0] if project_dirs else None
    return project_dir, resolve_rules_dir(profile_dir)


def _run_list(args: argparse.Namespace) -> int:
    """List rules from project and/or profile scopes."""
    import json

    profile_dir = get_active_profile_dir()
    project_dirs = discover_project_rules_dirs()

    rows = []
    scopes_to_list = [args.scope] if args.scope != "all" else ["project", "profile"]

    if "project" in scopes_to_list:
        for pdir in project_dirs:
            for rule in load_rules(pdir):
                rows.append({
                    "name": rule.rel_id,
                    "scope": "project",
                    "description": rule.description,
                    "alwaysApply": rule.always_apply,
                    "globs": rule.globs,
                    "path": str(rule.path),
                })

    if "profile" in scopes_to_list:
        from agent.rules_loader import resolve_rules_dir
        profile_rules_dir = resolve_rules_dir(profile_dir)
        for rule in load_rules(profile_rules_dir):
            rows.append({
                "name": rule.rel_id,
                "scope": "profile",
                "description": rule.description,
                "alwaysApply": rule.always_apply,
                "globs": rule.globs,
                "path": str(rule.path),
            })

    if not rows:
        print("No rules found.")
        return 0

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'NAME':<30} {'SCOPE':<10} {'ALWAYS':<8} {'DESCRIPTION'}")
        print("-" * 80)
        for r in rows:
            always = "yes" if r["alwaysApply"] else "no"
            globs = ", ".join(r["globs"]) if r["globs"] else "-"
            print(
                f"{r['name']:<30} {r['scope']:<10} {always:<8} "
                f"{r['description'] or '-'}"
            )

    return 0


def _run_create(args: argparse.Namespace) -> int:
    """Create or update a rule file."""
    from agent.auto_config import safe_path, safe_write
    from agent.rules_loader import resolve_rules_dir

    scope = args.scope
    cwd = Path.cwd()
    rules_dir = resolve_rules_dir_for_scope(scope, cwd)
    rules_dir.mkdir(parents=True, exist_ok=True)

    target = safe_path(rules_dir, args.name, ".md")
    if target.exists() and not args.overwrite:
        print(
            f"Rule {args.name!r} already exists at {target}.\n"
            "Use --overwrite to replace it, or use 'hermes rules create --update'.",
            file=sys.stderr
        )
        return 1

    meta = {}
    if args.description:
        meta["description"] = args.description
    if args.always_apply:
        meta["alwaysApply"] = True
    if args.globs:
        meta["globs"] = args.globs

    if meta:
        validate_rule_frontmatter(meta)

    body = args.body or ""
    content = render_frontmatter(meta, body)
    safe_write(target, content)
    print(f"Rule {args.name!r} written to {target}")
    return 0


def _run_show(args: argparse.Namespace) -> int:
    """Print a rule's full content."""
    import json

    from agent.rules_configure_tool import run as tool_run

    result = tool_run(action="read", name=args.name, scope=args.scope)
    if not result.get("ok"):
        print(
            f"hermes rules show: {result.get('error_code')}: "
            f"{result.get('error_message')}",
            file=sys.stderr
        )
        return 1

    rule = result["rule"]
    if args.format == "json":
        print(json.dumps(rule, indent=2))
    else:
        print(f"# {args.name}")
        if rule["frontmatter"]:
            import yaml
            print("---")
            print(yaml.safe_dump(rule["frontmatter"], default_flow_style=False).rstrip())
            print("---")
        print()
        print(rule["body"])
    return 0


def _run_delete(args: argparse.Namespace) -> int:
    """Delete a rule file."""
    from agent.rules_configure_tool import run as tool_run

    if not args.force:
        confirm = input(f"Delete rule {args.name!r}? [y/N] ")
        if confirm.lower() not in ("y", "yes"):
            print("Aborted.")
            return 1

    result = tool_run(action="delete", name=args.name, scope=args.scope)
    if not result.get("ok"):
        print(
            f"hermes rules delete: {result.get('error_code')}: "
            f"{result.get('error_message')}",
            file=sys.stderr
        )
        return 1

    print(f"Rule {args.name!r} deleted.")
    return 0


def _run_apply(args: argparse.Namespace) -> int:
    """Print the effective rules block for the current session context."""
    cwd = Path(args.cwd) if args.cwd else Path.cwd()
    profile_dir = get_active_profile_dir()
    rules = load_active_rules(profile_dir, cwd)
    always_on, glob_scoped = partition_rules(rules)

    if args.touched:
        matched = match_glob_rules(glob_scoped, args.touched)
        effective = always_on + matched
    else:
        effective = always_on

    if not effective:
        print("(no active rules)")
        return 0

    output = format_rules_for_prompt(effective)
    print(output)
    return 0
