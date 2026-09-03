"""``hermes recipe`` — export, preview, and install shareable setup bundles.

Inspired by Poke Recipes (poke.com/docs/creating-recipes): one shareable
YAML file that bundles cron automations, remote MCP integrations, skill
references, and a starter prompt. See ``hermes_cli/recipes.py`` for the
format and security model (secrets never travel; installs are consent-first;
no executable config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _split_csv(value: str) -> list:
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def cmd_recipe(args: argparse.Namespace) -> int:
    from hermes_cli.recipes import (
        RecipeError,
        build_recipe,
        describe_recipe,
        dump_recipe,
        install_recipe,
        load_recipe,
    )

    action = getattr(args, "recipe_action", None)
    try:
        if action == "export":
            recipe = build_recipe(
                name=args.name,
                description=args.description,
                author=args.author,
                starter_prompt=args.starter_prompt,
                job_ids=_split_csv(args.jobs),
                mcp_names=_split_csv(args.mcp),
                skills=_split_csv(args.skills),
            )
            text = dump_recipe(recipe)
            if args.output:
                out = Path(args.output).expanduser()
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(text, encoding="utf-8")
                print(f"Recipe written to {out}")
            else:
                print(text)
            return 0

        if action in ("show", "preview"):
            recipe = load_recipe(args.source)
            print(describe_recipe(recipe))
            return 0

        if action == "install":
            recipe = load_recipe(args.source)
            print(describe_recipe(recipe))
            print()
            if not args.yes:
                try:
                    answer = input("Install this recipe? [y/N] ").strip().lower()
                except EOFError:
                    answer = ""
                if answer not in ("y", "yes"):
                    print("Aborted — nothing was installed.")
                    return 1
            summary = install_recipe(recipe, enable_jobs=args.enable)
            for job in summary["cron_jobs"]:
                state = "enabled" if args.enable else "paused"
                print(f"✓ cron job created ({state}): {job.get('name')} [{job['id']}]")
            for sname in summary["mcp_servers"]:
                print(f"✓ MCP server added to config.yaml: {sname}")
            for sname in summary["mcp_skipped"]:
                print(f"— MCP server '{sname}' already exists; left untouched")
            required = recipe.get("required_secrets") or {}
            for sname, keys in required.items():
                print(
                    f"! MCP server '{sname}' needs credentials you must add "
                    f"yourself ({', '.join(keys)}) — recipes never carry secrets."
                )
            for skill in summary["skills"]:
                print(f"→ install skill with: hermes skills install {skill}")
            if summary["cron_jobs"] and not args.enable:
                print("Jobs are paused. Review them, then: hermes cron resume <id>")
            if recipe.get("starter_prompt"):
                print("\nSuggested first message:")
                print(f"  {recipe['starter_prompt']}")
            return 0

    except RecipeError as exc:
        print(f"Recipe error: {exc}", file=sys.stderr)
        return 1

    print("Usage: hermes recipe {export|show|install} … (see --help)", file=sys.stderr)
    return 1


def build_recipe_parser(subparsers) -> None:
    """Attach the ``recipe`` subcommand to ``subparsers``."""
    recipe_parser = subparsers.add_parser(
        "recipe",
        help="Share or adopt setup bundles (cron jobs + MCP servers + skills)",
        description=(
            "Export your automations/integrations as one shareable YAML "
            "recipe, or install someone else's from a file or URL. Secrets "
            "are never exported; installs preview everything and create "
            "cron jobs paused by default."
        ),
    )
    recipe_sub = recipe_parser.add_subparsers(dest="recipe_action")

    exp = recipe_sub.add_parser("export", help="Export a recipe from this install")
    exp.add_argument("--name", required=True, help="Recipe display name")
    exp.add_argument("--description", default="", help="What this setup does")
    exp.add_argument("--author", default="", help="Your name/handle (optional)")
    exp.add_argument(
        "--starter-prompt",
        default="",
        help="Suggested first message for adopters (optional)",
    )
    exp.add_argument(
        "--jobs",
        default="",
        help="Comma-separated cron job IDs or names to include",
    )
    exp.add_argument(
        "--mcp",
        default="",
        help="Comma-separated MCP server names from config.yaml to include "
        "(remote http/sse servers only; credentials are stripped)",
    )
    exp.add_argument(
        "--skills",
        default="",
        help="Comma-separated skill identifiers to recommend "
        "(e.g. official/research/arxiv)",
    )
    exp.add_argument("-o", "--output", default="", help="Write to file (default: stdout)")

    show = recipe_sub.add_parser(
        "show", aliases=["preview"], help="Preview a recipe without installing"
    )
    show.add_argument("source", help="Recipe file path or https URL")

    inst = recipe_sub.add_parser("install", help="Install a recipe (consent-first)")
    inst.add_argument("source", help="Recipe file path or https URL")
    inst.add_argument(
        "--yes", "-y", action="store_true", help="Skip the confirmation prompt"
    )
    inst.add_argument(
        "--enable",
        action="store_true",
        help="Enable installed cron jobs immediately (default: paused)",
    )

    recipe_parser.set_defaults(func=cmd_recipe)
