"""``hermes hki`` CLI for project-scoped knowledge artifacts."""

from __future__ import annotations

import argparse
import sys

from hki.inventory import build_inventory, load_inventory, write_inventory
from hki.manifest import build_manifest, load_manifest, write_manifest
from hki.paths import resolve_scope
from hki.report import write_sources_report


def build_parser(
    parent_subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Attach the ``hki`` subcommand tree. Returns the top parser."""

    parser = parent_subparsers.add_parser(
        "hki",
        help="Manage project-scoped Human Knowledge Infrastructure artifacts",
        description=(
            "Create project-scoped HKI workspace artifacts under .hermes/hki/. "
            "This first slice inventories sources, builds a stable manifest, "
            "and writes a basic source report."
        ),
    )
    sub = parser.add_subparsers(dest="hki_action")

    inventory = sub.add_parser("inventory", help="Scan the workspace and write inventory.json")
    _add_cwd_arg(inventory)

    manifest = sub.add_parser("manifest", help="Build manifest.json from inventory.json")
    _add_cwd_arg(manifest)

    report = sub.add_parser("report", help="Write HKI reports")
    report_sub = report.add_subparsers(dest="hki_report_action")
    sources = report_sub.add_parser("sources", help="Write reports/sources.md")
    _add_cwd_arg(sources)

    parser.set_defaults(_hki_parser=parser)
    report.set_defaults(_hki_report_parser=report)
    return parser


def hki_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes hki ...`` argparse dispatch."""

    action = getattr(args, "hki_action", None)
    if not action:
        parser = getattr(args, "_hki_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print("usage: hermes hki <action> [options]", file=sys.stderr)
        return 0

    try:
        if action == "inventory":
            return _cmd_inventory(args)
        if action == "manifest":
            return _cmd_manifest(args)
        if action == "report":
            report_action = getattr(args, "hki_report_action", None)
            if report_action == "sources":
                return _cmd_report_sources(args)
            parser = getattr(args, "_hki_report_parser", None)
            if parser is not None:
                parser.print_help()
                return 0
            print("usage: hermes hki report <report> [options]", file=sys.stderr)
            return 1
    except (OSError, ValueError) as exc:
        print(f"hki: {exc}", file=sys.stderr)
        return 2

    print(f"Unknown hki action: {action}", file=sys.stderr)
    return 1


def _add_cwd_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cwd",
        default=".",
        metavar="PATH",
        help="Workspace root to inspect (default: current directory)",
    )


def _cmd_inventory(args: argparse.Namespace) -> int:
    scope = resolve_scope(args.cwd)
    inventory = build_inventory(scope)
    path = write_inventory(inventory, scope)
    print(f"Wrote HKI inventory: {path}")
    print(f"  workspace: {scope.root}")
    print(f"  files:     {len(inventory.files)}")
    print(f"  skipped:   {inventory.skipped_count}")
    return 0


def _cmd_manifest(args: argparse.Namespace) -> int:
    scope = resolve_scope(args.cwd)
    inventory = _load_or_create_inventory(scope)
    manifest = build_manifest(inventory, inventory_path=scope.inventory_path)
    path = write_manifest(manifest, scope)
    print(f"Wrote HKI manifest: {path}")
    print(f"  workspace: {scope.root}")
    print(f"  sources:   {len(manifest.sources)}")
    return 0


def _cmd_report_sources(args: argparse.Namespace) -> int:
    scope = resolve_scope(args.cwd)
    manifest = _load_or_create_manifest(scope)
    path = write_sources_report(manifest, scope)
    print(f"Wrote HKI source report: {path}")
    print(f"  workspace: {scope.root}")
    print(f"  manifest:  {scope.manifest_path}")
    return 0


def _load_or_create_inventory(scope):
    if scope.inventory_path.exists():
        return load_inventory(scope.inventory_path)
    inventory = build_inventory(scope)
    write_inventory(inventory, scope)
    return inventory


def _load_or_create_manifest(scope):
    if scope.manifest_path.exists():
        return load_manifest(scope.manifest_path)
    inventory = _load_or_create_inventory(scope)
    manifest = build_manifest(inventory, inventory_path=scope.inventory_path)
    write_manifest(manifest, scope)
    return manifest
