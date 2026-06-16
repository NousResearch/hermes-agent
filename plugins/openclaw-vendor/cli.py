"""CLI for the openclaw-vendor Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="openclaw_vendor_command")

    subs.add_parser("status", help="Vendor mirror, extensions, packages, and skill link state")
    subs.add_parser("list", help="List all vendor extensions, packages, and sibling plugins")

    install = subs.add_parser(
        "install",
        help="Link all vendor extension skills into ~/.hermes/skills/",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Replace existing skill links",
    )
    install.add_argument(
        "--extension",
        default="",
        help="Sync only one extension id (e.g. hypura-harness)",
    )

    sync = subs.add_parser(
        "sync",
        help="Refresh ~/.hermes/skills links from vendor/openclaw-mirror",
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Replace existing skill links",
    )
    sync.add_argument(
        "--extension",
        default="",
        help="Sync only one extension id",
    )

    subparser.set_defaults(func=openclaw_vendor_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def openclaw_vendor_command(args: argparse.Namespace) -> int:
    command = getattr(args, "openclaw_vendor_command", None)
    if not command:
        print("usage: hermes openclaw-vendor {install,sync,status,list}")
        return 2

    ext = (getattr(args, "extension", "") or "").strip() or None

    if command == "status":
        return _print(core.status())
    if command == "list":
        return _print(core.list_units())
    if command == "install":
        return _print(core.install(force=getattr(args, "force", False), extension_id=ext))
    if command == "sync":
        return _print(core.sync_all_skills(force=getattr(args, "force", False), extension_id=ext))
    print(f"unknown openclaw-vendor subcommand: {command}")
    return 2
