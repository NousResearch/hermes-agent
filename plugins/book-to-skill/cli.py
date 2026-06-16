"""CLI for the book-to-skill Hermes plugin."""

from __future__ import annotations

import argparse
import json

from . import core


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="book_to_skill_command")

    subs.add_parser("status", help="Show upstream checkout and Hermes skill link state")
    subs.add_parser("check", help="Run upstream extract.py --check for optional dependencies")

    install = subs.add_parser(
        "install",
        help="Clone upstream book-to-skill and link it into ~/.hermes/skills/",
    )
    install.add_argument(
        "--force",
        action="store_true",
        help="Re-clone upstream and refresh the Hermes skills link",
    )
    install.add_argument(
        "--ref",
        default="",
        help=f"Git branch or tag to clone (default: {core.DEFAULT_REF})",
    )

    sync = subs.add_parser(
        "sync",
        help="Refresh the ~/.hermes/skills/book-to-skill link from vendor/",
    )
    sync.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing skills/book-to-skill link",
    )

    subparser.set_defaults(func=book_to_skill_command)


def _print(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("ok", True) else 1


def book_to_skill_command(args: argparse.Namespace) -> int:
    command = getattr(args, "book_to_skill_command", None)
    if not command:
        print(
            "usage: hermes book-to-skill {install,sync,status,check}",
        )
        return 2
    if command == "status":
        return _print(core.status())
    if command == "check":
        return _print(core.check_extractors())
    if command == "install":
        ref = (getattr(args, "ref", "") or "").strip() or None
        return _print(core.install(force=getattr(args, "force", False), ref=ref))
    if command == "sync":
        return _print(core.sync_skill_link(force=getattr(args, "force", False)))
    print(f"unknown book-to-skill subcommand: {command}")
    return 2
