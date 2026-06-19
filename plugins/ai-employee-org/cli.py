"""CLI for hermes ai-employees."""

from __future__ import annotations

import argparse

from . import core
from . import cron_install
from . import stack as stack_mod


def register_cli(subparser: argparse.ArgumentParser) -> None:
    subs = subparser.add_subparsers(dest="ai_employees_command")

    subs.add_parser("status", help="Profiles, skill link, ops dirs, cron scripts")

    setup = subs.add_parser("setup", help="Profiles + SOUL + kanban board only")
    setup.add_argument("--clone", action="store_true", help="Clone default profile on create")
    setup.add_argument("--board", default=core.BOARD_SLUG)

    skill = subs.add_parser("skill", help="Install bundled ai-employee-org skill")
    skill.add_argument("--force", action="store_true")
    skill.add_argument("--no-profiles", action="store_true", help="Default HERMES_HOME only")

    stack = subs.add_parser("stack", help="Apply bundled operator stack to config.yaml")
    stack.add_argument("--dry-run", action="store_true")

    cron = subs.add_parser("cron", help="Install all role cron jobs")
    cron_sub = cron.add_subparsers(dest="ai_employees_cron_command")
    inst = cron_sub.add_parser("install", help="Register 5 role crons in ~/.hermes/cron/jobs.json")
    inst.add_argument("--telegram-chat-id", default="", help="Override Telegram deliver chat_id")
    inst.add_argument("--dry-run", action="store_true")

    install = subs.add_parser("install", help="Full bootstrap: plugin, skill, profiles, crons, stack")
    install.add_argument("--clone", action="store_true")
    install.add_argument("--board", default=core.BOARD_SLUG)
    install.add_argument("--no-stack", action="store_true")
    install.add_argument("--no-cron", action="store_true")
    install.add_argument("--telegram-chat-id", default="")
    install.add_argument("--dry-run", action="store_true")

    subparser.set_defaults(func=ai_employees_command)


def ai_employees_command(args: argparse.Namespace) -> int:
    cmd = getattr(args, "ai_employees_command", None) or "status"

    if cmd == "status":
        print(core.json_dump(core.status()))
        return 0

    if cmd == "setup":
        payload = {
            "profiles": core.setup_profiles(clone=getattr(args, "clone", False)),
            "kanban": core.setup_kanban(board_slug=args.board),
            "ops": core.ensure_ops_dirs(),
        }
        print(core.json_dump(payload))
        return 0

    if cmd == "skill":
        payload = core.install_skill(
            profiles=not getattr(args, "no_profiles", False),
            force=getattr(args, "force", False),
        )
        print(core.json_dump(payload))
        return 0

    if cmd == "stack":
        payload = stack_mod.apply_ai_employee_stack(dry_run=getattr(args, "dry_run", False))
        print(core.json_dump(payload))
        return 0 if payload.get("ok", True) else 1

    if cmd == "cron":
        sub = getattr(args, "ai_employees_cron_command", None)
        if sub == "install":
            chat = (getattr(args, "telegram_chat_id", "") or "").strip() or None
            payload = cron_install.install_all_crons(
                telegram_chat_id=chat,
                dry_run=getattr(args, "dry_run", False),
            )
            print(core.json_dump(payload))
            return 0 if payload.get("ok") else 1
        print("Usage: hermes ai-employees cron install")
        return 2

    if cmd == "install":
        chat = (getattr(args, "telegram_chat_id", "") or "").strip() or None
        payload = core.install_all(
            clone=getattr(args, "clone", False),
            board_slug=args.board,
            apply_stack=not getattr(args, "no_stack", False),
            install_crons=not getattr(args, "no_cron", False),
            telegram_chat_id=chat,
            dry_run=getattr(args, "dry_run", False),
        )
        print(core.json_dump(payload))
        return 0 if payload.get("success", True) else 1

    print("Usage: hermes ai-employees {status|setup|skill|stack|cron|install}")
    return 2
