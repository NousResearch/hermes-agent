"""`hermes wisdom` command surface."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable


def _emit(value: Any, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str)
        )
    elif isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False, default=str))
    else:
        print(value)


def cmd_wisdom(args: argparse.Namespace) -> int:
    from hermes_wisdom.client import WisdomError
    from hermes_wisdom.package import PackagePolicyError
    from hermes_wisdom.service import WisdomService

    service = WisdomService()
    command = getattr(args, "wisdom_command", None)
    try:
        if command not in {"setup", "status", None}:
            service.require_setup()
        if command == "setup":
            accepted = bool(args.accept_disclosure)
            if not accepted:
                if not sys.stdin.isatty():
                    raise PackagePolicyError(
                        "noninteractive setup requires --accept-disclosure"
                    )
                from hermes_wisdom.service import WISDOM_DISCLOSURE

                print(WISDOM_DISCLOSURE)
                answer = input("Enable Collective Wisdom for this profile? [y/N] ")
                if answer.strip().lower() not in {"y", "yes"}:
                    return 7
                accepted = True
            result = service.setup(disclosure_accepted=accepted)
        elif command == "status":
            result = service.status()
        elif command == "scan":
            result = service.scan(getattr(args, "skill", None))
        elif command == "suggest":
            raw_specification = getattr(args, "system_specification", None)
            system_specification = None
            if raw_specification:
                try:
                    system_specification = json.loads(raw_specification)
                except json.JSONDecodeError as exc:
                    raise PackagePolicyError(
                        "--system-specification-json must be valid JSON"
                    ) from exc
            result = service.suggest(
                getattr(args, "skill", None),
                description=getattr(args, "description", None),
                system_specification=system_specification,
                allow_private_secret_review=getattr(
                    args, "private_secret_override", False
                ),
            )
        elif command == "candidates":
            result = {"candidates": service.scan_candidates()}
        elif command == "review":
            if not args.portal and not args.acknowledge and not sys.stdin.isatty():
                raise PackagePolicyError(
                    "noninteractive review cannot create consent; use --portal or an interactive --acknowledge"
                )
            acknowledge = bool(args.acknowledge)
            if not args.portal and not acknowledge:
                preview = service.review(args.draft_id, acknowledge=False)
                _emit(preview, as_json=args.json)
                answer = input(
                    "Review every raw file and the three hashes above. Record consent receipt? [y/N] "
                )
                if answer.strip().lower() not in {"y", "yes"}:
                    return 7
                acknowledge = True
            result = service.review(
                args.draft_id, acknowledge=acknowledge, portal=args.portal
            )
        elif command == "approve":
            result = service.approve(args.draft_id)
        elif command == "decline":
            result = service.decline(args.draft_id)
        elif command == "list":
            result = service.list_skills()
        elif command == "show":
            result = service.show(args.skill_id)
        elif command == "versions":
            result = {"versions": service.versions(args.skill_id)}
        elif command == "install":
            if args.apply_receipt:
                result = service.install_apply(
                    args.apply_receipt, accept_partial=args.accept_partial
                )
            else:
                plan = service.install_plan(
                    args.reference, update_mode=args.update_mode
                )
                if args.plan or args.json:
                    result = plan
                else:
                    _emit(plan, as_json=False)
                    if not sys.stdin.isatty():
                        raise PackagePolicyError(
                            "noninteractive install requires --plan then --apply-receipt"
                        )
                    answer = input("Apply this authenticated install plan? [y/N] ")
                    if answer.strip().lower() not in {"y", "yes"}:
                        return 7
                    result = service.install_apply(
                        plan["receipt"], accept_partial=args.accept_partial
                    )
        elif command == "check":
            result = service.check(apply_automatic=True)
        elif command == "update":
            if args.all:
                result = service.update_all(apply=True)
            elif args.apply_receipt:
                result = service.update_apply(
                    args.apply_receipt,
                    accept_sensitive=args.accept_sensitive,
                    accept_partial=args.accept_partial,
                    preserve_modified=args.preserve_modified,
                )
            elif args.skill_id:
                plan = service.update_plan(args.skill_id)
                if plan.get("state") == "current" or args.plan or args.json:
                    result = plan
                else:
                    _emit(plan, as_json=False)
                    if not sys.stdin.isatty():
                        raise PackagePolicyError(
                            "noninteractive update requires --plan then --apply-receipt"
                        )
                    answer = input("Apply this verified managed update plan? [y/N] ")
                    if answer.strip().lower() not in {"y", "yes"}:
                        return 7
                    result = service.update_apply(
                        plan["receipt"],
                        accept_sensitive=args.accept_sensitive,
                        accept_partial=args.accept_partial,
                        preserve_modified=args.preserve_modified,
                    )
            else:
                raise PackagePolicyError("update requires a skill id or --all")
        elif command == "uninstall":
            if not args.yes:
                if not sys.stdin.isatty():
                    raise PackagePolicyError("noninteractive uninstall requires --yes")
                answer = input(
                    "Move this managed skill to recoverable Wisdom trash? [y/N] "
                )
                if answer.strip().lower() not in {"y", "yes"}:
                    return 7
            result = service.uninstall(args.skill_id)
        elif command == "notifications":
            result = service.notifications(mark_seen=args.mark_seen)
        else:
            args._wisdom_parser.print_help()
            return 2
        _emit(result, as_json=bool(getattr(args, "json", False)))
        return 0
    except WisdomError as exc:
        _emit(
            {"ok": False, "error": str(exc), "category": exc.exit_code},
            as_json=bool(getattr(args, "json", False)),
        )
        return exc.exit_code
    except PackagePolicyError as exc:
        _emit(
            {"ok": False, "error": str(exc), "category": 6},
            as_json=bool(getattr(args, "json", False)),
        )
        return 6
    except KeyboardInterrupt:
        return 7


def build_wisdom_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "wisdom",
        help="Collective Wisdom — review, share, and install team skills",
        description=(
            "Local qualification stays on this device. Publication uploads only an owner-private "
            "instruction package and requires a complete, hash-bound owner review."
        ),
    )
    parser.set_defaults(func=cmd_wisdom, _wisdom_parser=parser)
    commands = parser.add_subparsers(dest="wisdom_command")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name, help=help_text)
        command.add_argument(
            "--json", action="store_true", help="Emit stable machine-readable JSON"
        )
        return command

    setup = add("setup", "Validate entitlement and initialize this profile")
    setup.add_argument(
        "--accept-disclosure",
        action="store_true",
        help="Accept the local telemetry and owner-private draft disclosure",
    )
    add("status", "Show local and Gateway Wisdom status")
    scan = add("scan", "Run local policy and advisory scans")
    scan.add_argument("skill", nargs="?")
    suggest = add("suggest", "Browse candidates or submit an owner-private draft")
    suggest.add_argument("skill", nargs="?")
    suggest.add_argument("--description", help="Owner-edited outcome description")
    suggest.add_argument(
        "--system-specification-json",
        dest="system_specification",
        help="Owner-reviewed declarative System Specification JSON",
    )
    suggest.add_argument(
        "--send-for-owner-only-server-review",
        dest="private_secret_override",
        action="store_true",
        help="Explicitly override a high-confidence local secret pause for owner-private review",
    )
    add("candidates", "List all manually selectable local candidates")
    review = add("review", "Review exact server draft bytes and hashes")
    review.add_argument("draft_id")
    review.add_argument(
        "--portal", action="store_true", help="Open the authenticated Portal review"
    )
    review.add_argument(
        "--acknowledge",
        action="store_true",
        help="Record a receipt after complete review",
    )
    approve = add("approve", "Approve a freshly reviewed draft and publish")
    approve.add_argument("draft_id")
    decline = add("decline", "Decline an owner-private draft")
    decline.add_argument("draft_id")
    add("list", "List published Collective Wisdom skills")
    show = add("show", "Show one published skill")
    show.add_argument("skill_id")
    versions = add("versions", "List published versions")
    versions.add_argument("skill_id")
    install = add("install", "Plan or apply an authenticated managed install")
    install.add_argument("reference", nargs="?", default="")
    install.add_argument(
        "--plan", action="store_true", help="Create a plan receipt without applying"
    )
    install.add_argument(
        "--apply-receipt", help="Apply a previously reviewed plan receipt"
    )
    install.add_argument(
        "--accept-partial",
        action="store_true",
        help="Accept partial/setup-required compatibility",
    )
    install.add_argument(
        "--update-mode", choices=["MANUAL", "AUTO_WITH_NOTICE", "REQUIRED"]
    )
    add("check", "Reconcile the feed and check managed installations")
    update = add("update", "Plan or apply verified managed updates")
    update.add_argument("skill_id", nargs="?")
    update.add_argument(
        "--all", action="store_true", help="Process every managed install"
    )
    update.add_argument(
        "--plan", action="store_true", help="Create a plan without applying"
    )
    update.add_argument(
        "--apply-receipt", help="Apply a previously reviewed update plan"
    )
    update.add_argument(
        "--accept-sensitive",
        action="store_true",
        help="Explicitly accept newly declared sensitive requirements",
    )
    update.add_argument(
        "--accept-partial",
        action="store_true",
        help="Accept partial/setup-required compatibility",
    )
    update.add_argument(
        "--preserve-modified",
        action="store_true",
        help="Preserve locally modified managed bytes as an unmanaged fork",
    )
    uninstall = add("uninstall", "Move a managed skill to recoverable trash")
    uninstall.add_argument("skill_id")
    uninstall.add_argument("--yes", action="store_true", help="Confirm uninstall")
    notifications = add("notifications", "Show durable local Wisdom notices")
    notifications.add_argument("--mark-seen", action="store_true")
