"""`hermes wisdom` command surface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
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
    from hermes_wisdom.entitlement import require_entitlement
    from hermes_wisdom.package import PackagePolicyError
    from hermes_wisdom.service import WisdomService

    command = getattr(args, "wisdom_command", None)
    try:
        require_entitlement()
        service = WisdomService()
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
        elif command == "installed-setup":
            from hermes_wisdom.installed_setup import inspect_installed_setup

            result = inspect_installed_setup(service.store, args.skill_id, version=args.version)
        elif command == "sync":
            result = service.retry_sync() if args.action == "retry" else service.sync_status()
        elif command == "inbox":
            from hermes_wisdom.mediation import WisdomMediation

            result = WisdomMediation(service).activity()
        elif command == "consent":
            from hermes_wisdom.consent import ConsentActor, WisdomConsent

            org = service.store.active_org_id()
            with service.store.transaction() as db:
                row = db.execute(
                    "SELECT owner_session,platform FROM wisdom_consent WHERE organization_id=? AND id=?",
                    (org, args.interaction_id),
                ).fetchone()
            if row is None or row["platform"] != "local":
                raise PackagePolicyError("Use this interaction's original native control")
            actor = ConsentActor(row["owner_session"], "local", "local-user", f"local:{row['owner_session']}")
            consent = WisdomConsent(service)
            preview = consent.resolve(org, args.interaction_id, actor, "inspect")
            if args.action == "confirm":
                if not sys.stdin.isatty():
                    raise PackagePolicyError("Wisdom consent requires an interactive native confirmation")
                _emit(preview, as_json=False)
                if input("Apply this exact Wisdom operation? [y/N] ").strip().lower() not in {"y", "yes"}:
                    return 7
            result = consent.resolve(org, args.interaction_id, actor, args.action)
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
        elif command == "browse":
            result = {"skills": service.search_skills(getattr(args, "query", None))}
        elif command == "review-week":
            from hermes_wisdom.weekly_queue import enqueue_weekly_review

            result = enqueue_weekly_review(service, dry_run=bool(getattr(args, "dry_run", False)))
        elif command == "act":
            from hermes_wisdom.agent_led.actions import handle_action

            result = handle_action(
                args.target, service=service, mute_choice=getattr(args, "mute", None)
            )
        elif command == "share":
            from hermes_wisdom.agent_led.share_flow import ShareFlow

            flow = ShareFlow(getattr(args, "flow_id", None))
            if args.share_step == "start":
                from tools.skill_usage import _find_skill_dir

                path = _find_skill_dir(args.skill)
                if path is None:
                    raise PackagePolicyError(f"skill not found: {args.skill}")
                result = flow.start(path)
            elif args.share_step == "package":
                result = flow.package()
            elif args.share_step == "request-changes":
                result = flow.request_changes(args.note or "")
            elif args.share_step == "cancel":
                result = flow.cancel()
            elif args.share_step == "approve":
                from hermes_wisdom.agent_led.share_flow import submit_via_service
                from hermes_constants import get_hermes_home

                staging = Path(get_hermes_home()) / "wisdom" / "share_staging"
                result = flow.approve(submit=submit_via_service(service, staging))
            else:
                result = flow.summary()
        elif command == "dismiss":
            from hermes_wisdom.agent_led.history import SuggestionHistory
            from hermes_wisdom.agent_led.policy import load_policy

            policy = load_policy(client=service.client)
            result = SuggestionHistory().record_dismissal(
                args.skill,
                args.content_hash,
                suppression_days=args.days or policy.dismiss_suppression_days,
                client=service.client,
            )
        elif command == "mute":
            from hermes_wisdom.preferences import WisdomPreferences

            result = WisdomPreferences(service).native_mute_command(args.duration)
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
    from hermes_wisdom.entitlement import is_entitled

    if not is_entitled():
        return
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
    installed_setup = add("installed-setup", "Inspect installed setup guidance without executing it")
    installed_setup.add_argument("skill_id")
    installed_setup.add_argument("--version", type=int, help="Require this exact installed version")
    sync = add("sync", "Check or retry saved notification receipts and operation reports")
    sync.add_argument("action", choices=("status", "retry"), nargs="?", default="status")
    add("inbox", "Read agent advice and pending native consent")
    consent = add("consent", "Inspect or confirm an exact local interaction")
    consent.add_argument("interaction_id")
    consent.add_argument("action", default="inspect", nargs="?", metavar="inspect[.PAGE]|defer|confirm")
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
    browse = add("browse", "Search the organization catalog by keyword")
    browse.add_argument("query", nargs="?")
    review_week = add(
        "review-week", "Queue this week's review for the active private agent session"
    )
    review_week.add_argument(
        "--force", action="store_true", help="Compatibility flag; does not bypass the weekly cap or consent"
    )
    review_week.add_argument(
        "--dry-run", action="store_true", help="Preview local evidence without queueing or calling a model"
    )
    act = add("act", "Open current review for an old recommendation control (does not apply)")
    act.add_argument("target")
    act.add_argument("--mute", choices=["1d", "1w", "30d", "forever"], help="Mute duration")
    share = add("share", "Agent-guided share flow (package, review, approve)")
    share.add_argument(
        "share_step",
        choices=["start", "package", "request-changes", "approve", "cancel", "status"],
    )
    share.add_argument("skill", nargs="?", help="Skill name (start step)")
    share.add_argument("--flow-id", dest="flow_id", help="Resume an existing flow")
    share.add_argument("--note", help="Requested changes (request-changes step)")
    dismiss = add("dismiss", "Record 'Not now' for a skill at its current content hash")
    dismiss.add_argument("skill")
    dismiss.add_argument("content_hash")
    dismiss.add_argument("--days", type=int, help="Suppression window (default from policy)")
    mute = add("mute", "Manage your organization-wide proactive Wisdom notifications")
    mute.add_argument("duration", nargs="?", default="status", choices=["status", "1d", "1w", "30d", "forever", "off"])
