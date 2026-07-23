"""Argument parser for the Communication Core CLI."""

from __future__ import annotations

from collections.abc import Callable


def _leaf(subparsers, name: str, help_text: str, **kwargs):
    return subparsers.add_parser(name, help=help_text, **kwargs)


def build_communication_parser(subparsers, *, cmd_communication: Callable) -> None:
    parser = subparsers.add_parser(
        "communication",
        help="Manage account-scoped contacts, routes, drafts, and sync",
    )
    domains = parser.add_subparsers(dest="communication_domain", required=True)

    init = _leaf(domains, "init", "Initialize or migrate Communication Core storage")
    init.set_defaults(func=cmd_communication)

    accounts = _leaf(domains, "accounts", "Manage connected accounts")
    account_actions = accounts.add_subparsers(dest="communication_action", required=True)
    for name in ("list", "show", "disable", "status", "capabilities"):
        leaf = _leaf(account_actions, name, f"{name.title()} connected accounts")
        if name != "list":
            leaf.add_argument("account_id")
        leaf.add_argument("--include-disabled", action="store_true") if name == "list" else None
        leaf.set_defaults(func=cmd_communication)
    add = _leaf(account_actions, "add", "Register one exact connected account")
    add.add_argument("--provider", required=True)
    add.add_argument("--namespace", required=True)
    add.add_argument("--label", required=True)
    add.add_argument("--owner-profile", required=True)
    add.add_argument("--credential-ref")
    add.add_argument("--browser-profile-ref")
    add.add_argument(
        "--write-policy",
        choices=("disabled", "draft_only", "approval_required"),
        default="disabled",
    )
    add.set_defaults(func=cmd_communication)

    sync = _leaf(domains, "sync", "Run and inspect account-scoped sync")
    sync_actions = sync.add_subparsers(dest="communication_action", required=True)
    run = _leaf(sync_actions, "run", "Run full or incremental sync")
    run.add_argument("account_id")
    run.add_argument("--mode", choices=("full", "incremental"), default="incremental")
    run.set_defaults(func=cmd_communication)
    for name in ("status", "retry"):
        leaf = _leaf(sync_actions, name, f"{name.title()} a sync")
        leaf.add_argument("account_id")
        leaf.set_defaults(func=cmd_communication)

    people = _leaf(domains, "people", "Search and reconcile canonical people")
    people_actions = people.add_subparsers(dest="communication_action", required=True)
    search = _leaf(people_actions, "search", "Search contacts and identities")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.set_defaults(func=cmd_communication)
    show = _leaf(people_actions, "show", "Show one canonical person")
    show.add_argument("person_id")
    show.set_defaults(func=cmd_communication)
    duplicates = _leaf(people_actions, "duplicates", "Show explainable duplicate candidates")
    duplicates.set_defaults(func=cmd_communication)
    merge = _leaf(people_actions, "merge", "Manually merge a duplicate person")
    merge.add_argument("winner_person_id")
    merge.add_argument("merged_person_id")
    merge.add_argument("--evidence", required=True)
    merge.set_defaults(func=cmd_communication)
    unmerge = _leaf(people_actions, "unmerge", "Reverse a prior manual merge")
    unmerge.add_argument("merge_audit_id")
    unmerge.set_defaults(func=cmd_communication)

    routes = _leaf(domains, "routes", "Manage directed account and person routes")
    route_actions = routes.add_subparsers(dest="communication_action", required=True)
    route_list = _leaf(route_actions, "list", "List person routes")
    route_list.add_argument("--person-id")
    route_list.set_defaults(func=cmd_communication)
    for name, allowed in (("allow", True), ("deny", False)):
        leaf = _leaf(route_actions, name, f"{name.title()} one directed account link")
        leaf.add_argument("source_account_id")
        leaf.add_argument("target_account_id")
        leaf.add_argument("--reason", required=True)
        leaf.set_defaults(func=cmd_communication, route_allowed=allowed)
    for name in ("set", "dry-run"):
        leaf = _leaf(route_actions, name, f"{name.title()} an exact person route")
        leaf.add_argument("person_id")
        leaf.add_argument("source_endpoint_id")
        leaf.add_argument("target_endpoint_id")
        leaf.set_defaults(func=cmd_communication)
    audit = _leaf(route_actions, "audit", "Show route decisions")
    audit.add_argument("--person-id")
    audit.set_defaults(func=cmd_communication)

    groups = _leaf(domains, "groups", "Manage explicit groups and smart segments")
    group_actions = groups.add_subparsers(dest="communication_action", required=True)
    for name in ("list", "show", "preview"):
        leaf = _leaf(group_actions, name, f"{name.title()} groups")
        if name != "list":
            leaf.add_argument("group_id")
        leaf.set_defaults(func=cmd_communication)
    create = _leaf(group_actions, "create", "Create an explicit group")
    create.add_argument("name")
    create.add_argument("--exclude", action="store_true")
    create.set_defaults(func=cmd_communication)

    timeline = _leaf(domains, "timeline", "Show a provenance-preserving timeline")
    timeline_actions = timeline.add_subparsers(dest="communication_action", required=True)
    show = _leaf(timeline_actions, "show", "Show one person's timeline")
    show.add_argument("person_id")
    show.add_argument("--endpoint-id")
    show.add_argument("--start-at")
    show.add_argument("--end-at")
    show.set_defaults(func=cmd_communication)

    brief = _leaf(domains, "brief", "Build relationship briefs")
    brief_actions = brief.add_subparsers(dest="communication_action", required=True)
    daily = _leaf(brief_actions, "daily", "Build today's relationship brief")
    daily.add_argument("--date")
    daily.set_defaults(func=cmd_communication)

    analyze = _leaf(domains, "analyze", "Analyze communication evidence")
    analyze_actions = analyze.add_subparsers(dest="communication_action", required=True)
    conversation = _leaf(analyze_actions, "conversation", "Analyze one conversation")
    conversation.add_argument("conversation_id")
    conversation.set_defaults(func=cmd_communication)

    drafts = _leaf(domains, "drafts", "Create and review safe drafts")
    draft_actions = drafts.add_subparsers(dest="communication_action", required=True)
    create = _leaf(draft_actions, "create", "Create a draft for an exact route")
    create.add_argument("person_id")
    create.add_argument("source_endpoint_id")
    create.add_argument("--text", required=True)
    create.set_defaults(func=cmd_communication)
    listing = _leaf(draft_actions, "list", "List drafts")
    listing.add_argument("--status")
    listing.set_defaults(func=cmd_communication)
    for name in ("show", "cancel"):
        leaf = _leaf(draft_actions, name, f"{name.title()} a draft")
        leaf.add_argument("draft_id")
        leaf.set_defaults(func=cmd_communication)

    approvals = _leaf(domains, "approvals", "Approve or reject an exact draft")
    approval_actions = approvals.add_subparsers(dest="communication_action", required=True)
    approve = _leaf(approval_actions, "approve", "Approve one exact draft")
    approve.add_argument("draft_id")
    approve.add_argument("--ttl-minutes", type=int, default=30)
    approve.set_defaults(func=cmd_communication)
    reject = _leaf(approval_actions, "reject", "Reject one active approval")
    reject.add_argument("approval_id")
    reject.set_defaults(func=cmd_communication)

    migration = _leaf(domains, "migration", "Migrate or roll back legacy communication data")
    migration_actions = migration.add_subparsers(
        dest="communication_action", required=True
    )
    facebook_import = _leaf(
        migration_actions,
        "facebook-import",
        "Import one legacy Facebook CRM database read-only",
    )
    facebook_import.add_argument("account_id")
    facebook_import.add_argument("source_db")
    facebook_import.set_defaults(func=cmd_communication)
    facebook_rollback = _leaf(
        migration_actions,
        "facebook-rollback",
        "Roll back one completed Facebook migration run",
    )
    facebook_rollback.add_argument("run_id")
    facebook_rollback.set_defaults(func=cmd_communication)

    greetings = _leaf(domains, "greetings", "Plan deduplicated greetings")
    greeting_actions = greetings.add_subparsers(dest="communication_action", required=True)
    for name in ("plan", "list"):
        leaf = _leaf(greeting_actions, name, f"{name.title()} greetings")
        leaf.add_argument("--date")
        leaf.set_defaults(func=cmd_communication)
