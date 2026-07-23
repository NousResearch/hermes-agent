"""Runtime handler for ``hermes communication``."""

from __future__ import annotations

import json
import sys
from typing import Any

from communication_core.errors import CommunicationError
from communication_core.migrations import FacebookMigrationBridge
from communication_core.repository import CommunicationRepository
from communication_core.service import CommunicationService


_SECRET_FIELDS = {"credential_ref", "browser_profile_ref", "rate_limit_state"}


def _public(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _public(item) for key, item in value.items() if key not in _SECRET_FIELDS}
    if isinstance(value, list):
        return [_public(item) for item in value]
    return value


def _emit(value: Any) -> None:
    print(json.dumps(_public(value), ensure_ascii=False, sort_keys=True, indent=2))


def _person_show(repository: CommunicationRepository, person_id: str) -> dict[str, Any]:
    return repository.person_detail(person_id)


def communication_command(args) -> int:
    repository = CommunicationRepository()
    service = CommunicationService(repository)
    domain = args.communication_domain
    action = getattr(args, "communication_action", None)
    try:
        if domain == "init":
            result = service.initialize()
        elif domain == "accounts":
            if action == "list":
                result = repository.list_accounts(include_disabled=args.include_disabled)
            elif action == "add":
                result = repository.add_account(
                    provider=args.provider,
                    account_namespace=args.namespace,
                    label=args.label,
                    owner_profile=args.owner_profile,
                    credential_ref=args.credential_ref,
                    browser_profile_ref=args.browser_profile_ref,
                    write_policy=args.write_policy,
                )
            elif action == "disable":
                repository.disable_account(args.account_id)
                result = {"account_id": args.account_id, "enabled": False, "fallback": False}
            elif action == "status":
                result = {"account": repository.get_account(args.account_id), "sync": repository.sync_status(args.account_id)}
            elif action == "capabilities":
                result = {"account_id": args.account_id, "capabilities": service.orchestrator.capabilities(args.account_id)}
            else:
                result = repository.get_account(args.account_id)
        elif domain == "sync":
            if action == "status":
                result = repository.sync_status(args.account_id)
            else:
                result = service.sync(
                    args.account_id,
                    mode="retry" if action == "retry" else args.mode,
                )
        elif domain == "people":
            if action == "search":
                result = repository.search_all(args.query, args.limit)
            elif action == "show":
                result = _person_show(repository, args.person_id)
            elif action == "duplicates":
                result = repository.find_duplicate_candidates()
            elif action == "merge":
                result = repository.merge_people(
                    args.winner_person_id,
                    args.merged_person_id,
                    actor="user",
                    evidence={"manual": args.evidence},
                )
            else:
                result = repository.unmerge_people(args.merge_audit_id, actor="user")
        elif domain == "routes":
            if action == "list":
                result = repository.list_routes(args.person_id)
            elif action in {"allow", "deny"}:
                repository.allow_account_link(
                    args.source_account_id,
                    args.target_account_id,
                    allowed=args.route_allowed,
                    actor="user",
                    reason=args.reason,
                )
                result = {"source_account_id": args.source_account_id, "target_account_id": args.target_account_id, "allowed": args.route_allowed}
            elif action in {"set", "dry-run"}:
                request = dict(
                    person_id=args.person_id,
                    source_endpoint_id=args.source_endpoint_id,
                    target_endpoint_id=args.target_endpoint_id,
                    actor="user",
                )
                result = service.apply_route(**request) if action == "set" else service.route_dry_run(**request)
            else:
                result = repository.list_route_audit(args.person_id)
        elif domain == "groups":
            if action == "list":
                result = repository.list_groups()
            elif action == "create":
                result = repository.create_group(args.name, exclusion=args.exclude)
            elif action == "show":
                result = repository.get_group(args.group_id)
            else:
                result = repository.group_preview(args.group_id)
        elif domain == "timeline":
            result = repository.timeline(
                args.person_id,
                endpoint_id=args.endpoint_id,
                start_at=args.start_at,
                end_at=args.end_at,
            )
        elif domain == "brief":
            result = repository.daily_brief(args.date)
        elif domain == "analyze":
            result = repository.analyze_conversation(args.conversation_id)
        elif domain == "drafts":
            if action == "create":
                result = service.create_draft(person_id=args.person_id, source_endpoint_id=args.source_endpoint_id, payload=args.text)
            elif action == "list":
                result = repository.list_drafts(status=args.status)
            elif action == "show":
                result = repository.get_draft(args.draft_id)
            else:
                repository.cancel_draft(args.draft_id)
                result = {"draft_id": args.draft_id, "status": "cancelled"}
        elif domain == "approvals":
            if action == "approve":
                result = service.approve_draft(args.draft_id, ttl_minutes=args.ttl_minutes)
            else:
                repository.reject_approval(args.approval_id)
                result = {"approval_id": args.approval_id, "status": "rejected"}
        elif domain == "greetings":
            result = repository.plan_greetings(args.date) if action == "plan" else repository.list_greetings(args.date)
        elif domain == "migration":
            bridge = FacebookMigrationBridge(repository)
            if action == "facebook-import":
                result = bridge.migrate(args.source_db, args.account_id)
            else:
                result = bridge.rollback(args.run_id)
        else:
            raise ValueError(f"unknown communication domain: {domain}")
        if result is None:
            raise KeyError("record not found")
        _emit(result)
        return 0
    except (CommunicationError, KeyError, ValueError) as error:
        print(f"communication: {error}", file=sys.stderr)
        return 2
