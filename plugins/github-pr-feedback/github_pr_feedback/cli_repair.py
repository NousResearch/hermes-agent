"""Scoped conflict intake using the existing governed repair controller."""
from __future__ import annotations

import json


def dispatch_repair(ctx, args) -> int:
    from .cli import (
        _FULL_SHA, _exclusive_scan_lock, _github_client, _load_policy_from_context,
        _scan_payload, get_default_hermes_root, KanbanSubprocessClient,
    )
    from .ledger import FeedbackLedger
    from .repair_controller import RepairController

    try:
        policy = _load_policy_from_context(ctx)
        if not _FULL_SHA.fullmatch(args.head_sha) or args.pr_number <= 0:
            raise ValueError("invalid exact repair identity")
        if policy.repair_steward is None or args.repository not in policy.repair_steward.repositories:
            raise ValueError("repository is not configured for repair")
    except ValueError:
        print(json.dumps({"status": "invalid_configuration"}, sort_keys=True))
        return 1
    with _exclusive_scan_lock() as acquired:
        if not acquired:
            print(json.dumps({"status": "scan_already_running"}, sort_keys=True))
            return 1
        ledger = FeedbackLedger.for_current_profile()
        try:
            result = RepairController(
                policy, ledger, _github_client(policy), KanbanSubprocessClient(),
                control_home=get_default_hermes_root(),
            ).scan(conflicts_only=True, scoped_target=(args.repository, args.pr_number, args.head_sha.lower()))
        finally:
            ledger.close()
    print(json.dumps(_scan_payload(result), sort_keys=True))
    return 1 if result.degraded else 0
