"""Read immutable CI evidence without running checks or downstream mutations."""
import json
import re


def inspect_ci(ctx, args):
    from .cli import _load_policy_from_context, _ci_receipt_payload
    from .ledger import FeedbackLedger, LedgerStateError

    try:
        policy = _load_policy_from_context(ctx)
        if not policy.enabled or args.repository not in policy.targets or not re.fullmatch(r"[0-9a-fA-F]{64}", args.receipt_id):
            raise ValueError("receipt target is not configured")
        ledger = FeedbackLedger.for_current_profile()
        try:
            receipt = ledger.ci_receipt_by_id(args.repository, args.pr_number, args.receipt_id)
        finally:
            ledger.close()
        if receipt is None:
            raise ValueError("receipt not found for this PR")
        payload = _ci_receipt_payload(receipt)
        payload["handoff_status"] = "not_evaluated"
        payload["failed_output_digests"] = [{"stdout": c.stdout_sha256, "stderr": c.stderr_sha256}
            for c in receipt.commands if c.returncode != 0 or c.timed_out or c.classification != "passed"]
        print(json.dumps(payload, sort_keys=True))
        return 0
    except (ValueError, LedgerStateError):
        print(json.dumps({"status": "ci_receipt_unavailable"}))
        return 1
