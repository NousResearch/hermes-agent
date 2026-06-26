from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_audit import collect_gmail_inbox_audit, write_json_artifact
from hermes_cli.signal_coo.email_hygiene import stage_hygiene_actions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = collect_gmail_inbox_audit(
        config_path=home / "config" / "google_accounts.yaml",
        relationship_context_path=home / "config" / "relationship_context.yaml",
        days=int(os.getenv("TORBEN_EMAIL_HYGIENE_LOOKBACK_DAYS", "60")),
        max_messages_per_account=int(os.getenv("TORBEN_EMAIL_HYGIENE_MAX_MESSAGES", "2500")),
        max_body_fetches_per_account=int(os.getenv("TORBEN_EMAIL_HYGIENE_MAX_BODY_FETCHES", "600")),
        fetch_workers=int(os.getenv("TORBEN_EMAIL_HYGIENE_WORKERS", "8")),
    )
    write_json_artifact(payload, state_dir / "torben-email-hygiene-review-latest.json")
    audit = payload.get("email_audit") or {}
    ledger = ActionLedger(state_dir / "torben-action-ledger.json")
    llm_review: dict = {}
    staged = stage_hygiene_actions(
        ledger=ledger,
        records=list(audit.get("messages") or []),
        enable_llm_review=_truthy(os.getenv("TORBEN_EMAIL_HYGIENE_LLM_REVIEW"), default=True),
        review_metadata_out=llm_review,
    )
    output = {
        "task": "torben_email_hygiene_weekly_review",
        "wakeAgent": bool(staged),
        "generated_at": _utc_now(),
        "mutation_boundary": "weekly review stages recommendations only; do not trash, archive, label, delete, unsubscribe, or send until Eric approves a handle",
        "approval_contract": [
            "Every recommendation must include a handle.",
            "Eric can approve a handle; only then may the Gmail hygiene apply script run.",
            "Trash means Gmail Trash, not permanent delete.",
            "Nudge recommendations do not mutate Gmail; they rebump the thread in the brief/action ledger.",
            "If uncertain, recommend review instead of deletion.",
        ],
        "recommendations": staged,
        "diagnostics": {
            "messages_scanned": audit.get("message_count", 0),
            "recommendation_count": len(staged),
            "gmail_reads": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_read_api_calls", 0
            ),
            "gmail_writes": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_write_api_calls", 0
            ),
            "external_mutations": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "external_mutations", 0
            ),
            "warnings": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "warnings", []
            ),
            "llm_review": llm_review,
        },
    }
    write_json_artifact(output, state_dir / "torben-email-hygiene-review-actions-latest.json")
    if not staged:
        print(json.dumps({"wakeAgent": False, "reason": "no weekly email hygiene recommendations"}))
        return 0
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
