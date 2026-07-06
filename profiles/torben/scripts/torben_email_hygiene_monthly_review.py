from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_audit import collect_gmail_inbox_audit, write_json_artifact
from hermes_cli.signal_coo.email_hygiene import build_inbox_filter_recommendations, stage_hygiene_actions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    query = os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_QUERY", "in:inbox")
    max_messages = _positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_MAX_MESSAGES"), 10000)
    max_body_fetches = _positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_MAX_BODY_FETCHES"), 1200)
    filter_min_messages = _positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_FILTER_MIN_MESSAGES"), 3)
    payload = collect_gmail_inbox_audit(
        config_path=home / "config" / "google_accounts.yaml",
        relationship_context_path=home / "config" / "relationship_context.yaml",
        days=_positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_LOOKBACK_DAYS"), 3650),
        gmail_query=query,
        max_messages_per_account=max_messages,
        max_body_fetches_per_account=max_body_fetches,
        fetch_workers=_positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_WORKERS"), 8),
    )
    write_json_artifact(payload, state_dir / "torben-email-hygiene-monthly-review-latest.json")
    audit = payload.get("email_audit") or {}
    records = list(audit.get("messages") or [])
    ledger = ActionLedger(state_dir / "torben-action-ledger.jsonl")
    llm_review: dict = {}
    cleanup_recommendations = stage_hygiene_actions(
        ledger=ledger,
        records=records,
        enable_llm_review=_truthy(os.getenv("TORBEN_EMAIL_HYGIENE_MONTHLY_LLM_REVIEW"), default=True),
        review_metadata_out=llm_review,
    )
    filter_recommendations = build_inbox_filter_recommendations(
        records,
        min_messages=filter_min_messages,
    )
    warnings = ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get("warnings", [])
    cleanup_item_count = sum(len(item.get("items") or []) for item in cleanup_recommendations)
    output = {
        "task": "torben_email_hygiene_monthly_review",
        "wakeAgent": bool(cleanup_recommendations or filter_recommendations or warnings),
        "generated_at": _utc_now(),
        "mutation_boundary": (
            "monthly review stages recommendations only; do not archive, trash, label, delete, "
            "unsubscribe, create filters, or send until Eric approves a handle or filter plan"
        ),
        "review_scope": {
            "gmail_query": audit.get("gmail_query") or query,
            "max_messages_per_account": max_messages,
            "max_body_fetches_per_account": max_body_fetches,
            "filter_min_messages": filter_min_messages,
        },
        "inbox_convergence": {
            "messages_scanned": audit.get("message_count", 0),
            "cleanup_group_count": len(cleanup_recommendations),
            "cleanup_item_count": cleanup_item_count,
            "filter_recommendation_count": len(filter_recommendations),
            "goal": "Converge each enabled Gmail inbox toward zero by staging bounded cleanup waves and durable filter recommendations.",
            "capped": any("max_messages_per_account" in str(warning) for warning in warnings),
        },
        "cleanup_recommendations": cleanup_recommendations,
        "filter_recommendations": filter_recommendations,
        "diagnostics": {
            "category_counts": audit.get("category_counts", {}),
            "juno_bucket_counts": audit.get("juno_bucket_counts", {}),
            "source_summary_count": len(audit.get("source_summaries") or []),
            "gmail_reads": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_read_api_calls", 0
            ),
            "gmail_writes": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_write_api_calls", 0
            ),
            "external_mutations": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "external_mutations", 0
            ),
            "warnings": warnings,
            "llm_review": llm_review,
        },
    }
    write_json_artifact(output, state_dir / "torben-email-hygiene-monthly-review-actions-latest.json")
    if not output["wakeAgent"]:
        print(json.dumps({"wakeAgent": False, "reason": "monthly inbox review found no cleanup or filter recommendations"}))
        return 0
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-email-hygiene-monthly-review", main))
