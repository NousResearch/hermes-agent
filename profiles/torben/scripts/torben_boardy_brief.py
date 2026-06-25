from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.email_audit import collect_gmail_inbox_audit, write_json_artifact


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_deferred_intro_targets(state_dir):
    path = state_dir / "torben-gtm-deferred-intros.json"
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict) and str(item.get("status") or "").lower() == "deferred"]


def main() -> int:
    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    payload = collect_gmail_inbox_audit(
        config_path=home / "config" / "google_accounts.yaml",
        relationship_context_path=home / "config" / "relationship_context.yaml",
        days=int(os.getenv("TORBEN_BOARDY_BRIEF_LOOKBACK_DAYS", "1")),
        max_messages_per_account=int(os.getenv("TORBEN_BOARDY_BRIEF_MAX_MESSAGES", "250")),
        max_body_fetches_per_account=int(os.getenv("TORBEN_BOARDY_BRIEF_MAX_BODY_FETCHES", "80")),
        fetch_workers=int(os.getenv("TORBEN_BOARDY_BRIEF_WORKERS", "6")),
    )
    write_json_artifact(payload, state_dir / "torben-boardy-brief-latest.json")
    audit = payload.get("email_audit") or {}
    morning_candidates = audit.get("morning_briefing_candidates") or {}
    boardy_digest = list(morning_candidates.get("boardy_digest") or [])
    deferred_intro_targets = _load_deferred_intro_targets(state_dir)
    if not boardy_digest:
        print(json.dumps({"wakeAgent": False, "reason": "no Boardy digest items"}))
        return 0
    output = {
        "task": "torben_boardy_brief",
        "wakeAgent": True,
        "generated_at": _utc_now(),
        "mutation_boundary": "read/summarize only; do not send email, archive, label, delete, unsubscribe, or mutate calendars",
        "policy": {
            "cadence": "twice daily",
            "realtime_exception": "Direct Boardy Intro threads may notify outside this brief only after screening for relationship fit, day-job sensitivity/conflict, and a concrete next move; never treat them as automatic yeses.",
            "brief_goal": "Summarize Boardy market/intro noise into useful signal without creating realtime interruptions or pushing low-fit intros.",
            "deferred_intro_rule": "If a Boardy suggestion matches a deferred_intro_target, do not ask Eric to greenlight it now. Treat it as parked until its defer_until condition is satisfied.",
        },
        "deferred_intro_targets": deferred_intro_targets,
        "boardy_digest": boardy_digest[:20],
        "llm_decision_contract": morning_candidates.get("llm_decision_contract") or {},
        "diagnostics": {
            "messages_scanned": audit.get("message_count", 0),
            "boardy_digest_count": len(boardy_digest),
            "gmail_reads": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_read_api_calls", 0
            ),
            "gmail_writes": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "gmail_write_api_calls", 0
            ),
            "external_mutations": ((payload.get("source_diagnostics") or {}).get("gmail") or {}).get("audit", {}).get(
                "external_mutations", 0
            ),
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
