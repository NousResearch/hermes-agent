from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from hermes_constants import get_hermes_home
from hermes_cli.signal_coo.action_ledger import ActionLedger
from hermes_cli.signal_coo.email_hygiene import HYGIENE_POLICY_VERSION, apply_hygiene_action


SAFE_AUTO_OPERATIONS = {"archive", "archive_and_label", "label"}


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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def summarize_auto_apply_for_signal(output: dict) -> str:
    errors = list(output.get("errors") or [])
    results = [result for result in (output.get("results") or []) if isinstance(result, dict)]
    if not results and not errors:
        return json.dumps({"wakeAgent": False, "reason": "no hygiene auto-apply candidates"})

    applied_count = sum(len(result.get("applied") or []) for result in results)
    skipped_count = sum(len(result.get("skipped") or []) for result in results)
    handles = [
        str(handle)
        for handle in (output.get("selected_handles") or [])
        if str(handle or "").strip()
    ]
    operations = sorted(
        {
            str(result.get("operation") or "unknown")
            for result in results
            if result.get("operation")
        }
    )
    apply_label = "Would apply" if bool(output.get("dry_run")) else "Applied"
    lines = [
        "Torben Gmail hygiene auto-apply completed.",
        f"Handles: {', '.join(handles) if handles else 'none'}",
        f"{apply_label}: {applied_count} message(s)"
        + (f" via {', '.join(operations)}" if operations else ""),
        f"Skipped: {skipped_count}",
        f"Gmail writes: {int(output.get('gmail_write_api_calls') or 0)}",
        f"External mutations: {int(output.get('external_mutations') or 0)}",
        f"Dry run: {bool(output.get('dry_run'))}",
    ]
    if errors:
        lines.append(f"Errors: {len(errors)}")
    lines.append("Full audit: state/torben-email-hygiene-auto-apply-latest.json")
    return "\n".join(lines)


def _is_auto_candidate(record, *, allow_trash: bool) -> tuple[bool, str]:
    state = record.executor_state or {}
    if state.get("mutation_type") != "gmail_hygiene":
        return False, "not_gmail_hygiene"
    if state.get("hygiene_policy_version") != HYGIENE_POLICY_VERSION:
        return False, "stale_hygiene_policy_version"
    if record.status != "approved":
        return False, f"status_{record.status}"
    if "approve_hygiene_apply" not in record.allowed_next_actions:
        return False, "missing_approve_hygiene_apply"
    if str(record.risk_class or "medium") != "low":
        return False, f"risk_{record.risk_class or 'medium'}"
    rung = str(state.get("automation_rung") or state.get("category_rung") or "").strip()
    if rung and rung != "auto_within_caps":
        return False, f"rung_{rung}"
    operation = str(state.get("operation") or "")
    if operation == "trash" and allow_trash:
        return True, "ok"
    if operation not in SAFE_AUTO_OPERATIONS:
        return False, f"operation_{operation or 'missing'}"
    return True, "ok"


def _operation_priority(record) -> int:
    operation = str((record.executor_state or {}).get("operation") or "")
    return {
        "archive_and_label": 0,
        "label": 1,
        "archive": 2,
        "trash": 3,
    }.get(operation, 9)


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservative Torben Gmail hygiene auto-apply loop.")
    parser.add_argument("--max-handles", type=int, default=None, help="Max low-risk hygiene handles to apply this run.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without mutating Gmail.")
    parser.add_argument(
        "--no-trash",
        action="store_true",
        help="Disable auto-trash even for low-risk v3 disposable handles.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON output instead of Signal text.")
    args = parser.parse_args()

    home = get_hermes_home()
    state_dir = home / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    ledger = ActionLedger(state_dir / "torben-action-ledger.jsonl")
    allow_trash = (
        not args.no_trash
        and not _truthy(os.getenv("TORBEN_EMAIL_HYGIENE_AUTO_DISABLE_TRASH"))
        and _truthy(os.getenv("TORBEN_EMAIL_HYGIENE_AUTO_ALLOW_TRASH"), default=False)
    )
    dry_run = args.dry_run or _truthy(os.getenv("TORBEN_EMAIL_HYGIENE_AUTO_APPLY_DRY_RUN"))
    max_handles = args.max_handles or _positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_AUTO_MAX_HANDLES"), 1)

    candidates = []
    skipped = []
    for record in ledger.load():
        allowed, reason = _is_auto_candidate(record, allow_trash=allow_trash)
        if not allowed:
            if (record.executor_state or {}).get("mutation_type") == "gmail_hygiene":
                skipped.append({"handle": record.handle, "reason": reason})
            continue
        candidates.append(record)
    selected = sorted(
        candidates,
        key=lambda record: (
            -_operation_priority(record),
            record.created_at,
            record.handle,
        ),
        reverse=True,
    )[:max_handles]

    results = []
    for record in selected:
        results.append(
            apply_hygiene_action(
                ledger=ledger,
                config_path=home / "config" / "google_accounts.yaml",
                handle=record.handle,
                approved_by="torben-email-hygiene-auto-apply",
                dry_run=dry_run,
                auto_apply=True,
                profile_home=home,
            )
        )

    output = {
        "task": "torben_email_hygiene_auto_apply",
        "generated_at": _utc_now(),
        "dry_run": dry_run,
        "allow_trash": allow_trash,
        "max_handles": max_handles,
        "selected_handles": [record.handle for record in selected],
        "results": results,
        "skipped_hygiene_handles": skipped[:25],
        "external_mutations": sum(int(result.get("external_mutations") or 0) for result in results),
        "gmail_write_api_calls": sum(int(result.get("gmail_write_api_calls") or 0) for result in results),
        "errors": [error for result in results for error in (result.get("errors") or [])],
        "wakeAgent": bool(results or any(result.get("errors") for result in results)),
        "policy_boundary": (
            "Auto-apply still requires ea.mutations.email_archive_delete_label.enabled=true "
            "and a positive max_per_run. Only low-risk trash is automatic; medium-risk "
            "or action-shaped mail remains approval-gated."
        ),
    }
    output_path = state_dir / "torben-email-hygiene-auto-apply-latest.json"
    tmp = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, output_path)
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(summarize_auto_apply_for_signal(output))
    return 1 if output["errors"] else 0


if __name__ == "__main__":
    if "--json" in sys.argv:
        raise SystemExit(main())
    from torben_job_contract import run_job

    raise SystemExit(run_job("torben-email-hygiene-auto-apply", main))
