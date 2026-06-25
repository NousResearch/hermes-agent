"""Approval-gated Gmail hygiene recommendations for Torben."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .action_ledger import ActionLedger, ActionRecord, parse_time
from .google_auth import account_for_alias
from .google_evidence import GMAIL_API_ROOT, _read_token


HYGIENE_POLICY_VERSION = 1
HYGIENE_OPEN_STATUSES = {"staged", "approval_required", "approved", "executing"}
ACTION_PREFIX = "email_hygiene"

MFA_TERMS = (
    "one time passcode",
    "one-time passcode",
    "verification code",
    "security code",
    "login code",
    "authentication code",
    "password reset",
)
NOISE_TERMS = (
    "privacy policy",
    "terms of service",
    "terms update",
    "policy update",
    "rebranding",
    "reward",
    "rewards",
    "lendingclub",
    "lending club",
    "mens wearhouse",
    "men's wearhouse",
    "unsubscribe",
)
ACTIONABLE_CATEGORIES = {
    "calendar_scheduling",
    "deadline_or_action",
    "founder_funding_customer",
    "human_review_reply_candidate",
}
ARCHIVABLE_CATEGORIES = {
    "newsletter_general",
    "promotions_noise",
    "receipt_vendor_ops",
    "account_security",
}


@dataclass
class HygieneGroup:
    key: str
    title: str
    operation: str
    risk_class: str
    rationale: str
    items: list[dict[str, Any]] = field(default_factory=list)

    def to_action_summary(self) -> str:
        return f"{self.title} ({len(self.items)} item{'s' if len(self.items) != 1 else ''})"

    def to_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "title": self.title,
            "operation": self.operation,
            "risk_class": self.risk_class,
            "rationale": self.rationale,
            "items": self.items,
        }


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _internal_date(record: dict[str, Any]) -> datetime | None:
    raw = str(record.get("internal_date_ms") or "").strip()
    if raw.isdigit():
        return datetime.fromtimestamp(int(raw) / 1000, tz=timezone.utc)
    return parse_time(str(record.get("date") or ""))


def _age_hours(record: dict[str, Any], now: datetime) -> float:
    internal = _internal_date(record)
    if not internal:
        return 0.0
    return max(0.0, (now - internal).total_seconds() / 3600)


def _text(record: dict[str, Any]) -> str:
    return " ".join(
        str(record.get(key) or "")
        for key in ("sender", "sender_email", "sender_domain", "subject", "snippet", "body_excerpt")
    ).lower()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _compact_item(record: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "account_alias": record.get("account_alias"),
        "message_id": record.get("message_id"),
        "thread_id": record.get("thread_id"),
        "sender": record.get("sender"),
        "subject": record.get("subject"),
        "category": record.get("category"),
        "juno_bucket": record.get("juno_bucket"),
        "reason": reason,
        "age_hours": round(float(record.get("age_hours") or 0), 1),
        "evidence_ids": list(record.get("evidence_ids") or []),
    }


def _group_candidate_records(records: list[dict[str, Any]], *, now: datetime) -> list[HygieneGroup]:
    groups = {
        "trash_mfa_codes": HygieneGroup(
            key="trash_mfa_codes",
            title="Trash stale MFA/password-code emails",
            operation="trash",
            risk_class="low",
            rationale="Security-code emails are disposable after the login window has passed.",
        ),
        "archive_policy_rewards_noise": HygieneGroup(
            key="archive_policy_rewards_noise",
            title="Archive policy/rebrand/rewards noise",
            operation="archive",
            risk_class="low",
            rationale="Policy updates, rewards mail, and rebrand notices rarely need inbox presence after review.",
        ),
        "archive_stale_receipts": HygieneGroup(
            key="archive_stale_receipts",
            title="Archive stale receipts and vendor confirmations",
            operation="archive",
            risk_class="low",
            rationale="Old receipts and confirmations should be searchable, not inbox-visible.",
        ),
        "nudge_stale_replies": HygieneGroup(
            key="nudge_stale_replies",
            title="Re-bump stale threads that may still matter",
            operation="nudge_only",
            risk_class="medium",
            rationale="Older unreplied action-shaped threads should be reviewed instead of silently decaying.",
        ),
    }

    for record in records:
        category = str(record.get("category") or "")
        juno_bucket = str(record.get("juno_bucket") or "")
        text = _text(record)
        age_hours = _age_hours(record, now)
        enriched = {**record, "age_hours": age_hours}
        if category == "account_security" and age_hours >= 1 and _contains_any(text, MFA_TERMS):
            groups["trash_mfa_codes"].items.append(
                _compact_item(enriched, reason="account-security code older than one hour")
            )
            continue
        if age_hours >= 24 and _contains_any(text, NOISE_TERMS) and category not in ACTIONABLE_CATEGORIES:
            groups["archive_policy_rewards_noise"].items.append(
                _compact_item(enriched, reason="policy/rebrand/rewards-style inbox noise older than one day")
            )
            continue
        if category == "receipt_vendor_ops" and age_hours >= 14 * 24:
            groups["archive_stale_receipts"].items.append(
                _compact_item(enriched, reason="receipt/vendor confirmation older than two weeks")
            )
            continue
        if category in ACTIONABLE_CATEGORIES and juno_bucket in {"reply", "deadline", "flag"} and age_hours >= 7 * 24:
            groups["nudge_stale_replies"].items.append(
                _compact_item(enriched, reason="action-shaped thread older than one week")
            )

    for group in groups.values():
        group.items = group.items[:50]
    return [group for group in groups.values() if group.items]


def _existing_hygiene_actions(ledger: ActionLedger) -> dict[str, ActionRecord]:
    existing: dict[str, ActionRecord] = {}
    for record in ledger.load():
        state = record.executor_state or {}
        key = state.get("hygiene_action_key")
        if (
            record.scope == "ea"
            and isinstance(key, str)
            and state.get("hygiene_policy_version") == HYGIENE_POLICY_VERSION
            and record.status in HYGIENE_OPEN_STATUSES
        ):
            existing[key] = record
    return existing


def stage_hygiene_actions(
    *,
    ledger: ActionLedger,
    records: list[dict[str, Any]],
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = (now or _now()).astimezone(timezone.utc)
    existing = _existing_hygiene_actions(ledger)
    staged: list[dict[str, Any]] = []
    for group in _group_candidate_records(records, now=now):
        action_key = f"{ACTION_PREFIX}:{group.key}:{now:%Y-%m-%d}"
        record = existing.get(action_key)
        if record is None:
            record = ledger.add_action(
                scope="EA",
                summary=group.to_action_summary(),
                evidence_ids=[
                    evidence
                    for item in group.items[:10]
                    for evidence in (item.get("evidence_ids") or [])
                ],
                allowed_next_actions=["approve_hygiene_apply", "revise", "discard"],
                status="approval_required",
                risk_class=group.risk_class,
                ttl_hours=14 * 24,
                now=now,
                executor_state={
                    "mutation_type": "gmail_hygiene",
                    "provider": "gmail",
                    "mutation_status": "not_applied",
                    "hygiene_policy_version": HYGIENE_POLICY_VERSION,
                    "hygiene_action_key": action_key,
                    "operation": group.operation,
                    "rationale": group.rationale,
                    "items": group.items,
                    "approval_required": True,
                },
            )
            existing[action_key] = record
        staged.append(
            {
                "handle": record.handle,
                "status": record.status,
                **group.to_payload(),
            }
        )
    return staged


def _gmail_post(url: str, token: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _validate_item_for_operation(operation: str, item: dict[str, Any]) -> None:
    category = str(item.get("category") or "")
    reason = str(item.get("reason") or "").lower()
    if operation == "trash" and category != "account_security":
        raise ValueError(f"Refusing to trash non-account-security item: {item.get('message_id')}")
    if operation == "archive" and category not in ARCHIVABLE_CATEGORIES:
        raise ValueError(f"Refusing to archive actionable category {category}: {item.get('message_id')}")
    if operation == "trash" and "older than one hour" not in reason:
        raise ValueError(f"Refusing to trash without stale-code reason: {item.get('message_id')}")


def apply_hygiene_action(
    *,
    ledger: ActionLedger,
    config_path: str | Path,
    handle: str,
    approved_by: str = "signal",
    dry_run: bool = False,
) -> dict[str, Any]:
    record = ledger.get(handle)
    if record is None:
        raise ValueError(f"No action found for handle: {handle}")
    state = dict(record.executor_state or {})
    if state.get("mutation_type") != "gmail_hygiene":
        raise ValueError(f"Action {handle} is not a Gmail hygiene action.")
    if "approve_hygiene_apply" not in record.allowed_next_actions:
        raise ValueError(f"Action {handle} is not approved for Gmail hygiene apply.")
    if record.status not in {"approval_required", "approved", "executing"}:
        raise ValueError(f"Action {handle} is not open for hygiene apply: {record.status}")
    operation = str(state.get("operation") or "")
    if operation not in {"trash", "archive", "nudge_only"}:
        raise ValueError(f"Unsupported hygiene operation for {handle}: {operation}")
    items = list(state.get("items") or [])
    if len(items) > 50:
        raise ValueError(f"Refusing to apply more than 50 items in one hygiene action: {len(items)}")

    now = _now().isoformat().replace("+00:00", "Z")
    result = {
        "handle": handle,
        "operation": operation,
        "dry_run": dry_run,
        "applied": [],
        "skipped": [],
        "errors": [],
        "external_mutations": 0,
        "gmail_write_api_calls": 0,
    }
    tokens: dict[str, str] = {}
    for item in items:
        account_alias = str(item.get("account_alias") or "")
        message_id = str(item.get("message_id") or "")
        if not account_alias or not message_id:
            result["skipped"].append({"item": item, "reason": "missing account_alias or message_id"})
            continue
        try:
            _validate_item_for_operation(operation, item)
        except ValueError as exc:
            result["errors"].append({"message_id": message_id, "error": str(exc)})
            continue
        if operation == "nudge_only":
            result["applied"].append(
                {
                    "account_alias": account_alias,
                    "message_id": message_id,
                    "nudge_only": True,
                    "reason": "rebumped in action ledger without Gmail mutation",
                }
            )
            continue
        if dry_run:
            result["applied"].append({"account_alias": account_alias, "message_id": message_id, "dry_run": True})
            continue
        token = tokens.get(account_alias)
        if token is None:
            token = _read_token(account_for_alias(config_path, account_alias))
            tokens[account_alias] = token
        try:
            if operation == "trash":
                _gmail_post(f"{GMAIL_API_ROOT}/messages/{message_id}/trash", token)
            elif operation == "archive":
                _gmail_post(
                    f"{GMAIL_API_ROOT}/messages/{message_id}/modify",
                    token,
                    {"removeLabelIds": ["INBOX"]},
                )
            result["gmail_write_api_calls"] += 1
            result["external_mutations"] += 1
            result["applied"].append({"account_alias": account_alias, "message_id": message_id})
        except urllib.error.HTTPError as exc:  # pragma: no cover - live API path
            result["gmail_write_api_calls"] += 1
            result["errors"].append({"message_id": message_id, "error": f"HTTP {exc.code}"})
        except Exception as exc:  # pragma: no cover - live API path
            result["errors"].append({"message_id": message_id, "error": type(exc).__name__})

    records = ledger.load()
    success = bool(result["applied"]) and not result["errors"] and not dry_run
    if dry_run:
        mutation_status = "dry_run"
    elif success and operation == "nudge_only":
        mutation_status = "nudged"
    elif success:
        mutation_status = "applied"
    else:
        mutation_status = "blocked"
    for candidate in records:
        if candidate.handle != record.handle:
            continue
        candidate.status = "executed" if success else candidate.status
        candidate.executor_state.update(
            {
                "mutation_status": mutation_status,
                "last_apply_result": result,
                "applied_at": now if success else None,
            }
        )
        candidate.resolution_history.append(
            {
                "at": now,
                "status": "dry_run" if dry_run else candidate.executor_state["mutation_status"],
                "reason": f"Email hygiene apply requested by {approved_by}.",
            }
        )
    ledger.save(records)
    return result
