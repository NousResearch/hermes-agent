"""Approval-gated Gmail hygiene recommendations for Torben."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .action_ledger import ActionLedger, ActionRecord, parse_time
from .google_auth import account_for_alias
from .google_evidence import GMAIL_API_ROOT, _read_token


HYGIENE_POLICY_VERSION = 1
HYGIENE_OPEN_STATUSES = {"staged", "approval_required", "approved", "executing"}
ACTION_PREFIX = "email_hygiene"
DEFAULT_LLM_PROVIDER = "openai-codex"
DEFAULT_LLM_MODEL = "gpt-5.5"
DEFAULT_LLM_TIMEOUT_SECONDS = 120

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
    llm_review: dict[str, Any] = field(default_factory=dict)

    def to_action_summary(self) -> str:
        return f"{self.title} ({len(self.items)} item{'s' if len(self.items) != 1 else ''})"

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "key": self.key,
            "title": self.title,
            "operation": self.operation,
            "risk_class": self.risk_class,
            "rationale": self.rationale,
            "items": self.items,
        }
        if self.llm_review:
            payload["llm_review"] = self.llm_review
        return payload


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


def _labels(record: dict[str, Any]) -> list[str]:
    labels = record.get("labels")
    if not isinstance(labels, list):
        return []
    return [str(label or "").strip() for label in labels if str(label or "").strip()]


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
        "labels": _labels(record),
        "snippet": record.get("snippet"),
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
        "archive_stale_low_signal_info": HygieneGroup(
            key="archive_stale_low_signal_info",
            title="Archive stale low-signal info mail",
            operation="archive",
            risk_class="low",
            rationale="Old low-signal info/newsletter/promotional mail can leave the inbox after review.",
        ),
        "trash_existing_spam": HygieneGroup(
            key="trash_existing_spam",
            title="Trash messages already marked spam",
            operation="trash",
            risk_class="low",
            rationale="Gmail already classed these as spam; Torben still stages the action for approval first.",
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
        label_set = {label.upper() for label in _labels(record)}
        if "SPAM" in label_set and age_hours >= 24:
            groups["trash_existing_spam"].items.append(
                _compact_item(enriched, reason="already marked spam and older than one day")
            )
            continue
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
        if category in ARCHIVABLE_CATEGORIES and age_hours >= 30 * 24 and juno_bucket == "info":
            groups["archive_stale_low_signal_info"].items.append(
                _compact_item(enriched, reason="low-signal info mail older than thirty days")
            )
            continue
        if category in ACTIONABLE_CATEGORIES and juno_bucket in {"reply", "deadline", "flag"} and age_hours >= 7 * 24:
            groups["nudge_stale_replies"].items.append(
                _compact_item(enriched, reason="action-shaped thread older than one week")
            )

    for group in groups.values():
        group.items = group.items[:50]
    return [group for group in groups.values() if group.items]


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = str(payload.get("output_text") or "").strip()
    if output_text:
        return output_text
    parts: list[str] = []
    for item in payload.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content", []) or []:
            if not isinstance(content, dict):
                continue
            if content.get("type") in {"output_text", "text"}:
                text = str(content.get("text") or "").strip()
                if text:
                    parts.append(text)
    return "\n\n".join(parts).strip()


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _llm_review_request(groups: list[HygieneGroup]) -> dict[str, Any]:
    compact_groups = []
    for group in groups:
        compact_groups.append(
            {
                "key": group.key,
                "operation": group.operation,
                "rationale": group.rationale,
                "risk_class": group.risk_class,
                "items": [
                    {
                        "message_id": item.get("message_id"),
                        "thread_id": item.get("thread_id"),
                        "account_alias": item.get("account_alias"),
                        "sender": item.get("sender"),
                        "subject": item.get("subject"),
                        "category": item.get("category"),
                        "juno_bucket": item.get("juno_bucket"),
                        "reason": item.get("reason"),
                        "age_hours": item.get("age_hours"),
                        "labels": item.get("labels"),
                        "snippet": item.get("snippet"),
                    }
                    for item in group.items[:12]
                ],
            }
        )
    return {"candidate_groups": compact_groups}


def _hygiene_review_prompt(groups: list[HygieneGroup]) -> str:
    return (
        "Review these Gmail cleanup candidates for Eric's Torben COO agent.\n"
        "This is an inbox hygiene review, not an apply step. You may keep or drop candidates. "
        "Do not invent messages. Treat sender identity, category, labels, reason, and snippet as evidence.\n\n"
        "Hard rules:\n"
        "- Keep important relationship, funding, customer, legal, admin, family, school, health, finance, or scheduling mail out of archive/trash.\n"
        "- If a stale thread may still matter, prefer the nudge_only group instead of archive/trash.\n"
        "- Trash means Gmail Trash, never permanent delete.\n"
        "- Trash existing spam only when labels include SPAM and the item already looks non-actionable.\n"
        "- Trash MFA/password-code mail only when it is a stale account-security code.\n"
        "- Archive only low-signal info, receipt/vendor ops, policy/rewards/rebrand noise, or general newsletter/promotional mail.\n"
        "- If unsure, drop the cleanup candidate and leave it for review.\n\n"
        "Return one JSON object with this schema only:\n"
        "{\n"
        '  "recommendations": [\n'
        '    {"key": "group key", "decision": "keep|drop", "reason": "short reason", "items": [\n'
        '      {"message_id": "id", "decision": "keep|drop", "reason": "short reason"}\n'
        "    ]}\n"
        "  ],\n"
        '  "global_notes": ["short note"]\n'
        "}\n\n"
        f"Candidates:\n{json.dumps(_llm_review_request(groups), ensure_ascii=False, sort_keys=True)}"
    )


def _call_hygiene_llm(groups: list[HygieneGroup]) -> tuple[dict[str, Any], dict[str, Any]]:
    from hermes_cli.runtime_provider import resolve_runtime_provider

    provider = str(os.getenv("TORBEN_EMAIL_HYGIENE_LLM_PROVIDER") or DEFAULT_LLM_PROVIDER).strip()
    runtime = resolve_runtime_provider(requested=provider)
    api_key = str(runtime.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError(f"{provider} credentials unavailable")
    base_url = str(runtime.get("base_url") or "https://api.openai.com/v1").strip().rstrip("/")
    model = str(
        os.getenv("TORBEN_EMAIL_HYGIENE_LLM_MODEL")
        or runtime.get("model")
        or runtime.get("default_model")
        or DEFAULT_LLM_MODEL
    ).strip()
    timeout = _positive_int(os.getenv("TORBEN_EMAIL_HYGIENE_LLM_TIMEOUT_SECONDS"), DEFAULT_LLM_TIMEOUT_SECONDS)
    request_payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are Torben, Eric Freeman's operational COO. "
                    "Review inbox cleanup candidates conservatively. Return JSON only."
                ),
            },
            {"role": "user", "content": _hygiene_review_prompt(groups)},
        ],
        "store": False,
        "max_output_tokens": 1600,
    }
    response = requests.post(
        f"{base_url}/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=request_payload,
        timeout=timeout,
    )
    response.raise_for_status()
    response_payload = response.json()
    generated = _parse_json_object(_extract_response_text(response_payload))
    return generated, {"provider": provider, "model": model, "base_url_host": base_url.split("//")[-1].split("/")[0]}


def _apply_llm_review(
    groups: list[HygieneGroup],
    generated: dict[str, Any],
    *,
    meta: dict[str, Any],
) -> list[HygieneGroup]:
    recommendations = generated.get("recommendations")
    if not isinstance(recommendations, list):
        raise ValueError("LLM hygiene review returned no recommendations list")
    by_key = {
        str(item.get("key") or ""): item
        for item in recommendations
        if isinstance(item, dict) and str(item.get("key") or "").strip()
    }
    reviewed: list[HygieneGroup] = []
    for group in groups:
        group_decision = by_key.get(group.key) or {}
        decision = str(group_decision.get("decision") or "keep").strip().lower()
        if decision == "drop":
            continue
        item_decisions = {
            str(item.get("message_id") or ""): item
            for item in group_decision.get("items", []) or []
            if isinstance(item, dict)
        }
        kept_items: list[dict[str, Any]] = []
        for item in group.items:
            item_decision = item_decisions.get(str(item.get("message_id") or "")) or {}
            if str(item_decision.get("decision") or "keep").strip().lower() == "drop":
                continue
            kept = dict(item)
            kept["llm_decision"] = str(item_decision.get("decision") or decision or "keep")
            kept["llm_reason"] = str(item_decision.get("reason") or group_decision.get("reason") or "").strip()
            kept_items.append(kept)
        if not kept_items:
            continue
        reviewed.append(
            HygieneGroup(
                key=group.key,
                title=group.title,
                operation=group.operation,
                risk_class=group.risk_class,
                rationale=group.rationale,
                items=kept_items,
                llm_review={
                    **meta,
                    "decision": decision or "keep",
                    "reason": str(group_decision.get("reason") or "").strip(),
                },
            )
        )
    return reviewed


def _review_hygiene_groups_with_llm(
    groups: list[HygieneGroup],
    *,
    enabled: bool,
) -> tuple[list[HygieneGroup], dict[str, Any]]:
    if not groups:
        return groups, {"invoked": False, "status": "no_candidates"}
    if not enabled:
        return groups, {"invoked": False, "status": "disabled"}
    try:
        generated, runtime_meta = _call_hygiene_llm(groups)
        meta = {
            "invoked": True,
            "status": "accepted",
            "fallback": False,
            "global_notes": list(generated.get("global_notes") or [])[:5],
            **runtime_meta,
        }
        reviewed = _apply_llm_review(groups, generated, meta=meta)
        return reviewed, {**meta, "groups_before": len(groups), "groups_after": len(reviewed)}
    except Exception as exc:  # noqa: BLE001 - cleanup must fail closed and keep review visible.
        meta = {
            "invoked": True,
            "status": "failed_deterministic_fallback",
            "fallback": True,
            "error_type": type(exc).__name__,
            "error": str(exc)[:240],
            "groups_before": len(groups),
            "groups_after": len(groups),
        }
        for group in groups:
            group.llm_review = meta
        return groups, meta


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
    enable_llm_review: bool = False,
    review_metadata_out: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    now = (now or _now()).astimezone(timezone.utc)
    existing = _existing_hygiene_actions(ledger)
    groups, review_meta = _review_hygiene_groups_with_llm(
        _group_candidate_records(records, now=now),
        enabled=enable_llm_review,
    )
    if review_metadata_out is not None:
        review_metadata_out.update(review_meta)
    staged: list[dict[str, Any]] = []
    records_to_save: list[ActionRecord] | None = None
    for group in groups:
        action_key = f"{ACTION_PREFIX}:{group.key}:{now:%Y-%m-%d}"
        record = existing.get(action_key)
        evidence_ids = [
            evidence
            for item in group.items[:10]
            for evidence in (item.get("evidence_ids") or [])
        ]
        executor_state = {
            "mutation_type": "gmail_hygiene",
            "provider": "gmail",
            "mutation_status": "not_applied",
            "hygiene_policy_version": HYGIENE_POLICY_VERSION,
            "hygiene_action_key": action_key,
            "operation": group.operation,
            "rationale": group.rationale,
            "items": group.items,
            "llm_review": group.llm_review or review_meta,
            "approval_required": True,
        }
        if record is None:
            record = ledger.add_action(
                scope="EA",
                summary=group.to_action_summary(),
                evidence_ids=evidence_ids,
                allowed_next_actions=["approve_hygiene_apply", "revise", "discard"],
                status="approval_required",
                risk_class=group.risk_class,
                ttl_hours=14 * 24,
                now=now,
                executor_state=executor_state,
            )
            existing[action_key] = record
        else:
            if records_to_save is None:
                records_to_save = ledger.load()
            for candidate in records_to_save:
                if candidate.handle != record.handle:
                    continue
                candidate.summary = group.to_action_summary()
                candidate.evidence_ids = evidence_ids
                candidate.risk_class = group.risk_class
                candidate.executor_state.update(executor_state)
                candidate.resolution_history.append(
                    {
                        "at": now.isoformat().replace("+00:00", "Z"),
                        "status": "recommendation_refreshed",
                        "reason": "Weekly email hygiene recommendation refreshed with current inbox evidence.",
                    }
                )
                record = candidate
                existing[action_key] = candidate
                break
        staged.append(
            {
                "handle": record.handle,
                "status": record.status,
                **group.to_payload(),
            }
        )
    if records_to_save is not None:
        ledger.save(records_to_save)
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
    labels = {str(label or "").upper() for label in item.get("labels") or []}
    stale_account_code = category == "account_security" and "older than one hour" in reason
    existing_spam = "SPAM" in labels and "already marked spam" in reason
    if operation == "trash" and not (stale_account_code or existing_spam):
        raise ValueError(f"Refusing to trash item outside approved stale-code/spam classes: {item.get('message_id')}")
    if operation == "archive" and category not in ARCHIVABLE_CATEGORIES:
        raise ValueError(f"Refusing to archive actionable category {category}: {item.get('message_id')}")


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
