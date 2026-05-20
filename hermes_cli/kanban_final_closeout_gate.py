"""Read-only final closeout gate for GitHub-backed Kanban handoffs.

The command-line entry point is ``kanban-final-closeout-gate``.  It consumes
TianGongKaiWu evidence manifests (executor handoff, review decision, optional
final-closeout draft) plus the #161 duplicate-child guard receipt, then emits a
single JSON receipt and optional Markdown report.  It deliberately performs no
external writes: no merge, no issue close, no label transition, no trace send,
and no Kanban mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import yaml

SCHEMA_VERSION = "kanban-final-closeout-gate:receipt:v1"
ENTRYPOINT = "kanban-final-closeout-gate"
EXECUTOR_SCHEMA = "tiangongkaiwu.executor_handoff.v1"
REVIEW_SCHEMA = "tiangongkaiwu.review_decision.v1"
CLOSEOUT_SCHEMA = "tiangongkaiwu.final_closeout_gate.v1"
DUPLICATE_GUARD_SCHEMA = "kanban-duplicate-child-guard:receipt:v1"

GITHUB_URL_RE = re.compile(r"https://github\.com/[^\s)\]>'\"]+", re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:ya?ml|json)?\s*\n(.*?)\n```", re.IGNORECASE | re.DOTALL)

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("github_token", re.compile(r"\bgh[opsu]_[A-Za-z0-9_]{20,}\b")),
    ("openai_style_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "credential_field",
        re.compile(
            r"(?i)['\"]?\b(?:oauth_token|api[_-]?key|access[_-]?token|refresh[_-]?token|secret|password)\b['\"]?\s*[:=]\s*['\"]?[A-Za-z0-9_./+=-]{16,}"
        ),
    ),
)
RAW_LOCATOR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("telegram_raw_chat_or_topic", re.compile(r"\btelegram:-?\d{6,}(?::-?\d{1,})?\b", re.IGNORECASE)),
    ("chat_id_field", re.compile(r"(?i)['\"]?\b(?:chat_id|thread_id|topic_id)\b['\"]?\s*[:=]\s*-?\d{5,}\b")),
    ("discord_raw_ids", re.compile(r"\bdiscord:\d{8,}(?::\d{8,})?\b", re.IGNORECASE)),
    ("slack_raw_channel", re.compile(r"\bslack:[CGD][A-Z0-9]{8,}\b", re.IGNORECASE)),
)
SENSITIVE_FIELD_RE = re.compile(
    r"(?i)^(?:oauth_token|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer[_-]?token|secret|password|authorization)$"
)
RAW_LOCATOR_FIELD_RE = re.compile(r"(?i)^(?:chat_id|thread_id|topic_id|channel_id|guild_id)$")

PASS_STATUSES = {"passed", "pass", "ok", "success", "waived", "not_applicable"}
FAIL_STATUSES = {"failed", "fail", "error", "blocked", "returned"}
TRACE_OK_STATUSES = {"sent", "dry_run", "skipped", "observed", "passed"}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _get(data: Any, path: str, default: Any = None) -> Any:
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _normalize_status(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace(" ", "_") or "unknown"


def _normalize_issue(issue: Any) -> Optional[str]:
    if issue is None:
        return None
    text = str(issue).strip()
    if not text:
        return None
    match = re.search(r"/issues/(\d+)", text)
    if match:
        return f"#{int(match.group(1))}"
    if text.startswith("#"):
        text = text[1:]
    return f"#{int(text)}" if text.isdigit() else str(issue).strip()


def _normalize_pr(pr: Any) -> Optional[str]:
    if pr is None:
        return None
    text = str(pr).strip()
    if not text:
        return None
    match = re.search(r"/pull/(\d+)", text)
    if match:
        return f"PR#{int(match.group(1))}"
    text = text.removeprefix("PR#").removeprefix("#")
    return f"PR#{int(text)}" if text.isdigit() else str(pr).strip()


def _runtime_profile_smoke(actor_profile: Optional[str]) -> dict[str, Any]:
    actual = os.environ.get("HERMES_PROFILE") or os.environ.get("HERMES_AGENT_PROFILE") or ""
    expected = actor_profile or ""
    return {
        "actual_profile": actual,
        "expected_actor_profile": expected,
        "actor_profile_match": (not expected) or actual == expected,
        "current_kanban_task_id": os.environ.get("HERMES_KANBAN_TASK") or "",
        "hermes_home_profile_hint": Path(os.environ.get("HERMES_HOME", "")).name if os.environ.get("HERMES_HOME") else "",
    }


def _load_yaml_documents(text: str) -> list[Any]:
    return [doc for doc in yaml.safe_load_all(text) if doc is not None]


def _load_json_or_yaml(text: str, *, source: str) -> list[dict[str, Any]]:
    docs: list[Any] = []
    stripped = text.strip()
    if not stripped:
        return []

    parse_errors: list[str] = []
    if stripped[:1] in "[{":
        try:
            parsed = json.loads(stripped)
            docs = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as exc:
            parse_errors.append(str(exc))
    if not docs:
        try:
            docs = _load_yaml_documents(text)
        except yaml.YAMLError as exc:
            parse_errors.append(str(exc))
    if not docs:
        for block in FENCE_RE.findall(text):
            try:
                docs.extend(_load_yaml_documents(block))
            except yaml.YAMLError as exc:
                parse_errors.append(str(exc))

    out: list[dict[str, Any]] = []
    for index, doc in enumerate(docs):
        if isinstance(doc, dict):
            item = dict(doc)
            item.setdefault("_source", {"path": source, "document_index": index})
            out.append(item)
    if not out and parse_errors:
        raise ValueError(f"no manifest/receipt documents found in {source}: {'; '.join(parse_errors[-2:])}")
    return out


def load_documents(paths: Iterable[str]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        text = path.read_text(encoding="utf-8")
        documents.extend(_load_json_or_yaml(text, source=str(path)))
    return documents


def _first_by_schema(manifests: Iterable[dict[str, Any]], schema: str) -> Optional[dict[str, Any]]:
    for manifest in manifests:
        if manifest.get("schema") == schema:
            return manifest
    return None


def _manifest_sources(manifests: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []
    for manifest in manifests:
        sources.append(
            {
                "schema": manifest.get("schema"),
                "source": manifest.get("_source", {}),
                "issue_url": _get(manifest, "issue.url"),
                "artifact_url": _get(manifest, "artifact.url") or _get(manifest, "artifact.artifact_url") or _get(manifest, "pr.url"),
            }
        )
    return sources


def _dedupe(values: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _github_urls_from_payload(value: Any) -> list[str]:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    return _dedupe(GITHUB_URL_RE.findall(text))


def _scan_public_text(texts: Iterable[Any]) -> dict[str, Any]:
    joined = "\n".join(str(text) for text in texts if text is not None)
    secret_hits = [(name, len(pattern.findall(joined))) for name, pattern in SECRET_PATTERNS]
    locator_hits = [(name, len(pattern.findall(joined))) for name, pattern in RAW_LOCATOR_PATTERNS]
    secret_hits = [(name, count) for name, count in secret_hits if count]
    locator_hits = [(name, count) for name, count in locator_hits if count]
    return {
        "secret_hit_count": sum(count for _, count in secret_hits),
        "secret_hit_types": [name for name, _ in secret_hits],
        "raw_locator_hit_count": sum(count for _, count in locator_hits),
        "raw_locator_hit_types": [name for name, _ in locator_hits],
    }


def _public_text_payload(manifests: list[dict[str, Any]], public_text: Optional[Iterable[str]]) -> list[str]:
    payload = [json.dumps(manifest, ensure_ascii=False, sort_keys=True) for manifest in manifests]
    payload.extend(public_text or [])
    return payload


def _redact_public_text(text: str) -> str:
    redacted = text
    for name, pattern in (*SECRET_PATTERNS, *RAW_LOCATOR_PATTERNS):
        redacted = pattern.sub(f"<redacted:{name}>", redacted)
    return redacted


def _redact_public_value(value: Any, *, field_name: Optional[str] = None) -> Any:
    if field_name and SENSITIVE_FIELD_RE.match(field_name):
        return "<redacted:credential>"
    if field_name and RAW_LOCATOR_FIELD_RE.match(field_name):
        return "<redacted:raw_platform_locator>"
    if isinstance(value, str):
        return _redact_public_text(value)
    if isinstance(value, dict):
        return {key: _redact_public_value(child, field_name=str(key)) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_public_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_redact_public_value(child) for child in value)
    return value


def _sanitize_public_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Remove raw locators/secrets from any structure that may be pasted publicly."""
    return _redact_public_value(receipt)


def _make_gap(code: str, message: str, *, evidence: Optional[Iterable[Any]] = None, next_owner: str, next_action: str) -> dict[str, Any]:
    return {
        "code": code,
        "severity": "blocking",
        "message": message,
        "evidence": _dedupe(evidence or []),
        "next_owner": next_owner,
        "next_action": next_action,
    }


def _check_from_declared_status(status: Any) -> str:
    norm = _normalize_status(status)
    if norm in PASS_STATUSES:
        return "passed" if norm != "waived" else "waived"
    if norm in FAIL_STATUSES:
        return "failed"
    return "unknown"


def _artifact_check(
    executor: Optional[dict[str, Any]],
    review: Optional[dict[str, Any]],
    closeout: Optional[dict[str, Any]],
    current_head: Optional[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declared_statuses = [
        _check_from_declared_status(_get(closeout, "gate_checks.artifact_matches_approval.status")),
        _check_from_declared_status(_get(review, "review_input.required_gates.head_or_artifact_matches")),
    ]
    approved = (
        _get(closeout, "artifact.approved_head_sha_or_artifact_id")
        or _get(review, "artifact.submitted_head_sha_or_artifact_id")
        or _get(executor, "artifact.commit_or_head_sha")
        or _get(executor, "pr.head_sha")
    )
    current = (
        current_head
        or _get(closeout, "artifact.current_head_sha_or_artifact_id")
        or _get(executor, "pr.head_sha")
        or _get(executor, "artifact.commit_or_head_sha")
    )
    declared_unchanged = _get(closeout, "artifact.head_unchanged_since_approval")
    evidence = {
        "approved_head_or_artifact_id": approved,
        "current_head_or_artifact_id": current,
        "declared_head_unchanged_since_approval": declared_unchanged,
        "declared_statuses": declared_statuses,
        "urls": _dedupe(
            [
                _get(closeout, "artifact.pr_url"),
                _get(closeout, "artifact.artifact_url"),
                _get(review, "artifact.pr_url"),
                _get(review, "artifact.artifact_url"),
                _get(executor, "artifact.url"),
                _get(executor, "pr.url"),
            ]
        ),
    }
    if "failed" in declared_statuses:
        return {"status": "failed", "evidence": evidence}, [
            _make_gap(
                "artifact_head_declared_failed",
                "Manifest evidence explicitly marks the artifact/head match gate as failed.",
                evidence=[evidence],
                next_owner="reviewer",
                next_action="re_review",
            )
        ]
    if approved and current and str(approved) == str(current) and declared_unchanged is not False:
        return {"status": "passed", "evidence": evidence}, []
    if approved and current:
        return {"status": "failed", "evidence": evidence}, [
            _make_gap(
                "head_changed_after_approval",
                "Current PR/head/artifact id differs from the approved review artifact.",
                evidence=[evidence],
                next_owner="reviewer",
                next_action="re_review",
            )
        ]
    return {"status": "unknown", "evidence": evidence}, [
        _make_gap(
            "artifact_head_missing",
            "Approved or current artifact/head id is missing, so closeout cannot prove the artifact is unchanged.",
            evidence=[evidence],
            next_owner="executor",
            next_action="provide_evidence",
        )
    ]


def _review_decision_check(review: Optional[dict[str, Any]], closeout: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decision = _get(review, "review_decision")
    blocking_findings = _as_list(_get(review, "blocking_findings"))
    evidence = {
        "review_decision": decision,
        "closeout_approved_review_decision_claim": _get(closeout, "approved_inputs.approved_review_decision"),
        "blocking_findings": blocking_findings,
        "review_url_or_comment_url": _get(closeout, "approved_inputs.review_decision_url_or_comment_url"),
        "review_source": _get(review, "_source"),
    }
    if str(decision).strip().lower() == "approved" and not blocking_findings:
        return {"status": "passed", "evidence": evidence}, []
    return {"status": "failed", "evidence": evidence}, [
        _make_gap(
            "review_not_approved",
            "Review decision is missing or not approved.",
            evidence=[evidence],
            next_owner="reviewer",
            next_action="re_review",
        )
    ]


def _tests_check(executor: Optional[dict[str, Any]], review: Optional[dict[str, Any]], closeout: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declared = [
        _get(executor, "tests.status"),
        _get(review, "review_input.required_gates.tests_evidence_present"),
        _get(closeout, "gate_checks.required_tests_or_checks.status"),
    ]
    statuses = [_check_from_declared_status(s) for s in declared if s is not None]
    evidence = _dedupe(
        _as_list(_get(executor, "tests.commands"))
        + _as_list(_get(executor, "tests.passed"))
        + _as_list(_get(review, "checks_reviewed.tests_or_ci_reviewed"))
        + _as_list(_get(closeout, "gate_checks.required_tests_or_checks.evidence"))
    )
    if "failed" in statuses:
        status = "failed"
    elif any(s in {"passed", "waived"} for s in statuses):
        status = "passed" if "passed" in statuses else "waived"
    else:
        status = "unknown"
    check = {"status": status, "evidence": evidence, "declared_statuses": declared}
    if status in {"passed", "waived"}:
        return check, []
    return check, [
        _make_gap(
            "tests_or_checks_missing_or_failed",
            "Required tests/checks evidence is missing or failed.",
            evidence=evidence or declared,
            next_owner="executor",
            next_action="provide_evidence",
        )
    ]


def _labels_check(executor: Optional[dict[str, Any]], review: Optional[dict[str, Any]], closeout: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    labels_after = _dedupe(
        _as_list(_get(executor, "labels.after"))
        + _as_list(_get(review, "labels.after"))
        + _as_list(_get(closeout, "labels.after"))
        + _as_list(_get(closeout, "gate_checks.lifecycle_labels_readable.labels_after"))
    )
    declared = _check_from_declared_status(_get(closeout, "gate_checks.lifecycle_labels_readable.status"))
    if declared == "failed":
        status = "failed"
    elif labels_after or declared == "passed":
        status = "passed"
    else:
        status = "unknown"
    check = {
        "status": status,
        "labels_before": _dedupe(
            _as_list(_get(executor, "labels.before"))
            + _as_list(_get(review, "labels.before"))
            + _as_list(_get(closeout, "gate_checks.lifecycle_labels_readable.labels_before"))
        ),
        "labels_after": labels_after,
    }
    if status == "passed":
        return check, []
    return check, [
        _make_gap(
            "lifecycle_label_missing_or_unreadable",
            "Lifecycle labels are missing or explicitly failed readback.",
            evidence=[check],
            next_owner="orchestrator",
            next_action="provide_evidence",
        )
    ]


def _trace_receipts_from_manifest(manifest: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for item in _as_list(_get(manifest, "trace.receipts")):
        if isinstance(item, dict):
            receipts.append(item)
    for item in _as_list(_get(manifest, "gate_checks.trace_receipts_complete.receipts")):
        if isinstance(item, dict):
            receipts.append(item)
    return receipts


def _trace_check(executor: Optional[dict[str, Any]], review: Optional[dict[str, Any]], closeout: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_receipts = [
        (EXECUTOR_SCHEMA, _trace_receipts_from_manifest(executor)) if executor else None,
        (REVIEW_SCHEMA, _trace_receipts_from_manifest(review)) if review else None,
        (CLOSEOUT_SCHEMA, _trace_receipts_from_manifest(closeout)) if closeout else None,
    ]
    manifest_receipts = [item for item in manifest_receipts if item is not None]
    receipts = _dedupe([receipt for _, schema_receipts in manifest_receipts for receipt in schema_receipts])
    missing_receipt_schemas = [schema for schema, schema_receipts in manifest_receipts if not schema_receipts]
    declared = [
        _get(review, "review_input.required_gates.trace_receipts_readable"),
        _get(closeout, "gate_checks.trace_receipts_complete.status"),
    ]
    declared_statuses = [_check_from_declared_status(s) for s in declared if s is not None]
    bad_receipts = [r for r in receipts if _normalize_status(r.get("delivery_status")) not in TRACE_OK_STATUSES]
    if "failed" in declared_statuses or bad_receipts or missing_receipt_schemas or not receipts:
        status = "failed"
    else:
        status = "passed"
    check = {"status": status, "receipts": receipts, "declared_statuses": declared, "missing_receipt_schemas": missing_receipt_schemas}
    if status == "passed":
        return check, []
    return check, [
        _make_gap(
            "trace_receipt_missing",
            "Actor-owned trace receipts are missing, failed, or not readable.",
            evidence=receipts or declared,
            next_owner="executor",
            next_action="provide_evidence",
        )
    ]


def _kanban_terminal_check(closeout: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    declared = _get(closeout, "kanban.child_tasks_terminal")
    gate_status = _check_from_declared_status(_get(closeout, "gate_checks.kanban_task_graph_terminal.status"))
    evidence = _get(closeout, "gate_checks.kanban_task_graph_terminal.evidence")
    if declared is True or gate_status in {"passed", "waived"}:
        return {"status": "passed" if gate_status != "waived" else "waived", "evidence": evidence, "child_tasks_terminal": declared}, []
    if declared is False or gate_status == "failed":
        check = {"status": "failed", "evidence": evidence, "child_tasks_terminal": declared}
        return check, [
            _make_gap(
                "child_not_terminal",
                "Kanban child task graph is not terminal or closeout precondition failed.",
                evidence=[check],
                next_owner="orchestrator",
                next_action="wait_for_children",
            )
        ]
    check = {"status": "unknown", "evidence": evidence, "child_tasks_terminal": declared}
    return check, [
        _make_gap(
            "child_terminal_state_missing",
            "Kanban child terminal state is missing from closeout manifest.",
            evidence=[check],
            next_owner="orchestrator",
            next_action="provide_evidence",
        )
    ]


def _public_sanitation_check(manifests: list[dict[str, Any]], guard_receipts: list[dict[str, Any]], public_text: Optional[Iterable[str]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scan_payload = _public_text_payload(manifests + guard_receipts, public_text)
    scan = _scan_public_text(scan_payload)
    declared_secret_statuses = [
        _check_from_declared_status(_get(manifest, "security_scan.secret_scan.status"))
        for manifest in manifests
        if _get(manifest, "security_scan.secret_scan.status") is not None
    ]
    declared_raw_statuses = [
        _check_from_declared_status(_get(manifest, "security_scan.raw_locator_scan.status"))
        for manifest in manifests
        if _get(manifest, "security_scan.raw_locator_scan.status") is not None
    ]
    declared_gate = [_check_from_declared_status(_get(manifest, "gate_checks.public_text_sanitation.status")) for manifest in manifests if _get(manifest, "gate_checks.public_text_sanitation.status") is not None]
    failed = (
        scan["secret_hit_count"] > 0
        or scan["raw_locator_hit_count"] > 0
        or "failed" in declared_secret_statuses
        or "failed" in declared_raw_statuses
        or "failed" in declared_gate
    )
    check = {
        "status": "failed" if failed else "passed",
        "secret_scan_summary": (
            f"failed: {scan['secret_hit_count']} potential secret hit(s) by type {scan['secret_hit_types']}"
            if scan["secret_hit_count"]
            else "passed: no credential/token pattern found"
        ),
        "raw_locator_scan_summary": (
            f"failed: {scan['raw_locator_hit_count']} raw platform locator hit(s) by type {scan['raw_locator_hit_types']}"
            if scan["raw_locator_hit_count"]
            else "passed: no raw platform locator pattern found"
        ),
        "declared_secret_statuses": declared_secret_statuses,
        "declared_raw_locator_statuses": declared_raw_statuses,
    }
    if not failed:
        return check, []
    return check, [
        _make_gap(
            "public_text_scan_failed",
            "Public text contains potential credential material, raw platform locator, or a declared security scan failure.",
            evidence=[check],
            next_owner="executor",
            next_action="fix_public_text",
        )
    ]


def _identity_check(manifests: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    required_schemas = {EXECUTOR_SCHEMA, REVIEW_SCHEMA, CLOSEOUT_SCHEMA}
    failures: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []
    for manifest in manifests:
        schema = manifest.get("schema")
        identity = manifest.get("identity_block")
        if schema in required_schemas and not isinstance(identity, dict):
            item = {"schema": schema, "status": "failed", "reason": "missing_identity_block"}
            checked.append(item)
            failures.append(item)
            continue
        if not isinstance(identity, dict):
            checked.append({"schema": schema, "status": "waived", "reason": "identity_block_not_required"})
            continue
        required = identity.get("required") is True
        location = identity.get("location")
        if not required or location == "not_applicable":
            checked.append({"schema": manifest.get("schema"), "status": "waived", "location": location})
            continue
        header = str(identity.get("exact_header") or "")
        actor = manifest.get("actor") or {}
        missing = []
        for field in ("agent_name", "provider", "model"):
            value = actor.get(field)
            if not value or str(value) not in header:
                missing.append(field)
        verified = identity.get("verified_against_runtime") is True
        status = "passed" if header and verified and not missing else "failed"
        item = {
            "schema": schema,
            "status": status,
            "location": location,
            "verified_against_runtime": identity.get("verified_against_runtime"),
            "missing_actor_fields_in_header": missing,
        }
        checked.append(item)
        if status == "failed":
            failures.append(item)
    check = {"status": "failed" if failures else "passed", "evidence": checked}
    if not failures:
        return check, []
    return check, [
        _make_gap(
            "identity_block_mismatch",
            "Identity block is missing, not runtime-verified, or inconsistent with actor/model/provider fields.",
            evidence=failures,
            next_owner="executor",
            next_action="provide_evidence",
        )
    ]


def _duplicate_guard_check(guard_receipts: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not guard_receipts:
        check = {"status": "failed", "receipts": [], "summary": "missing #161 duplicate-child guard receipt"}
        return check, [
            _make_gap(
                "duplicate_guard_receipt_missing",
                "#161 duplicate-child guard receipt is required but missing.",
                evidence=[check],
                next_owner="orchestrator",
                next_action="provide_evidence",
            )
        ]

    failures: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for receipt in guard_receipts:
        duplicate_groups = _as_list(_get(receipt, "detector.duplicate_groups"))
        insufficient = _as_list(_get(receipt, "detector.insufficiently_marked_review_children"))
        actions = _as_list(_get(receipt, "dry_run_plan.actions"))
        public_safety = receipt.get("public_safety") or {}
        summary = {
            "schema": receipt.get("schema"),
            "ok": receipt.get("ok"),
            "mode": receipt.get("mode"),
            "board_read_only": _get(receipt, "board.read_only"),
            "duplicate_groups": len(duplicate_groups),
            "insufficiently_marked_review_children": len(insufficient),
            "actions": len(actions),
            "apply_enabled": _get(receipt, "dry_run_plan.apply_enabled"),
            "apply_supported": _get(receipt, "dry_run_plan.apply_supported"),
            "no_mutations_performed": public_safety.get("no_mutations_performed"),
            "raw_platform_locator_included": public_safety.get("raw_platform_locator_included"),
            "credentials_or_tokens_touched": public_safety.get("credentials_or_tokens_touched", False),
            "source": receipt.get("_source", {}),
        }
        summaries.append(summary)
        safe = (
            receipt.get("schema") == DUPLICATE_GUARD_SCHEMA
            and receipt.get("mode") == "dry_run"
            and receipt.get("ok") is True
            and _get(receipt, "board.read_only") is True
            and _get(receipt, "dry_run_plan.apply_enabled") is False
            and public_safety.get("no_mutations_performed") is True
            and public_safety.get("raw_platform_locator_included") is False
            and public_safety.get("credentials_or_tokens_touched", False) is False
            and len(duplicate_groups) == 0
            and len(insufficient) == 0
            and len(actions) == 0
        )
        if not safe:
            failures.append(summary)
    check = {"status": "failed" if failures else "passed", "receipts": summaries}
    if not failures:
        return check, []
    return check, [
        _make_gap(
            "duplicate_child_guard_failed",
            "#161 duplicate-child guard receipt is unsafe, reports duplicates, or is not clean/read-only.",
            evidence=failures,
            next_owner="orchestrator",
            next_action="provide_evidence",
        )
    ]


def _missing_manifest_gaps(executor: Optional[dict[str, Any]], review: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    gaps = []
    if executor is None:
        gaps.append(
            _make_gap(
                "executor_manifest_missing",
                "Executor handoff manifest is required for final closeout gate.",
                next_owner="executor",
                next_action="provide_evidence",
            )
        )
    if review is None:
        gaps.append(
            _make_gap(
                "review_manifest_missing",
                "Approved review decision manifest is required for final closeout gate.",
                next_owner="reviewer",
                next_action="provide_evidence",
            )
        )
    return gaps


def _next_from_gaps(gaps: list[dict[str, Any]]) -> dict[str, Any]:
    if not gaps:
        return {
            "owner": "closeout",
            "action": "closeout",
            "suggested_next_step": "Gate passed. Closeout actor may merge/close/label/trace only if separately authorized by the task contract.",
        }
    first = gaps[0]
    return {
        "owner": first.get("next_owner") or "orchestrator",
        "action": first.get("next_action") or "provide_evidence",
        "suggested_next_step": first.get("message") or "Resolve the blocking gate gap, then rerun the read-only closeout check.",
    }


def build_closeout_gate_receipt(
    *,
    manifests: Sequence[dict[str, Any]],
    guard_receipts: Optional[Sequence[dict[str, Any]]] = None,
    issue: Any = None,
    pr: Any = None,
    task_id: Optional[str] = None,
    repo: str = "GTZhou/TianGongKaiWu",
    board: str = "tiangongkaiwu",
    current_head: Optional[str] = None,
    public_text: Optional[Iterable[str]] = None,
    actor_profile: Optional[str] = None,
) -> dict[str, Any]:
    """Build a no-side-effect closeout-gate receipt from manifests.

    The function is intentionally pure/read-only: it only examines supplied
    Python dictionaries and public text strings, then returns a receipt.
    """
    manifest_list = [dict(manifest) for manifest in manifests]
    guard_list = [dict(receipt) for receipt in (guard_receipts or [])]
    executor = _first_by_schema(manifest_list, EXECUTOR_SCHEMA)
    review = _first_by_schema(manifest_list, REVIEW_SCHEMA)
    closeout = _first_by_schema(manifest_list, CLOSEOUT_SCHEMA)

    gate_checks: dict[str, dict[str, Any]] = {}
    gaps: list[dict[str, Any]] = _missing_manifest_gaps(executor, review)

    check, check_gaps = _artifact_check(executor, review, closeout, current_head)
    gate_checks["artifact_matches_approval"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _review_decision_check(review, closeout)
    gate_checks["review_decision_approved"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _tests_check(executor, review, closeout)
    gate_checks["required_tests_or_checks"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _labels_check(executor, review, closeout)
    gate_checks["lifecycle_labels_readable"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _trace_check(executor, review, closeout)
    gate_checks["trace_receipts_complete"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _kanban_terminal_check(closeout)
    gate_checks["kanban_task_graph_terminal"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _public_sanitation_check(manifest_list, guard_list, public_text)
    gate_checks["public_text_sanitation"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _identity_check(manifest_list)
    gate_checks["identity_block_verified"] = check
    gaps.extend(check_gaps)

    check, check_gaps = _duplicate_guard_check(guard_list)
    gate_checks["duplicate_child_guard_receipt"] = check
    gaps.extend(check_gaps)

    # Remove duplicate gap objects while preserving the first, highest-priority cause.
    unique_gaps: list[dict[str, Any]] = []
    seen_gap_codes: set[str] = set()
    for gap in gaps:
        code = gap.get("code", "")
        if code not in seen_gap_codes:
            seen_gap_codes.add(code)
            unique_gaps.append(gap)

    evidence_urls = _dedupe(_github_urls_from_payload(manifest_list) + _github_urls_from_payload(guard_list))
    passed = not unique_gaps
    receipt = {
        "schema": SCHEMA_VERSION,
        "mode": "check",
        "ok": passed,
        "closeout_gate": "passed" if passed else "failed",
        "generated_at": _utc_now(),
        "input": {
            "repo": repo,
            "issue": _normalize_issue(issue),
            "pr": _normalize_pr(pr),
            "task_id": task_id,
            "board": board,
            "current_head_override": current_head,
        },
        "source_manifests": _manifest_sources(manifest_list),
        "guard_receipts": [
            {
                "schema": receipt.get("schema"),
                "source": receipt.get("_source", {}),
                "ok": receipt.get("ok"),
                "mode": receipt.get("mode"),
            }
            for receipt in guard_list
        ],
        "gate_checks": gate_checks,
        "gaps": unique_gaps,
        "evidence_urls": evidence_urls,
        "next": _next_from_gaps(unique_gaps),
        "runtime_profile_smoke": _runtime_profile_smoke(actor_profile),
        "public_safety": {
            "no_mutations_performed": True,
            "write_targets": [],
            "merge_performed": False,
            "issue_closed": False,
            "label_transition_performed": False,
            "trace_sent": False,
            "credentials_or_tokens_touched": False,
            "raw_platform_locator_included": False,
            "no_reaudit_performed": True,
        },
    }
    return _sanitize_public_receipt(receipt)


def render_markdown_report(receipt: dict[str, Any]) -> str:
    lines = [
        "# kanban-final-closeout-gate check",
        "",
        f"- closeout_gate: {receipt.get('closeout_gate')}",
        f"- ok: {str(receipt.get('ok')).lower()}",
        f"- issue: {receipt.get('input', {}).get('issue')}",
        f"- pr: {receipt.get('input', {}).get('pr')}",
        f"- task_id: {receipt.get('input', {}).get('task_id')}",
        f"- No mutations performed: {str(receipt.get('public_safety', {}).get('no_mutations_performed')).lower()}",
        "",
        "## Gate checks",
    ]
    for name, check in receipt.get("gate_checks", {}).items():
        lines.append(f"- {name}: {check.get('status')}")
    lines.extend(["", "## Gaps"])
    gaps = receipt.get("gaps", [])
    if gaps:
        for gap in gaps:
            lines.append(f"- {gap.get('code')}: {gap.get('message')} -> {gap.get('next_owner')}/{gap.get('next_action')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Evidence URLs"])
    urls = receipt.get("evidence_urls", [])
    if urls:
        for url in urls:
            lines.append(f"- {url}")
    else:
        lines.append("- none")
    next_step = receipt.get("next", {})
    lines.extend(
        [
            "",
            "## Next",
            f"- owner: {next_step.get('owner')}",
            f"- action: {next_step.get('action')}",
            f"- suggested_next_step: {next_step.get('suggested_next_step')}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_receipt(receipt: dict[str, Any], *, receipt_file: Optional[str], json_output: bool) -> None:
    encoded = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    if receipt_file:
        Path(receipt_file).expanduser().write_text(encoded + "\n", encoding="utf-8")
    if json_output:
        print(encoded)
    else:
        print(f"{ENTRYPOINT}: closeout_gate={receipt['closeout_gate']} gaps={len(receipt['gaps'])}; no mutations performed")
        if receipt_file:
            print(f"receipt: {receipt_file}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=ENTRYPOINT,
        description="Read-only final closeout gate over TianGongKaiWu manifests and #161 guard receipts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="evaluate closeout gate and emit JSON/Markdown evidence")
    check.add_argument("--issue", help="GitHub issue number or #N")
    check.add_argument("--pr", help="GitHub PR number")
    check.add_argument("--task-id", help="Kanban task id being checked")
    check.add_argument("--repo", default="GTZhou/TianGongKaiWu", help="GitHub repo, default GTZhou/TianGongKaiWu")
    check.add_argument("--board", default="tiangongkaiwu", help="Kanban board slug")
    check.add_argument("--manifest", action="append", default=[], help="YAML/JSON/Markdown manifest file; repeatable")
    check.add_argument("--guard-receipt", action="append", default=[], help="#161 duplicate-child guard receipt JSON/YAML; repeatable")
    check.add_argument("--current-head", help="override current PR/head/artifact id for head-unchanged check")
    check.add_argument("--public-text", action="append", default=[], help="additional public text to scan; repeatable")
    check.add_argument("--public-text-file", action="append", default=[], help="file whose contents are public text to scan; repeatable")
    check.add_argument("--actor-profile", help="expected profile for runtime/profile smoke evidence")
    check.add_argument("--receipt-file", help="write JSON receipt to this path")
    check.add_argument("--markdown-report-file", help="write Markdown report to this path")
    check.add_argument("--json", action="store_true", help="print full JSON receipt to stdout")

    doctor = sub.add_parser("doctor", help="emit no-side-effect runtime/profile smoke evidence")
    doctor.add_argument("--actor-profile", help="expected worker/profile name")
    doctor.add_argument("--json", action="store_true", help="print JSON instead of one-line summary")
    return parser


def _read_public_text_files(paths: Iterable[str]) -> list[str]:
    texts = []
    for raw_path in paths:
        texts.append(Path(raw_path).expanduser().read_text(encoding="utf-8"))
    return texts


def _doctor_receipt(actor_profile: Optional[str]) -> dict[str, Any]:
    receipt = {
        "schema": SCHEMA_VERSION,
        "mode": "doctor",
        "ok": True,
        "generated_at": _utc_now(),
        "runtime_profile_smoke": _runtime_profile_smoke(actor_profile),
        "public_safety": {
            "no_mutations_performed": True,
            "write_targets": [],
            "merge_performed": False,
            "issue_closed": False,
            "label_transition_performed": False,
            "trace_sent": False,
            "credentials_or_tokens_touched": False,
        },
    }
    return _sanitize_public_receipt(receipt)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "doctor":
        receipt = _doctor_receipt(args.actor_profile)
        if args.json:
            print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            smoke = receipt["runtime_profile_smoke"]
            print(
                f"{ENTRYPOINT}: profile={smoke['actual_profile'] or '(unset)'} "
                f"expected={smoke['expected_actor_profile'] or '(none)'} "
                f"match={smoke['actor_profile_match']} no mutations performed"
            )
        return 0

    try:
        manifests = load_documents(args.manifest)
        guard_receipts = load_documents(args.guard_receipt)
        public_text = list(args.public_text or []) + _read_public_text_files(args.public_text_file or [])
        receipt = build_closeout_gate_receipt(
            manifests=manifests,
            guard_receipts=guard_receipts,
            issue=args.issue,
            pr=args.pr,
            task_id=args.task_id,
            repo=args.repo,
            board=args.board,
            current_head=args.current_head,
            public_text=public_text,
            actor_profile=args.actor_profile,
        )
        _write_receipt(receipt, receipt_file=args.receipt_file, json_output=args.json)
        if args.markdown_report_file:
            Path(args.markdown_report_file).expanduser().write_text(render_markdown_report(receipt), encoding="utf-8")
        return 0 if receipt["ok"] else 1
    except Exception as exc:
        error_receipt = _sanitize_public_receipt(
            {
                "schema": SCHEMA_VERSION,
                "mode": "check",
                "ok": False,
                "closeout_gate": "failed",
                "error": str(exc),
                "public_safety": {"no_mutations_performed": True, "write_targets": []},
            }
        )
        if getattr(args, "json", False):
            print(json.dumps(error_receipt, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        else:
            print(f"{ENTRYPOINT}: {error_receipt['error']}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
