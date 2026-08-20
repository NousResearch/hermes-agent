#!/usr/bin/env python3
"""Canonical fail-closed privacy context, preflight, and egress decisions.

Repository guarantee only. Host/provider retention remains NOT_COMPUTABLE.
Never grants execution authority. Never offers a global privacy-disable switch.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "1.0.0"
SUPPORTED_CONTRACT_VERSIONS = frozenset({"1.0.0"})
DECISIONS = ("ALLOW", "REDACT_THEN_ALLOW", "REQUEST_CONSENT", "BLOCK")
MODES = frozenset({"local_only", "external_research_allowed", "custom"})
SOURCE_CLASSES = frozenset(
    {
        "operator_supplied",
        "workspace_context",
        "public_research",
        "model_prior",
        "host_tool",
        "derived",
        "unknown",
    }
)
CLASSIFICATIONS = frozenset(
    {
        "public",
        "private",
        "credential",
        "secret",
        "mixed",
        "unknown",
    }
)
REDACTION_STATUSES = frozenset(
    {
        "none",
        "not_required",
        "applied",
        "blocked",
        "consent_required",
    }
)
EXTERNAL_ACTION_REQUIRED = (
    "provider",
    "tool_class",
    "destination",
    "purpose",
    "source_class",
    "classification",
    "decision",
    "redaction_status",
    "recorded_at",
)
# Purpose-bound consent only — never global_disable / disable_privacy.
CONSENT_REQUIRED = (
    "consent_id",
    "scope",
    "purpose",
    "categories_allowed",
    "source_class",
    "issued_at",
)
FORBIDDEN_CONSENT_SCOPES = frozenset(
    {
        "global",
        "global_disable",
        "disable_privacy",
        "all",
        "unrestricted",
        "bypass",
    }
)

_FINDERS = (
    ("private_key", "high", r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----", "block_egress"),
    ("github_token", "high", r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b", "block_egress"),
    ("api_key", "high", r"\b(?:sk|pk)_[A-Za-z0-9]{20,}\b", "block_egress"),
    ("bearer_token", "high", r"\bBearer\s+[A-Za-z0-9._~-]{16,}\b", "block_egress"),
    (
        "connection_string",
        "high",
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s]+",
        "block_egress",
    ),
    ("payment_card", "high", r"\b(?:\d[ -]*?){13,16}\b", "block_egress"),
)

_REDACT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("PRIVATE_KEY", re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----.*?-----END (?:[A-Z ]+ )?PRIVATE KEY-----", re.S | re.I)),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b", re.I)),
    ("API_KEY", re.compile(r"\b(?:sk|pk)_[A-Za-z0-9]{20,}\b", re.I)),
    ("BEARER", re.compile(r"\bBearer\s+[A-Za-z0-9._~-]{16,}\b", re.I)),
    ("CONNECTION", re.compile(r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis)://[^\s]+", re.I)),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b")),
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def assert_contract_version(version: str | None) -> None:
    if version is None:
        raise ValueError("NG-PRIVACY-010: privacy contract_version required")
    if version not in SUPPORTED_CONTRACT_VERSIONS:
        raise ValueError(
            f"NG-PRIVACY-011: unsupported privacy contract_version {version!r}; "
            f"supported={sorted(SUPPORTED_CONTRACT_VERSIONS)}"
        )


def default_privacy_context(
    artifact_path: str = "out/neon-genie/example",
    mode: str = "local_only",
    *,
    approved_domains: list[str] | None = None,
    approved_tool_classes: list[str] | None = None,
    consents: list[dict[str, Any]] | None = None,
    data_sources_used: list[dict[str, Any]] | None = None,
    external_actions: list[dict[str, Any]] | None = None,
    privacy_warnings: list[str] | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError("privacy mode must be local_only, external_research_allowed, or custom")
    allowed = mode == "external_research_allowed"
    default_tools = ["web_search"] if allowed else []
    tools = list(approved_tool_classes) if approved_tool_classes is not None else default_tools
    domains = list(approved_domains or [])
    validated_consents = [validate_consent_record(c) for c in (consents or [])]
    return {
        "contract_version": CONTRACT_VERSION,
        "mode": mode,
        "telemetry": "disabled",
        "training_use": "none_by_neon_genie",
        "retention": {
            "artifact_path": artifact_path,
            "automatic_expiry": None,
            "host_provider_retention": "NOT_COMPUTABLE",
            "repository_retention": "operator_selected_output_path",
        },
        "egress": {
            "allowed": allowed,
            "approved_domains": domains,
            "approved_tool_classes": tools,
        },
        "redaction": {
            "enabled": True,
            "blocked_categories": ["credentials", "secrets", "government_ids", "private_keys"],
        },
        "consents": validated_consents,
        "data_sources_used": list(data_sources_used or []),
        "external_actions": list(external_actions or []),
        "privacy_warnings": list(privacy_warnings or []),
        "deletion_instructions": f"Delete the operator-selected output directory: {artifact_path}.",
        "unknowns": {
            "host_providers": "NOT_COMPUTABLE",
            "provider_retention": "NOT_COMPUTABLE",
            "provider_processing": "NOT_COMPUTABLE",
        },
        "learning_ledger": {
            "location": "local_operator_selected_path",
            "default_path": "out/neon-genie/learning-ledger.jsonl",
            "auto_apply_forbidden": True,
            "ships_with_skill": False,
            "disclosure": (
                "Learning ledger entries are local operator artifacts under the chosen "
                "output path; they are never auto-applied to the skill corpus and must "
                "not contain raw secrets or private source payloads."
            ),
        },
    }


def privacy_findings(payload: str) -> list[dict[str, str]]:
    """Return classifications only — never include matched secret substrings."""
    findings: list[dict[str, str]] = []
    for category, confidence, pattern, action in _FINDERS:
        for match in re.finditer(pattern, payload, re.IGNORECASE):
            findings.append(
                {
                    "category": category,
                    "confidence": confidence,
                    "action": action,
                    "location": f"character:{match.start()}-{match.end()}",
                }
            )
    emails = list(re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", payload))
    if len(emails) >= 3:
        findings.append(
            {
                "category": "private_email_list",
                "confidence": "high",
                "action": "request_consent",
                "location": "multiple_email_addresses",
            }
        )
    elif emails:
        findings.append(
            {
                "category": "email_address",
                "confidence": "medium",
                "action": "redact",
                "location": f"character:{emails[0].start()}-{emails[-1].end()}",
            }
        )
    # Customer-list style signals (names/emails/phones enumerations)
    if re.search(r"\b(customer\s+list|crm\s+export|mailing\s+list)\b", payload, re.I):
        findings.append(
            {
                "category": "private_customer_list",
                "confidence": "high",
                "action": "request_consent",
                "location": "customer_list_marker",
            }
        )
    return findings


def redact_payload(payload: str) -> dict[str, Any]:
    """Produce a minimized query representation without private source content.

    Secrets and private identifiers are replaced with category tokens. The
    original private source content is never returned and must not be persisted.
    """
    text = payload
    categories: list[str] = []
    for label, pattern in _REDACT_PATTERNS:
        if pattern.search(text):
            categories.append(label.lower())
            text = pattern.sub(f"[{label}_REDACTED]", text)
    # Collapse whitespace for a stable minimized query
    minimized = re.sub(r"\s+", " ", text).strip()
    applied = bool(categories)
    return {
        "safe_query": minimized,
        "redaction_applied": applied,
        "redacted_categories": sorted(set(categories)),
        "content_fingerprint": "sha256:"
        + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        # Never store original payload.
        "private_source_persisted": False,
    }


def _consent_matches(
    consent: dict[str, Any],
    *,
    purpose: str,
    destination: str | None,
    category: str | None = None,
) -> bool:
    if consent.get("revoked") is True:
        return False
    expires = consent.get("expires_at")
    if expires:
        # Lexicographic ISO-Z comparison is safe for our timestamp format.
        if str(expires) < utc_now():
            return False
    c_purpose = str(consent.get("purpose") or "").strip().lower()
    if c_purpose and c_purpose not in purpose.lower() and purpose.lower() not in c_purpose:
        # Require purpose substring match either direction for purpose-bound scope.
        return False
    dests = consent.get("destinations") or []
    if dests and destination and destination not in dests:
        return False
    allowed = set(consent.get("categories_allowed") or [])
    denied = set(consent.get("categories_denied") or [])
    if category and category in denied:
        return False
    if category and allowed and category not in allowed and "public_query" not in allowed:
        return False
    return True


def find_matching_consent(
    privacy_context: dict[str, Any],
    *,
    purpose: str,
    destination: str | None,
    category: str | None = None,
) -> dict[str, Any] | None:
    for consent in privacy_context.get("consents") or []:
        if _consent_matches(consent, purpose=purpose, destination=destination, category=category):
            return consent
    return None


def privacy_egress_check(
    payload: str,
    destination: str | None,
    purpose: str | None,
    privacy_context: dict[str, Any],
) -> str:
    """Return one decision; callers must never execute BLOCK/REQUEST_CONSENT."""
    if not destination or not purpose:
        return "BLOCK"
    assert_contract_version(privacy_context.get("contract_version"))
    egress = privacy_context.get("egress") or {}
    if privacy_context.get("mode") == "local_only" or not egress.get("allowed"):
        return "BLOCK"
    findings = privacy_findings(payload)
    if any(f["action"] == "block_egress" for f in findings):
        return "BLOCK"
    consent_needed = [f for f in findings if f["action"] == "request_consent"]
    if consent_needed:
        # Require purpose-bound consent covering the private category.
        ok = True
        for f in consent_needed:
            if not find_matching_consent(
                privacy_context,
                purpose=purpose,
                destination=destination,
                category=f["category"],
            ):
                ok = False
                break
        if not ok:
            return "REQUEST_CONSENT"
        # Even with consent, never send raw private list content.
        return "REDACT_THEN_ALLOW"
    if any(f["action"] == "redact" for f in findings):
        return "REDACT_THEN_ALLOW"
    domains = egress.get("approved_domains") or []
    if domains and destination not in domains:
        return "BLOCK"
    return "ALLOW"


def prepare_egress(
    payload: str,
    destination: str | None,
    purpose: str | None,
    privacy_context: dict[str, Any],
    *,
    provider: str = "host",
    tool_class: str = "web_search",
    source_class: str = "public_research",
    recorded_at: str | None = None,
) -> dict[str, Any]:
    """Decide + optionally build a safe minimized query (never private source)."""
    decision = privacy_egress_check(payload, destination, purpose, privacy_context)
    findings = privacy_findings(payload)
    # Never include raw secret text in the result.
    safe_findings = [
        {k: v for k, v in f.items() if k != "match"} for f in findings
    ]
    result: dict[str, Any] = {
        "decision": decision,
        "findings": safe_findings,
        "safe_query": None,
        "redaction": None,
        "external_action": None,
    }
    if decision == "BLOCK":
        result["redaction"] = {"redaction_status": "blocked", "private_source_persisted": False}
        return result
    if decision == "REQUEST_CONSENT":
        result["redaction"] = {
            "redaction_status": "consent_required",
            "private_source_persisted": False,
        }
        return result

    redaction = redact_payload(payload)
    if decision == "REDACT_THEN_ALLOW":
        result["safe_query"] = redaction["safe_query"]
        result["redaction"] = {
            **redaction,
            "redaction_status": "applied",
        }
        classification = "mixed"
        redaction_status = "applied"
    else:
        # ALLOW: still never persist private source; safe_query is the payload if clean.
        result["safe_query"] = redaction["safe_query"] if redaction["redaction_applied"] else payload.strip()
        result["redaction"] = {
            **redaction,
            "redaction_status": "applied" if redaction["redaction_applied"] else "not_required",
        }
        classification = "public"
        redaction_status = result["redaction"]["redaction_status"]

    action = build_external_action(
        provider=provider,
        tool_class=tool_class,
        destination=str(destination),
        purpose=str(purpose),
        source_class=source_class,
        classification=classification,
        decision=decision,
        redaction_status=redaction_status,
        recorded_at=recorded_at,
        safe_query=result["safe_query"],
    )
    result["external_action"] = action
    return result


def validate_consent_record(consent: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(consent, dict):
        raise ValueError("NG-PRIVACY-020: consent must be an object")
    missing = [k for k in CONSENT_REQUIRED if k not in consent or consent.get(k) in (None, "")]
    if missing:
        raise ValueError(f"NG-PRIVACY-021: consent missing required fields: {', '.join(missing)}")
    scope = str(consent.get("scope") or "").strip().lower()
    if scope in FORBIDDEN_CONSENT_SCOPES:
        raise ValueError(
            "NG-PRIVACY-022: global privacy disable / unrestricted consent is forbidden; "
            "use purpose_bound consent only"
        )
    if scope not in {"purpose_bound", "purpose-bound", "single_purpose"}:
        raise ValueError(
            "NG-PRIVACY-023: consent.scope must be purpose_bound (never a global disable switch)"
        )
    # Normalize scope spelling
    out = dict(consent)
    out["scope"] = "purpose_bound"
    if "categories_allowed" not in out or not isinstance(out["categories_allowed"], list):
        raise ValueError("NG-PRIVACY-024: consent.categories_allowed must be an array")
    if "revoked" not in out:
        out["revoked"] = False
    return out


def build_consent_record(
    *,
    purpose: str,
    categories_allowed: list[str],
    destinations: list[str] | None = None,
    categories_denied: list[str] | None = None,
    source_class: str = "operator_explicit",
    issued_at: str | None = None,
    expires_at: str | None = None,
    consent_id: str | None = None,
) -> dict[str, Any]:
    """Construct a purpose-bound consent record (never global disable)."""
    cid = consent_id or "consent_" + hashlib.sha256(
        f"{purpose}|{','.join(categories_allowed)}|{issued_at or ''}".encode()
    ).hexdigest()[:12]
    record = {
        "consent_id": cid,
        "scope": "purpose_bound",
        "purpose": purpose,
        "destinations": list(destinations or []),
        "categories_allowed": list(categories_allowed),
        "categories_denied": list(categories_denied or ["credentials", "secrets", "private_keys"]),
        "source_class": source_class,
        "issued_at": issued_at or utc_now(),
        "expires_at": expires_at,
        "revoked": False,
    }
    return validate_consent_record(record)


def build_external_action(
    *,
    provider: str,
    tool_class: str,
    destination: str,
    purpose: str,
    source_class: str,
    classification: str,
    decision: str,
    redaction_status: str,
    recorded_at: str | None = None,
    safe_query: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if decision not in {"ALLOW", "REDACT_THEN_ALLOW"}:
        raise ValueError(
            f"NG-PRIVACY-030: external action decision must be ALLOW or REDACT_THEN_ALLOW, got {decision!r}"
        )
    if not provider or not tool_class or not destination or not purpose:
        raise ValueError(
            "NG-PRIVACY-031: external action requires provider, tool_class, destination, purpose"
        )
    if source_class not in SOURCE_CLASSES:
        raise ValueError(f"NG-PRIVACY-032: invalid source_class {source_class!r}")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"NG-PRIVACY-033: invalid classification {classification!r}")
    if redaction_status not in REDACTION_STATUSES:
        raise ValueError(f"NG-PRIVACY-034: invalid redaction_status {redaction_status!r}")
    ts = recorded_at or utc_now()
    # Deterministic timestamp format check
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts):
        raise ValueError(
            f"NG-PRIVACY-035: recorded_at must be UTC ISO Zulu YYYY-MM-DDTHH:MM:SSZ, got {ts!r}"
        )
    action: dict[str, Any] = {
        "provider": provider,
        "tool_class": tool_class,
        "destination": destination,
        "purpose": purpose,
        "source_class": source_class,
        "classification": classification,
        "decision": decision,
        "egress_decision": decision,  # alias for envelope assert
        "redaction_status": redaction_status,
        "recorded_at": ts,
    }
    if safe_query is not None:
        # safe_query only — never private source content
        action["safe_query"] = safe_query
        action["private_source_persisted"] = False
    if extra:
        for k, v in extra.items():
            if k in action:
                continue
            action[k] = v
    return validate_external_action(action)


def validate_external_action(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise ValueError("NG-PRIVACY-036: external_action must be an object")
    missing = [k for k in EXTERNAL_ACTION_REQUIRED if k not in action or action.get(k) in (None, "")]
    if missing:
        raise ValueError(
            f"NG-PRIVACY-037: external_action missing required fields: {', '.join(missing)}"
        )
    decision = action.get("decision") or action.get("egress_decision")
    if decision not in {"ALLOW", "REDACT_THEN_ALLOW"}:
        raise ValueError("NG-PRIVACY-038: external_action.decision must be ALLOW or REDACT_THEN_ALLOW")
    if action.get("source_class") not in SOURCE_CLASSES:
        raise ValueError(f"NG-PRIVACY-032: invalid source_class {action.get('source_class')!r}")
    if action.get("classification") not in CLASSIFICATIONS:
        raise ValueError(f"NG-PRIVACY-033: invalid classification {action.get('classification')!r}")
    if action.get("redaction_status") not in REDACTION_STATUSES:
        raise ValueError(
            f"NG-PRIVACY-034: invalid redaction_status {action.get('redaction_status')!r}"
        )
    ts = str(action.get("recorded_at") or "")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", ts):
        raise ValueError(
            f"NG-PRIVACY-035: recorded_at must be UTC ISO Zulu YYYY-MM-DDTHH:MM:SSZ, got {ts!r}"
        )
    # Ensure alias consistency
    out = dict(action)
    out["decision"] = decision
    out["egress_decision"] = decision
    if "private_source_persisted" not in out:
        out["private_source_persisted"] = False
    if out.get("private_source_persisted") is True:
        raise ValueError("NG-PRIVACY-039: external_action must not persist private source content")
    return out


def assert_envelope_privacy(privacy: dict[str, Any]) -> None:
    assert_contract_version(privacy.get("contract_version"))
    if privacy.get("mode") not in MODES:
        raise ValueError("NG-PRIVACY-012: invalid privacy mode")
    if privacy.get("telemetry") != "disabled":
        raise ValueError("NG-PRIVACY-013: repository telemetry must be disabled")
    if privacy.get("mode") == "local_only" and privacy.get("external_actions"):
        raise ValueError("NG-PRIVACY-001: local_only envelopes cannot contain external_actions")
    for consent in privacy.get("consents") or []:
        validate_consent_record(consent)
    for action in privacy.get("external_actions") or []:
        if action.get("egress_decision") not in {"ALLOW", "REDACT_THEN_ALLOW"} and action.get(
            "decision"
        ) not in {"ALLOW", "REDACT_THEN_ALLOW"}:
            raise ValueError("NG-PRIVACY-002: unapproved external action")
        validate_external_action(action)
        # Fail closed if raw secret-like material appears in serialized action.
        blob = json.dumps(action, sort_keys=True)
        if privacy_findings(blob) and any(
            f["action"] == "block_egress" for f in privacy_findings(blob)
        ):
            raise ValueError("NG-PRIVACY-040: external_action contains secret-like material")


def privacy_receipt_fields(privacy: dict[str, Any]) -> dict[str, Any]:
    """Flat privacy fields mirrored onto receipt/envelope for operator visibility."""
    return {
        "privacy": privacy,
        "privacy_mode": privacy["mode"],
        "data_sources_used": privacy.get("data_sources_used") or [],
        "external_actions": privacy.get("external_actions") or [],
        "artifact_paths": [privacy.get("retention", {}).get("artifact_path") or ""],
        "telemetry_status": privacy.get("telemetry") or "disabled",
        "retention_statement": (
            "Artifacts remain at the operator-selected output location. "
            "Host/provider retention is NOT_COMPUTABLE."
        ),
        "privacy_warnings": privacy.get("privacy_warnings") or [],
        "deletion_instructions": privacy.get("deletion_instructions")
        or "Delete the operator-selected output directory.",
        "privacy_unknowns": privacy.get("unknowns")
        or {
            "host_providers": "NOT_COMPUTABLE",
            "provider_retention": "NOT_COMPUTABLE",
            "provider_processing": "NOT_COMPUTABLE",
        },
        "learning_ledger_disclosure": (privacy.get("learning_ledger") or {}).get("disclosure"),
    }


def learning_ledger_privacy_fields(privacy: dict[str, Any] | None) -> dict[str, Any]:
    """Minimal privacy disclosure for learning-ledger entries (no private payloads)."""
    ctx = privacy or default_privacy_context()
    return {
        "privacy_mode": ctx.get("mode"),
        "privacy_contract_version": ctx.get("contract_version"),
        "privacy_telemetry": ctx.get("telemetry"),
        "learning_local_only": True,
        "learning_auto_apply_forbidden": True,
        "learning_disclosure": (ctx.get("learning_ledger") or {}).get("disclosure")
        or default_privacy_context()["learning_ledger"]["disclosure"],
        "host_provider_retention": "NOT_COMPUTABLE",
    }


def _strip_yaml_scalar(raw: str) -> Any:
    s = raw.strip()
    if not s or s == "~" or s == "null":
        return None
    if s.startswith("#"):
        return None
    # drop inline comments when not inside quotes
    if s[0] not in "'\"" and " #" in s:
        s = s.split(" #", 1)[0].rstrip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s in {"true", "True"}:
        return True
    if s in {"false", "False"}:
        return False
    return s


def _parse_indent_yaml(text: str) -> dict[str, Any]:
    """Minimal indented YAML subset (maps + string/bool lists) — stdlib only."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        # count leading spaces (tabs → 2 spaces)
        expanded = raw.replace("\t", "  ")
        indent = len(expanded) - len(expanded.lstrip(" "))
        lines.append((indent, expanded.lstrip(" ")))

    def parse_block(start: int, base_indent: int) -> tuple[Any, int]:
        if start >= len(lines):
            return {}, start
        # list block?
        if lines[start][1].startswith("- ") and lines[start][0] == base_indent:
            items: list[Any] = []
            i = start
            while i < len(lines) and lines[i][0] == base_indent and lines[i][1].startswith("- "):
                item_body = lines[i][1][2:].strip()
                if not item_body:
                    # nested map under list item
                    child, i = parse_block(i + 1, base_indent + 2)
                    items.append(child)
                    continue
                if ":" in item_body and not item_body.startswith("http"):
                    # inline key: value starting a map item
                    key, _, rest = item_body.partition(":")
                    node: dict[str, Any] = {key.strip(): _strip_yaml_scalar(rest)}
                    i += 1
                    while i < len(lines) and lines[i][0] > base_indent:
                        if lines[i][0] == base_indent + 2 and not lines[i][1].startswith("- "):
                            k, _, r = lines[i][1].partition(":")
                            k = k.strip()
                            r = r.strip()
                            if r == "" or r is None:
                                # nested
                                nested, i = parse_block(i + 1, lines[i][0] + 2)
                                node[k] = nested
                            else:
                                # could be list marker on following lines
                                if i + 1 < len(lines) and lines[i + 1][0] > lines[i][0] and lines[i + 1][1].startswith("- "):
                                    nested, i = parse_block(i + 1, lines[i + 1][0])
                                    node[k] = nested
                                else:
                                    node[k] = _strip_yaml_scalar(r)
                                    i += 1
                        elif lines[i][1].startswith("- ") and lines[i][0] >= base_indent + 2:
                            # shouldn't normally hit for map item values without key
                            break
                        else:
                            break
                    items.append(node)
                    continue
                items.append(_strip_yaml_scalar(item_body))
                i += 1
            return items, i

        # mapping block
        result: dict[str, Any] = {}
        i = start
        while i < len(lines) and lines[i][0] == base_indent and not lines[i][1].startswith("- "):
            line = lines[i][1]
            if ":" not in line:
                i += 1
                continue
            key, _, rest = line.partition(":")
            key = key.strip()
            rest = rest.strip()
            if rest != "":
                result[key] = _strip_yaml_scalar(rest)
                i += 1
            else:
                # nested block
                if i + 1 >= len(lines) or lines[i + 1][0] <= base_indent:
                    result[key] = {}
                    i += 1
                else:
                    child_indent = lines[i + 1][0]
                    nested, i = parse_block(i + 1, child_indent)
                    result[key] = nested
        return result, i

    root, _ = parse_block(0, lines[0][0] if lines else 0)
    return root if isinstance(root, dict) else {}


def _parse_simple_yaml_map(text: str) -> dict[str, Any]:
    """Parse brief YAML for privacy/research (stdlib first; optional PyYAML)."""
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        return _parse_indent_yaml(text)
    except Exception:
        # last-resort regex extraction
        result: dict[str, Any] = {}
        m = re.search(r"(?m)^privacy:\s*$", text)
        if m:
            block = text[m.end() :]
            next_top = re.search(r"(?m)^[a-zA-Z_]", block)
            if next_top:
                block = block[: next_top.start()]
            mode_m = re.search(r"(?m)^\s+mode:\s*[\"']?([a-z_]+)[\"']?", block)
            if mode_m:
                result.setdefault("privacy", {})["mode"] = mode_m.group(1)
            ad_m = re.search(r"approved_domains:\s*\n((?:\s+-\s+.+\n?)*)", block)
            if ad_m:
                domains = [
                    re.sub(r'^\s*-\s*[\"\']?|[\"\']\s*$', "", ln).strip()
                    for ln in ad_m.group(1).splitlines()
                    if ln.strip().startswith("-")
                ]
                result.setdefault("privacy", {})["approved_domains"] = [d for d in domains if d]
            tools_m = re.search(r"approved_tool_classes:\s*\n((?:\s+-\s+.+\n?)*)", block)
            if tools_m:
                tools = [
                    re.sub(r'^\s*-\s*[\"\']?|[\"\']\s*$', "", ln).strip()
                    for ln in tools_m.group(1).splitlines()
                    if ln.strip().startswith("-")
                ]
                result.setdefault("privacy", {})["approved_tool_classes"] = [t for t in tools if t]
        rm = re.search(r"(?m)^research:\s*\n((?:\s+.+\n?)*)", text)
        if rm:
            rblock = rm.group(1)
            research: dict[str, Any] = {}
            em = re.search(r"(?m)^\s+enabled:\s*(true|false)", rblock)
            if em:
                research["enabled"] = em.group(1) == "true"
            om = re.search(r"(?m)^\s+offline:\s*(true|false)", rblock)
            if om:
                research["offline"] = om.group(1) == "true"
            if research:
                result["research"] = research
        return result



def resolve_privacy_config_from_brief(
    brief: dict[str, Any] | str | Path | None,
    *,
    artifact_path: str = "out/neon-genie/example",
) -> dict[str, Any]:
    """Resolve deterministic privacy context from a brief mapping or YAML path/text."""
    data: dict[str, Any] = {}
    if brief is None:
        return default_privacy_context(artifact_path)
    if isinstance(brief, Path):
        text = brief.read_text(encoding="utf-8")
        data = _parse_simple_yaml_map(text)
        # Prefer full parse if available
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            pass
    elif isinstance(brief, str):
        # path or raw yaml
        p = Path(brief)
        if p.is_file():
            return resolve_privacy_config_from_brief(p, artifact_path=artifact_path)
        data = _parse_simple_yaml_map(brief)
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(brief)
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            pass
    elif isinstance(brief, dict):
        data = brief
    else:
        return default_privacy_context(artifact_path)

    privacy_cfg = data.get("privacy") if isinstance(data.get("privacy"), dict) else {}
    research = data.get("research") if isinstance(data.get("research"), dict) else {}

    mode = privacy_cfg.get("mode")
    # Offline / research disabled always force local_only (fail closed).
    if research.get("offline") is True or research.get("enabled") is False:
        mode = "local_only"
    if mode is None:
        # Default repository boundary remains local_only even if research.enabled=true
        # (research is host-exposed; packaging does not auto-elevate egress).
        mode = "local_only"
    if mode not in MODES:
        raise ValueError(f"NG-PRIVACY-050: invalid privacy.mode in brief: {mode!r}")

    # Reject global disable flags if present
    if privacy_cfg.get("disable") or privacy_cfg.get("disabled") or privacy_cfg.get("bypass"):
        raise ValueError(
            "NG-PRIVACY-051: global privacy disable is forbidden; use purpose_bound consents"
        )
    if str(privacy_cfg.get("scope") or "").lower() in FORBIDDEN_CONSENT_SCOPES:
        raise ValueError("NG-PRIVACY-051: global privacy disable is forbidden")

    consents_raw = privacy_cfg.get("consents") or []
    consents = [validate_consent_record(c) for c in consents_raw]

    return default_privacy_context(
        artifact_path,
        mode=str(mode),
        approved_domains=privacy_cfg.get("approved_domains"),
        approved_tool_classes=privacy_cfg.get("approved_tool_classes"),
        consents=consents,
        privacy_warnings=list(privacy_cfg.get("privacy_warnings") or []),
    )


def merge_privacy_context(
    base: dict[str, Any],
    *,
    external_actions: list[dict[str, Any]] | None = None,
    data_sources_used: list[dict[str, Any]] | None = None,
    privacy_warnings: list[str] | None = None,
) -> dict[str, Any]:
    out = dict(base)
    if external_actions is not None:
        out["external_actions"] = [validate_external_action(a) for a in external_actions]
    if data_sources_used is not None:
        out["data_sources_used"] = list(data_sources_used)
    if privacy_warnings is not None:
        out["privacy_warnings"] = list(privacy_warnings)
    assert_envelope_privacy(out)
    return out


def enumerate_external_actions(privacy: dict[str, Any]) -> list[dict[str, Any]]:
    """Return validated external action enumeration for receipts (empty if none)."""
    actions = []
    for a in privacy.get("external_actions") or []:
        actions.append(validate_external_action(a))
    return actions


def sanitize_error_message(message: str) -> str:
    """Strip secret-like material from errors/logs before emission."""
    red = redact_payload(message)
    return red["safe_query"]


def assert_no_raw_secrets(blob: str, *, context: str = "artifact") -> None:
    """Fail if credential-like material appears in serialized artifacts.

    Payment-card digit runs are excluded from full-document scans (too many
    false positives on hashes/ids); card checks still apply to egress payloads.
    """
    findings = privacy_findings(blob)
    credential_cats = {
        "private_key",
        "github_token",
        "api_key",
        "bearer_token",
        "connection_string",
    }
    if any(f["action"] == "block_egress" and f["category"] in credential_cats for f in findings):
        raise ValueError(
            f"NG-PRIVACY-060: raw secret-like material must not appear in {context}"
        )
