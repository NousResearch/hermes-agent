#!/usr/bin/env python3
"""Reusable HASOS policy engine for Hermes Harness runtime.

This module is intentionally side-effect-light: it classifies payloads, redacts
secret-like data, returns structured policy decisions, and writes sanitized audit
reports. Runtime hooks call this module before deciding whether to allow or block.
Boundary: this is a Hermes policy engine, not an OS sandbox.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "hasos.policy_decision.v1"

SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|passwd|pwd|private[_-]?key|auth|credential|cookie|jwt|p8|pem)")
SAFE_NONSECRET_KEYS = {
    "session_id",
    "secrets_redacted",
    "redacted_secret_present",
    "redacted_marker_present",
    "secret_redaction_check",
}
SECRET_VAL_RE = re.compile(
    r"(?ix)("
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"AIza[0-9A-Za-z_-]{20,}|"
    r"TEST_FAKE_VALUE_[A-Za-z0-9_-]{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----"
    r")",
    re.S,
)

DANGEROUS_TERMINAL_RE = re.compile(
    r"(?ix)("
    # Broad destructive deletion patterns.  These are deliberately conservative
    # for Hermes tool-call runtime: even user-home or absolute-path recursive
    # force deletes should require a safer, explicit scoped workflow.
    r"\brm\s+-(?=[^\n;]*r)(?=[^\n;]*f)[^\n;]*\s+(?:\\?['\"]?\$HOME\\?['\"]?|~/?|/(?:$|[^\s;]+)|(?:\\?['\"]?)?(?:\.|\.\.|\.\/[^\s;\\?'\"]*|\.\/\*|\.\.\/[^\s;\\?'\"]*|\*)\\?['\"]?(?=$|[\s,;}]))|"
    r"\bsudo\b|"
    r"\bchmod\s+-R\s+777\b|"
    r"\bgit\s+push\s+--force(?:-with-lease)?\b|"
    r"\bgit\s+reset\s+--hard\b|"
    r"\bgit\s+clean\s+-f?fdx\b|"
    r"\bsecurity\s+find-generic-password\b|"
    r"\bcat\s+[^\n;]*(\.env|auth\.json|\.p8|id_rsa|\.pem)\b|"
    r"\bcurl\b[^\n;]*(\.env|auth\.json|\.p8|id_rsa|token|secret|password)|"
    r"sh\s+-c\s+['\"][^'\"]*\brm\s+-(?=[^'\"]*r)(?=[^'\"]*f)[^'\"]*(?:/|~|\$HOME)|"
    r"python\s+-c\s+[^\n]*(?:subprocess\.run|os\.system|Path\([^\)]*\)\.unlink)|"
    r"python\s+-c\s+['\"][^'\"]*(shutil\.rmtree|os\.remove|os\.unlink|os\.system|subprocess\.run|Path\([^\)]*\)\.unlink|rm\s*-\s*rf|['\"]rm['\"]\s*,\s*['\"]-[a-z]*f[a-z]*['\"])|"
    r"shutil\.rmtree\s*\(|"
    r"\bfind\b[^\n;]*\s-delete\b|"
    r"\bxargs\s+rm\b|"
    r"\bcurl\b[^\n;]*\|\s*(?:sh|bash)\b|"
    r"\$\([^\)]*\brm\s+-rf[^\)]*\)|"
    r"\bosascript\b[^\n;]*(?:delete|erase|rm\s+-rf)|"
    r"license\s*(bypass|crack)|security\s*bypass|leaked\s+proprietary|unauthorized\s+claude\s+code"
    r")"
)
RELEASE_OR_UPLOAD_RE = re.compile(
    r"(?ix)("
    r"\bfastlane\s+(deliver|pilot|upload_to_app_store|precheck.*--submit)|"
    r"\bxcrun\s+(altool|notarytool)|"
    r"\bnotarytool\s+submit\b|"
    r"\btransporter\b|iTMSTransporter|"
    r"submit\s+for\s+review|external\s+testflight|release\s+to\s+public|appstoreconnect[^\n]*(submit|release|upload)|"
    r"\bgh\s+release\s+upload\b|"
    r"\baws\s+s3\s+(cp|sync)\b|\bgcloud\s+storage\s+cp\b|\baz\s+storage\b"
    r")"
)
CREDENTIAL_MUTATION_RE = re.compile(
    r"(?ix)("
    r"revoke\s+certificate|create\s+certificate|modify\s+secret|write\s+secret|"
    r"request\s+new\s+scope|mint\s+.*token|exchange\s+.*token|"
    r"security\s+(add|delete|set)-|chmod\s+[^\n;]*(\.p8|\.pem|id_rsa|auth\.json|\.env)"
    r")"
)
COST_OR_BILLING_RE = re.compile(r"(?i)(paid service|billing|post[- ]?billing|cost increase|pricing change|availability change|upgrade plan)")
PUBLIC_EXTERNAL_RE = re.compile(r"(?i)(public share|public publish|public channel|social announcement|new external audience|production irreversible)")
SAFE_AUDIT_RE = re.compile(r"(?i)(read[- ]?only|dry[- ]?run|audit|status|lint|verify|doctor|health|report)")
FORBIDDEN_IN_SAFE_AUTOMATION_RE = re.compile(r"(?i)(upload|submit|release|publish|credential|secret value|print secret|billing|paid|public|delete|force push|reset --hard)")
HASOS_RUNTIME_POLICY_WRITE_RE = re.compile(
    r"(?ix)("
    r"(?:~|\$HOME|\$\{HOME\}|\$HERMES_HOME|\$\{HERMES_HOME\}|/[^\s'\"`;]*)/"
    r"(?:"
    r"\.hermes/(?:standards/HASOS/policies/runtime-release-gate-evidence\.json|release-gate-evidence/active-runtime-policy\.json|policies/runtime-release-gate-evidence\.json)|"
    r"standards/HASOS/policies/runtime-release-gate-evidence\.json|"
    r"release-gate-evidence/active-runtime-policy\.json|"
    r"policies/runtime-release-gate-evidence\.json"
    r")"
    r")"
)
HARNESS_TRIGGER_RE = re.compile(
    r"(?i)(harness|long[- ]running|background|subagent|external cli|gateway|cron|mcp|plugin|release|signing|credential|secret|upload|hook|event persistence|scorecard|permission engine|memory summarization)"
)

EXPECTED_EVENTS = {
    "on_session_start",
    "on_session_end",
    "on_session_finalize",
    "on_session_reset",
    "pre_tool_call",
    "post_tool_call",
    "pre_llm_call",
    "post_llm_call",
    "subagent_stop",
    "pre_api_request",
    "post_api_request",
}
SIDE_EFFECT_TOOLS = {"terminal", "execute_code", "cronjob", "send_message", "browser_click", "browser_type"}
WORKSPACE_WRITE_TOOLS = {"patch", "write_file", "skill_manage"}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sanitize_text(text: Any, limit: int | None = None) -> str:
    """Best-effort high-confidence sanitizer for display/log/report text.

    Boundary: DLP v1. It redacts known secret-like values and secret-bearing
    assignments, but it is not complete DLP and must not be described as such.
    """
    raw = "" if text is None else str(text)
    raw = re.sub(
        r"(?i)(\bauthorization\s*:\s*(?:bearer|basic)\s+)[^\s,;]+",
        r"\1[REDACTED]",
        raw,
    )
    raw = re.sub(
        r"(?ix)(\b[A-Za-z0-9_-]*(?:api[_-]?key|access[_-]?token|refresh[_-]?token|token|secret|password|passwd|pwd|private[_-]?key|credential|cookie|jwt)[A-Za-z0-9_-]*['\"]?\s*[:=]\s*['\"]?)([^'\"\s,;}\]]+)",
        r"\1[REDACTED]",
        raw,
    )
    raw = SECRET_VAL_RE.sub("[REDACTED]", raw)
    return raw[:limit] if limit is not None else raw


def sanitize_data(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if str(k) in SAFE_NONSECRET_KEYS:
                out[k] = sanitize_data(v)
            elif SECRET_KEY_RE.search(str(k)):
                out[k] = "[REDACTED]"
            else:
                out[k] = sanitize_data(v)
        return out
    if isinstance(obj, list):
        return [sanitize_data(v) for v in obj]
    if isinstance(obj, str):
        return sanitize_text(obj)
    return obj


def detect_secret_findings(obj: Any) -> list[dict[str, Any]]:
    """Return metadata-only high-confidence secret findings; never returns values."""
    findings: list[dict[str, Any]] = []

    def walk(value: Any, path: str = "$") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k)
                child = f"{path}.{key}"
                if key not in SAFE_NONSECRET_KEYS and SECRET_KEY_RE.search(key):
                    findings.append({"path": child, "kind": "secret-key-name", "value": "[REDACTED]"})
                walk(v, child)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, f"{path}[{i}]")
        elif isinstance(value, str):
            if SECRET_VAL_RE.search(value) or sanitize_text(value) != value:
                findings.append({"path": path, "kind": "secret-like-value", "value": "[REDACTED]"})

    walk(obj)
    return findings


def redact(obj: Any) -> Any:
    return sanitize_data(obj)


def stable_id(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)[:4000]
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def tool_and_args(payload: dict[str, Any]) -> tuple[str, Any, str]:
    tool = payload.get("tool_name") or payload.get("extra", {}).get("tool_name") or ""
    args = payload.get("tool_input") or payload.get("args") or payload.get("extra", {}).get("args") or {}
    text = json.dumps(args, ensure_ascii=False, default=str)
    return str(tool), args, text


def classify(event: str, payload: dict[str, Any]) -> tuple[int, list[str]]:
    tool, args, text = tool_and_args(payload)
    level = 1
    reasons: list[str] = []
    if event in EXPECTED_EVENTS:
        reasons.append("known-hook-event")
    if event in ("pre_tool_call", "post_tool_call"):
        level = 2
        reasons.append("tool-call")
    if tool in SIDE_EFFECT_TOOLS | WORKSPACE_WRITE_TOOLS:
        level = max(level, 3)
        reasons.append(f"tool:{tool}")
    if tool == "cronjob":
        action = str(args.get("action", "")).lower() if isinstance(args, dict) else ""
        level = max(level, 4)
        reasons.append(f"cron:{action or 'unknown'}")
    if tool == "send_message":
        level = max(level, 4)
        reasons.append("external-message")
    if tool in SIDE_EFFECT_TOOLS and RELEASE_OR_UPLOAD_RE.search(text):
        level = max(level, 4)
        reasons.append("level4c-release-upload-public-or-production-action")
    if tool in SIDE_EFFECT_TOOLS and CREDENTIAL_MUTATION_RE.search(text):
        level = max(level, 4)
        reasons.append("level4c-credential-or-signing-mutation")
    if tool in SIDE_EFFECT_TOOLS and COST_OR_BILLING_RE.search(text):
        level = max(level, 4)
        reasons.append("cost-or-post-billing-hard-stop-signal")
    if tool in SIDE_EFFECT_TOOLS and PUBLIC_EXTERNAL_RE.search(text):
        level = max(level, 4)
        reasons.append("public-or-external-audience-signal")
    if tool in ("terminal", "execute_code") and HASOS_RUNTIME_POLICY_WRITE_RE.search(text):
        level = 5
        reasons.append("level5-hasos-runtime-policy-evidence-write")
    if tool in ("terminal", "execute_code") and DANGEROUS_TERMINAL_RE.search(text):
        level = 5
        reasons.append("level5-dangerous-command-or-secret-exfiltration-pattern")
    return level, reasons


def has_policy_authorization(payload: dict[str, Any], text: str) -> bool:
    """Return whether a trusted runtime-controlled policy context authorizes Level 4.

    Security boundary: ``text`` and ``payload['extra']`` are untrusted for
    authorization.  A command/prompt/tool argument must not be able to authorize
    itself by embedding strings such as ``hasos_policy_authorized=true`` or by
    being copied into generic ``extra`` kwargs.  Prefer the explicit
    ``policy_context`` / ``runtime_policy`` containers.  Top-level flags are
    retained only for local synthetic tests and future runtime-controlled call
    sites that write the hook payload envelope directly.
    """
    trusted_values: list[Any] = [
        payload.get("hasos_policy_authorized"),
        payload.get("policy_authorized"),
    ]
    for container_name in ("policy_context", "runtime_policy"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            trusted_values.extend([
                container.get("hasos_policy_authorized"),
                container.get("policy_authorized"),
            ])
    return any(v is True or (isinstance(v, str) and v.lower() in {"true", "yes", "1"}) for v in trusted_values)




def _truthy(value: Any) -> bool:
    return value is True or (isinstance(value, str) and value.lower() in {"true", "yes", "1", "passed"})


def _parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _runtime_policy_files() -> list[Path]:
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    return [
        home / "standards" / "HASOS" / "policies" / "runtime-release-gate-evidence.json",
        home / "release-gate-evidence" / "active-runtime-policy.json",
        home / "policies" / "runtime-release-gate-evidence.json",
    ]


def _load_runtime_release_policy_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in _runtime_policy_files():
        try:
            if not path.exists():
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            raw_entries = data.get("entries") or data.get("policies") or []
        elif isinstance(data, list):
            raw_entries = data
        else:
            raw_entries = []
        for entry in raw_entries:
            if isinstance(entry, dict):
                entry = {**entry, "_policy_file": str(path)}
                entries.append(entry)
    return entries


def _string_list(value: Any) -> list[str] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return []
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            return None
        items.append(item.strip())
    return items


def _preauthorized_runtime_policy_for_payload(payload: dict[str, Any], text: str) -> dict[str, Any]:
    """Return file-backed runtime policy for exact allowlisted release scopes.

    This is a local controlled-policy bridge for gateway/tool contexts that
    cannot pass ``runtime_policy`` directly. The policy file supplies the trust;
    tool/command text is used only to match the bounded scope. Raw command text
    markers still do not authorize anything.
    """
    tool, args, _ = tool_and_args(payload)
    combined_parts = [text, str(tool)]
    if isinstance(args, dict):
        combined_parts.extend(str(args.get(k, "")) for k in ("command", "workdir", "script", "prompt", "name"))
    combined = "\n".join(combined_parts)
    now = datetime.now(timezone.utc)
    for entry in _load_runtime_release_policy_entries():
        if str(entry.get("status", "")).lower() not in {"active", "approved"}:
            continue
        tools = _string_list(entry.get("tool_names"))
        if not tools or tool not in set(tools):
            continue
        expires = _parse_utc(entry.get("expires_at"))
        if expires is not None and expires < now:
            continue
        forbidden = _string_list(entry.get("forbidden_substrings", []))
        if forbidden is None:
            continue
        if any(s in combined for s in forbidden):
            continue
        required = _string_list(entry.get("required_substrings", []))
        patterns = _string_list(entry.get("required_patterns", []))
        if required is None or patterns is None:
            continue
        if not required and not patterns:
            continue
        if required and not all(s in combined for s in required):
            continue
        try:
            if patterns and not all(re.search(p, combined, re.I | re.S) for p in patterns):
                continue
        except re.error:
            continue
        policy = entry.get("runtime_policy") or entry.get("policy_context") or {}
        if not isinstance(policy, dict):
            continue
        merged = {
            "policy_authorized": True,
            "approval_source": "local_runtime_release_gate_evidence_file",
            **policy,
        }
        return merged
    return {}


def trusted_policy_context(payload: dict[str, Any], text: str = "") -> dict[str, Any]:
    merged: dict[str, Any] = {}
    preauthorized = _preauthorized_runtime_policy_for_payload(payload, text)
    if preauthorized:
        merged.update(preauthorized)
    for name in ("policy_context", "runtime_policy"):
        value = payload.get(name)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _evidence_reference_exists(ctx: dict[str, Any]) -> bool:
    evidence_path = ctx.get("evidence_path")
    if evidence_path:
        return Path(str(evidence_path)).expanduser().is_file()
    for key in ("evidence_id_or_path", "evidence_id"):
        value = ctx.get(key)
        if not value:
            continue
        text = str(value)
        if "/" in text or "\\" in text or re.search(r"\.[A-Za-z0-9]{2,8}$", text):
            return Path(text).expanduser().is_file()
        return True
    return False


def _shell_text_without_comments(text: str) -> str:
    out_lines: list[str] = []
    for line in str(text or "").splitlines():
        buf: list[str] = []
        quote: str | None = None
        escaped = False
        for ch in line:
            if escaped:
                buf.append(ch)
                escaped = False
                continue
            if ch == "\\":
                buf.append(ch)
                escaped = True
                continue
            if quote:
                if ch == quote:
                    quote = None
                buf.append(ch)
                continue
            if ch in {"'", '"'}:
                quote = ch
                buf.append(ch)
                continue
            if ch == "#":
                break
            buf.append(ch)
        out_lines.append("".join(buf))
    return "\n".join(out_lines)


def _scope_requirements_match(ctx: dict[str, Any], text: str) -> tuple[bool, list[str]]:
    source = str(ctx.get("approval_source", ""))
    if not source.startswith("latest_user_"):
        return True, []
    required = ctx.get("scope_required_substrings")
    patterns = ctx.get("scope_required_patterns")
    if not isinstance(required, list):
        required = []
    if not isinstance(patterns, list):
        patterns = []
    required = [s for s in required if isinstance(s, str) and s.strip()]
    patterns = [s for s in patterns if isinstance(s, str) and s.strip()]
    if not required and not patterns:
        return False, ["scope_required_substrings"]
    candidates = _release_scope_segments(text or "")
    missing: list[str] = []
    for candidate in candidates:
        candidate_missing: list[str] = []
        lowered = candidate.lower()
        for item in required:
            if item.lower() not in lowered:
                candidate_missing.append(f"scope:{item}")
        for pattern in patterns:
            try:
                if not re.search(pattern, candidate, re.I | re.S):
                    candidate_missing.append(f"scope_pattern:{pattern}")
            except re.error:
                candidate_missing.append("scope_required_patterns")
        if not candidate_missing:
            return True, []
        if not missing:
            missing = candidate_missing
    return False, missing


def _split_shell_segments(text: str) -> list[str]:
    cleaned = _shell_text_without_comments(text)
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    escaped = False
    i = 0
    while i < len(cleaned):
        ch = cleaned[i]
        if escaped:
            buf.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\":
            buf.append(ch)
            escaped = True
            i += 1
            continue
        if quote:
            if ch == quote:
                quote = None
            buf.append(ch)
            i += 1
            continue
        if ch in {"'", '"'}:
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch in {";", "\n"} or cleaned.startswith("&&", i) or cleaned.startswith("||", i):
            segment = "".join(buf).strip()
            if segment:
                segments.append(segment)
            buf = []
            i += 2 if cleaned.startswith(("&&", "||"), i) else 1
            continue
        buf.append(ch)
        i += 1
    segment = "".join(buf).strip()
    if segment:
        segments.append(segment)
    return segments or [cleaned]


def _release_scope_segments(text: str) -> list[str]:
    segments = _split_shell_segments(text)
    release_segments = [s for s in segments if RELEASE_OR_UPLOAD_RE.search(s)]
    return release_segments or segments


def _tool_scope_text(payload: dict[str, Any]) -> str:
    args = payload.get("tool_input") or payload.get("args") or {}
    if isinstance(args, dict):
        for key in ("command", "code", "script", "message"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
    return json.dumps(args, ensure_ascii=False, default=str)


def validate_level4_gate(payload: dict[str, Any], action_kind: str) -> tuple[bool, list[str], list[str]]:
    """Validate runtime-controlled Level-4 gate evidence.

    This v1 verifier checks that trusted runtime ``policy_context`` or
    ``runtime_policy`` carries the minimum gate evidence. It intentionally does
    not trust raw tool input or generic ``extra`` fields.
    """
    ctx = trusted_policy_context(payload, json.dumps(payload.get("tool_input") or payload.get("args") or {}, ensure_ascii=False, default=str))
    required = [
        "policy_authorized",
        "runbook_id",
        "runbook_version",
        "owner",
        "target",
        "evidence_id_or_path",
        "stop_rules_checked",
        "redaction_checked",
    ]
    missing: list[str] = []
    if not (_truthy(ctx.get("policy_authorized")) or _truthy(ctx.get("hasos_policy_authorized"))):
        missing.append("policy_authorized")
    for key in ("runbook_id", "runbook_version", "owner", "target"):
        if not ctx.get(key):
            missing.append(key)
    if not _evidence_reference_exists(ctx):
        missing.append("evidence_id_or_path")
    for key in ("stop_rules_checked", "redaction_checked"):
        if not _truthy(ctx.get(key)):
            missing.append(key)
    if action_kind == "release_upload" and not _truthy(ctx.get("release_security_gate_passed")):
        missing.append("release_security_gate_passed")
    if action_kind == "release_upload":
        scope_ok, scope_missing = _scope_requirements_match(
            ctx,
            _tool_scope_text(payload),
        )
        if not scope_ok:
            missing.extend(scope_missing)
    if action_kind == "privacy_tracking" and not _truthy(ctx.get("privacy_consistency_gate_passed")):
        missing.append("privacy_consistency_gate_passed")
    if action_kind == "credential_mutation":
        if not _truthy(ctx.get("credential_scope_reviewed")):
            missing.append("credential_scope_reviewed")
        for key in ("rollback_plan", "rollback_verification"):
            if not ctx.get(key):
                missing.append(key)
    if action_kind == "external_public" and not (ctx.get("external_audience_allowlist") or ctx.get("public_target_allowlist")):
        missing.append("external_or_public_allowlist")
    return not missing, required, missing


def _block_for_missing_level4_gates(action_kind: str, missing: list[str], required: list[str]) -> tuple[str, str, str, list[str], list[str]]:
    message = f"HASOS Level 4C {action_kind} blocked: missing runtime-controlled gate evidence: {', '.join(missing)}."
    return ("block", "block", message, required, missing)

def safe_recurring_audit(args: Any, text: str) -> bool:
    if not isinstance(args, dict):
        return False
    action = str(args.get("action", "")).lower()
    if action not in {"create", "update", "run", "resume", "pause", "list"}:
        return False
    if action == "list":
        return True
    prompt = str(args.get("prompt", ""))
    script = str(args.get("script", ""))
    name = str(args.get("name", ""))
    combined = "\n".join([name, prompt, script, text])
    normalized = re.sub(r"(?i)\b(no|without|do not|不得|不要)\s+(upload|submit|release|publish|credential|secret|billing|paid|public|delete)", "", combined)
    return bool(SAFE_AUDIT_RE.search(normalized)) and not FORBIDDEN_IN_SAFE_AUTOMATION_RE.search(normalized)


def message_target_is_origin_or_home(args: Any) -> bool:
    if not isinstance(args, dict):
        return False
    target = str(args.get("target", "") or "")
    return target in {"", "origin", "local", "telegram", "discord", "slack", "signal", "matrix"}


def evaluate_policy(event: str, payload: dict[str, Any], level: int, reasons: list[str]) -> tuple[str, str, str, list[str], list[str]]:
    """Return (decision, action, message, required_gates, missing_gates)."""
    tool, args, text = tool_and_args(payload)
    runtime_ctx = trusted_policy_context(payload, text)
    policy_authorized = has_policy_authorization({**payload, "runtime_policy": runtime_ctx} if runtime_ctx else payload, text)
    runtime_side_effect_tool = tool in SIDE_EFFECT_TOOLS

    if level >= 5:
        gates = ["deny-by-default", "safer-scoped-workflow", "explicit-risk-review"]
        return ("block", "block", "HASOS Level 5 blocked by runtime harness: destructive/security-sensitive command requires a safer, explicit scoped workflow.", gates, gates)

    if COST_OR_BILLING_RE.search(text) and runtime_side_effect_tool:
        gates = ["traditional-chinese-user-decision", "cost-risk-review"]
        return ("block", "block", "HASOS payment hard-stop: explicit user decision in Traditional Chinese is required before execution.", gates, gates)

    if CREDENTIAL_MUTATION_RE.search(text) and runtime_side_effect_tool:
        ok, required, missing = validate_level4_gate(payload, "credential_mutation")
        if not (policy_authorized and ok):
            if not policy_authorized and "policy_authorized" not in missing:
                missing.insert(0, "policy_authorized")
            return _block_for_missing_level4_gates("credential/signing mutation", missing, required)
        return ("allow-policy-authorized-4c-credential", "allow", "", required, [])

    if RELEASE_OR_UPLOAD_RE.search(text) and runtime_side_effect_tool:
        ok, required, missing = validate_level4_gate(payload, "release_upload")
        if not (policy_authorized and ok):
            if not policy_authorized and "policy_authorized" not in missing:
                missing.insert(0, "policy_authorized")
            return _block_for_missing_level4_gates("release/upload/public-production action", missing, required)
        return ("allow-policy-authorized-4c-release", "allow", "", required, [])

    if PUBLIC_EXTERNAL_RE.search(text) and runtime_side_effect_tool:
        ok, required, missing = validate_level4_gate(payload, "external_public")
        if not (policy_authorized and ok):
            if not policy_authorized and "policy_authorized" not in missing:
                missing.insert(0, "policy_authorized")
            return _block_for_missing_level4_gates("public/external-audience action", missing, required)
        return ("allow-policy-authorized-4c-public", "allow", "", required, [])

    if event == "pre_tool_call" and tool == "cronjob":
        if safe_recurring_audit(args, text):
            return ("allow-policy-authorized-4d", "allow", "", [], [])
        ok, required, missing = validate_level4_gate(payload, "cron")
        if not (policy_authorized and ok):
            if not policy_authorized and "policy_authorized" not in missing:
                missing.insert(0, "policy_authorized")
            return _block_for_missing_level4_gates("cron action", missing, required)
        return ("allow-policy-authorized-4c-cron", "allow", "", required, [])

    if event == "pre_tool_call" and tool == "send_message":
        if message_target_is_origin_or_home(args):
            return ("allow-policy-authorized-4c-home", "allow", "", [], [])
        ok, required, missing = validate_level4_gate(payload, "external_public")
        if not (policy_authorized and ok):
            if not policy_authorized and "policy_authorized" not in missing:
                missing.insert(0, "policy_authorized")
            return _block_for_missing_level4_gates("external message", missing, required)
        return ("allow-policy-authorized-4c-external-message", "allow", "", required, [])

    return ("allow", "allow", "", [], [])


def scorecard(event: str, payload: dict[str, Any], level: int, reasons: list[str], decision: str) -> dict[str, Any]:
    hard_blocked = decision == "block"
    return {
        "lifecycle_evidence": event in EXPECTED_EVENTS,
        "permission_classified": level is not None,
        "level4_precision_checked": any(r.startswith("cron:") or "level4c" in r or "cost" in r or "external-message" in r for r in reasons) or level < 4,
        "policy_authorization_checked": level >= 4,
        "secrets_redacted": True,
        "hard_block_level5_or_hardstop": hard_blocked or level < 5,
    }


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    event = str(payload.get("hook_event_name") or payload.get("event") or "")
    level, reasons = classify(event, payload)
    decision, action, message, required_gates, missing_gates = evaluate_policy(event, payload, level, reasons)
    reason_codes = list(reasons)
    if decision == "allow-policy-authorized-4d" and "level4d-preauthorized-read-only-audit" not in reason_codes:
        reason_codes.append("level4d-preauthorized-read-only-audit")
    if decision == "allow-policy-authorized-4c-home" and "level4c-home-delivery-allowlist" not in reason_codes:
        reason_codes.append("level4c-home-delivery-allowlist")
    return {
        "schema_version": SCHEMA_VERSION,
        "event": event,
        "tool_name": tool_and_args(payload)[0] or None,
        "level": level,
        "decision": decision,
        "action": action,
        "message": message,
        "reason_codes": reason_codes,
        "required_gates": required_gates,
        "missing_gates": missing_gates,
        "secret_findings": detect_secret_findings(payload),
        "scorecard": scorecard(event, payload, level, reasons, decision),
    }


def event_record(payload: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    safe = redact(payload)
    return {
        "ts": utc(),
        "event": decision["event"],
        "id": stable_id({"event": decision["event"], "payload": safe, "ts_bucket": int(time.time() // 60)}),
        "session_id": safe.get("session_id") or safe.get("extra", {}).get("session_id"),
        "tool_name": decision.get("tool_name"),
        "hasos_level": decision["level"],
        "reasons": decision["reason_codes"],
        "decision": decision["decision"],
        "policy_decision": {k: v for k, v in decision.items() if k != "scorecard"},
        "payload": {**safe, "hasos_scorecard": decision["scorecard"]},
    }


def harness_context_for_message(message: str) -> str:
    if not HARNESS_TRIGGER_RE.search(message or ""):
        return ""
    return (
        "HASOS Harness runtime is active: classify Level 0-5, persist lifecycle evidence, redact secrets, "
        "check Level-4 policy authorization, block Level-5 and obvious hard-stop patterns, use hermes-harness "
        "for gated workflows, and verify before finalizing. Boundary: hook-level enforcement, not OS sandbox."
    )


def sanitize_audit_result(result: dict[str, Any]) -> dict[str, Any]:
    clean = redact(result)
    clean.setdefault("report_schema", "hasos.sanitized_audit_report.v1")
    synthetic = clean.get("synthetic")
    if isinstance(synthetic, dict):
        if "hook_outputs" in synthetic:
            synthetic.pop("hook_outputs", None)
            synthetic["hook_outputs_removed"] = "[REDACTED]"
    return clean


def write_sanitized_audit_report(result: dict[str, Any], root: str | Path | None = None) -> Path:
    base = Path(root) if root is not None else Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser() / "audit-reports" / "hasos-harness-runtime"
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    status = str(result.get("status", "unknown"))
    ts = str(result.get("ts") or utc()).replace(":", "").replace("+", "_")
    path = base / f"{datetime.now(timezone.utc).strftime('%Y-%m-%d')}-{status}-{stable_id(result)}.json"
    sanitized = sanitize_audit_result(result)
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return path


def audit_report_status(root: str | Path | None = None) -> dict[str, Any]:
    base = Path(root) if root is not None else Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser() / "audit-reports" / "hasos-harness-runtime"
    reports = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime if p.exists() else 0) if base.exists() else []
    latest = reports[-1] if reports else None
    latest_data: dict[str, Any] = {}
    if latest is not None:
        try:
            latest_data = json.loads(latest.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            latest_data = {"status": "unreadable"}
    return {
        "root": str(base),
        "report_count": len(reports),
        "latest_report": str(latest) if latest else "",
        "latest_status": latest_data.get("status", ""),
        "latest_report_schema": latest_data.get("report_schema", ""),
    }


def prune_sanitized_audit_reports(root: str | Path | None = None, *, keep_last: int = 90, max_age_days: int = 30) -> dict[str, Any]:
    """Prune only sanitized HASOS audit JSON reports under the controlled root."""
    base = Path(root) if root is not None else Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser() / "audit-reports" / "hasos-harness-runtime"
    if not base.exists():
        return {"root": str(base), "removed": 0, "kept": 0, "keep_last": keep_last, "max_age_days": max_age_days}
    now = datetime.now(timezone.utc).timestamp()
    reports = sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime)
    removable: list[Path] = []
    if keep_last > 0 and len(reports) > keep_last:
        removable.extend(reports[:-keep_last])
    cutoff_seconds = max_age_days * 24 * 60 * 60
    for candidate in reports:
        try:
            if now - candidate.stat().st_mtime > cutoff_seconds:
                removable.append(candidate)
        except OSError:
            continue
    removed = 0
    for candidate in sorted(set(removable)):
        if candidate.parent != base or candidate.suffix != ".json":
            continue
        try:
            candidate.unlink()
            removed += 1
        except OSError:
            pass
    return {"root": str(base), "removed": removed, "kept": len(list(base.glob('*.json'))), "keep_last": keep_last, "max_age_days": max_age_days}
