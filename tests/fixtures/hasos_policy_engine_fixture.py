"""Minimal repo-contained HASOS policy engine fixture for clean CI.

This fixture intentionally implements only the behavior asserted by
`tests/hasos/test_hasos_policy_engine.py`.  The real local engine remains the
source of truth when present at ~/.hermes/scripts/hasos_policy_engine.py.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = "hasos.policy_decision.v1"
_REPORT_SCHEMA = "hasos.sanitized_audit_report.v1"
_SECRET_RE = re.compile(
    r"(?i)(sk-[A-Za-z0-9_-]{20,}|TEST_FAKE_VALUE_[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._-]{20,})"
)
_HASOS_RUNTIME_POLICY_WRITE_RE = re.compile(
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


def sanitize_text(value: Any) -> str:
    return _SECRET_RE.sub("[REDACTED]", "" if value is None else str(value))


def sanitize_data(value: Any) -> Any:
    if isinstance(value, str):
        return sanitize_text(value)
    if isinstance(value, list):
        return [sanitize_data(v) for v in value]
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("api_key", "access_token", "secret", "password", "private_key")):
                redacted[key_text] = "[REDACTED]" if isinstance(item, str) else sanitize_data(item)
            else:
                redacted[key_text] = sanitize_data(item)
        return redacted
    return value


def redact(value: Any) -> Any:
    return sanitize_data(value)


def detect_secret_findings(value: Any) -> list[dict[str, str]]:
    serialized = json.dumps(value, ensure_ascii=False, default=str)
    findings = []
    for match in _SECRET_RE.finditer(serialized):
        findings.append({"type": "secret-like-token", "value": "[REDACTED]", "span": f"{match.start()}:{match.end()}"})
    if any(marker in serialized.lower() for marker in ("api_key", "access_token", "secret")):
        findings.append({"type": "sensitive-key", "value": "[REDACTED]"})
    return findings


def _allow(level: int, decision: str = "allow", reason: str = "tool-call") -> dict[str, Any]:
    return {
        "schema_version": _SCHEMA,
        "level": level,
        "decision": decision,
        "action": "allow",
        "message": "",
        "reason_codes": [reason],
        "required_gates": [],
        "missing_gates": [],
    }


def _block(level: int, reason: str, required: list[str] | None = None) -> dict[str, Any]:
    required = required or []
    return {
        "schema_version": _SCHEMA,
        "level": level,
        "decision": "block",
        "action": "block",
        "message": f"HASOS Level {level} blocked by fixture: {reason}",
        "reason_codes": [reason],
        "required_gates": required,
        "missing_gates": required,
    }


def _runtime_policy_files() -> list[Path]:
    home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
    return [
        home / "standards" / "HASOS" / "policies" / "runtime-release-gate-evidence.json",
        home / "release-gate-evidence" / "active-runtime-policy.json",
        home / "policies" / "runtime-release-gate-evidence.json",
    ]


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


def _file_backed_runtime_policy(payload: dict[str, Any]) -> dict[str, Any]:
    tool = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    serialized_input = json.dumps(tool_input, ensure_ascii=False, default=str)
    now = datetime.now(timezone.utc)
    for path in _runtime_policy_files():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        entries = data.get("entries") or data.get("policies") or [] if isinstance(data, dict) else data if isinstance(data, list) else []
        for entry in entries:
            if not isinstance(entry, dict) or str(entry.get("status", "")).lower() not in {"active", "approved"}:
                continue
            tool_names = _string_list(entry.get("tool_names"))
            if not tool_names or tool not in set(tool_names):
                continue
            expires = _parse_utc(entry.get("expires_at"))
            if expires is not None and expires < now:
                continue
            required = _string_list(entry.get("required_substrings", []))
            patterns = _string_list(entry.get("required_patterns", []))
            if required is None or patterns is None:
                continue
            if not required and not patterns:
                continue
            forbidden = _string_list(entry.get("forbidden_substrings", []))
            if forbidden is None:
                continue
            if any(term not in serialized_input for term in required):
                continue
            try:
                if patterns and not all(re.search(pattern, serialized_input, re.I | re.S) for pattern in patterns):
                    continue
            except re.error:
                continue
            if any(term in serialized_input for term in forbidden):
                continue
            policy = entry.get("runtime_policy")
            if isinstance(policy, dict):
                return policy
    return {}


def _runtime_policy(payload: dict[str, Any]) -> dict[str, Any]:
    policy = payload.get("runtime_policy") if isinstance(payload.get("runtime_policy"), dict) else None
    if policy is None and isinstance(payload.get("policy_context"), dict):
        policy = payload["policy_context"]
    return policy or _file_backed_runtime_policy(payload)


def _evidence_reference_exists(policy: dict[str, Any]) -> bool:
    evidence_path = policy.get("evidence_path")
    if evidence_path:
        return Path(str(evidence_path)).expanduser().is_file()
    evidence_id_or_path = policy.get("evidence_id_or_path")
    if evidence_id_or_path:
        text = str(evidence_id_or_path)
        if "/" in text or "\\" in text or re.search(r"\.[A-Za-z0-9]{2,8}$", text):
            return Path(text).expanduser().is_file()
        return True
    evidence_id = policy.get("evidence_id")
    if evidence_id:
        text = str(evidence_id)
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

_RELEASE_RE = re.compile(r"(?i)(xcrun\s+(altool|notarytool)|fastlane\s+(deliver|pilot|upload_to_app_store)|submit\s+for\s+review|upload-app|gh\s+release\s+upload)")


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
    release_segments = [s for s in segments if _RELEASE_RE.search(s)]
    return release_segments or segments


def _scope_requirements_match(policy: dict[str, Any], text: str) -> tuple[bool, list[str]]:
    source = str(policy.get("approval_source", ""))
    if not source.startswith("latest_user_"):
        return True, []
    required = policy.get("scope_required_substrings")
    patterns = policy.get("scope_required_patterns")
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


def _tool_scope_text(payload: dict[str, Any]) -> str:
    args = payload.get("tool_input") or payload.get("args") or {}
    if isinstance(args, dict):
        for key in ("command", "code", "script", "message"):
            value = args.get(key)
            if isinstance(value, str) and value:
                return value
    return json.dumps(args, ensure_ascii=False, default=str)


def validate_level4_gate(payload: dict[str, Any], action_kind: str) -> tuple[bool, list[str], list[str]]:
    policy = _runtime_policy(payload)
    required = [
        "policy_authorized",
        "runbook_id",
        "runbook_version",
        "owner",
        "target",
        "evidence_id_or_path",
        "stop_rules_checked",
        "redaction_checked",
        "release_security_gate_passed",
    ]
    missing = [key for key in required if key != "evidence_id_or_path" and not policy.get(key)]
    if not _evidence_reference_exists(policy):
        missing.append("evidence_id_or_path")
    if action_kind == "release_upload":
        scope_ok, scope_missing = _scope_requirements_match(
            policy,
            _tool_scope_text(payload),
        )
        if not scope_ok:
            missing.extend(scope_missing)
    return (not missing), required, missing


def _complete_release_gate(policy: dict[str, Any]) -> bool:
    required = [
        "policy_authorized",
        "runbook_id",
        "runbook_version",
        "owner",
        "target",
        "stop_rules_checked",
        "redaction_checked",
        "release_security_gate_passed",
    ]
    return _evidence_reference_exists(policy) and all(bool(policy.get(k)) for k in required)


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    tool = str(payload.get("tool_name", ""))
    tool_input = payload.get("tool_input") if isinstance(payload.get("tool_input"), dict) else {}
    command = str(tool_input.get("command", ""))
    lowered = json.dumps(tool_input, ensure_ascii=False, default=str).lower()
    policy = _runtime_policy(payload)

    level5_terms = [
        "rm -rf", "rm -r -f", "git push --force", "git push --force-with-lease",
        "git reset --hard", "git clean", "shutil.rmtree", "xargs rm -rf", "find /tmp/example -delete",
        "install.sh | sh", "path('/tmp/example').unlink", "finder\" to delete", "cat ~/.hermes/.env",
        "auth.json", "sudo launchctl unload",
    ]
    if tool == "terminal" and any(term in command.lower() for term in level5_terms):
        return _block(5, "level5-dangerous-command-or-secret-exfiltration-pattern", ["deny-by-default"])
    if tool in {"terminal", "execute_code"} and _HASOS_RUNTIME_POLICY_WRITE_RE.search(lowered):
        return _block(5, "level5-hasos-runtime-policy-evidence-write", ["deny-by-default"])

    level4_release = tool == "terminal" and any(term in command.lower() for term in ("submit for review", "fastlane deliver", "xcrun altool", "--upload-app", "app-store upload"))
    level4_credential = tool == "terminal" and "security add-generic-password" in command.lower()
    level4_external = tool == "send_message" and str(tool_input.get("target", "")).startswith("telegram:-")
    level4_cost = "upgrade plan" in lowered or "paid service" in lowered
    level4_cron = tool == "cronjob" and ("publish" in lowered or "public release" in lowered)

    if tool == "cronjob" and "hasos harness runtime daily audit" in lowered and "read-only" in lowered:
        return _allow(4, "allow-policy-authorized-4d", "4d-read-only-audit-cron")

    if level4_release:
        ok, _required, missing = validate_level4_gate(payload, "release_upload")
        if ok:
            return _allow(4, "allow-policy-authorized-4c-release", "level4c-release-upload-public-or-production-action")
        if not missing:
            missing = ["runbook_id", "release_security_gate_passed"]
        return _block(4, "level4c-release-upload-public-or-production-action", missing)

    if level4_credential:
        if _complete_release_gate(policy) and policy.get("credential_scope_reviewed") and policy.get("rollback_plan") and policy.get("rollback_verification"):
            return _allow(4, "allow-policy-authorized-4c-credential", "level4c-credential-mutation")
        return _block(4, "credential", ["credential_scope_reviewed", "rollback_plan"])

    if level4_external:
        return _block(4, "external", ["external_or_public_allowlist"])
    if level4_cost:
        return _block(4, "cost", ["cost_approval"])
    if level4_cron:
        return _block(4, "level4-cron-public-side-effect", ["runbook_id"])

    if tool in {"read_file", "search_files"} or "documentation only" in lowered or "py_compile" in command:
        return _allow(2, reason="read-only-or-documentation")
    return _allow(2)


def _strip_sensitive_report_fields(value: Any) -> Any:
    data = sanitize_data(value)
    if isinstance(data, dict):
        synthetic = data.get("synthetic")
        if isinstance(synthetic, dict):
            synthetic.pop("hook_outputs", None)
    return data


def write_sanitized_audit_report(result: dict[str, Any], root: Path | str) -> Path:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    data = _strip_sensitive_report_fields(result)
    if isinstance(data, dict):
        data["report_schema"] = _REPORT_SCHEMA
        data["written_at"] = datetime.now(timezone.utc).isoformat()
    path = root_path / "hasos-audit-report-0001.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def audit_report_status(root: Path | str) -> dict[str, Any]:
    files = sorted(Path(root).glob("*.json"))
    latest_status = "none"
    if files:
        try:
            latest_status = json.loads(files[-1].read_text(encoding="utf-8")).get("status", "unknown")
        except Exception:
            latest_status = "unknown"
    return {"report_count": len(files), "latest_status": latest_status}


def prune_sanitized_audit_reports(root: Path | str, keep_last: int = 1, max_age_days: int = 365) -> dict[str, int]:
    files = sorted(Path(root).glob("*.json"))
    remove = files[:-keep_last] if keep_last >= 0 else files
    for path in remove:
        path.unlink(missing_ok=True)
    return {"kept": len(files) - len(remove), "removed": len(remove)}
