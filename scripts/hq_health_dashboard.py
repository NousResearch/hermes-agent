#!/usr/bin/env python
"""HQ health dashboard for supervised Hermes/HQ operations.

Runs the local HQ harness suite, reads their report JSON/Markdown outputs,
and writes a single dashboard with readiness, control/eval, audit/privacy,
remote-control, memory/skill, and approval-request status.

This script is intentionally conservative:
- It does not execute remote SSH commands.
- It does not read .env/auth.json secret values.
- It does not approve or deny anything.
- Approval status is best-effort from logs because the live approval queue is
  in the Gateway process memory and is not externally queryable from this script.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

HERMES_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / "AppData" / "Local" / "hermes"))
SCRIPTS = HERMES_HOME / "scripts"
REPORTS = HERMES_HOME / "reports"
LOGS = HERMES_HOME / "logs"
PYTHON = Path(sys.executable)
POLICY_VERSION = "hq-health-dashboard-v0.1"
DASHBOARD_MD = REPORTS / "hq_health_dashboard_latest.md"
DASHBOARD_JSON = REPORTS / "hq_health_dashboard_latest.json"
HARNESS_EVIDENCE_REPORT = "hq_harness_eval_latest.json"


@dataclass
class HarnessSpec:
    key: str
    title: str
    script: str
    json_report: str | None
    md_report: str | None
    ok_statuses: tuple[str, ...]
    review_statuses: tuple[str, ...] = ("REVIEW",)


HARNESS_SPECS: List[HarnessSpec] = [
    HarnessSpec(
        key="readiness",
        title="HQ readiness smoke check",
        script="hq_readiness_check.py",
        json_report=None,
        md_report="hq_readiness_latest.md",
        ok_statuses=("OK",),
        review_statuses=("CHECK",),
    ),
    HarnessSpec(
        key="control_eval",
        title="Control-decision eval",
        script="hq_control_eval.py",
        json_report="hq_control_eval_latest.json",
        md_report="hq_control_eval_latest.md",
        ok_statuses=("OK", "PASS"),
    ),
    HarnessSpec(
        key="audit_trail",
        title="Audit trail summarizer",
        script="hq_audit_trail_summarizer.py",
        json_report="hq_audit_trail_latest.json",
        md_report="hq_audit_trail_latest.md",
        ok_statuses=("OK",),
    ),
    HarnessSpec(
        key="privacy_budget",
        title="Privacy budget / disclosure tracker",
        script="hq_privacy_budget_tracker.py",
        json_report="hq_privacy_budget_latest.json",
        md_report="hq_privacy_budget_latest.md",
        ok_statuses=("OK",),
    ),
    HarnessSpec(
        key="remote_control_gate",
        title="Remote-control readiness gate",
        script="hq_remote_control_gate.py",
        json_report="hq_remote_control_gate_latest.json",
        md_report="hq_remote_control_gate_latest.md",
        # BLOCKED_NO_ENABLED_HOSTS is the safe default before user explicitly
        # enrolls remote hosts. It is not a failure.
        ok_statuses=("OK", "BLOCKED_NO_ENABLED_HOSTS", "READY_FOR_SUPERVISED_DRY_RUN"),
    ),
    HarnessSpec(
        key="memory_skill_quality",
        title="Memory/skill quality gate",
        script="hq_memory_skill_quality_gate.py",
        json_report="hq_memory_skill_quality_latest.json",
        md_report="hq_memory_skill_quality_latest.md",
        ok_statuses=("OK",),
        review_statuses=("REVIEW",),
    ),
]

TS_RE = re.compile(r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})")
APPROVAL_REQUEST_RE = re.compile(
    r"(?i)(Dangerous command requires approval|send_exec_approval|Approval text-send|approval request)"
)
APPROVED_RE = re.compile(r"(?i)User approved (?P<count>\d+) dangerous command\(s\) via /approve(?: \((?P<mode>[^)]+)\))?")
DENIED_RE = re.compile(r"(?i)User denied (?P<count>\d+) dangerous command\(s\) via /deny")
FAILED_APPROVAL_SEND_RE = re.compile(r"(?i)Failed to send approval request|Button-based approval failed")
HARDLINE_RE = re.compile(r"(?i)Hardline block|BLOCKED \(hardline\)|unconditional blocklist")
BLOCKED_RE = re.compile(r"(?i)(execute_code script denied by user|The user has NOT consented|approval denied|dangerous command.*denied|BLOCKED \(hardline\)|Hardline block)")
SLASH_APPROVE_RE = re.compile(r"slash '/approve(?P<args>[^']*)' invoked by user=(?P<user>.*?) id=(?P<uid>\d+)")
SLASH_DENY_RE = re.compile(r"slash '/deny(?P<args>[^']*)' invoked by user=(?P<user>.*?) id=(?P<uid>\d+)")


def run_cmd(cmd: List[str], timeout: int = 180) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr).strip()
    except Exception as e:
        return 999, f"{type(e).__name__}: {e}"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": f"{type(e).__name__}: {e}"}


def parse_readiness_md(path: Path) -> Dict[str, Any]:
    checks: Dict[str, str] = {}
    if not path.exists():
        return {"gate_status": "MISSING", "checks": checks}
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"- \*\*(?P<name>[^*]+)\*\*: `(?P<status>OK|CHECK)`", text):
        checks[m.group("name")] = m.group("status")
    gate = "OK" if checks and all(v == "OK" for v in checks.values()) else "CHECK"
    return {"gate_status": gate, "checks": checks}


def extract_status(spec: HarnessSpec, data: Dict[str, Any], md_path: Path) -> Tuple[str, Dict[str, Any]]:
    if spec.key == "readiness":
        parsed = parse_readiness_md(md_path)
        return parsed.get("gate_status", "MISSING"), parsed
    summary = data.get("summary") if isinstance(data.get("summary"), dict) else data
    if not isinstance(summary, dict):
        return "MISSING", {}
    if spec.key == "control_eval":
        status = "OK" if summary.get("failed") == 0 and summary.get("total", 0) else "REVIEW"
        return status, summary
    status = str(summary.get("gate_status") or summary.get("status") or "UNKNOWN")
    return status, summary


def classify_status(spec: HarnessSpec, status: str) -> str:
    if status in spec.ok_statuses:
        return "OK"
    if status in spec.review_statuses:
        return "REVIEW"
    if status in {"MISSING", "UNKNOWN"}:
        return "MISSING"
    return "FAIL"


def classify_harness_evidence_status(status: str) -> str:
    """Map adapter OK/REVIEW/BLOCKED/RECOVER into dashboard classes."""

    if status == "OK":
        return "OK"
    if status in {"REVIEW", "RECOVER"}:
        return "REVIEW"
    if status == "BLOCKED":
        return "FAIL"
    if status in {"MISSING", "UNKNOWN"}:
        return "MISSING"
    return "FAIL"


def run_harness_evidence_adapter() -> Dict[str, Any]:
    """Run the dry-run harness evidence adapter and normalize its status.

    This remains read-only: it passes an explicit report path to the adapter,
    parses stdout JSON, and never changes Gateway, cron, tool permissions, or
    live enforcement.
    """

    script_path = SCRIPTS / "hq_harness_dashboard_adapter.py"
    report_path = REPORTS / HARNESS_EVIDENCE_REPORT
    if not script_path.exists():
        status = "MISSING"
        summary: Dict[str, Any] = {
            "status": status,
            "reason": f"adapter script missing: {script_path}",
            "next_action": "restore the dry-run harness dashboard adapter before using evidence status",
        }
        return {
            "title": "Harness evidence dashboard adapter",
            "script": str(script_path),
            "return_code": 127,
            "status": status,
            "classification": classify_harness_evidence_status(status),
            "summary": summary,
            "json_report": str(report_path),
            "md_report": None,
            "output_tail": "adapter script missing",
        }

    rc, out = run_cmd([str(PYTHON), str(script_path), "--report", str(report_path), "--json"], timeout=120)
    try:
        summary = json.loads(out) if out else {}
    except json.JSONDecodeError as exc:
        summary = {
            "status": "RECOVER",
            "reason": f"adapter emitted malformed JSON: {exc.msg}",
            "next_action": "inspect adapter output before using evidence status",
        }
    if not isinstance(summary, dict):
        summary = {
            "status": "REVIEW",
            "reason": "adapter JSON root must be an object",
            "next_action": "inspect adapter output before using evidence status",
        }
    status = str(summary.get("status") or "UNKNOWN")
    return {
        "title": "Harness evidence dashboard adapter",
        "script": str(script_path),
        "return_code": rc,
        "status": status,
        "classification": classify_harness_evidence_status(status),
        "summary": summary,
        "json_report": str(report_path),
        "md_report": None,
        "output_tail": out[-1200:],
    }


def run_harnesses() -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for spec in HARNESS_SPECS:
        script_path = SCRIPTS / spec.script
        rc = 127
        out = "script missing"
        if script_path.exists():
            rc, out = run_cmd([str(PYTHON), str(script_path)], timeout=300)
        json_path = REPORTS / spec.json_report if spec.json_report else None
        md_path = REPORTS / spec.md_report if spec.md_report else REPORTS / "missing.md"
        data = read_json(json_path) if json_path else {}
        status, summary = extract_status(spec, data, md_path)
        classification = classify_status(spec, status)
        results[spec.key] = {
            "title": spec.title,
            "script": str(script_path),
            "return_code": rc,
            "status": status,
            "classification": classification,
            "summary": summary,
            "json_report": str(json_path) if json_path else None,
            "md_report": str(md_path) if md_path else None,
            "output_tail": out[-1200:],
        }
    results["harness_evidence_dashboard"] = run_harness_evidence_adapter()
    return results


def iter_log_lines(max_lines_per_file: int = 3000) -> List[str]:
    lines: List[str] = []
    for name in ["gateway.log", "agent.log", "errors.log"]:
        p = LOGS / name
        if not p.exists():
            continue
        try:
            tail = p.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines_per_file:]
        except Exception:
            continue
        for line in tail:
            lines.append(f"[{name}] {line}")
    return lines


def _ts(line: str) -> str:
    m = TS_RE.search(line)
    return m.group("ts") if m else ""


def sanitize(line: str) -> str:
    # Keep dashboard useful without leaking secret-shaped material.
    line = re.sub(r"(?i)(token|api[_ -]?key|password|secret)(\s*[:=]\s*)\S+", r"\1\2<redacted>", line)
    line = re.sub(r"(?<![A-Za-z0-9])(sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{12,}|xox[baprs]-[A-Za-z0-9-]{12,})", "<secret-like-redacted>", line)
    return line[-500:]


def collect_approval_events() -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    request_count = approved_count = denied_count = hardline_count = blocked_count = send_fail_count = 0
    slash_approve_count = slash_deny_count = 0
    for line in iter_log_lines():
        event_type = None
        detail: Dict[str, Any] = {}
        if APPROVAL_REQUEST_RE.search(line):
            event_type = "approval_request_or_prompt"
            request_count += 1
        if m := APPROVED_RE.search(line):
            event_type = "approved"
            approved_count += int(m.group("count") or 1)
            detail["count"] = int(m.group("count") or 1)
            detail["mode"] = m.group("mode") or "once"
        if m := DENIED_RE.search(line):
            event_type = "denied"
            denied_count += int(m.group("count") or 1)
            detail["count"] = int(m.group("count") or 1)
        if FAILED_APPROVAL_SEND_RE.search(line):
            event_type = "approval_send_warning"
            send_fail_count += 1
        if HARDLINE_RE.search(line):
            event_type = "hardline_block"
            hardline_count += 1
        elif BLOCKED_RE.search(line):
            event_type = event_type or "blocked_or_denied"
            blocked_count += 1
        if m := SLASH_APPROVE_RE.search(line):
            event_type = "slash_approve_invoked"
            slash_approve_count += 1
            detail.update({"user": m.group("user"), "uid": m.group("uid"), "args": (m.group("args") or "").strip()})
        if m := SLASH_DENY_RE.search(line):
            event_type = "slash_deny_invoked"
            slash_deny_count += 1
            detail.update({"user": m.group("user"), "uid": m.group("uid"), "args": (m.group("args") or "").strip()})
        if event_type:
            events.append({
                "timestamp": _ts(line),
                "type": event_type,
                "detail": detail,
                "evidence": sanitize(line),
            })

    # Best-effort only. Button-based successful prompt sends are intentionally
    # not always logged, and live pending queues are process-local to Gateway.
    unresolved_estimate = max(0, request_count - approved_count - denied_count - send_fail_count)
    return {
        "live_pending_visibility": "not_available_cross_process; use /approve or /deny in Discord for live queue",
        "best_effort_unresolved_from_logs": unresolved_estimate,
        "request_or_prompt_events": request_count,
        "approved_count": approved_count,
        "denied_count": denied_count,
        "slash_approve_invocations": slash_approve_count,
        "slash_deny_invocations": slash_deny_count,
        "approval_send_warnings": send_fail_count,
        "hardline_blocks": hardline_count,
        "blocked_or_denied_events": blocked_count,
        "recent_events": events[-30:],
    }


def overall_status(harnesses: Dict[str, Any], approvals: Dict[str, Any]) -> str:
    classes = [h.get("classification") for h in harnesses.values()]
    if any(c in {"FAIL", "MISSING"} for c in classes):
        return "FAIL"
    if approvals.get("best_effort_unresolved_from_logs", 0) > 0:
        return "REVIEW_APPROVALS"
    if any(c == "REVIEW" for c in classes):
        return "OK_WITH_REVIEW"
    return "OK"


def write_reports(harnesses: Dict[str, Any], approvals: Dict[str, Any]) -> Dict[str, Any]:
    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "policy_version": POLICY_VERSION,
        "overall_status": overall_status(harnesses, approvals),
        "harness_counts": {
            "ok": sum(1 for h in harnesses.values() if h.get("classification") == "OK"),
            "review": sum(1 for h in harnesses.values() if h.get("classification") == "REVIEW"),
            "fail": sum(1 for h in harnesses.values() if h.get("classification") == "FAIL"),
            "missing": sum(1 for h in harnesses.values() if h.get("classification") == "MISSING"),
        },
        "approval_best_effort_unresolved": approvals.get("best_effort_unresolved_from_logs", 0),
    }
    payload = {"summary": summary, "harnesses": harnesses, "approvals": approvals}
    DASHBOARD_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: List[str] = [
        "# HQ Health Dashboard",
        "",
        f"Generated: `{summary['generated_at']}`",
        f"Policy version: `{POLICY_VERSION}`",
        f"Overall status: `{summary['overall_status']}`",
        "",
        "## Harness Summary",
        "",
        "| Harness | Status | Class | Return code | Report |",
        "|---|---:|---:|---:|---|",
    ]
    for key, h in harnesses.items():
        report = h.get("md_report") or h.get("json_report") or ""
        lines.append(f"| {h['title']} | `{h['status']}` | `{h['classification']}` | `{h['return_code']}` | `{report}` |")

    lines.extend([
        "",
        "## Approval Requests / 승인 요청",
        "",
        "Live pending approval queues are stored inside the running Gateway process, so this dashboard cannot directly inspect the in-memory queue from a separate process. It provides a best-effort list from logs and points the user to `/approve` or `/deny` for live resolution.",
        "",
        f"- Best-effort unresolved from logs: `{approvals.get('best_effort_unresolved_from_logs')}`",
        f"- Request/prompt events seen: `{approvals.get('request_or_prompt_events')}`",
        f"- Approved count: `{approvals.get('approved_count')}`",
        f"- Denied count: `{approvals.get('denied_count')}`",
        f"- Slash /approve invocations: `{approvals.get('slash_approve_invocations')}`",
        f"- Slash /deny invocations: `{approvals.get('slash_deny_invocations')}`",
        f"- Approval send warnings: `{approvals.get('approval_send_warnings')}`",
        f"- Hardline blocks: `{approvals.get('hardline_blocks')}`",
        "",
        "### Recent approval-related events",
        "",
    ])
    recent = approvals.get("recent_events") or []
    if not recent:
        lines.append("No recent approval-related events found in scanned logs.")
    else:
        for ev in recent[-15:]:
            lines.extend([
                f"#### {ev.get('timestamp') or '(no timestamp)'} — `{ev.get('type')}`",
                "",
                f"- Detail: `{json.dumps(ev.get('detail') or {}, ensure_ascii=False)}`",
                f"- Evidence: `{ev.get('evidence')}`",
                "",
            ])

    lines.extend([
        "## Gate Interpretation",
        "",
        "- `OK`: passed or safe default.",
        "- `OK_WITH_REVIEW`: no blocker, but at least one harness has review notes.",
        "- `REVIEW_APPROVALS`: approval logs suggest unresolved approval prompts; user should check Discord and use `/approve` or `/deny`.",
        "- `FAIL`/`MISSING`: fix before expanding autonomy.",
        "",
    ])
    DASHBOARD_MD.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> int:
    harnesses = run_harnesses()
    approvals = collect_approval_events()
    payload = write_reports(harnesses, approvals)
    print(DASHBOARD_MD)
    print(json.dumps(payload["summary"], ensure_ascii=False))
    status = payload["summary"]["overall_status"]
    return 0 if status in {"OK", "OK_WITH_REVIEW"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
